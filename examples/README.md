# Incremental correction and fine tuning

[`incremental_fine_tuning.py`](https://github.com/royerlab/hoct/blob/main/examples/incremental_fine_tuning.py) is a runnable,
UI-independent starting point for the feedback loop in [HOCT, Section 4.5
and Appendix D](https://arxiv.org/pdf/2607.11754): predict tracks, review candidate
links, accumulate labels, fit a linear classification head on frozen edge
embeddings, and rerun tracking.

## Run the demo

From this checkout, install the dependencies (Python 3.11 or later):

```bash
uv sync --extra dev --extra demo
```

A working Gurobi licence is required by the tracking solver. Download and unzip
[a CTC training dataset](https://celltrackingchallenge.net/datasets/), for example
**Fluo-C2DL-Huh7**, then run:

```bash
uv run python examples/incremental_fine_tuning.py /data/Fluo-C2DL-Huh7 \
    --sequence 01 --frames 10 --output incremental-demo
```

Alternatively, this file can be used outside the checkout after installing
`pip install 'hoct[correction]' 'napari[pyqt6]' click`. The checkout's `demo`
extra installs Napari, a Qt backend, Click, and correction dependencies.
Use a desktop environment that can open a Qt window.
The model is downloaded and cached on first use. Pass `--model /path/model.pt`
to use your own TorchScript checkpoint, or `--model ctc_v0` for CTC weights.

The expected input layout is:

```text
Fluo-C2DL-Huh7/
  01/t000.tif, t001.tif, ...
  01_GT/TRA/man_track000.tif, man_track001.tif, ...
```

The TRA masks are used only as instance detections; `man_track.txt` and
cross-frame mask IDs are never used to provide correction labels. Some CTC TRA
masks are markers rather than complete segmentations, so this is a convenient
input example rather than a faithful segmentation benchmark. Supply your own
complete instance masks with `--labels-dir /data/segmentation`. Every loaded
image frame needs a matching mask frame. Sparse CTC SEG masks are unsuitable.
Images and masks are lazy Dask arrays with one frame per chunk. The loader reads
one sample frame per stack to infer shape and dtype via
`dask.array.image.imread`; graph construction and Napari request the remaining
pixels as needed.
`--frames` limits graph size and processing time. Graph features and the model
still consume memory; lazy image loading does not remove those costs.
Frames are indexed from zero in the graph and annotation UI.

Napari shows the image stack, instance segmentation, current tracks, and red
centroid markers for the proposed source and target. The viewer jumps to the
source first; inspect both endpoints before labeling:

- **y**: the source is the target's biological parent, including a division.
- **n**: the source is not the target's parent.
- **s**: skip an ambiguous link; it remains eligible for a future batch.
- **t**: jump between the source and target time/z coordinates.
- **b**: submit the judgments collected so far in the current batch.
- **q**: submit this batch and finish correction; keep the viewer open for inspection.

Use normal Napari time/z sliders, pan, and zoom to inspect context. Full batches
submit automatically. Annotation keys are disabled while a worker fits the head
and reruns tracking. The resulting tracks layer updates on the Qt thread.
After `--rounds` submissions, or a batch with no judgments, correction stops.
Close the viewer to export. Closing with a partial batch saves those labels
without starting another fit; exported tracks then reflect the last completed fit.
An in-flight worker finishes before export. Existing output directories are not overwritten.

The batch contains up to ten selected and ten unselected candidate links nearest
score 0.5. Either group may contain mistakes. Confirming a parent automatically
labels all competing incoming links incorrect, so some proposals are skipped and
the accumulated training-label count can exceed the number of manual judgments.
Two daughters may share the same parent.

## Connect your UI

`CorrectionSession` owns the candidate graph, current model, selected solution,
and annotation state. Its `propose` method returns a Polars table with stable
edge and node IDs, score, current selection, and both endpoints' `t,z,y,x`
coordinates in image pixels. Your UI can use this table to navigate to each link.
Return `{edge_id: True/False}` and omit skipped links. Keep one outstanding batch
per session and serialize submissions.

```python
from hoct import load_model
from hoct.features import create_graph
from examples.incremental_fine_tuning import CorrectionSession

# Adapt your loader: images/labels have shape (T, [Z,] Y, X).
graph = create_graph(
    labels, images=images, distance_threshold=300, n_neighbors=5, delta_t=3
)
session = CorrectionSession(graph, load_model(device="cpu"))
proposals = session.propose(20)

# Send proposals to your UI. Later, in its batch-submit callback:
def on_submit(annotations):
    global proposals
    fitted = session.submit(proposals, annotations)
    if fitted:
        # Replace with your viewer's refresh call; the solution may change globally.
        update_tracks_layer(session.solution)
    proposals = session.propose(20)
    show_annotation_batch(proposals)
```

`update_tracks_layer` and `show_annotation_batch` above are placeholders for your
application. The standalone script implements this pattern in `NapariCorrectionUI`: endpoint
navigation, keyboard judgments, one outstanding batch, worker submission,
and track-layer refresh. Replace that adapter to integrate your own controls.
Initial graph construction and tracking run before opening the viewer; later
fitting/inference run on a Napari `thread_worker`. Only returned/error signal
handlers update viewer layers. Do not mutate the session concurrently.
`td.functional.to_napari_format(session.solution, shape=images.shape)` converts
tracks to the displayed image dimensionality, including 2D data.

Adapt `load_ctc` for another dataset, or start with a full candidate GEFF:

```python
import tracksdata as td

graph, _ = td.graph.InMemoryGraph.from_geff("candidates.geff")
session = CorrectionSession(graph, load_model(device="cpu"))
```

A solution-only GEFF lacks the rejected candidates needed for correction. Keep
all HOCT input features on the candidate graph. `is_labeled` distinguishes
unknown links from known negatives in `is_correct`; both are Boolean attributes.
Existing annotations are preserved. Conflicting labels are rejected before a
batch is written. To revise an established parent, clear both attributes on all
incoming links for that target (including implied negatives), then submit the
replacement. This demo does not include an undo/history UI.

## Fitting and outputs

Each fit uses **all accumulated explicit and implied labels**, not just the most
recent batch. It waits for at least one positive and one negative label. The
backbone stays fixed; `fit_from_labels` initializes a fresh linear head from the
original backbone head each round and uses class-balanced binary cross-entropy,
L-BFGS (up to 500 iterations), adaptive L2 regularization (weight 1.0), and
consistency with the previous ILP solution (weight 0.25). Feedback changes the
learned costs; it does **not** enforce hard ILP constraints, so a confirmed link
is not guaranteed to be selected after solving.

The default uses no random test-time augmentation for a faster demo. Pass
`--test-time-augs 5 --rounds 20 --batch-size 20` for the paper's augmentation and
round settings. This CTC demo does not reproduce the paper's bacteria validation
experiment, oracle feedback, or reported metrics. It uses the single-pass solver
settings from the starting template; tune candidate generation and solver
parameters for your data. For large volumes, add the same `tiling_scheme` to
`session.predict_kwargs` so fitting and prediction use matching spatial windows.

At completion, `--output` contains:

- `candidates.geff`: all candidates, updated scores, selection and annotation attributes.
- `tracks.geff`: the selected tracking solution, if one was returned.

The adapted model remains accessible as `session.model` for further predictions.
The export saves graphs, not model weights. To continue from saved annotations,
load `candidates.geff` and the same original pretrained model, construct a session
(which initializes scores/selection), and call `fit_from_labels` with the saved
mask/label keys and session prediction settings before `session.refresh()`.
