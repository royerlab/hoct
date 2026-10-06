# Tracking

## From images and segmentation

```bash
hoct track images.tif segmentation.tif -o tracks.geff
```

Both inputs may be a whole-time-series TIFF, a Zarr/OME-Zarr store, or a folder
of frame files sorted alphabetically. Use the same layout for both inputs.
Images and segmentation must have matching shapes. Python arrays use
`(T, Y, X)` or `(T, Z, Y, X)`; segmentation values are integer instance IDs,
with zero as background.

For a CTC sequence with your own segmentation:

```bash
hoct track /data/Fluo-C2DL-Huh7/01 /data/segmentation/01 -o tracks.geff
```

Use masks for every frame. Sparse CTC SEG annotations do not provide a complete
segmentation time series.

## Useful options

| Option | Purpose |
| --- | --- |
| `--model ctc_v0` | Select registered weights, or pass a local `.pt` path |
| `--device cpu` | Select the inference device |
| `--max-distance 300 --neighbors 5 --max-dt 3` | Control candidate links |
| `--window 5` | Set the temporal inference window |
| `--tile auto` | Automatically tile large candidate graphs; `on` and `off` also work |
| `--full-graph` | Keep all candidates and predicted attributes for later correction |
| `--config solver.yaml` | Use a custom solver configuration |
| `--overwrite` | Replace an existing output directory |

Generate an editable configuration from the current solver defaults:

```bash
hoct init-config -o solver.yaml
hoct track images.tif segmentation.tif -o tracks.geff --config solver.yaml
```

For physical voxel sizes, repeat `--scale` in axis order `t, [z,] y, x`:

```bash
hoct track images.tif segmentation.tif -o tracks.geff \
    --scale 1 --scale 2 --scale 0.5 --scale 0.5
```

The maximum candidate distance then uses physical spatial units. See
`hoct track --help` for all options.

## Outputs and existing graphs

GEFF is the default output. CTC export writes per-frame `maskNNN.tif` files
and `res_track.txt`:

```bash
hoct track images.tif segmentation.tif -o 01_RES --format ctc
```

To score and solve a full candidate GEFF with HOCT features:

```bash
hoct predict candidates.geff -o scored.geff
hoct predict candidates.geff --solution -o tracks.geff
```

A candidate graph contains all plausible links; a solution graph contains only
selected tracks. Preserve the candidate graph for
[incremental correction](correction.md).

## Python

```python
import numpy as np
import torch
from hoct import load_model, predict

images = np.load("images.npy")
labels = np.load("labels.npy")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device=device)
solution = predict(model, images=images, labels=labels)
solution.to_geff("tracks.geff")
```

Dask arrays are also accepted. To keep candidates and their updated scores,
create the graph explicitly:

```python
from hoct.features import create_graph

graph = create_graph(labels, images=images, distance_threshold=300, n_neighbors=5, delta_t=3)
solution = predict(model, graph=graph)
graph.to_geff("candidates.geff")
```

See the [API reference](api.md) for tiling, test-time augmentation, and solver
parameters, or the
[basic Napari example](https://github.com/royerlab/hoct/blob/main/examples/basic_tracking.py).
