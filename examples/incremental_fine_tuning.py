"""Interactive incremental correction with CTC images and segmentation masks.

Quick start from a checkout (Python >= 3.11, working Gurobi licence required)::

    uv sync --extra dev --extra demo
    uv run python examples/incremental_fine_tuning.py /data/Fluo-C2DL-Huh7

Download and unzip a training dataset from https://celltrackingchallenge.net/datasets/
first. By default, sequence 01's TRA masks are used as detections for a small
10-frame demo. No ground-truth links or oracle labels are read. Use
``--labels-dir`` for your own complete segmentations; sparse SEG annotations are
not suitable. ``--model`` accepts a registered model name or a TorchScript path.

Napari displays lazy images, segmentation masks, tracks and proposed endpoints.
Press y/n to label, s to skip, t to switch endpoints, b to submit a partial
batch, or q to submit and finish correction. Use Napari's time/z sliders to
inspect cells. Close the viewer to export the candidate graph and tracks.

Adaptation points: replace ``load_ctc`` for your data or ``NapariCorrectionUI``
for your viewer and annotation callbacks. ``CorrectionSession`` owns the graph,
model and annotations independently of the UI. See examples/README.md for an
asynchronous UI-loop sketch and details of the paper's correction protocol.
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import dask.array as da
import numpy as np
import polars as pl
import tifffile
import torch
import tracksdata as td
from dask import delayed

from hoct import load_model, predict
from hoct.correction import fit_from_labels, label_edge
from hoct.features import create_graph
from hoct.inference import EdgeModel
from hoct.tracking import ILPSolverConfig

EDGE_ID = td.DEFAULT_ATTR_KEYS.EDGE_ID
SOURCE = td.DEFAULT_ATTR_KEYS.EDGE_SOURCE
TARGET = td.DEFAULT_ATTR_KEYS.EDGE_TARGET
NODE_ID = td.DEFAULT_ATTR_KEYS.NODE_ID
SOLUTION = td.DEFAULT_ATTR_KEYS.SOLUTION
LABEL_MASK = "is_labeled"
LABEL = "is_correct"


def load_ctc(
    root: Path, sequence: str = "01", labels_dir: Path | None = None, n_frames: int = 10
) -> tuple[da.Array, da.Array]:
    """Load the first n_frames of a local CTC sequence as (T, [Z,] Y, X).

    TIFF names must end in their numeric frame index, e.g. t000.tif and
    man_track000.tif. Require matching indices, consecutive frames, equal
    shapes, and integer instance labels (0 is background).
    The TRA masks supply detections only; man_track.txt is never loaded.
    Only TIFF headers are read here; pixels load lazily, one frame per chunk.
    Negative instance labels are rejected when each mask frame is computed.
    """
    if n_frames < 5:
        raise ValueError("Use at least 5 frames for the default temporal window")
    labels_dir = labels_dir if labels_dir is not None else root / f"{sequence}_GT" / "TRA"

    def frame_files(directory: Path) -> dict[int, Path]:
        result = {}
        for path in sorted(directory.glob("*.tif*")):
            match = re.search(r"(\d+)$", path.stem)
            if match is None:
                raise ValueError(f"TIFF name must end with a frame index: {path}")
            index = int(match[1])
            if index in result:
                raise ValueError(f"Duplicate frame {index} in {directory}")
            result[index] = path
        if not result:
            raise FileNotFoundError(f"No frame TIFFs in {directory}")
        return result

    image_files = frame_files(root / sequence)
    label_files = frame_files(labels_dir)
    indices = sorted(image_files)[:n_frames]
    if len(indices) < 5 or indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("Images must contain at least 5 consecutive frames")
    missing = set(indices) - set(label_files)
    if missing:
        raise ValueError(f"Missing segmentation frames: {sorted(missing)}")

    def lazy_stack(files: dict[int, Path], *, masks: bool = False) -> da.Array:
        frames = []
        for index in indices:
            path = files[index]
            with tifffile.TiffFile(path) as tif:
                shape, dtype = tif.series[0].shape, tif.series[0].dtype
            if masks and not np.issubdtype(dtype, np.integer):
                raise ValueError("Segmentations must be integer instance labels")
            frames.append(da.from_delayed(delayed(_read_frame)(path, masks), shape=shape, dtype=dtype))
        return da.stack(frames)

    images = lazy_stack(image_files)
    labels = lazy_stack(label_files, masks=True)
    if images.shape != labels.shape or labels.ndim not in (3, 4):
        raise ValueError("Images and labels must have equal shape (T, [Z,] Y, X)")
    return images, labels


def _read_frame(path: Path, masks: bool) -> np.ndarray:
    """Read one delayed TIFF frame and validate mask values at compute time."""
    frame = tifffile.imread(path)
    if masks and np.any(frame < 0):
        raise ValueError(f"Segmentations must be nonnegative: {path}")
    return frame


def sample_uncertainty(graph: td.graph.BaseGraph, n: int) -> pl.DataFrame:
    """Select up to n/2 unlabeled links per solver group, nearest score 0.5.

    Balance selected/unselected links, with deterministic edge-ID tie breaking.
    A depleted group produces a smaller batch; skipped links can be proposed again.
    """
    if n < 2 or n % 2:
        raise ValueError("n must be an even integer >= 2")
    candidates = graph.edge_attrs(attr_keys=[SOURCE, TARGET, "similarity", SOLUTION, LABEL_MASK]).filter(
        ~pl.col(LABEL_MASK)
    )
    ranked = candidates.sort((pl.col("similarity") - 0.5).abs(), EDGE_ID)
    return pl.concat([ranked.filter(pl.col(SOLUTION) == selected).head(n // 2) for selected in (True, False)])


def apply_annotations(graph: td.graph.BaseGraph, annotations: dict[int, bool]) -> None:
    """Validate a batch before writing explicit and implied labels.

    A confirmed parent makes every other incoming candidate incorrect via
    label_edge, including candidates at other temporal gaps. Mark those implied
    negatives labeled too. Reject conflicting parents within/across batches;
    daughters may share a source (division). Unknown IDs and non-booleans are
    errors. To revise a confirmed parent, a UI must explicitly clear the labels
    for that target first, including its implied negatives, then resubmit.
    """
    edges = graph.edge_attrs(attr_keys=[SOURCE, TARGET, LABEL_MASK, LABEL])
    rows = {row[EDGE_ID]: row for row in edges.to_dicts()}
    confirmed = {row[TARGET]: row[EDGE_ID] for row in rows.values() if row[LABEL_MASK] and row[LABEL]}
    for edge_id, value in annotations.items():
        if edge_id not in rows:
            raise ValueError(f"Unknown edge ID: {edge_id}")
        if type(value) is not bool:
            raise ValueError("Annotations must be Python booleans")
        row = rows[edge_id]
        if row[LABEL_MASK] and row[LABEL] != value:
            raise ValueError(f"Edge {edge_id} conflicts with an existing label")
        if value:
            if row[TARGET] in confirmed and confirmed[row[TARGET]] != edge_id:
                raise ValueError(f"Cannot confirm two parents for target {row[TARGET]}")
            confirmed[row[TARGET]] = edge_id

    for edge_id, value in annotations.items():
        row = rows[edge_id]
        label_edge(graph, row[SOURCE], row[TARGET], attr_key=LABEL, value=value)
        affected_ids = edges.filter(pl.col(TARGET) == row[TARGET])[EDGE_ID].to_list() if value else [edge_id]
        graph.update_edge_attrs(edge_ids=affected_ids, attrs={LABEL_MASK: True})


class CorrectionSession:
    """UI-independent state: initialize, propose, submit, display, repeat.

    Keep ``graph`` (all candidates and annotations) distinct from ``solution``
    (selected tracks). predict updates scores/selection in graph in place.
    Pass the same window/augmentation/tiling settings to prediction and fitting.
    For large data, extend predict_kwargs with a tracksdata TilingScheme.
    """

    def __init__(self, graph: td.graph.BaseGraph, model: EdgeModel, test_time_augs: int = 0) -> None:
        self.graph = graph
        self.model = model.eval()
        self.round = 0
        self.predict_kwargs = {"window_size": 5, "test_time_augs": test_time_augs}
        # Single-pass settings from the starting template; tune for your data.
        self.solver = ILPSolverConfig(
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=0.25,
            delta_t_weight=0.5,
            edge_bias=0.5,
            node_weight=-10.0,
            timeout=600.0,
            tracklet_solver=False,
        )
        for key in (LABEL_MASK, LABEL):
            if key not in graph.edge_attr_keys():
                graph.add_edge_attr_key(key, pl.Boolean, False)
        self.solution = self.refresh()

    def refresh(self) -> td.graph.InMemoryGraph | None:
        """Rescore all candidates and rerun the ILP with the current model."""
        self.solution = predict(self.model, graph=self.graph, solver_config=self.solver, **self.predict_kwargs)
        return self.solution

    def propose(self, n: int = 20) -> pl.DataFrame:
        """Return links to review, with source/target t,z,y,x in image pixels.

        Coordinate t is the zero-based index in the loaded stack, even if the
        original TIFF numbering starts later. z is 0 for 2D data.
        """
        edges = sample_uncertainty(self.graph, n)
        nodes = self.graph.node_attrs(attr_keys=[NODE_ID, "t", "z", "y", "x"])
        for endpoint in (SOURCE, TARGET):
            prefix = "source" if endpoint == SOURCE else "target"
            renamed = nodes.rename({NODE_ID: endpoint, **{c: f"{prefix}_{c}" for c in ("t", "z", "y", "x")}})
            edges = edges.join(renamed, on=endpoint, how="left", maintain_order="left")
        return edges

    def submit(self, proposals: pl.DataFrame, annotations: dict[int, bool]) -> bool:
        """Apply UI feedback; refit on all labels and refresh if both classes exist.

        Return whether fitting ran. An empty batch is a no-op. Use a single
        outstanding proposal batch per session to avoid stale UI submissions.
        Labels influence learned costs; they do not pin the ILP solution.
        """
        if not set(annotations).issubset(set(proposals[EDGE_ID].to_list())):
            raise ValueError("Annotate only the proposed edge IDs")
        if not annotations:
            return False
        apply_annotations(self.graph, annotations)
        self.round += 1
        labels = self.graph.edge_attrs(attr_keys=[LABEL_MASK, LABEL]).filter(pl.col(LABEL_MASK))
        if labels[LABEL].n_unique() < 2:
            print("Need correct and incorrect links before fitting; collect another batch.")
            return False
        self.model = fit_from_labels(
            graph=self.graph,
            model=self.model,
            label_mask_key=LABEL_MASK,
            label_key=LABEL,
            **self.predict_kwargs,
        ).eval()
        self.refresh()
        print(
            f"Round {self.round}: {len(annotations)} judgments, {len(labels)} total labels including implied negatives."
        )
        return True

    def save(self, directory: Path) -> None:
        """Export candidates with annotations and selected tracks to a new directory.

        To resume, load candidates.geff and the original model, construct a
        session, then refit from its saved labels. The adapted model remains
        available as session.model; this export saves graphs, not model weights.
        """
        directory.mkdir(parents=True, exist_ok=False)
        self.graph.to_geff(str(directory / "candidates.geff"))
        if self.solution is not None:
            self.solution.to_geff(str(directory / "tracks.geff"))


class NapariCorrectionUI:
    """Napari adapter for a single outstanding proposal batch.

    y/n/s label or skip the current link; t switches between its endpoints.
    b submits a partial batch and q submits then ends correction. Full batches
    submit automatically. Fitting runs on a worker; viewer updates stay on the
    Qt thread and annotation keys are disabled while a submission is running.
    Closing the viewer preserves pending judgments without starting a new fit.
    """

    def __init__(
        self,
        viewer,
        session: CorrectionSession,
        images: da.Array,
        labels: da.Array,
        batch_size: int = 20,
        rounds: int = 20,
    ) -> None:
        self.viewer = viewer
        self.session = session
        self.batch_size = batch_size
        self.rounds = rounds
        self.ndim = images.ndim
        self.closed = False
        self.busy = False
        self.finished = False
        self.worker = None
        self.annotations: dict[int, bool] = {}
        self.rows: list[dict] = []
        self.index = 0
        self.endpoint = 0
        viewer.add_image(images, name="images")
        viewer.add_labels(labels, name="segmentation")
        self.points = viewer.add_points(
            np.empty((0, self.ndim)),
            ndim=self.ndim,
            name="proposed link",
            size=12,
            face_color="transparent",
            border_color="red",
        )
        self.tracks = None
        self.show_solution()
        # Callbacks close over this adapter, not the model or graph.
        viewer.bind_key("y", lambda v: self.decide(True), overwrite=True)
        viewer.bind_key("n", lambda v: self.decide(False), overwrite=True)
        viewer.bind_key("s", lambda v: self.decide(None), overwrite=True)
        viewer.bind_key("t", lambda v: self.navigate(), overwrite=True)
        viewer.bind_key("b", lambda v: self.submit(), overwrite=True)
        viewer.bind_key("q", lambda v: self.submit(finish=True), overwrite=True)
        self.next_batch()

    def show_solution(self) -> None:
        """Refresh the tracks layer after inference using image dimensionality."""
        if self.session.solution is None:
            return
        shape = tuple(self.viewer.layers["images"].data.shape)
        data, lineage = td.functional.to_napari_format(self.session.solution, shape=shape)
        if data.is_empty():
            if self.tracks is not None:
                self.viewer.layers.remove(self.tracks)
                self.tracks = None
            return
        if self.tracks is None:
            self.tracks = self.viewer.add_tracks(data.to_numpy(), graph=lineage, name="tracks")
        else:
            # Clear old lineage first: track IDs can change after fitting.
            self.tracks.graph = {}
            self.tracks.data = data.to_numpy()
            self.tracks.graph = lineage

    def next_batch(self) -> None:
        self.annotations = {}
        if self.session.round >= self.rounds:
            self.finish()
            return
        self.proposals = self.session.propose(self.batch_size)
        self.rows = self.proposals.to_dicts()
        self.index = 0
        if not self.rows:
            self.finish()
            return
        self.show_edge()

    def show_edge(self) -> None:
        row = self.rows[self.index]
        columns = ("t", "z", "y", "x") if self.ndim == 4 else ("t", "y", "x")
        self.points.data = np.array([[row[f"{prefix}_{c}"] for c in columns] for prefix in ("source", "target")])
        self.endpoint = 0
        self.navigate(toggle=False)
        self.viewer.status = (
            f"Round {self.session.round + 1}, link {self.index + 1}/{len(self.rows)}: "
            f"{row[SOURCE]} -> {row[TARGET]}, score={row['similarity']:.3f}, selected={row[SOLUTION]} | "
            "y/n: label, s: skip, t: other endpoint, b: submit, q: finish"
        )

    def navigate(self, toggle: bool = True) -> None:
        """Jump to source or target time/z; retain normal pan/zoom controls."""
        if self.busy or self.finished or not self.rows:
            return
        if toggle:
            self.endpoint = 1 - self.endpoint
        point = self.points.data[self.endpoint]
        for axis in range(self.ndim - 2):
            self.viewer.dims.set_point(axis, round(point[axis]))
        self.viewer.camera.center = tuple(point[-2:])

    def decide(self, value: bool | None) -> None:
        if self.busy or self.finished:
            return
        row = self.rows[self.index]
        if value is not None:
            self.annotations[row[EDGE_ID]] = value
        self.index += 1
        # A confirmed parent implies negatives for competing proposals.
        confirmed = {r[TARGET] for r in self.rows if self.annotations.get(r[EDGE_ID]) is True}
        while self.index < len(self.rows) and self.rows[self.index][TARGET] in confirmed:
            self.index += 1
        if self.index == len(self.rows):
            self.submit()
        else:
            self.show_edge()

    def submit(self, finish: bool = False) -> None:
        """Serialize session updates on a worker; signal callbacks update Napari."""
        if self.busy or self.finished:
            return
        if not self.annotations:
            # All skipped: stop instead of immediately proposing the same batch.
            self.finish()
            return
        from napari.qt.threading import thread_worker

        self.busy = True
        self.viewer.status = "Fitting corrections and refreshing tracking..."
        worker = thread_worker(self.session.submit, ignore_errors=True)(self.proposals, dict(self.annotations))
        self.worker = worker  # Keep alive until the worker finishes.

        def returned(fitted):
            self.busy = False
            if self.closed:
                return
            self.show_solution()
            if finish:
                self.finish()
            else:
                self.next_batch()

        def errored(error):
            from napari.utils.notifications import show_error

            self.busy = False
            if not self.closed:
                self.finish()
                show_error(str(error))

        worker.returned.connect(returned)
        worker.errored.connect(errored)
        worker.start()

    def finish(self) -> None:
        self.finished = True
        self.points.data = np.empty((0, self.ndim))
        self.viewer.status = "Correction finished. Inspect tracks, then close the viewer to export."


@click.command()
@click.argument("ctc_root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sequence", type=click.Choice(["01", "02"]), default="01", show_default=True)
@click.option(
    "--labels-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Use your segmentation TIFF directory instead of TRA masks.",
)
@click.option(
    "--frames",
    type=click.IntRange(min=5),
    default=10,
    show_default=True,
    help="First N frames to include in the lazy stack.",
)
@click.option("--model", default="general_v1", show_default=True, help="Registered model name or TorchScript .pt.")
@click.option("--rounds", type=click.IntRange(min=1), default=20, show_default=True)
@click.option("--batch-size", type=click.IntRange(min=2), default=20, show_default=True)
@click.option("--test-time-augs", type=click.IntRange(min=0), default=0, show_default=True)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="incremental-demo",
    show_default=True,
    help="New output directory for candidates and selected tracks.",
)
def main(
    ctc_root: Path,
    sequence: str,
    labels_dir: Path | None,
    frames: int,
    model: str,
    rounds: int,
    batch_size: int,
    test_time_augs: int,
    output: Path,
) -> None:
    """Review links and fine-tune HOCT on a local CTC_ROOT sequence in Napari."""
    if batch_size % 2:
        raise click.BadParameter("must be even", param_hint="--batch-size")
    if output.exists():
        raise click.BadParameter("directory already exists; choose a new path", param_hint="--output")
    import napari
    from napari.qt.threading import WorkerBase

    images, labels = load_ctc(ctc_root, sequence, labels_dir, frames)
    graph = create_graph(labels, images=images, distance_threshold=300.0, n_neighbors=5, delta_t=3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session = CorrectionSession(graph, load_model(model, device=device), test_time_augs)
    viewer = napari.Viewer()
    ui = NapariCorrectionUI(viewer, session, images, labels, batch_size, rounds)
    napari.run()
    ui.closed = True
    # Finish an in-flight fit before exporting; never race the session's graph.
    WorkerBase.await_workers()
    if ui.annotations and not ui.busy:
        apply_annotations(session.graph, ui.annotations)
    session.save(output)
    click.echo(f"Saved candidates and tracks to {output}")


if __name__ == "__main__":
    main()
