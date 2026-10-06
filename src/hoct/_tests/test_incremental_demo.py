"""Exercise the demo's feedback boundary and its full synthetic correction loop."""

import importlib.util
from pathlib import Path

import dask.array as da
import numpy as np
import polars as pl
import pytest
import tifffile
import tracksdata as td

from hoct._tests.test_correction import FakeEdgeModel
from hoct.correction import ProbedModel
from hoct.features import create_graph

_SPEC = importlib.util.spec_from_file_location(
    "incremental_demo", Path(__file__).resolve().parents[3] / "examples" / "incremental_fine_tuning.py"
)
demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(demo)


@pytest.fixture
def graph():
    labels = np.zeros((5, 32, 32), dtype=np.uint16)
    labels[:, 3:8, 3:8] = 1
    labels[:, 20:25, 20:25] = 2
    result = create_graph(labels, distance_threshold=100, n_neighbors=5, delta_t=3)
    for key in (demo.LABEL_MASK, demo.LABEL):
        result.add_edge_attr_key(key, pl.Boolean, False)
    return result


def competitors(graph):
    edges = graph.edge_attrs(attr_keys=[])
    target = edges.group_by(demo.TARGET).len().filter(pl.col("len") > 1)[demo.TARGET][0]
    return edges.filter(pl.col(demo.TARGET) == target)[demo.EDGE_ID].to_list()


def test_confirmation_masks_competitors_and_rejects_conflicts_atomically(graph):
    incoming_ids = competitors(graph)
    first, second, *_ = incoming_ids
    before = graph.edge_attrs(attr_keys=[demo.LABEL_MASK, demo.LABEL])
    with pytest.raises(ValueError, match="two parents"):
        demo.apply_annotations(graph, {first: True, second: True})
    assert graph.edge_attrs(attr_keys=[demo.LABEL_MASK, demo.LABEL]).equals(before)

    demo.apply_annotations(graph, {first: True})
    rows = graph.edge_attrs(attr_keys=[demo.LABEL_MASK, demo.LABEL])
    incoming = rows.filter(pl.col(demo.EDGE_ID).is_in(incoming_ids))
    assert incoming[demo.LABEL_MASK].all()
    assert incoming[demo.LABEL].sum() == 1
    with pytest.raises(ValueError, match="existing label"):
        demo.apply_annotations(graph, {second: True})
    with pytest.raises(ValueError, match="Unknown edge"):
        demo.apply_annotations(graph, {999999: False})
    with pytest.raises(ValueError, match="booleans"):
        demo.apply_annotations(graph, {second: 0})


def test_balanced_sampling_excludes_implied_labels_and_breaks_ties(graph):
    edges = graph.edge_attrs(attr_keys=[])
    ids = edges[demo.EDGE_ID].to_list()
    graph.add_edge_attr_key("similarity", pl.Float32, 0.5)
    graph.add_edge_attr_key(demo.SOLUTION, pl.Boolean, False)
    graph.update_edge_attrs(edge_ids=ids[::2], attrs={demo.SOLUTION: True})
    demo.apply_annotations(graph, {competitors(graph)[0]: True})
    batch = demo.sample_uncertainty(graph, 4)
    assert len(batch) == 4
    assert batch[demo.SOLUTION].sum() == 2
    assert not batch[demo.LABEL_MASK].any()
    for selected in (True, False):
        group = batch.filter(pl.col(demo.SOLUTION) == selected)[demo.EDGE_ID].to_list()
        assert group == sorted(group)


def test_session_fit_predict_export_and_accumulate(graph, tmp_path):
    session = demo.CorrectionSession(graph, FakeEdgeModel())
    batch = session.propose(100)
    assert {"source_t", "target_z", "source_x"}.issubset(batch.columns)
    first = competitors(graph)[0]
    assert session.submit(batch, {first: True})
    assert isinstance(session.model, ProbedModel)
    backbone = session.model._edge_model
    remaining = session.propose(100)
    assert first not in remaining[demo.EDGE_ID].to_list()
    assert session.submit(remaining, {remaining[demo.EDGE_ID][0]: False})
    assert session.model._edge_model is backbone  # no nested probes / warm start
    assert session.round == 2
    with pytest.raises(ValueError, match="only the proposed"):
        session.submit(remaining, {first: True})
    session.save(tmp_path / "result")
    saved, _ = td.graph.InMemoryGraph.from_geff(str(tmp_path / "result" / "candidates.geff"))
    assert saved.edge_attrs(attr_keys=[demo.LABEL_MASK])[demo.LABEL_MASK].sum() > 1
    assert (tmp_path / "result" / "tracks.geff").exists()


def test_ctc_loader_matches_numeric_frames_and_rejects_missing_masks(tmp_path, monkeypatch):
    images = tmp_path / "01"
    masks = tmp_path / "01_GT" / "TRA"
    images.mkdir()
    masks.mkdir(parents=True)
    for index in range(5, 10):
        tifffile.imwrite(images / f"t{index:03}.tif", np.full((16, 16), index, dtype=np.uint16))
        tifffile.imwrite(masks / f"man_track{index:03}.tif", np.ones((16, 16), dtype=np.uint16))
    reads = []
    original_read = tifffile.imread

    def read(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(tifffile, "imread", read)
    stack, labels = demo.load_ctc(tmp_path, n_frames=5)
    assert isinstance(stack, da.Array) and isinstance(labels, da.Array)
    assert not reads  # Header inspection does not decode any pixel arrays.
    assert stack.chunks[0] == (1,) * 5
    assert stack.shape == labels.shape == (5, 16, 16)
    assert stack[:, 0, 0].compute().tolist() == list(range(5, 10))
    (masks / "man_track007.tif").unlink()
    with pytest.raises(ValueError, match="Missing segmentation"):
        demo.load_ctc(tmp_path, n_frames=5)


@pytest.mark.parametrize("volume", [False, True])
def test_napari_navigation_and_worker_submission(graph, volume):
    napari = pytest.importorskip("napari")
    from qtpy.QtCore import QEventLoop, QTimer
    from qtpy.QtWidgets import QApplication

    session = demo.CorrectionSession(graph, FakeEdgeModel())
    shape = (5, 1, 32, 32) if volume else (5, 32, 32)
    app = QApplication.instance() or QApplication([])
    viewer = napari.components.ViewerModel()
    try:
        ui = demo.NapariCorrectionUI(viewer, session, da.zeros(shape), da.zeros(shape, dtype=np.uint16))
        assert isinstance(viewer.layers["images"].data, da.Array)
        assert "tracks" in viewer.layers
        row = ui.rows[0]
        assert viewer.dims.current_step[0] == row["source_t"]
        ui.navigate()
        assert viewer.dims.current_step[0] == row["target_t"]
        assert ui.points.data.shape == (2, len(shape))
        ui.decide(True)
        ui.submit(finish=True)
        assert ui.busy
        loop = QEventLoop()
        ui.worker.returned.connect(loop.quit)
        ui.worker.errored.connect(loop.quit)
        QTimer.singleShot(10000, loop.quit)
        loop.exec()
        app.processEvents()
        assert not ui.busy
        assert ui.finished
        assert session.round == 1
        assert isinstance(session.model, ProbedModel)
        assert viewer.layers["tracks"].data.shape[1] == len(shape) + 1
    finally:
        viewer.layers.clear()


def test_click_cli_validates_options_before_starting_gui(tmp_path):
    from click.testing import CliRunner

    runner = CliRunner()
    assert runner.invoke(demo.main, ["--help"]).exit_code == 0
    result = runner.invoke(demo.main, [str(tmp_path), "--batch-size", "3"])
    assert result.exit_code == 2 and "must be even" in result.output
    result = runner.invoke(demo.main, [str(tmp_path), "--frames", "4"])
    assert result.exit_code == 2
    result = runner.invoke(demo.main, [str(tmp_path), "--output", str(tmp_path)])
    assert result.exit_code == 2 and "already exists" in result.output
