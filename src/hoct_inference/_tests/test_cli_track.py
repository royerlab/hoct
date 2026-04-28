"""End-to-end tests for the ``hoct-inference track`` CLI command."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
import tracksdata as td
from typer.testing import CliRunner

pytest.importorskip("bioio")  # CLI requires the bioio extra

from hoct_inference.cli import app

MODEL_PATH = Path(__file__).resolve().parents[3] / "weights" / "2026_01_30_09_23_41_job_26961657.pt"

runner = CliRunner()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_2d_movie(n_frames: int = 4, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Two cells drifting linearly across ``n_frames`` frames of size ``size x size``.

    Uses ``uint16`` for both arrays since ImageJ-tagged TIFFs only support a
    limited set of dtypes and we want labels and images to round-trip cleanly.
    """
    labels = np.zeros((n_frames, size, size), dtype=np.uint16)
    for t in range(n_frames):
        labels[t, 10 + t : 20 + t, 10:20] = 1
        labels[t, 40 - t : 50 - t, 40:50] = 2
    images = (labels.astype(np.float32) * 100.0).astype(np.uint16)
    return images, labels


def _write_imagej_tiff(path: Path, array: np.ndarray, axes: str) -> None:
    tifffile.imwrite(path, array, imagej=True, metadata={"axes": axes})


def _write_folder(folder: Path, stack: np.ndarray, frame_axes: str) -> None:
    folder.mkdir()
    for t, frame in enumerate(stack):
        _write_imagej_tiff(folder / f"frame_{t:03d}.tif", frame, frame_axes)


@pytest.fixture
def model_path() -> Path:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model checkpoint not found: {MODEL_PATH}")
    return MODEL_PATH


@pytest.fixture
def movie_2d_files(tmp_path):
    """Single-file 2D+t image and segmentation TIFFs."""
    images, labels = _make_2d_movie()
    img_path = tmp_path / "image.tif"
    seg_path = tmp_path / "segm.tif"
    _write_imagej_tiff(img_path, images, "TYX")
    _write_imagej_tiff(seg_path, labels, "TYX")
    return img_path, seg_path


@pytest.fixture
def movie_2d_folders(tmp_path):
    """Folder-of-frames 2D+t image and segmentation."""
    images, labels = _make_2d_movie()
    img_dir = tmp_path / "images"
    seg_dir = tmp_path / "segm"
    _write_folder(img_dir, images, "YX")
    _write_folder(seg_dir, labels, "YX")
    return img_dir, seg_dir


# ---------------------------------------------------------------------------
# Successful runs
# ---------------------------------------------------------------------------


def _assert_solution_geff(output: Path) -> td.graph.InMemoryGraph:
    assert output.exists(), "track did not write the output GEFF directory"
    graph, _ = td.graph.InMemoryGraph.from_geff(str(output))
    assert graph.num_nodes() > 0
    assert graph.num_edges() > 0
    return graph


def test_track_single_file_runs_end_to_end(tmp_path, model_path, movie_2d_files):
    """Running ``track`` on single-file inputs writes a non-empty solution GEFF."""
    img_path, seg_path = movie_2d_files
    output = tmp_path / "tracks.geff"

    result = runner.invoke(
        app,
        [
            "track",
            str(img_path),
            str(seg_path),
            "--model",
            str(model_path),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, f"track failed:\n{result.stdout}"
    assert "Building candidate tracking graph" in result.stdout
    assert "Results saved successfully" in result.stdout
    _assert_solution_geff(output)


def test_track_folder_inputs_match_single_file(tmp_path, model_path, movie_2d_files, movie_2d_folders):
    """Folder-of-frames input produces the same solution as the equivalent single file."""
    file_img, file_seg = movie_2d_files
    folder_img, folder_seg = movie_2d_folders

    out_file = tmp_path / "from_file.geff"
    out_folder = tmp_path / "from_folder.geff"

    for img, seg, out in [(file_img, file_seg, out_file), (folder_img, folder_seg, out_folder)]:
        result = runner.invoke(
            app,
            [
                "track",
                str(img),
                str(seg),
                "-m",
                str(model_path),
                "-o",
                str(out),
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, f"track failed for {img}:\n{result.stdout}"

    g_file = _assert_solution_geff(out_file)
    g_folder = _assert_solution_geff(out_folder)
    assert g_file.num_nodes() == g_folder.num_nodes()
    assert g_file.num_edges() == g_folder.num_edges()


def test_track_full_graph_flag_saves_more_edges_than_solution(tmp_path, model_path, movie_2d_files):
    """``--full-graph`` saves the candidate graph (always ≥ solution edge count)."""
    img_path, seg_path = movie_2d_files

    out_solution = tmp_path / "solution.geff"
    out_full = tmp_path / "full.geff"

    common = ["-m", str(model_path), "--device", "cpu"]
    runner.invoke(app, ["track", str(img_path), str(seg_path), *common, "-o", str(out_solution)])
    runner.invoke(app, ["track", str(img_path), str(seg_path), *common, "-o", str(out_full), "--full-graph"])

    sol_graph, _ = td.graph.InMemoryGraph.from_geff(str(out_solution))
    full_graph, _ = td.graph.InMemoryGraph.from_geff(str(out_full))

    # Full candidate graph has every candidate edge; solution is a subset.
    assert full_graph.num_edges() >= sol_graph.num_edges()
    # The full graph carries the predicted attributes.
    assert "solution" in full_graph.edge_attr_keys()
    assert "similarity" in full_graph.edge_attr_keys()


def test_track_overwrite_replaces_existing_output(tmp_path, model_path, movie_2d_files):
    img_path, seg_path = movie_2d_files
    output = tmp_path / "tracks.geff"
    output.mkdir()
    (output / "stale.txt").write_text("old content")

    result = runner.invoke(
        app,
        [
            "track",
            str(img_path),
            str(seg_path),
            "-m",
            str(model_path),
            "-o",
            str(output),
            "--overwrite",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, f"track failed:\n{result.stdout}"
    assert not (output / "stale.txt").exists(), "old contents should have been removed"
    _assert_solution_geff(output)


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------


def test_track_existing_output_without_overwrite_fails(tmp_path, model_path, movie_2d_files):
    img_path, seg_path = movie_2d_files
    output = tmp_path / "tracks.geff"
    output.mkdir()

    result = runner.invoke(
        app,
        [
            "track",
            str(img_path),
            str(seg_path),
            "-m",
            str(model_path),
            "-o",
            str(output),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code != 0
    assert "already exists" in result.stdout


def test_track_mismatched_layout_file_vs_folder_fails(tmp_path, model_path, movie_2d_files, movie_2d_folders):
    file_img, _ = movie_2d_files
    _, folder_seg = movie_2d_folders
    output = tmp_path / "tracks.geff"

    result = runner.invoke(
        app,
        [
            "track",
            str(file_img),
            str(folder_seg),
            "-m",
            str(model_path),
            "-o",
            str(output),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code != 0
    assert "same layout" in result.stdout


def test_track_mismatched_shapes_fail(tmp_path, model_path):
    images, _ = _make_2d_movie(n_frames=4, size=64)
    _, labels = _make_2d_movie(n_frames=4, size=32)  # different size
    img_path = tmp_path / "image.tif"
    seg_path = tmp_path / "segm.tif"
    _write_imagej_tiff(img_path, images, "TYX")
    _write_imagej_tiff(seg_path, labels, "TYX")
    output = tmp_path / "tracks.geff"

    result = runner.invoke(
        app,
        [
            "track",
            str(img_path),
            str(seg_path),
            "-m",
            str(model_path),
            "-o",
            str(output),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code != 0
    assert "does not match" in result.stdout


def test_track_missing_image_path_fails(tmp_path, model_path, movie_2d_files):
    _, seg_path = movie_2d_files
    output = tmp_path / "tracks.geff"

    result = runner.invoke(
        app,
        [
            "track",
            str(tmp_path / "missing.tif"),
            str(seg_path),
            "-m",
            str(model_path),
            "-o",
            str(output),
        ],
    )

    assert result.exit_code != 0
