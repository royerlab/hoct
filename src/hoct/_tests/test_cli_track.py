"""End-to-end tests for the ``hoct track`` CLI command."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
import tracksdata as td
import zarr
from typer.testing import CliRunner

from hoct._tests.conftest import MODEL_PATH
from hoct.cli import app

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


def _write_ome_zarr(path: Path, stack: np.ndarray) -> None:
    """Write a (T, Y, X) stack as an OME-Zarr group with (t, c, z, y, x) axes."""
    data = stack[:, np.newaxis, np.newaxis]  # (T, 1, 1, Y, X)
    group = zarr.open_group(str(path), mode="w")
    level = group.create_array("0", shape=data.shape, dtype=data.dtype)
    level[:] = data
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": name} for name in "tczyx"],
            "datasets": [{"path": "0"}],
        }
    ]


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
def movie_2d_ome_zarr(tmp_path):
    """OME-Zarr (t, c=1, z=1, y, x) image and segmentation stores."""
    images, labels = _make_2d_movie()
    img_path = tmp_path / "image.ome.zarr"
    seg_path = tmp_path / "segm.ome.zarr"
    _write_ome_zarr(img_path, images)
    _write_ome_zarr(seg_path, labels)
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


def test_track_ome_zarr_inputs_match_single_file(tmp_path, model_path, movie_2d_files, movie_2d_ome_zarr):
    """OME-Zarr (t, c, z, y, x) input produces the same solution as the equivalent TIFF."""
    file_img, file_seg = movie_2d_files
    zarr_img, zarr_seg = movie_2d_ome_zarr

    out_file = tmp_path / "from_file.geff"
    out_zarr = tmp_path / "from_zarr.geff"

    for img, seg, out in [(file_img, file_seg, out_file), (zarr_img, zarr_seg, out_zarr)]:
        result = runner.invoke(
            app,
            ["track", str(img), str(seg), "-m", str(model_path), "-o", str(out), "--device", "cpu"],
        )
        assert result.exit_code == 0, f"track failed for {img}:\n{result.stdout}"

    g_file = _assert_solution_geff(out_file)
    g_zarr = _assert_solution_geff(out_zarr)
    assert g_file.num_nodes() == g_zarr.num_nodes()
    assert g_file.num_edges() == g_zarr.num_edges()


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


def test_track_ctc_format_writes_label_tiffs_and_track_table(tmp_path, model_path, movie_2d_files):
    """``-f ctc`` writes a Cell Tracking Challenge folder: per-frame masks + res_track.txt."""
    img_path, seg_path = movie_2d_files
    output = tmp_path / "01_RES"

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
            "-f",
            "ctc",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, f"track failed:\n{result.stdout}"
    assert output.is_dir()

    track_file = output / "res_track.txt"
    assert track_file.exists(), "CTC export must produce res_track.txt"
    rows = [line.split() for line in track_file.read_text().splitlines() if line.strip()]
    assert rows, "res_track.txt should not be empty"
    # Each row is: tracklet_id start_t end_t parent_id (4 ints)
    assert all(len(r) == 4 and all(c.lstrip("-").isdigit() for c in r) for r in rows)

    masks = sorted(output.glob("*.tif"))
    assert len(masks) > 0, "CTC export must write per-frame label TIFFs"

    arr = tifffile.imread(masks[0])
    # CTC label TIFFs have the spatial shape of the input frames (here: 64x64).
    assert arr.shape[-2:] == (64, 64), f"unexpected CTC mask shape: {arr.shape}"


def test_track_ctc_with_full_graph_is_rejected(tmp_path, model_path, movie_2d_files):
    """CTC export only makes sense for the solution graph."""
    img_path, seg_path = movie_2d_files
    output = tmp_path / "01_RES"

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
            "-f",
            "ctc",
            "--full-graph",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code != 0
    assert "only valid for the solution graph" in result.stdout


# ---------------------------------------------------------------------------
# Auto-tiling decision
# ---------------------------------------------------------------------------


class _StubGraph:
    """Tiny duck-typed stand-in for ``td.graph.BaseGraph`` for the threshold logic."""

    def __init__(self, n_edges: int, n_time: int) -> None:
        self._n_edges = n_edges
        self._n_time = n_time

    def num_edges(self) -> int:
        return self._n_edges

    def time_points(self) -> list[int]:
        return list(range(self._n_time))


def test_auto_tiling_disabled_below_threshold():
    from hoct.cli import TileMode, _maybe_build_tiling_scheme

    # Just under the 2500 edges/time threshold.
    scheme = _maybe_build_tiling_scheme(_StubGraph(n_edges=2_499 * 10, n_time=10), TileMode.AUTO)
    assert scheme is None


def test_auto_tiling_enabled_above_threshold():
    from hoct.cli import TileMode, _maybe_build_tiling_scheme

    scheme = _maybe_build_tiling_scheme(_StubGraph(n_edges=2_501 * 10, n_time=10), TileMode.AUTO)
    assert scheme is not None
    assert scheme.tile_shape == (1, 64, 256, 256)
    assert scheme.overlap_shape == (2, 24, 64, 64)


def test_tile_off_returns_none_regardless_of_density():
    from hoct.cli import TileMode, _maybe_build_tiling_scheme

    assert _maybe_build_tiling_scheme(_StubGraph(n_edges=10**6, n_time=1), TileMode.OFF) is None


def test_tile_on_forces_scheme_below_threshold():
    from hoct.cli import TileMode, _maybe_build_tiling_scheme

    scheme = _maybe_build_tiling_scheme(_StubGraph(n_edges=10, n_time=10), TileMode.ON)
    assert scheme is not None
    assert scheme.tile_shape == (1, 64, 256, 256)


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
