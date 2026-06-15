"""Tests for hoct._io image loading helpers.

TIFF and Zarr inputs are read with always-available dependencies, so these
tests do not require the optional ``bioio`` extra.
"""

import dask.array as da
import numpy as np
import pytest
import tifffile
import zarr

from hoct._io import is_frame_folder, load_array


def _write_imagej_tiff(path, array, axes):
    """Write ``array`` as an ImageJ-tagged TIFF carrying the dim order."""
    tifffile.imwrite(path, array, imagej=True, metadata={"axes": axes})


def _write_zarr_array(path, array):
    """Write ``array`` as a plain ("simple") Zarr array."""
    z = zarr.create_array(store=str(path), shape=array.shape, dtype=array.dtype)
    z[:] = array


def _write_ome_zarr(path, array, axes):
    """Write ``array`` as a minimal single-scale OME-Zarr group."""
    group = zarr.open_group(str(path), mode="w")
    level = group.create_array("0", shape=array.shape, dtype=array.dtype)
    level[:] = array
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "axes": [{"name": name} for name in axes],
            "datasets": [{"path": "0"}],
        }
    ]


# ---------------------------------------------------------------------------
# Single TIFF
# ---------------------------------------------------------------------------


def test_load_single_2d_timeseries(tmp_path):
    """A single TYX TIFF round-trips to a (T, Y, X) array."""
    arr = np.random.randint(0, 100, (4, 16, 16), dtype=np.uint16)
    path = tmp_path / "movie.tif"
    _write_imagej_tiff(path, arr, "TYX")

    result = load_array(path)

    assert result.shape == (4, 16, 16)
    np.testing.assert_array_equal(result, arr)


def test_load_single_3d_timeseries(tmp_path):
    """A single TZYX TIFF round-trips to a (T, Z, Y, X) array."""
    arr = np.random.randint(0, 100, (3, 5, 8, 8), dtype=np.uint16)
    path = tmp_path / "volume.tif"
    _write_imagej_tiff(path, arr, "TZYX")

    result = load_array(path)

    assert result.shape == (3, 5, 8, 8)
    np.testing.assert_array_equal(result, arr)


def test_load_tiff_collapses_singleton_z(tmp_path):
    """A TZYX TIFF with Z=1 collapses to (T, Y, X)."""
    arr = np.random.randint(0, 100, (4, 1, 8, 8), dtype=np.uint16)
    path = tmp_path / "movie.tif"
    _write_imagej_tiff(path, arr, "TZYX")

    result = load_array(path)

    assert result.shape == (4, 8, 8)
    np.testing.assert_array_equal(result, arr[:, 0])


def test_load_tiff_keeps_first_channel(tmp_path):
    """A TCYX TIFF keeps the first channel and drops the channel axis."""
    arr = np.random.randint(0, 100, (4, 2, 8, 8), dtype=np.uint16)
    path = tmp_path / "movie.tif"
    _write_imagej_tiff(path, arr, "TCYX")

    result = load_array(path)

    assert result.shape == (4, 8, 8)
    np.testing.assert_array_equal(result, arr[:, 0])


# ---------------------------------------------------------------------------
# Simple Zarr
# ---------------------------------------------------------------------------


def test_load_simple_zarr_2d(tmp_path):
    """A plain (T, Y, X) Zarr array round-trips unchanged as a lazy dask array."""
    arr = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
    path = tmp_path / "movie.zarr"
    _write_zarr_array(path, arr)

    result = load_array(path)

    assert isinstance(result, da.Array), "Zarr should load lazily via dask"
    assert result.shape == (4, 8, 8)
    np.testing.assert_array_equal(result, arr)


def test_load_simple_zarr_collapses_singleton_axes(tmp_path):
    """A plain (T, 1, Y, X) Zarr array collapses to (T, Y, X)."""
    arr = np.arange(4 * 1 * 8 * 8, dtype=np.uint16).reshape(4, 1, 8, 8)
    path = tmp_path / "movie.zarr"
    _write_zarr_array(path, arr)

    result = load_array(path)

    assert result.shape == (4, 8, 8)
    np.testing.assert_array_equal(result, arr[:, 0])


# ---------------------------------------------------------------------------
# OME-Zarr (t, c, z, y, x)
# ---------------------------------------------------------------------------


def test_load_ome_zarr_collapses_to_2d(tmp_path):
    """A (t, c=1, z=1, y, x) OME-Zarr collapses to (T, Y, X), loaded lazily."""
    arr = np.arange(4 * 1 * 1 * 8 * 8, dtype=np.uint16).reshape(4, 1, 1, 8, 8)
    path = tmp_path / "movie.ome.zarr"
    _write_ome_zarr(path, arr, "tczyx")

    result = load_array(path)

    assert isinstance(result, da.Array), "OME-Zarr should load lazily via dask"
    assert result.shape == (4, 8, 8)
    np.testing.assert_array_equal(result, arr[:, 0, 0])


def test_load_ome_zarr_keeps_real_z(tmp_path):
    """A (t, c=1, z>1, y, x) OME-Zarr collapses to (T, Z, Y, X)."""
    arr = np.arange(3 * 1 * 5 * 8 * 8, dtype=np.uint16).reshape(3, 1, 5, 8, 8)
    path = tmp_path / "volume.ome.zarr"
    _write_ome_zarr(path, arr, "tczyx")

    result = load_array(path)

    assert result.shape == (3, 5, 8, 8)
    np.testing.assert_array_equal(result, arr[:, 0])


def test_load_ome_zarr_keeps_first_channel(tmp_path):
    """A multichannel (t, c>1, z=1, y, x) OME-Zarr keeps the first channel."""
    arr = np.arange(3 * 2 * 1 * 8 * 8, dtype=np.uint16).reshape(3, 2, 1, 8, 8)
    path = tmp_path / "movie.ome.zarr"
    _write_ome_zarr(path, arr, "tczyx")

    result = load_array(path)

    assert result.shape == (3, 8, 8)
    np.testing.assert_array_equal(result, arr[:, 0, 0])


def test_ome_zarr_group_without_multiscales_raises(tmp_path):
    """A Zarr group lacking OME 'multiscales' metadata is rejected clearly."""
    path = tmp_path / "plain_group.zarr"
    group = zarr.open_group(str(path), mode="w")
    level = group.create_array("0", shape=(4, 8, 8), dtype=np.uint16)
    level[:] = np.zeros((4, 8, 8), dtype=np.uint16)

    with pytest.raises(ValueError, match="multiscales"):
        load_array(path)


# ---------------------------------------------------------------------------
# Folder of frames
# ---------------------------------------------------------------------------


def test_load_folder_of_2d_frames(tmp_path):
    """Folder of YX frames stacks alphabetically into a lazy (T, Y, X) array."""
    folder = tmp_path / "frames"
    folder.mkdir()
    frames = [np.full((8, 8), t, dtype=np.uint8) for t in range(5)]
    for t, frame in enumerate(frames):
        _write_imagej_tiff(folder / f"frame_{t:03d}.tif", frame, "YX")

    result = load_array(folder)

    assert isinstance(result, da.Array), "a folder of TIFFs should load lazily via dask"
    # One dask chunk per frame, so frames load on demand rather than all at once.
    assert result.chunks[0] == (1,) * 5
    assert result.shape == (5, 8, 8)
    for t in range(5):
        np.testing.assert_array_equal(result[t], frames[t])


def test_load_folder_of_3d_frames(tmp_path):
    """Folder of ZYX frames stacks alphabetically into (T, Z, Y, X)."""
    folder = tmp_path / "volumes"
    folder.mkdir()
    frames = [np.full((3, 4, 4), t, dtype=np.uint8) for t in range(3)]
    for t, frame in enumerate(frames):
        _write_imagej_tiff(folder / f"vol_{t:03d}.tif", frame, "ZYX")

    result = load_array(folder)

    assert result.shape == (3, 3, 4, 4)
    for t in range(3):
        np.testing.assert_array_equal(result[t], frames[t])


def test_folder_sorts_files_alphabetically(tmp_path):
    """File ordering follows sorted filenames, not filesystem order."""
    folder = tmp_path / "frames"
    folder.mkdir()
    # Names chosen so alphabetical != insertion order.
    for name, value in [("c.tif", 2), ("a.tif", 0), ("b.tif", 1)]:
        _write_imagej_tiff(folder / name, np.full((4, 4), value, dtype=np.uint8), "YX")

    result = load_array(folder)

    # Sorted: a, b, c → 0, 1, 2
    assert result[0, 0, 0] == 0
    assert result[1, 0, 0] == 1
    assert result[2, 0, 0] == 2


def test_folder_skips_dotfiles(tmp_path):
    """Hidden files are ignored when listing folder contents."""
    folder = tmp_path / "frames"
    folder.mkdir()
    _write_imagej_tiff(folder / "a.tif", np.zeros((4, 4), dtype=np.uint8), "YX")
    _write_imagej_tiff(folder / ".hidden.tif", np.zeros((4, 4), dtype=np.uint8), "YX")

    result = load_array(folder)
    assert result.shape == (1, 4, 4)


def test_empty_folder_raises(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(ValueError, match="No image files"):
        load_array(folder)


def test_inconsistent_frame_shapes_raise(tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    _write_imagej_tiff(folder / "a.tif", np.zeros((8, 8), dtype=np.uint8), "YX")
    _write_imagej_tiff(folder / "b.tif", np.zeros((4, 4), dtype=np.uint8), "YX")

    with pytest.raises(ValueError, match="inconsistent shapes"):
        load_array(folder)


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_array(tmp_path / "does_not_exist.tif")


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------


def test_is_frame_folder_distinguishes_zarr_from_frame_folder(tmp_path):
    """A Zarr store is a single input; a plain folder is a frame folder."""
    zarr_path = tmp_path / "movie.zarr"
    _write_zarr_array(zarr_path, np.zeros((4, 8, 8), dtype=np.uint16))

    folder = tmp_path / "frames"
    folder.mkdir()
    _write_imagej_tiff(folder / "a.tif", np.zeros((4, 4), dtype=np.uint8), "YX")

    tiff_path = tmp_path / "movie.tif"
    _write_imagej_tiff(tiff_path, np.zeros((4, 8, 8), dtype=np.uint16), "TYX")

    assert is_frame_folder(folder) is True
    assert is_frame_folder(zarr_path) is False
    assert is_frame_folder(tiff_path) is False
