"""Tests for hoct._io image loading helpers."""

import numpy as np
import pytest
import tifffile

pytest.importorskip("bioio")

from hoct._io import load_array


def _write_imagej_tiff(path, array, axes):
    """Write ``array`` as an ImageJ-tagged TIFF so bioio resolves the dim order."""
    tifffile.imwrite(path, array, imagej=True, metadata={"axes": axes})


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


def test_load_folder_of_2d_frames(tmp_path):
    """Folder of YX frames stacks alphabetically into (T, Y, X)."""
    folder = tmp_path / "frames"
    folder.mkdir()
    frames = [np.full((8, 8), t, dtype=np.uint8) for t in range(5)]
    for t, frame in enumerate(frames):
        _write_imagej_tiff(folder / f"frame_{t:03d}.tif", frame, "YX")

    result = load_array(folder)

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
