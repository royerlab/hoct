"""Image loading helpers for the high-level CLI.

:func:`load_array` accepts a single file, a Zarr store, or a folder of
single-frame files, and returns a ``(T, Y, X)`` or ``(T, Z, Y, X)`` array
suitable for :func:`hoct.predict`.

Supported inputs
----------------
* **Single TIFF** — a multi-page ``(T, [Z,] Y, X)`` stack, read with ``tifffile``.
* **Zarr array** ("simple zarr") — any layout; length-1 axes are collapsed.
* **OME-Zarr** — a ``(t, c, z, y, x)`` multiscale group; the highest-resolution
  level is read, the first channel is kept, and length-1 axes are collapsed.
* **Folder of frames** — one single-timepoint file per timepoint, sorted
  alphabetically and stacked along a new leading T axis.

Zarr stores and folders of TIFFs are returned as **lazy** ``dask`` arrays, so
large datasets are never fully materialized in memory; downstream code reads
them frame by frame. Other inputs are returned as eager numpy arrays.

TIFF and Zarr are read with always-available dependencies. Any other
single-file format falls back to ``bioio`` (optional extra ``hoct[bioio]``).
"""

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

_TIFF_SUFFIXES = {".tif", ".tiff"}
_ZARR_MARKERS = ("zarr.json", ".zarray", ".zgroup")
# Channel ("c") and RGB-sample ("s") axes are reduced to their first index.
_CHANNEL_AXES = ("c", "s")


def _is_zarr(path: Path) -> bool:
    """Return True if ``path`` is a Zarr store (``.zarr`` suffix or zarr metadata)."""
    if path.suffix == ".zarr":
        return True
    if path.is_dir():
        return any((path / marker).exists() for marker in _ZARR_MARKERS)
    return False


def is_frame_folder(path: Path) -> bool:
    """Return True if ``path`` is a folder of single-frame files (not a Zarr store)."""
    path = Path(path)
    return path.is_dir() and not _is_zarr(path)


def _collapse_singleton_axes(data: ArrayLike) -> ArrayLike:
    """Drop length-1 axes, always keeping the trailing two (Y, X).

    Uses ``.squeeze`` so the operation stays lazy on dask arrays.
    """
    drop = tuple(axis for axis in range(data.ndim - 2) if data.shape[axis] == 1)
    return data.squeeze(axis=drop) if drop else data


def _select_first_channel(data: ArrayLike, axes: str) -> ArrayLike:
    """Keep only the first index of any channel/sample axis named in ``axes``.

    Uses basic indexing (not ``.take``, which dask arrays lack) so the result
    stays lazy on dask arrays.
    """
    names = [a.lower() for a in axes]
    for channel in _CHANNEL_AXES:
        if channel in names:
            index = names.index(channel)
            data = data[(slice(None),) * index + (0,)]
            names.pop(index)
    return data


def _reduce_to_movie(data: ArrayLike, axes: str) -> ArrayLike:
    """Reduce a labelled array to ``(T, [Z,] Y, X)``.

    Keeps the first channel of any channel axis, then collapses length-1 axes.
    ``axes`` is a per-dimension code such as ``"TCZYX"`` or ``"TYX"``.
    """
    data = _select_first_channel(data, axes)
    return _collapse_singleton_axes(data)


def _reduced_shape(shape: tuple[int, ...], axes: str) -> tuple[int, ...]:
    """Frame shape after :func:`_reduce_to_movie` (channel dropped, length-1 axes removed).

    Mirrors :func:`_reduce_to_movie` on the shape alone, so a folder of frames
    can be validated from TIFF headers without reading any pixels.
    """
    names = [a.lower() for a in axes]
    dims = list(shape)
    for channel in _CHANNEL_AXES:
        if channel in names:
            index = names.index(channel)
            names.pop(index)
            dims.pop(index)
    keep_from = len(dims) - 2
    return tuple(dim for i, dim in enumerate(dims) if i >= keep_from or dim != 1)


def _load_tiff(path: Path) -> np.ndarray:
    """Read a single TIFF stack and reduce it to ``(T, [Z,] Y, X)``."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        data = np.asarray(series.asarray())
        axes = series.axes
    return _reduce_to_movie(data, axes)


def _tiff_reduced_shape(path: Path) -> tuple[int, ...]:
    """Return a TIFF's reduced frame shape, reading headers only (no pixel data)."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        return _reduced_shape(tuple(series.shape), series.axes)


def _ome_multiscales(attrs: dict) -> list | None:
    """Return the OME-NGFF ``multiscales`` list, supporting v0.4 and v0.5 layouts."""
    if "multiscales" in attrs:
        return attrs["multiscales"]
    ome = attrs.get("ome")
    if isinstance(ome, dict) and "multiscales" in ome:
        return ome["multiscales"]
    return None


def _load_ome_zarr(group, multiscales: list) -> ArrayLike:
    """Lazily read the highest-resolution level of an OME-Zarr group as ``(T, [Z,] Y, X)``."""
    import dask.array as da

    metadata = multiscales[0]
    axes = "".join(axis["name"] if isinstance(axis, dict) else axis for axis in metadata["axes"])
    # OME datasets are ordered from highest to lowest resolution.
    dataset_path = metadata["datasets"][0]["path"]
    data = da.from_zarr(group[dataset_path])
    return _reduce_to_movie(data, axes)


def _load_zarr(path: Path) -> ArrayLike:
    """Lazily read a Zarr store: an OME-Zarr group or a plain ("simple") array."""
    import dask.array as da
    import zarr

    node = zarr.open(str(path), mode="r")
    if isinstance(node, zarr.Group):
        multiscales = _ome_multiscales(dict(node.attrs))
        if multiscales is None:
            raise ValueError(f"Zarr group at {path} has no OME 'multiscales' metadata.")
        return _load_ome_zarr(node, multiscales)
    return _collapse_singleton_axes(da.from_zarr(node))


def _load_with_bioio(path: Path) -> np.ndarray:
    """Fallback reader for single files in non-TIFF, non-Zarr formats."""
    try:
        from bioio import BioImage  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only when extra missing
        raise ImportError(
            f"Reading '{path.suffix}' files requires the 'bioio' extra. Install it with: pip install 'hoct[bioio]'"
        ) from exc
    img = BioImage(str(path))
    data = np.asarray(img.get_image_data("TZYX", C=0))  # (T, Z, Y, X)
    return _collapse_singleton_axes(data)


def _is_image_file(path: Path) -> bool:
    """Skip dotfiles and directories when listing a frame folder."""
    return path.is_file() and not path.name.startswith(".")


def _read_file(path: Path) -> ArrayLike:
    """Read a single file (Zarr, TIFF, or via bioio) and collapse length-1 axes."""
    if _is_zarr(path):
        return _load_zarr(path)
    if path.suffix.lower() in _TIFF_SUFFIXES:
        return _load_tiff(path)
    return _load_with_bioio(path)


def _load_tiff_folder(path: Path, files: list[Path]) -> ArrayLike:
    """Lazily stack a folder of single-frame TIFFs along a new leading T axis.

    Frame shapes are validated up front from TIFF headers (no pixel reads); the
    pixels themselves are loaded lazily, one frame per dask chunk.
    """
    from dask.array.image import imread as dask_imread

    shapes = {_tiff_reduced_shape(f) for f in files}
    if len(shapes) > 1:
        raise ValueError(f"Frames in {path} have inconsistent shapes: {shapes}")

    # Single shared suffix lets dask glob exactly the validated files, in the
    # same sorted order. ``_load_tiff`` applies the per-frame reduction.
    suffix = files[0].suffix.lower()
    return dask_imread(str(path / f"*{suffix}"), imread=_load_tiff)


def load_array(path: Path) -> ArrayLike:
    """Load image data from a file, a Zarr store, or a folder of frames.

    Conventions
    -----------
    * **Single file / Zarr store**: holds the entire time series. Returns
      ``(T, Y, X)`` for 2D+t or ``(T, Z, Y, X)`` for 3D+t. Length-1 axes (e.g. a
      singleton channel or Z) are collapsed.
    * **Folder**: each file is one timepoint, sorted alphabetically and stacked
      along a new T axis. Returns the same shapes as above.

    Zarr stores and folders of TIFFs are returned as lazy ``dask`` arrays; other
    inputs are returned as eager numpy arrays.

    Parameters
    ----------
    path
        Path to an image file, a ``.zarr`` store, or a folder of single-frame
        image files.

    Returns
    -------
    ArrayLike
        ``(T, Y, X)`` or ``(T, Z, Y, X)`` numpy or dask array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if is_frame_folder(path):
        files = sorted(p for p in path.iterdir() if _is_image_file(p))
        if not files:
            raise ValueError(f"No image files found in folder: {path}")

        suffixes = {f.suffix.lower() for f in files}
        if suffixes <= _TIFF_SUFFIXES and len(suffixes) == 1:
            return _load_tiff_folder(path, files)

        # Eager fallback for non-TIFF (or mixed-suffix) frame folders.
        frames = [_read_file(p) for p in files]
        shapes = {np.shape(f) for f in frames}
        if len(shapes) > 1:
            raise ValueError(f"Frames in {path} have inconsistent shapes: {shapes}")
        return np.stack(frames, axis=0)

    return _read_file(path)
