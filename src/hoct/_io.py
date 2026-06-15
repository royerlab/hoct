"""Image loading helpers for the high-level CLI.

Provides a single :func:`load_array` helper that accepts either a file or a
folder of frames and returns a ``(T, Y, X)`` or ``(T, Z, Y, X)`` numpy array
suitable for :func:`hoct.predict`.

Image reading uses ``bioio`` (optional dependency, ``hoct[bioio]``).
"""

from pathlib import Path

import numpy as np


def _require_bioio():
    """Import ``bioio.BioImage`` lazily and surface a clear install hint on failure."""
    try:
        from bioio import BioImage  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only when extra missing
        raise ImportError(
            "Reading raw images requires the 'bioio' extra. Install it with: pip install 'hoct[bioio]'"
        ) from exc
    return BioImage


def _is_image_file(path: Path) -> bool:
    """Skip dotfiles, directories, and obvious non-image siblings."""
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    return True


def _load_timeseries(path: Path) -> np.ndarray:
    """Load a single file as a (T, [Z,] Y, X) array via bioio.

    Reads the first channel (``C=0``) and drops the channel axis. If the data
    has a singleton Z dimension, it is squeezed to yield a 2D+t volume.
    """
    BioImage = _require_bioio()
    img = BioImage(str(path))
    data = np.asarray(img.get_image_data("TZYX", C=0))  # (T, Z, Y, X)
    if data.shape[1] == 1:
        data = data[:, 0]  # (T, Y, X)
    return data


def _load_frame(path: Path) -> np.ndarray:
    """Load a single file as one frame and return a ([Z,] Y, X) array.

    Used for folder-of-frames inputs where each file represents one timepoint.
    Reads ``C=0, T=0`` and squeezes a singleton Z to produce 2D frames.
    """
    BioImage = _require_bioio()
    img = BioImage(str(path))
    data = np.asarray(img.get_image_data("ZYX", C=0, T=0))  # (Z, Y, X)
    if data.shape[0] == 1:
        data = data[0]  # (Y, X)
    return data


def load_array(path: Path) -> np.ndarray:
    """Load image data from a file or folder.

    Conventions
    -----------
    * **File**: the entire time series lives in one file. Returns
      ``(T, Y, X)`` for 2D+t or ``(T, Z, Y, X)`` for 3D+t.
    * **Folder**: each file is one timepoint, sorted alphabetically. Frames are
      stacked along a new T axis. Returns the same shapes as above.

    Parameters
    ----------
    path
        Path to an image file or to a folder of single-frame image files.

    Returns
    -------
    np.ndarray
        ``(T, Y, X)`` or ``(T, Z, Y, X)`` array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if _is_image_file(p))
        if not files:
            raise ValueError(f"No image files found in folder: {path}")
        frames = [_load_frame(p) for p in files]
        shapes = {f.shape for f in frames}
        if len(shapes) > 1:
            raise ValueError(f"Frames in {path} have inconsistent shapes: {shapes}")
        return np.stack(frames, axis=0)

    return _load_timeseries(path)
