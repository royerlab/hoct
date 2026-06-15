#!/usr/bin/env python
"""Download CTC datasets and build GEFF candidate graphs for the test suite.

Usage
-----
    uv run python scripts/prepare_test_data.py [--force]

The generated graphs back the data-dependent tests (those marked
``requires_geff_data``). Without them, those tests are skipped. The data lands
in ``.test-data/`` (gitignored) and is never bundled into the package.

Each graph is built from a Cell Tracking Challenge training set: the ``TRA``
ground-truth masks serve as both the segmentation labels and the source of the
ground-truth lineage graph, so the resulting candidate graph carries the
``edge_is_gt`` labels the tracking tests need.
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import tifffile
import tracksdata as td
from tracksdata.io import from_ctc

from hoct.features import create_graph

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / ".test-data"
ZIP_DIR = DATA_DIR / "zips"
EXTRACT_DIR = DATA_DIR / "extracted"
GEFF_DIR = DATA_DIR / "geff"

BASE_URL = "https://data.celltrackingchallenge.net/training-datasets"

# (dataset, sequence, output GEFF name)
DATASETS: list[tuple[str, str, str]] = [
    ("Fluo-C2DL-Huh7", "01", "huh7_2d.geff"),
    ("Fluo-C3DL-MDA231", "01", "mda231_3d.geff"),
]

GRAPH_KWARGS = {"distance_threshold": 300.0, "n_neighbors": 5, "delta_t": 3}


def _download(dataset: str) -> Path:
    """Download the dataset zip into ``ZIP_DIR`` (skipping if already present)."""
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ZIP_DIR / f"{dataset}.zip"
    if not zip_path.exists():
        url = f"{BASE_URL}/{dataset}.zip"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


def _extract(zip_path: Path, dataset: str) -> Path:
    """Extract the dataset zip into ``EXTRACT_DIR`` (skipping if already present)."""
    target = EXTRACT_DIR / dataset
    if not target.exists():
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(EXTRACT_DIR)
    return target


def _load_stack(directory: Path, pattern: str) -> np.ndarray:
    """Stack a sorted glob of single-frame TIFFs into a ``(T, ...)`` array."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {directory / pattern}")
    return np.stack([tifffile.imread(f) for f in files])


def _gt_tra_dir(tra_dir: Path, labels: np.ndarray, stack: contextlib.ExitStack) -> Path:
    """Return a CTC TRA directory whose mask dimensionality matches the candidates.

    For 2D+t data ``create_graph`` expands labels to a singleton-z volume, so the
    candidate masks are 3D. Build the ground-truth graph from matching 3D masks
    to avoid a 2D/3D mask-intersection mismatch during ``graph.match()``.
    """
    if labels.ndim != 3:  # already 3D+t
        return tra_dir
    dst = Path(stack.enter_context(tempfile.TemporaryDirectory()))
    shutil.copy(tra_dir / "man_track.txt", dst / "man_track.txt")
    for f in sorted(tra_dir.glob("man_track*.tif")):
        tifffile.imwrite(dst / f.name, tifffile.imread(f)[None, ...])
    return dst


def build_geff(dataset: str, sequence: str, out_name: str, *, force: bool = False) -> Path:
    """Build a single GEFF candidate graph from a CTC sequence."""
    out_path = GEFF_DIR / out_name
    if out_path.exists() and not force:
        print(f"\u2713 {out_path} already exists (use --force to rebuild)")
        return out_path

    root = _extract(_download(dataset), dataset)
    seq_dir = root / sequence
    tra_dir = root / f"{sequence}_GT" / "TRA"

    images = _load_stack(seq_dir, "t*.tif")
    labels = _load_stack(tra_dir, "man_track*.tif")

    with contextlib.ExitStack() as stack:
        gt_graph = td.graph.RustWorkXGraph()
        from_ctc(str(_gt_tra_dir(tra_dir, labels, stack)), gt_graph)
        graph = create_graph(labels, images=images, gt_graph=gt_graph, **GRAPH_KWARGS)

        GEFF_DIR.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            shutil.rmtree(out_path)
        graph.to_geff(str(out_path))
        print(f"\u2713 wrote {out_path}  ({graph.num_nodes()} nodes, {graph.num_edges()} edges)")
    return out_path


def main() -> None:
    """Build all configured GEFF fixtures."""
    warnings.filterwarnings("ignore")
    force = "--force" in sys.argv
    for dataset, sequence, out_name in DATASETS:
        build_geff(dataset, sequence, out_name, force=force)


if __name__ == "__main__":
    main()
