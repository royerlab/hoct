# HOCT

Inference and tracking for the Higher-Order Cell Tracking Transformer (HOCT)
model with JIT-compiled models.

---

## Quick start (for biologists)

If all you want is to track cells from images and segmentation masks, you do
not need to write any Python. The steps below take you from "nothing
installed" to "a tracking result on disk" in about a minute.

### 1. Install `uv` (one-time)

`uv` is a small Python launcher. It downloads everything else automatically.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen your terminal so `uv` is on your `PATH`. Verify:

```bash
uv --version
```

### 2. Run tracking

`uvx` runs the CLI in a temporary, isolated environment — nothing is
installed permanently and there is no virtual environment to manage:

```bash
uvx --from "hoct[bioio]" hoct track \
    <IMAGES> <SEGMENTATION> \
    -o <OUTPUT.geff>
```

* `<IMAGES>`: a single image file (whole time series) **or** a folder of
  single-frame files sorted alphabetically.
* `<SEGMENTATION>`: same format as `<IMAGES>` (file with file, folder with
  folder), with one integer label per object.
* `<OUTPUT.geff>`: where to write the result. Default is a folder in the
  [GEFF](https://github.com/live-image-tracking-tools/geff) format. Pass
  `-f ctc` to instead write a [Cell Tracking Challenge][ctc] folder of
  per-frame label TIFFs plus `res_track.txt`.

[ctc]: https://celltrackingchallenge.net/datasets/

No model is specified above, so HOCT downloads the default pre-trained model
(`general_v0`) and caches it; later runs reuse the cache. To use your own
checkpoint, pass `-m /path/to/model.pt`. Set `HOCT_CACHE_DIR` to change where
downloads are cached.

The first run also downloads the Python dependencies; later runs start almost
instantly.

### Example: a CTC dataset

```bash
uvx --from "hoct[bioio]" hoct track \
    /data/Fluo-C3DL-MDA231/01 \
    /data/Fluo-C3DL-MDA231/01_ERR_SEG \
    -o tracks.geff
```

This loads 12 frames of 3D images and segmentation masks, builds the
candidate graph, runs the model on the GPU (falling back to CPU if none is
available), solves the tracking ILP, and writes `tracks.geff/`.

To benchmark against the Cell Tracking Challenge ground truth, write the
result directly in CTC format:

```bash
uvx --from "hoct[bioio]" hoct track \
    /data/Fluo-C3DL-MDA231/01 \
    /data/Fluo-C3DL-MDA231/01_ERR_SEG \
    -o /data/Fluo-C3DL-MDA231/01_RES \
    -f ctc
```

This produces the standard CTC layout (`maskNNN.tif` per timepoint plus
`res_track.txt`).

### Useful flags

| Flag | Default | What it does |
|---|---|---|
| `-o, --output` | *required* | Output GEFF directory |
| `-m, --model` | `general_v0` | Checkpoint path or registered model name; the default is downloaded on first use |
| `-f, --format` | `geff` | `geff` or `ctc` (Cell Tracking Challenge folder) |
| `-d, --device` | `cuda` | `cuda`, `mps`, or `cpu` (auto-falls back to CPU if needed) |
| `--tile` | `auto` | `auto`/`on`/`off`. Tiled inference for large data; auto-enables when the candidate graph has more than 2500 edges per timepoint. Tile shape `(t, z, y, x) = (1, 64, 256, 256)`, overlap `(2, 24, 64, 64)`. |
| `-ow, --overwrite` | off | Overwrite an existing output directory |
| `--full-graph` | off | Save the full candidate graph (with predicted scores) instead of just the solution |
| `--scale` | none | Physical voxel size, repeat the flag per axis: `--scale 1 --scale 0.5 --scale 0.5 --scale 0.5` for `t z y x` |
| `--max-distance` | `300.0` | Largest spatial distance to consider as a candidate edge |
| `--neighbors` | `5` | Maximum candidate neighbors per cell |
| `--max-dt` | `3` | Maximum temporal gap (in frames) for candidate edges |
| `--window, -w` | `5` | Temporal window size used by the model |
| `--config, -c` | none | Path to an ILP solver config YAML (see `init-config`) |

Run `uvx --from "hoct[bioio]" hoct track --help` for
the full list.

### Customising the solver

Generate a template config you can edit and pass with `-c`:

```bash
uvx --from hoct hoct init-config -o solver_config.yaml
# ...edit the file...
uvx --from "hoct[bioio]" hoct track ... -c solver_config.yaml
```

### Tracking from an existing GEFF

If you already have a candidate graph in GEFF form (e.g. produced by
`hoct.features.create_graph`), use `predict` instead of `track`:

```bash
uvx --from hoct hoct predict candidate.geff -o tracks.geff
```

---

## Installation

```bash
pip install "hoct[bioio]"
```

The `bioio` extra is needed for the `track` CLI (reading image/label files).

## Installation (for developers)

```bash
git clone https://github.com/royerlab/hoct
cd hoct
uv sync --extra dev --extra bioio
```

### Test data

Most of the suite runs on tiny synthetic inputs. The data/tracking tests need
candidate-graph GEFF fixtures built from two small Cell Tracking Challenge
training sets; they are skipped until you build them:

```bash
uv run python scripts/prepare_test_data.py
```

This downloads the datasets into `.test-data/` (gitignored) and writes the GEFF
fixtures. To use existing graphs instead, point `HOCT_TEST_GEFF_2D` /
`HOCT_TEST_GEFF_3D` at them.

## Python API

```python
import numpy as np
from hoct import load_model, predict

# Downloads and caches the default pre-trained model on first use. Pass a name
# (see hoct.available_models()) or a local .pt path to use a different one.
model = load_model(device="cuda")

# labels: (T, Y, X) or (T, Z, Y, X) integer array; images: same shape (optional)
labels = np.load("labels.npy")
images = np.load("images.npy")

solution_graph = predict(model, labels=labels, images=images)
solution_graph.to_geff("tracks.geff")
```

See `hoct.predict` for the full signature (custom solver config,
tiled inference, test-time augmentation, etc.).

## Pre-trained models

`load_model()` (and the CLI without `-m`) fetch a JIT-compiled checkpoint from
the project's GitHub releases, verify its SHA256, and cache it under the OS
cache directory (override with `HOCT_CACHE_DIR`). List the available names with
`hoct.available_models()`. The registry lives in `src/hoct/_models.py`.

### Publishing new weights (maintainers)

1. Create a GitHub release whose tag matches `_RELEASE_BASE` in
   `src/hoct/_models.py` (e.g. `weights-v0`) and upload the `.pt` asset.
2. Compute its hash: `shasum -a 256 model.pt`.
3. Add an entry to `MODELS` in `src/hoct/_models.py` with the asset URL and
   hash, and bump `DEFAULT_MODEL` if it should become the default.

## Development

```bash
# Run tests
pytest

# Run linting
ruff check .

# Format code
ruff format .
```
