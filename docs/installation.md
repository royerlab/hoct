# Installation

HOCT requires Python 3.11 or later.

## Command line

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed,
run the CLI in an isolated environment:

```bash
uvx --from 'hoct[bioio]' hoct track images.tif segmentation.tif -o tracks.geff
```

For an existing Python environment:

```bash
pip install 'hoct[bioio]'
hoct --help
```

## Python and optional dependencies

```bash
pip install hoct
```

| Extra | Purpose |
| --- | --- |
| `bioio` | Image-file support for the tracking CLI |
| `correction` | Sparse-label fine tuning |
| `demo` | Correction dependencies, Napari with Qt, and Click for the demo |
| `dev` | Tests, linting, and development dependencies |

The demo script lives in the repository; use a checkout to run it. See the
[incremental correction guide](correction.md).

## Tracking solver

HOCT uses tracksdata's ILP solver. Gurobi is included as a dependency and needs
a valid licence. The solver attempts SCIP as a fallback when Gurobi is
unavailable; the fallback must be available in your environment. A successful
package installation alone does not guarantee that a particular solver is usable.

## Models and devices

`load_model()` downloads `general_v1` by default, verifies its SHA256, and caches
it. Available names include `general_v1`, `ctc_v0`, and `general_v0`; use
`hoct.available_models()` to query the registry. A local TorchScript `.pt` path
also works. Set `HOCT_CACHE_DIR` to choose the download directory.

The CLI tries CUDA and falls back when it is unavailable. Choose `--device cpu`
or `--device mps` explicitly if appropriate. In Python, pass an available device
to `load_model(device=...)`; the Python loader does not automatically select one.
