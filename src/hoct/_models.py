"""Pre-trained HOCT model weights: registry and on-demand download.

Weights are distributed as GitHub release assets rather than bundled in the
wheel. The first time a model is requested it is downloaded and cached (with
SHA256 verification) into the OS cache directory; later runs reuse the cache.

Set ``HOCT_CACHE_DIR`` to override the cache location.
"""

from __future__ import annotations

import os
from pathlib import Path

import pooch
import torch

# Base URL for release assets. Bump the tag when publishing new weights.
_RELEASE_BASE = "https://github.com/royerlab/hoct/releases/download/weights-v0"

# Registry of distributed models: name -> {url, sha256}.
MODELS: dict[str, dict[str, str]] = {
    "general_v0": {
        "url": f"{_RELEASE_BASE}/general_v0.pt",
        "sha256": "024c2e4606275c96667907abfc9e0c27487b543480caf99d9ebd1d267cef8e4a",
    },
}

#: Model used when none is specified.
DEFAULT_MODEL = "general_v0"


def available_models() -> list[str]:
    """Return the names of the registered pre-trained models."""
    return list(MODELS)


def _cache_dir() -> Path:
    """Return the directory used to cache downloaded weights."""
    override = os.environ.get("HOCT_CACHE_DIR")
    return Path(override) if override else Path(pooch.os_cache("hoct"))


def resolve_model(model: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a model specification to a local checkpoint path.

    Parameters
    ----------
    model : str | os.PathLike | None
        One of:

        - ``None`` — the default pre-trained model (:data:`DEFAULT_MODEL`),
          downloaded and cached on first use.
        - a registered model name (see :func:`available_models`) — downloaded
          and cached on first use.
        - a path to an existing ``.pt`` checkpoint — returned unchanged.

    Returns
    -------
    Path
        Path to a local checkpoint file.
    """
    if model is None:
        model = DEFAULT_MODEL

    path = Path(model)
    if path.exists():
        return path

    name = str(model)
    if name not in MODELS:
        known = ", ".join(MODELS) or "(none)"
        if path.suffix:  # looked like a file path, but it does not exist
            raise FileNotFoundError(f"Model checkpoint not found: {name}")
        raise ValueError(f"Unknown model {name!r}. Use a path to a .pt file or one of: {known}.")

    entry = MODELS[name]
    cached = pooch.retrieve(
        url=entry["url"],
        known_hash=f"sha256:{entry['sha256']}",
        fname=f"{name}.pt",
        path=_cache_dir(),
        progressbar=True,
    )
    return Path(cached)


def load_model(
    model: str | os.PathLike[str] | None = None,
    *,
    device: str = "cpu",
) -> torch.jit.ScriptModule:
    """Load a JIT-compiled HOCT model, downloading it on demand.

    Parameters
    ----------
    model : str | os.PathLike | None
        Model specification, forwarded to :func:`resolve_model`.
    device : str
        Device to map the model onto (e.g. ``"cuda"``, ``"mps"``, ``"cpu"``).

    Returns
    -------
    torch.jit.ScriptModule
        The model in eval mode on ``device``.
    """
    path = resolve_model(model)
    module = torch.jit.load(path, map_location=device).to(device)
    module.eval()
    return module
