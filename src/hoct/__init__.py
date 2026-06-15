"""Inference API for Higher-Order Cell Tracking Transformer (HOCT) model."""

from hoct.__about__ import __version__
from hoct._api import predict

__all__ = ["__version__", "predict"]
