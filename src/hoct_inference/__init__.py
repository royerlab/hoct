"""Inference API for Higher-Order Cell Tracking Transformer (HOCT) model."""

from hoct_inference.__about__ import __version__
from hoct_inference._api import predict

__all__ = ["__version__", "predict"]
