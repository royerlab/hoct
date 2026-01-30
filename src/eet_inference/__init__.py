"""Inference API for Edge Embedding Tracking (EET) model."""

from eet_inference.__about__ import __version__
from eet_inference._api import predict

__all__ = ["__version__", "predict"]
