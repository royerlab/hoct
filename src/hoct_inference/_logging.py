"""Logging configuration for HOCT inference."""

import logging

from rich.logging import RichHandler

# Create logger for hoct_inference package
LOG = logging.getLogger(__name__)
LOG.addHandler(RichHandler(rich_tracebacks=True))
