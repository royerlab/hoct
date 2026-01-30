"""Logging configuration for EET inference."""

import logging
from rich.logging import RichHandler

# Create logger for eet_inference package
LOG = logging.getLogger(__name__)
LOG.addHandler(RichHandler(rich_tracebacks=True))
