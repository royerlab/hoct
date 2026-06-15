"""Shared test configuration and fixtures for hoct tests."""

import os
from pathlib import Path

import pytest

# Candidate-graph GEFF fixtures used by the data/tracking tests. Build them with
# `uv run python scripts/prepare_test_data.py`, or point the env vars at
# existing copies (e.g. on a cluster) to run these tests against other data.
_GEFF_DIR = Path(__file__).resolve().parents[3] / ".test-data" / "geff"
GEFF_2D = os.environ.get("HOCT_TEST_GEFF_2D", str(_GEFF_DIR / "huh7_2d.geff"))
GEFF_3D = os.environ.get("HOCT_TEST_GEFF_3D", str(_GEFF_DIR / "mda231_3d.geff"))

# Skip data-dependent tests when the GEFF fixtures are unreachable, so the suite
# stays green for contributors and CI without cluster access.
requires_geff_data = pytest.mark.skipif(
    not (Path(GEFF_2D).exists() and Path(GEFF_3D).exists()),
    reason="GEFF test data not available (set HOCT_TEST_GEFF_2D / HOCT_TEST_GEFF_3D)",
)
