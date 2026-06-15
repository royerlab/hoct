"""Tests for hoct.features.constants module."""

from hoct.features import constants


def test_constants_values():
    """Test that constants have expected values and types."""
    # Verify EDGE_GT_KEY
    assert constants.EDGE_GT_KEY == "edge_is_gt"

    # Verify REGIONPROPS contains expected properties
    expected_props = {
        "equivalent_diameter_area",
        "intensity_min",
        "intensity_max",
        "intensity_std",
        "intensity_mean",
        "inertia_tensor",
        "border_dist",
    }
    assert set(constants.REGIONPROPS) == expected_props
