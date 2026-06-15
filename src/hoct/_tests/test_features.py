"""Tests for hoct.features.features module."""

import numpy as np

from hoct.features import features


class TestBorderDistance:
    """Tests for border distance calculation functions."""

    def test_border_dist_2d_behavior(self):
        """Test 2D border distance with various positions."""
        shape = (100, 100)

        # Test corner (should be 1.0 - at border)
        corner_dist = features.border_dist_2d(np.array([0]), np.array([0]), shape=shape)
        assert corner_dist[0] == 1.0

        # Test center (should be 0.0 - far from border with default cutoff=5)
        center_dist = features.border_dist_2d(np.array([50]), np.array([50]), shape=shape)
        assert center_dist[0] == 0.0

        # Test near border (should be between 0 and 1)
        near_border = features.border_dist_2d(np.array([2]), np.array([50]), shape=shape)
        assert 0 < near_border[0] < 1

        # Test multiple points at once
        y = np.array([0, 50, 2])
        x = np.array([0, 50, 50])
        dists = features.border_dist_2d(y, x, shape=shape)
        assert len(dists) == 3
        assert dists[0] == 1.0  # corner
        assert dists[1] == 0.0  # center
        assert 0 < dists[2] < 1  # near border

    def test_border_dist_3d_behavior(self):
        """Test 3D border distance calculation."""
        shape = (100, 100, 100)

        # Corner should be at border
        corner_dist = features.border_dist_3d(np.array([0]), np.array([0]), np.array([0]), shape=shape)
        assert corner_dist[0] == 1.0

        # Center should be far from border
        center_dist = features.border_dist_3d(np.array([50]), np.array([50]), np.array([50]), shape=shape)
        assert center_dist[0] == 0.0

    def test_border_dist_currying(self):
        """Test that border distance functions support currying."""
        shape = (100, 100)

        # Partially apply shape parameter
        border_func = features.border_dist_2d(shape=shape)

        # Apply remaining parameters
        result = border_func(np.array([50]), np.array([50]))
        assert len(result) == 1
        assert result[0] == 0.0


class TestNormalizeImage:
    """Tests for image normalization."""

    def test_normalize_range(self):
        """Test that normalized images are in correct range."""
        image = np.array([[0, 50, 100], [150, 200, 255]], dtype=np.float32)

        # Without clipping
        normalized = features.normalize_image(image, clip=False)
        assert normalized.shape == image.shape
        assert normalized.min() >= 0

        # With clipping
        normalized_clip = features.normalize_image(image, clip=True)
        assert 0 <= normalized_clip.min() <= normalized_clip.max() <= 1

    def test_normalize_quantile(self):
        """Test normalization respects quantile parameter."""
        # Create image with outlier
        image = np.ones((10, 10)) * 100
        image[0, 0] = 1000  # outlier

        # With default quantile (0.999), outlier should be clipped
        normalized = features.normalize_image(image, uq=0.95, clip=True)
        assert normalized.max() <= 1.0
        assert normalized.shape == image.shape


class TestGraphAttributeFunctions:
    """Tests for graph attribute addition functions (integration-level)."""

    def test_add_is_div_exists(self):
        """Verify add_is_div function exists and has correct signature."""
        import inspect

        assert callable(features.add_is_div)
        sig = inspect.signature(features.add_is_div)
        assert "graph" in sig.parameters
        assert "gt_graph" in sig.parameters

    def test_add_delta_t_exists(self):
        """Verify add_delta_t function exists and has correct signature."""
        import inspect

        assert callable(features.add_delta_t)
        sig = inspect.signature(features.add_delta_t)
        assert "graph" in sig.parameters

    def test_add_border_dist_exists(self):
        """Verify add_border_dist function exists and has correct signature."""
        import inspect

        assert callable(features.add_border_dist)
        sig = inspect.signature(features.add_border_dist)
        params = sig.parameters
        assert "graph" in params
        assert "shape" in params
        assert "was_2d" in params
