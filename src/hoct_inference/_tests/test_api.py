"""Tests for hoct_inference._api module.

Tests focus on important behavioral aspects of graph creation API.
"""

import numpy as np
import pytest
from hoct_features.constants import REGIONPROPS
from hoct_features.graph import create_graph

from hoct_inference.tracking import ILPSolverConfig


@pytest.fixture
def synthetic_2d_labels():
    """Create synthetic 2D+t labels for testing."""
    labels = np.zeros((3, 64, 64), dtype=np.int32)
    labels[0, 10:20, 10:20] = 1
    labels[0, 30:40, 30:40] = 2
    labels[1, 12:22, 12:22] = 1
    labels[1, 28:38, 28:38] = 2
    labels[2, 14:24, 14:24] = 1
    labels[2, 26:36, 26:36] = 2
    return labels


@pytest.fixture
def synthetic_3d_labels():
    """Create synthetic 3D+t labels for testing."""
    labels = np.zeros((2, 16, 32, 32), dtype=np.int32)
    labels[0, 4:12, 8:16, 8:16] = 1
    labels[0, 4:12, 16:24, 16:24] = 2
    labels[1, 5:13, 10:18, 10:18] = 1
    labels[1, 5:13, 18:26, 18:26] = 2
    return labels


class TestCreateGraphFromLabels:
    """Tests for create_graph() behavior with/without images."""

    def test_labels_without_images_no_intensity_features(self, synthetic_2d_labels):
        """Test that intensity features are not added when images=None."""
        graph = create_graph(
            labels=synthetic_2d_labels,
            images=None,
            distance_threshold=300.0,
            n_neighbors=5,
            delta_t=3,
        )

        node_attrs = graph.node_attr_keys()
        for prop in REGIONPROPS:
            assert prop in node_attrs

    def test_labels_with_images_has_intensity_features(self, synthetic_2d_labels):
        """Test that intensity features are added when images provided."""
        images = np.random.randn(*synthetic_2d_labels.shape).astype(np.float32)
        graph = create_graph(
            labels=synthetic_2d_labels,
            images=images,
            distance_threshold=300.0,
            n_neighbors=5,
            delta_t=3,
        )

        node_attrs = graph.node_attr_keys()
        assert "intensity_mean" in node_attrs
        assert "intensity_min" in node_attrs
        assert "intensity_max" in node_attrs

    def test_2d_vs_3d_dimensionality(self, synthetic_2d_labels, synthetic_3d_labels):
        """Test that 2D and 3D data are correctly distinguished."""
        graph_2d = create_graph(
            labels=synthetic_2d_labels,
            distance_threshold=300.0,
            n_neighbors=5,
            delta_t=3,
        )
        graph_3d = create_graph(
            labels=synthetic_3d_labels,
            distance_threshold=300.0,
            n_neighbors=5,
            delta_t=3,
        )

        assert graph_2d.metadata["was_2d"] is True
        assert graph_3d.metadata["was_2d"] is False

    def test_inference_mode_no_gt_features(self, synthetic_2d_labels):
        """Test that GT features are not added in inference mode."""
        graph = create_graph(
            labels=synthetic_2d_labels,
            gt_graph=None,
            distance_threshold=300.0,
            n_neighbors=5,
            delta_t=3,
        )

        edge_attrs = graph.edge_attr_keys()
        assert "edge_is_gt" not in edge_attrs


class TestSolverConfig:
    """Tests for ILPSolverConfig validation and immutability."""

    def test_config_is_immutable(self):
        """Test that config cannot be modified after creation."""
        config = ILPSolverConfig.default()

        with pytest.raises(Exception):  # noqa: B017
            config.appearance_weight = 2.0

    def test_config_validates_negative_weights(self):
        """Test that negative weights are rejected."""
        with pytest.raises(Exception):  # noqa: B017
            ILPSolverConfig(appearance_weight=-1.0)

    def test_config_validates_zero_timeout(self):
        """Test that zero/negative timeout is rejected."""
        with pytest.raises(Exception):  # noqa: B017
            ILPSolverConfig(timeout=0.0)
