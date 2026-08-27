"""Tests for hoct._api module.

Tests focus on important behavioral aspects of graph creation API.
"""

import numpy as np
import pytest
import tracksdata as td

from hoct import predict, solution_to_tracking_result
from hoct.features.constants import REGIONPROPS
from hoct.features.graph import create_graph
from hoct.tracking import ILPSolverConfig


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

    def test_preserves_sparse_and_reused_label_ids(self):
        labels = np.zeros((2, 16, 16), dtype=np.uint32)
        labels[0, 1:4, 1:4] = 7
        labels[0, 8:11, 8:11] = 42
        labels[1, 2:5, 2:5] = 7

        graph = create_graph(labels, distance_threshold=30.0, n_neighbors=5, delta_t=1)
        identities = graph.node_attrs(attr_keys=["t", "label_id"]).select("t", "label_id").rows()

        assert sorted(identities) == [(0, 7), (0, 42), (1, 7)]
        assert "label_id" not in REGIONPROPS

    def test_scale_controls_candidate_geometry_without_changing_raw_coordinates(self):
        labels = np.zeros((2, 12, 12), dtype=np.uint16)
        labels[0, 1:3, 1:3] = 7
        labels[1, 1:3, 5:7] = 42

        unscaled = create_graph(labels, distance_threshold=5.0, n_neighbors=3, delta_t=1)
        scaled = create_graph(labels, distance_threshold=5.0, n_neighbors=3, delta_t=1, scale=(1.0, 1.0, 2.0))

        assert unscaled.num_edges() == 1
        assert scaled.num_edges() == 0
        assert (
            unscaled.node_attrs(attr_keys=["z", "y", "x"])
            .select("z", "y", "x")
            .equals(scaled.node_attrs(attr_keys=["z", "y", "x"]).select("z", "y", "x"))
        )
        scaled_positions = scaled.node_attrs(attr_keys=["x", "scaled_x"])
        np.testing.assert_allclose(scaled_positions["scaled_x"], scaled_positions["x"] * 2.0)
        assert scaled.metadata["scale"] == (1.0, 1.0, 1.0, 2.0)

    @pytest.mark.parametrize(
        ("shape", "scale", "expected"),
        [
            ((2, 8, 8), (1.0, 1.0, 1.0, 1.0), "3 elements"),
            ((2, 3, 8, 8), (1.0, 1.0, 1.0), "4 elements"),
        ],
    )
    def test_scale_length_is_validated_for_original_dimensions(self, shape, scale, expected):
        labels = np.zeros(shape, dtype=np.uint16)

        with pytest.raises(ValueError, match=expected):
            create_graph(labels, distance_threshold=5.0, n_neighbors=3, delta_t=1, scale=scale)

    def test_out_graph_is_updated_in_place_with_interop_schema(self):
        labels = np.zeros((2, 8, 8), dtype=np.uint16)
        labels[0, 1:3, 1:3] = 7
        labels[1, 2:4, 2:4] = 42
        out_graph = td.graph.InMemoryGraph()

        result = create_graph(
            labels,
            out_graph=out_graph,
            distance_threshold=5.0,
            n_neighbors=3,
            delta_t=1,
        )

        assert result is out_graph
        assert {"label_id", "scaled_z", "scaled_y", "scaled_x"} <= set(result.node_attr_keys())

    @pytest.mark.parametrize("shape", [(2, 8, 8), (2, 3, 8, 8)])
    def test_empty_graph_has_interop_and_scaled_coordinate_schema(self, shape):
        labels = np.zeros(shape, dtype=np.uint16)

        graph = create_graph(labels, distance_threshold=5.0, n_neighbors=3, delta_t=1)

        assert graph.num_nodes() == 0
        assert graph.num_edges() == 0
        assert {"label_id", "scaled_z", "scaled_y", "scaled_x"} <= set(graph.node_attr_keys())

    def test_predict_returns_an_exact_empty_result_without_running_the_model(self):
        labels = np.zeros((2, 8, 8), dtype=np.uint16)

        graph = predict(None, labels=labels, distance_threshold=5.0, n_neighbors=3, max_delta_t=1)
        result = solution_to_tracking_result(graph)

        assert result.detections.shape == (0, 2)
        assert result.links.shape == (0, 4)
        assert result.similarities.shape == (0,)

    def test_predict_returns_isolated_detections_when_no_candidate_link_exists(self):
        labels = np.zeros((2, 16, 16), dtype=np.uint16)
        labels[0, 1:3, 1:3] = 7
        labels[1, 12:14, 12:14] = 42

        graph = predict(None, labels=labels, distance_threshold=1.0, n_neighbors=3, max_delta_t=1)
        result = solution_to_tracking_result(graph)

        np.testing.assert_array_equal(result.detections, [[0, 7], [1, 42]])
        assert result.links.shape == (0, 4)
        assert result.similarities.shape == (0,)


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
