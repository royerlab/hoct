"""Tests for hoct.data._batching module."""

import pytest
import torch
import tracksdata as td

from hoct._tests.conftest import GEFF_2D, GEFF_3D, requires_geff_data
from hoct.data._batching import (
    DataKeys,
    _pad_tensor,
    collate_varying_length,
    item_from_filter,
)


class TestPadTensor:
    """Tests for _pad_tensor function."""

    def test_pad_1d_tensor(self):
        """Test padding 1D tensor."""
        tensor = torch.tensor([1, 2, 3], dtype=torch.long)
        padded = _pad_tensor(tensor, n_samples=5)

        assert padded.shape == (5,)
        assert torch.equal(padded[:3], tensor)
        assert torch.equal(padded[3:], torch.zeros(2))
        assert padded.dtype == torch.long

    def test_pad_2d_tensor_samples_only(self):
        """Test padding 2D tensor on sample dimension only."""
        tensor = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
        padded = _pad_tensor(tensor, n_samples=4, n_slots=None)

        assert padded.shape == (4, 2)
        assert torch.equal(padded[:2], tensor)
        assert torch.equal(padded[2:], torch.zeros(2, 2))

    def test_pad_2d_tensor_both_dimensions(self):
        """Test padding 2D tensor on both dimensions."""
        tensor = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
        padded = _pad_tensor(tensor, n_samples=4, n_slots=5)

        assert padded.shape == (4, 5)
        assert torch.equal(padded[:2, :2], tensor)
        # Check padding is zeros
        assert padded[2:, :].sum() == 0
        assert padded[:, 2:].sum() == 0

    def test_pad_3d_tensor(self):
        """Test padding 3D tensor preserves higher dimensions."""
        tensor = torch.ones(2, 3, 4)  # (samples, slots, features)
        padded = _pad_tensor(tensor, n_samples=5, n_slots=6)

        assert padded.shape == (5, 6, 4)
        assert torch.equal(padded[:2, :3, :], tensor)
        # Original features preserved
        assert padded[:2, :3, :].sum() == 2 * 3 * 4


@requires_geff_data
class TestItemFromFilter:
    """Tests for item_from_filter function with real GEFF data."""

    @pytest.mark.parametrize(
        "geff_path,expected_dims",
        [
            (GEFF_2D, 2),  # DIC-C2DH-HeLa is 2D
            (GEFF_3D, 3),  # Fluo-C3DL-MDA231 is 3D
        ],
    )
    def test_item_from_filter_structure(self, geff_path, expected_dims):
        """Test that item_from_filter returns properly structured DataItem."""
        # Load graph
        graph, _ = td.graph.InMemoryGraph.from_geff(geff_path)

        # Get all nodes at first timepoint
        t0_nodes = list(graph._time_to_nodes[0])
        sp_filter = graph.filter(node_ids=t0_nodes)

        # Determine spatial columns based on dimensions
        spatial_cols = ["x", "y"] if expected_dims == 2 else ["x", "y", "z"]
        properties = ["intensity_mean", "intensity_std"]

        # Create item
        item = item_from_filter(
            sp_filter,
            spatial_cols=spatial_cols,
            properties=properties,
            df_transforms=[],
            dict_transforms=[],
        )

        # Verify DataItem structure
        assert isinstance(item, dict)
        n_nodes = len(t0_nodes)

        # Check node tensors
        assert item[DataKeys.NODE_ID].shape == (n_nodes,)
        assert item[DataKeys.NODE_POS].shape[0] == n_nodes
        assert item[DataKeys.NODE_POS].shape[1] == expected_dims
        assert item[DataKeys.NODE_FEATS].ndim == 2
        assert item[DataKeys.NODE_FEATS].shape[0] == n_nodes
        assert item[DataKeys.T].shape == (n_nodes,)

        # Check edge tensors exist
        assert DataKeys.EDGE_ID in item
        assert DataKeys.EDGE_BATCH_ID in item
        assert DataKeys.EDGE_POS in item
        assert DataKeys.DELTA_T in item
        assert DataKeys.SOURCE_T in item

        # Check edge batch IDs are 0-indexed and valid
        if len(item[DataKeys.EDGE_BATCH_ID]) > 0:
            edge_batch_ids = item[DataKeys.EDGE_BATCH_ID]
            assert edge_batch_ids.shape[1] == 2  # (source, target)
            assert edge_batch_ids.min() >= 0
            assert edge_batch_ids.max() < n_nodes

        # Check graph references
        assert item[DataKeys.GRAPH] is None
        assert item[DataKeys.GT_GRAPH] is None

    def test_item_from_filter_with_gt_edges(self):
        """Test that ground truth edges are loaded when present."""
        # Load graph with ground truth
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        # Get nodes from multiple timepoints to get edges
        nodes = []
        for t in range(min(3, len(graph._time_to_nodes))):
            nodes.extend(graph._time_to_nodes[t])

        sp_filter = graph.filter(node_ids=nodes)

        item = item_from_filter(
            sp_filter,
            spatial_cols=["x", "y"],
            properties=["intensity_mean"],
            df_transforms=[],
            dict_transforms=[],
        )

        # Should have edges between timepoints
        assert len(item[DataKeys.EDGE_ID]) > 0

        # Check if ground truth labels exist
        if item[DataKeys.EDGE_TARGETS] is not None:
            assert item[DataKeys.EDGE_TARGETS].shape[0] == len(item[DataKeys.EDGE_ID])
            assert item[DataKeys.EDGE_TARGETS].shape[1] == 1
            # GT labels should be 0 or 1
            assert torch.all((item[DataKeys.EDGE_TARGETS] == 0) | (item[DataKeys.EDGE_TARGETS] == 1))

    def test_item_from_filter_edge_positions(self):
        """Test that edge positions are computed as midpoint of source and target."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        # Get nodes from first two timepoints
        nodes = list(graph._time_to_nodes[0]) + list(graph._time_to_nodes[1])
        sp_filter = graph.filter(node_ids=nodes)

        item = item_from_filter(
            sp_filter,
            spatial_cols=["x", "y"],
            properties=[],
            df_transforms=[],
            dict_transforms=[],
        )

        # Verify edge positions are between node positions
        if len(item[DataKeys.EDGE_ID]) > 0:
            node_pos = item[DataKeys.NODE_POS]
            edge_pos = item[DataKeys.EDGE_POS]
            edge_batch_ids = item[DataKeys.EDGE_BATCH_ID]

            # Check first edge
            source_idx = edge_batch_ids[0, 0]
            target_idx = edge_batch_ids[0, 1]
            expected_pos = (node_pos[source_idx] + node_pos[target_idx]) * 0.5
            assert torch.allclose(edge_pos[0], expected_pos, atol=1e-5)

    def test_item_from_filter_no_nans(self):
        """Test that nulls and nans are filled with zeros."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_3D)

        t0_nodes = list(graph._time_to_nodes[0])
        sp_filter = graph.filter(node_ids=t0_nodes)

        item = item_from_filter(
            sp_filter,
            spatial_cols=["x", "y", "z"],
            properties=["intensity_mean", "intensity_std", "intensity_min", "intensity_max"],
            df_transforms=[],
            dict_transforms=[],
        )

        # No nans should exist in features
        assert not torch.isnan(item[DataKeys.NODE_FEATS]).any()
        assert not torch.isnan(item[DataKeys.NODE_POS]).any()
        assert not torch.isnan(item[DataKeys.EDGE_POS]).any() if len(item[DataKeys.EDGE_POS]) > 0 else True


class TestCollateVaryingLength:
    """Tests for collate_varying_length function."""

    def test_collate_varying_sizes(self):
        """Test batching items with different numbers of nodes and edges."""
        # Create mock items with varying sizes
        items = [
            {
                DataKeys.NODE_ID: torch.tensor([0, 1, 2]),
                DataKeys.NODE_POS: torch.randn(3, 2),
                DataKeys.NODE_FEATS: torch.randn(3, 5),
                DataKeys.EDGE_BATCH_ID: torch.tensor([[0, 1], [1, 2]]),
                DataKeys.EDGE_POS: torch.randn(2, 2),
                DataKeys.EDGE_TARGETS: torch.tensor([[1], [0]]),
                DataKeys.T: torch.tensor([0, 0, 0]),
                DataKeys.DELTA_T: torch.tensor([1.0, 1.0]),
            },
            {
                DataKeys.NODE_ID: torch.tensor([3, 4, 5, 6]),
                DataKeys.NODE_POS: torch.randn(4, 2),
                DataKeys.NODE_FEATS: torch.randn(4, 5),
                DataKeys.EDGE_BATCH_ID: torch.tensor([[0, 1]]),
                DataKeys.EDGE_POS: torch.randn(1, 2),
                DataKeys.EDGE_TARGETS: torch.tensor([[1]]),
                DataKeys.T: torch.tensor([0, 0, 0, 0]),
                DataKeys.DELTA_T: torch.tensor([1.0]),
            },
        ]

        batch = collate_varying_length(items)

        # Check batch dimensions
        assert batch[DataKeys.NODE_MASK].shape == (2, 4)  # 2 items, max 4 nodes
        assert batch[DataKeys.EDGE_MASK].shape == (2, 2)  # 2 items, max 2 edges

        # Check node masks are correct
        assert torch.all(batch[DataKeys.NODE_MASK][0, :3])  # First item has 3 nodes
        assert not batch[DataKeys.NODE_MASK][0, 3]  # Padding is masked
        assert torch.all(batch[DataKeys.NODE_MASK][1, :])  # Second item has 4 nodes

        # Check edge masks are correct
        assert torch.all(batch[DataKeys.EDGE_MASK][0, :])  # First item has 2 edges
        assert batch[DataKeys.EDGE_MASK][1, 0]  # Second item has 1 edge
        assert not batch[DataKeys.EDGE_MASK][1, 1]  # Padding is masked

        # Check tensors are properly stacked
        assert batch[DataKeys.NODE_POS].shape == (2, 4, 2)
        assert batch[DataKeys.NODE_FEATS].shape == (2, 4, 5)
        assert batch[DataKeys.EDGE_POS].shape == (2, 2, 2)

    @requires_geff_data
    def test_collate_preserves_non_tensors(self):
        """Test that non-tensor items (graphs) are preserved as lists."""
        graph1, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        items = [
            {
                DataKeys.NODE_ID: torch.tensor([0]),
                DataKeys.NODE_POS: torch.randn(1, 2),
                DataKeys.NODE_FEATS: torch.randn(1, 3),
                DataKeys.EDGE_BATCH_ID: torch.tensor([[0, 0]]),
                DataKeys.EDGE_POS: torch.randn(1, 2),
                DataKeys.T: torch.tensor([0]),
                DataKeys.DELTA_T: torch.tensor([1.0]),
                DataKeys.GRAPH: graph1,
                DataKeys.GT_GRAPH: None,
            },
            {
                DataKeys.NODE_ID: torch.tensor([1]),
                DataKeys.NODE_POS: torch.randn(1, 2),
                DataKeys.NODE_FEATS: torch.randn(1, 3),
                DataKeys.EDGE_BATCH_ID: torch.tensor([[0, 0]]),
                DataKeys.EDGE_POS: torch.randn(1, 2),
                DataKeys.T: torch.tensor([0]),
                DataKeys.DELTA_T: torch.tensor([1.0]),
                DataKeys.GRAPH: None,
                DataKeys.GT_GRAPH: None,
            },
        ]

        batch = collate_varying_length(items)

        # Graphs should be in a list
        assert isinstance(batch[DataKeys.GRAPH], list)
        assert len(batch[DataKeys.GRAPH]) == 2
        assert batch[DataKeys.GRAPH][0] is graph1
        assert batch[DataKeys.GRAPH][1] is None

    def test_collate_padding_is_zeros(self):
        """Test that padded regions are filled with zeros."""
        items = [
            {
                DataKeys.NODE_ID: torch.tensor([0, 1]),
                DataKeys.NODE_POS: torch.ones(2, 2),
                DataKeys.NODE_FEATS: torch.ones(2, 3),
                DataKeys.EDGE_BATCH_ID: torch.tensor([[0, 1]]),
                DataKeys.EDGE_POS: torch.ones(1, 2),
                DataKeys.T: torch.ones(2),
                DataKeys.DELTA_T: torch.ones(1),
            },
            {
                DataKeys.NODE_ID: torch.tensor([2, 3, 4]),
                DataKeys.NODE_POS: torch.ones(3, 2),
                DataKeys.NODE_FEATS: torch.ones(3, 3),
                DataKeys.EDGE_BATCH_ID: torch.tensor([[0, 1], [1, 2]]),
                DataKeys.EDGE_POS: torch.ones(2, 2),
                DataKeys.T: torch.ones(3),
                DataKeys.DELTA_T: torch.ones(2),
            },
        ]

        batch = collate_varying_length(items)

        # Check padded node is zeros
        assert torch.all(batch[DataKeys.NODE_POS][0, 2] == 0)
        assert torch.all(batch[DataKeys.NODE_FEATS][0, 2] == 0)

        # Check non-padded values are preserved
        assert torch.all(batch[DataKeys.NODE_POS][0, :2] == 1)
        assert torch.all(batch[DataKeys.NODE_FEATS][1, :] == 1)
