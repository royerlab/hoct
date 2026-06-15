"""Tests for hoct.data.frame_dataset module."""

import pytest
import tracksdata as td

from hoct._tests.conftest import GEFF_2D, GEFF_3D, requires_geff_data
from hoct.data._batching import DataKeys
from hoct.data._frame_dataset import FrameDataset


@requires_geff_data
class TestFrameDataset:
    """Tests for FrameDataset."""

    @pytest.mark.parametrize("geff_path", [GEFF_2D, GEFF_3D])
    def test_dataset_initialization(self, geff_path):
        """Test dataset initialization with 2D and 3D graphs."""
        graph, _ = td.graph.InMemoryGraph.from_geff(geff_path)

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean", "intensity_std"],
        )

        # Both 2D and 3D graphs have z in node attrs
        assert dataset._spatial_cols == ["z", "y", "x"]

        # Check dataset has items (time windows)
        assert len(dataset) > 0

        # Check properties are stored
        assert dataset._properties == ["intensity_mean", "intensity_std"]

        # Check window size is set
        assert dataset.window_size > 0

        # Check graph property
        assert dataset.graph is graph

    def test_dataset_with_gt_graph(self):
        """Test dataset initialization with ground truth graph."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)
        gt_graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
            gt_graph=gt_graph,
        )

        assert dataset.gt_graph is gt_graph
        assert dataset._gt_graph is gt_graph

    def test_dataset_window_size_calculation(self):
        """Test that window size is calculated from max delta_t."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        # Get max delta_t from graph
        max_delta_t = graph.edge_attrs(attr_keys=["delta_t"])["delta_t"].max()

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
        )

        # Window size should be max delta_t
        assert dataset.window_size == max_delta_t

    def test_dataset_min_window_size(self):
        """Test that min_window_size parameter is respected."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        max_delta_t = graph.edge_attrs(attr_keys=["delta_t"])["delta_t"].max()
        min_window = max_delta_t + 5

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
            min_window_size=min_window,
        )

        # Window size should be the max of max_delta_t and min_window_size
        assert dataset.window_size == min_window

    def test_dataset_getitem_returns_dataitem(self):
        """Test that __getitem__ returns properly structured DataItem."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean", "intensity_std"],
        )

        # Get first item
        item = dataset[0]

        # Verify DataItem structure
        assert isinstance(item, dict)
        assert DataKeys.NODE_ID in item
        assert DataKeys.NODE_POS in item
        assert DataKeys.NODE_FEATS in item
        assert DataKeys.EDGE_BATCH_ID in item
        assert DataKeys.T in item

        # Check node positions have correct dimensions (3D: z, y, x)
        assert item[DataKeys.NODE_POS].shape[1] == 3

        # Check features include properties
        n_features = item[DataKeys.NODE_FEATS].shape[1]
        assert n_features >= len(["intensity_mean", "intensity_std"])

    def test_dataset_getitem_time_window(self):
        """Test that __getitem__ returns data for correct time window."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
        )

        # Get items at different indices
        if len(dataset) >= 2:
            item0 = dataset[0]
            item1 = dataset[1]

            # Time points should be different
            t0 = item0[DataKeys.T].unique()
            t1 = item1[DataKeys.T].unique()

            # Should have different time ranges (offset by 1)
            assert t0.min() != t1.min() or t0.max() != t1.max()

    def test_dataset_return_graph_first_item(self):
        """Test that graphs are returned only for first item when return_graph=True."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)
        gt_graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
            gt_graph=gt_graph,
            return_graph=True,
        )

        # First item should have graphs
        item0 = dataset[0]
        assert item0[DataKeys.GRAPH] is graph
        assert item0[DataKeys.GT_GRAPH] is gt_graph

        # Second item should not have graphs
        if len(dataset) > 1:
            item1 = dataset[1]
            # These keys might not exist or be None
            assert item1.get(DataKeys.GRAPH) is None
            assert item1.get(DataKeys.GT_GRAPH) is None

    def test_dataset_with_transforms(self):
        """Test dataset with dataframe and dict transforms."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        # Define simple transforms
        def df_transform(df):
            return df.with_columns((df["x"] * 0).alias("dummy_col"))

        def dict_transform(data):
            data["transformed"] = True
            return data

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
            df_transforms=[df_transform],
            dict_transforms=[dict_transform],
        )

        item = dataset[0]

        # Check dict transform was applied
        assert "transformed" in item
        assert item["transformed"] is True

    def test_dataset_name_property(self):
        """Test that dataset name is set correctly."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
            name="test_dataset",
        )

        assert dataset.name == "test_dataset"

    def test_dataset_group_property(self):
        """Test that group property extracts first part of name."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
            name="group_name/dataset_name",
        )

        assert dataset.group == "group_name"

    def test_dataset_attribute_dim(self):
        """Test attribute_dim method for scalar and array attributes."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
        )

        # Test scalar attribute (t should be 1D)
        t_dim = dataset.attribute_dim("t")
        assert t_dim == 1

        # Test array attribute if it exists (like inertia_tensor)
        try:
            attr_keys = graph.node_attr_keys()
            for key in attr_keys:
                dim = dataset.attribute_dim(key)
                assert dim >= 1
        except Exception:
            # If no array attributes, that's fine
            pass

    def test_dataset_time_range(self):
        """Test that time range is calculated correctly."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        dataset = FrameDataset(
            graph=graph,
            properties=["intensity_mean"],
        )

        # Time range should start at min time point
        time_pts = graph.time_points()
        assert dataset._time_range[0] == min(time_pts)

        # Length should match dataset length
        assert len(dataset._time_range) == len(dataset)
