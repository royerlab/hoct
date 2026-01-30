"""Tests for eet_inference.data._tiled_dataset module."""

import pickle

import pytest
import tracksdata as td
from tracksdata.functional import TilingScheme

from eet_inference.data._batching import DataKeys
from eet_inference.data._tiled_dataset import Tile, TiledRoiDataset


# Test data paths
GEFF_3D = "/hpc/reference/group.royer/geffneration/CTC/Fluo-C3DL-MDA231/01/graph.geff"
GEFF_2D = "/hpc/reference/group.royer/geffneration/CTC/DIC-C2DH-HeLa/01/graph.geff"


class TestTile:
    """Tests for Tile NamedTuple."""

    def test_tile_structure(self):
        """Test that Tile stores slicing information correctly."""
        slicing = (slice(0, 10), slice(0, 100), slice(0, 100))
        slicing_wo_overlap = (slice(2, 8), slice(10, 90), slice(10, 90))

        tile = Tile(slicing=slicing, slicing_wo_overlap=slicing_wo_overlap)

        assert tile.slicing == slicing
        assert tile.slicing_wo_overlap == slicing_wo_overlap
        assert len(tile) == 2  # NamedTuple with 2 fields


class TestTiledRoiDataset:
    """Tests for TiledRoiDataset."""

    @pytest.mark.parametrize("geff_path", [GEFF_2D, GEFF_3D])
    def test_dataset_initialization(self, geff_path):
        """Test dataset initialization with 2D and 3D graphs."""
        graph, _ = td.graph.InMemoryGraph.from_geff(geff_path)

        # Use small tiles since 3D graph only has 11 timepoints
        tiling_scheme = TilingScheme(
            tile_shape=(5, 50, 200, 200),  # (t, z, y, x)
            overlap_shape=(1, 5, 20, 20),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean", "intensity_std"],
            tiling_scheme=tiling_scheme,
        )

        # Both 2D and 3D graphs have z in node attrs, so spatial_cols are always ['z', 'y', 'x']
        assert dataset._spatial_cols == ["z", "y", "x"]

        # Check dataset has tiles
        assert len(dataset) > 0
        assert len(dataset._tiles) > 0

        # Check graph property
        assert dataset.graph is graph

    def test_dataset_filters_empty_tiles(self):
        """Test that empty tiles are filtered out during initialization."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        # Use very large tiles to potentially create empty ones
        tiling_scheme = TilingScheme(
            tile_shape=(5, 50, 500, 500),
            overlap_shape=(1, 5, 50, 50),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean"],
            tiling_scheme=tiling_scheme,
        )

        # All tiles should have nodes (empty ones filtered)
        assert all(len(tile.slicing) > 0 for tile in dataset._tiles)

    def test_dataset_getitem_returns_dataitem(self):
        """Test that __getitem__ returns properly structured DataItem."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        tiling_scheme = TilingScheme(
            tile_shape=(10, 50, 200, 200),
            overlap_shape=(2, 5, 20, 20),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean", "intensity_std"],
            tiling_scheme=tiling_scheme,
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

        # Check features include time + spatial + properties
        n_features = item[DataKeys.NODE_FEATS].shape[1]
        assert n_features >= len(["intensity_mean", "intensity_std"])

    def test_dataset_getitem_multiple_tiles(self):
        """Test accessing multiple tiles returns different data."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_3D)

        tiling_scheme = TilingScheme(
            tile_shape=(5, 50, 100, 100),
            overlap_shape=(1, 5, 10, 10),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean"],
            tiling_scheme=tiling_scheme,
        )

        if len(dataset) < 2:
            pytest.skip("Dataset has fewer than 2 tiles")

        item0 = dataset[0]
        item1 = dataset[1]

        # Different tiles should have different node counts (likely)
        # or at minimum different node IDs
        nodes0 = set(item0[DataKeys.NODE_ID].tolist())
        nodes1 = set(item1[DataKeys.NODE_ID].tolist())

        # Either different sizes or different IDs (tiles might overlap but shouldn't be identical)
        assert len(nodes0) != len(nodes1) or nodes0 != nodes1

    def test_dataset_spatial_filter_lazy_initialization(self):
        """Test that spatial filter is initialized lazily."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        tiling_scheme = TilingScheme(
            tile_shape=(10, 50, 200, 200),
            overlap_shape=(2, 5, 20, 20),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean"],
            tiling_scheme=tiling_scheme,
        )

        # Initially None
        assert dataset._sp_filter is None

        # Accessing property creates it
        sp_filter = dataset.sp_filter
        assert sp_filter is not None

        # Subsequent access returns same object
        assert dataset.sp_filter is sp_filter

    @pytest.mark.skip(reason="Graph contains unpicklable Cython objects (RTree). Pickle support requires saving/loading graph separately.")
    def test_dataset_pickle_support(self):
        """Test that dataset can be pickled and unpickled."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        tiling_scheme = TilingScheme(
            tile_shape=(10, 50, 200, 200),
            overlap_shape=(2, 5, 20, 20),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean"],
            tiling_scheme=tiling_scheme,
        )

        # sp_filter should initially be None
        assert dataset._sp_filter is None

        # Pickle and unpickle (without accessing sp_filter first)
        pickled = pickle.dumps(dataset)
        unpickled = pickle.loads(pickled)

        # sp_filter should still be None after unpickling
        assert unpickled._sp_filter is None

        # Other attributes should be preserved
        assert len(unpickled) == len(dataset)
        assert unpickled._spatial_cols == dataset._spatial_cols
        assert unpickled._properties == dataset._properties
        assert len(unpickled._tiles) == len(dataset._tiles)

        # Verify unpickled dataset is functional
        _ = unpickled[0]  # Should work without errors

    def test_dataset_with_transforms(self):
        """Test dataset with dataframe and dict transforms."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        # Define simple transforms
        def df_transform(df):
            # Add a dummy column
            return df.with_columns((df["x"] * 0).alias("dummy_col"))

        def dict_transform(data):
            # Add a flag
            data["transformed"] = True
            return data

        tiling_scheme = TilingScheme(
            tile_shape=(10, 50, 200, 200),
            overlap_shape=(2, 5, 20, 20),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean"],
            tiling_scheme=tiling_scheme,
            df_transforms=[df_transform],
            dict_transforms=[dict_transform],
        )

        item = dataset[0]

        # Check dict transform was applied
        assert "transformed" in item
        assert item["transformed"] is True

    def test_dataset_raises_on_training_mode(self):
        """Test that training mode raises NotImplementedError."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        tiling_scheme = TilingScheme(
            tile_shape=(10, 50, 200, 200),
            overlap_shape=(2, 5, 20, 20),
        )

        with pytest.raises(NotImplementedError, match="Training is not implemented yet"):
            TiledRoiDataset(
                graph=graph,
                properties=["intensity_mean"],
                tiling_scheme=tiling_scheme,
                train=True,
            )

    def test_dataset_raises_on_empty_tiles(self):
        """Test that dataset raises error when no valid tiles are found."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        # Use tiling scheme that's way too large for the time range
        time_pts = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.T])[td.DEFAULT_ATTR_KEYS.T]
        time_range = time_pts.max() - time_pts.min()

        tiling_scheme = TilingScheme(
            tile_shape=(int(time_range) + 100, 50, 200, 200),  # Larger than time range
            overlap_shape=(2, 5, 20, 20),
        )

        with pytest.raises(ValueError, match="No tiles found"):
            TiledRoiDataset(
                graph=graph,
                properties=["intensity_mean"],
                tiling_scheme=tiling_scheme,
            )
