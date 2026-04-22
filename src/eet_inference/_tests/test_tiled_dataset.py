"""Tests for eet_inference.data._tiled_dataset module."""

import itertools

import pytest
import tracksdata as td
from torch.utils.data import IterableDataset
from tracksdata.functional import TilingScheme

from eet_inference._tests.conftest import GEFF_2D, GEFF_3D
from eet_inference.data._batching import DataKeys
from eet_inference.data._tiled_dataset import Tile, TiledRoiDataset


class TestTile:
    """Tests for Tile NamedTuple."""

    def test_tile_structure(self):
        slicing = (slice(0, 10), slice(0, 100), slice(0, 100))
        slicing_wo_overlap = (slice(2, 8), slice(10, 90), slice(10, 90))

        tile = Tile(slicing=slicing, slicing_wo_overlap=slicing_wo_overlap)

        assert tile.slicing == slicing
        assert tile.slicing_wo_overlap == slicing_wo_overlap
        assert len(tile) == 2


class TestTiledRoiDataset:
    """Tests for TiledRoiDataset."""

    @pytest.mark.parametrize("geff_path", [GEFF_2D, GEFF_3D])
    def test_dataset_initialization(self, geff_path):
        """Test dataset is an IterableDataset and has correct spatial cols."""
        graph, _ = td.graph.InMemoryGraph.from_geff(geff_path)

        tiling_scheme = TilingScheme(
            tile_shape=(5, 50, 200, 200),
            overlap_shape=(1, 5, 20, 20),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean", "intensity_std"],
            tiling_scheme=tiling_scheme,
        )

        assert isinstance(dataset, IterableDataset)
        assert dataset._spatial_cols == ["z", "y", "x"]
        assert dataset.graph is graph

    def test_dataset_iter_returns_dataitem(self):
        """Test that iterating returns properly structured DataItem."""
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

        item = next(iter(dataset))

        assert isinstance(item, dict)
        assert DataKeys.NODE_ID in item
        assert DataKeys.NODE_POS in item
        assert DataKeys.NODE_FEATS in item
        assert DataKeys.EDGE_BATCH_ID in item
        assert DataKeys.T in item
        assert item[DataKeys.NODE_POS].shape[1] == 3

    def test_dataset_filters_empty_tiles(self):
        """Test that empty tiles are skipped during iteration."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        tiling_scheme = TilingScheme(
            tile_shape=(5, 50, 500, 500),
            overlap_shape=(1, 5, 50, 50),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean"],
            tiling_scheme=tiling_scheme,
        )

        items = list(dataset)
        assert len(items) > 0
        for item in items:
            assert item[DataKeys.NODE_ID].numel() > 0

    def test_dataset_iter_multiple_tiles(self):
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

        items = list(itertools.islice(dataset, 2))
        if len(items) < 2:
            pytest.skip("Dataset has fewer than 2 non-empty tiles")

        nodes0 = set(items[0][DataKeys.NODE_ID].tolist())
        nodes1 = set(items[1][DataKeys.NODE_ID].tolist())
        assert len(nodes0) != len(nodes1) or nodes0 != nodes1

    def test_dataset_iter_is_repeatable(self):
        """Test that iterating the dataset twice yields the same results."""
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

        items_first = [item[DataKeys.NODE_ID].tolist() for item in dataset]
        items_second = [item[DataKeys.NODE_ID].tolist() for item in dataset]
        assert items_first == items_second

    def test_dataset_with_transforms(self):
        """Test dataset with dataframe and dict transforms."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        def dict_transform(data):
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
            dict_transforms=[dict_transform],
        )

        item = next(iter(dataset))
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

    def test_dataset_large_tile_yields_items(self):
        """Test that a tile covering the entire graph still yields at least one item."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)

        tiling_scheme = TilingScheme(
            tile_shape=(1_000_000,) * 4,
            overlap_shape=(2, 5, 20, 20),
        )

        dataset = TiledRoiDataset(
            graph=graph,
            properties=["intensity_mean"],
            tiling_scheme=tiling_scheme,
        )

        items = list(dataset)
        assert len(items) == 1
