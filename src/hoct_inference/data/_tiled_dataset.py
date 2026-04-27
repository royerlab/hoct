from collections.abc import Callable, Generator, Sequence
from typing import NamedTuple

import polars as pl
import tracksdata as td
from torch.utils.data import IterableDataset
from tracksdata.functional import TilingScheme, apply_tiled

from hoct_inference.data._batching import DataItem, item_from_filter
from hoct_inference.data._transforms import Translate


class Tile(NamedTuple):
    slicing: tuple[slice, ...]
    slicing_wo_overlap: tuple[slice, ...]


class TiledRoiDataset(IterableDataset):
    def __init__(
        self,
        graph: td.graph.InMemoryGraph,
        properties: list[str],
        tiling_scheme: TilingScheme,
        df_transforms: Sequence[Callable[[pl.DataFrame], pl.DataFrame]] = (),
        dict_transforms: Sequence[Callable[[DataItem], DataItem]] = (),
        train: bool = False,
    ):
        IterableDataset.__init__(self)
        self._graph = graph
        self._df_transforms = df_transforms
        self._dict_transforms = dict_transforms
        self._properties = properties
        self._tiling_scheme = tiling_scheme

        if "z" in self._graph.node_attr_keys():
            self._spatial_cols = ["z", "y", "x"]
        else:
            self._spatial_cols = ["y", "x"]

        if train:
            raise NotImplementedError("Training is not implemented yet")

    @property
    def graph(self) -> td.graph.InMemoryGraph:
        return self._graph

    def _iter_tiles(self) -> Generator[DataItem, None, None]:
        for tile in apply_tiled(
            graph=self._graph,
            tiling_scheme=self._tiling_scheme,
            func=lambda x: x,
        ):
            if tile.graph_filter.num_edges() == 0:
                continue
            df_transforms = [
                Translate(
                    values=[-s.start for s in tile.slicing[1:]],  # skipping time
                    columns=self._spatial_cols,
                ),
                *self._df_transforms,
            ]
            yield item_from_filter(
                tile.graph_filter,
                self._spatial_cols,
                self._properties,
                df_transforms,
                self._dict_transforms,
            )

    def __iter__(self) -> Generator[DataItem, None, None]:
        yield from self._iter_tiles()
