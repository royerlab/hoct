from collections.abc import Callable, Sequence
from typing import NamedTuple

import polars as pl
import tracksdata as td
from torch.utils.data import Dataset
from tracksdata.functional import TilingScheme, apply_tiled
from tracksdata.graph._rustworkx_graph import RXFilter

from eet_inference.data._batching import DataItem, item_from_filter
from eet_inference.data._transforms import Translate


class Tile(NamedTuple):
    slicing: tuple[slice, ...]
    slicing_wo_overlap: tuple[slice, ...]


class TiledRoiDataset(Dataset):
    def __init__(
        self,
        graph: td.graph.InMemoryGraph,
        properties: list[str],
        tiling_scheme: TilingScheme,
        df_transforms: Sequence[Callable[[pl.DataFrame], pl.DataFrame]] = (),
        dict_transforms: Sequence[Callable[[DataItem], DataItem]] = (),
        train: bool = False,
    ):
        Dataset.__init__(self)
        self._graph = graph
        self._df_transforms = df_transforms
        self._dict_transforms = dict_transforms
        self._properties = properties

        time_pts = self._graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.T])[td.DEFAULT_ATTR_KEYS.T]
        min_t = time_pts.min()
        max_t = time_pts.max()

        self._tiles: list[Tile] = [
            Tile(slicing=t.slicing, slicing_wo_overlap=t.slicing_wo_overlap)
            for t in apply_tiled(
                graph=self._graph,
                tiling_scheme=tiling_scheme,
                func=lambda x: x,
            )
            if len(t.graph_filter.edge_ids()) > 0  # removing empty tiles
        ]

        if len(self._tiles) == 0:
            raise ValueError(
                f"No tiles found. Tiling scheme {tiling_scheme} is too big for {max_t - min_t} time points"
            )

        if "z" in self._graph.node_attr_keys():
            self._spatial_cols = ["z", "y", "x"]
        else:
            self._spatial_cols = ["y", "x"]
        self._sp_filter = None
        self._train = train

        if train:
            raise NotImplementedError("Training is not implemented yet")

    def __len__(self) -> int:
        return len(self._tiles)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_sp_filter"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    @property
    def graph(self) -> td.graph.InMemoryGraph:
        return self._graph

    @property
    def sp_filter(self) -> RXFilter:
        if self._sp_filter is None:
            self._sp_filter = self._graph.spatial_filter()
        return self._sp_filter

    def __getitem__(self, index: int) -> DataItem:
        tile = self._tiles[index]
        sp_filter = self.sp_filter[tile.slicing]
        df_transforms = [
            Translate(
                values=[-s.start for s in tile.slicing[1:]],  # skipping time
                columns=self._spatial_cols,
            ),
            *self._df_transforms,
        ]
        return item_from_filter(sp_filter, self._spatial_cols, self._properties, df_transforms, self._dict_transforms)
