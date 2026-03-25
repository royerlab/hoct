import itertools
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import tracksdata as td
from torch.utils.data import ConcatDataset, Dataset

from eet_inference.data._batching import DataItem, DataKeys, item_from_filter


class FrameDataset(Dataset):
    def __init__(
        self,
        graph: td.graph.InMemoryGraph,
        properties: list[str],
        gt_graph: td.graph.InMemoryGraph | None = None,
        min_window_size: int | None = None,
        df_transforms: Sequence[Callable[[pl.DataFrame], pl.DataFrame]] = (),
        dict_transforms: Sequence[Callable[[DataItem], DataItem]] = (),
        return_graph: bool = False,
        delete_masks: bool = False,
        name: str = "",
    ):
        super().__init__()

        self._graph = graph
        self._gt_graph = gt_graph
        self._sp_filter = None
        self._properties = properties
        self.name = name

        # Calculate window size from max delta_t in edges
        max_delta_t = self._graph.edge_attrs(attr_keys=["delta_t"])["delta_t"].max()
        if min_window_size is not None:
            self.window_size = int(max(max_delta_t, min_window_size))
        else:
            self.window_size = int(max_delta_t)

        time_pts = self._graph.time_points()

        min_time = min(time_pts)

        self._time_range = list(
            range(
                min_time,
                max(max(time_pts) + 2 - self.window_size, min_time + 1),
            )
        )

        if "z" in self._graph.node_attr_keys():
            self._spatial_cols = ["z", "y", "x"]
        else:
            self._spatial_cols = ["y", "x"]

        self._df_transforms = df_transforms
        self._dict_transforms = dict_transforms
        self._return_graph = return_graph

        if delete_masks:
            if td.DEFAULT_ATTR_KEYS.MASK in self._graph.node_attr_keys():
                self._graph.update_node_attrs(attrs={td.DEFAULT_ATTR_KEYS.MASK: [None] * self._graph.num_nodes()})

            if self._gt_graph is not None and td.DEFAULT_ATTR_KEYS.MASK in self._gt_graph.node_attr_keys():
                self._gt_graph.update_node_attrs(attrs={td.DEFAULT_ATTR_KEYS.MASK: [None] * self._gt_graph.num_nodes()})

    def __len__(self) -> int:
        return len(self._time_range)

    def __getitem__(
        self,
        index: int,
        **kwargs: Any,
    ) -> DataItem:
        t = index + self._time_range[0]
        sp_filter = self.graph.filter(
            node_ids=list(
                itertools.chain.from_iterable(
                    self.graph._time_to_nodes.get(i, []) for i in range(t, t + self.window_size)
                )
            ),
        )
        data = item_from_filter(
            sp_filter,
            self._spatial_cols,
            self._properties,
            self._df_transforms,
            self._dict_transforms,
            **kwargs,
        )
        if index == 0 and self._return_graph:
            data[DataKeys.GRAPH] = self.graph
            data[DataKeys.GT_GRAPH] = self.gt_graph

        return data

    @property
    def graph(self) -> td.graph.InMemoryGraph:
        return self._graph

    @property
    def gt_graph(self) -> td.graph.InMemoryGraph | None:
        return self._gt_graph

    @staticmethod
    def from_geff(
        path: Path,
        properties: list[str],
        min_window_size: int | None = None,
        df_transforms: Sequence[Callable[[pl.DataFrame], pl.DataFrame]] = (),
        dict_transforms: Sequence[Callable[[DataItem], DataItem]] = (),
        return_graph: bool = False,
        delete_masks: bool = False,
    ) -> "FrameDataset":
        graph_path = path / "graph.geff"
        gt_graph_path = path / "gt_graph.geff"

        node_props = [
            td.DEFAULT_ATTR_KEYS.T,
            "z",
            "y",
            "x",
            *properties,
            "is_div",
        ]

        if not delete_masks:
            node_props.extend([td.DEFAULT_ATTR_KEYS.MASK, td.DEFAULT_ATTR_KEYS.BBOX])

        graph, _ = td.graph.IndexedRXGraph.from_geff(graph_path, geff_read_kwargs={"node_props": node_props})
        if return_graph:
            gt_graph, _ = td.graph.IndexedRXGraph.from_geff(gt_graph_path)
        else:
            gt_graph = None

        obj = FrameDataset(
            graph=graph,
            properties=properties,
            gt_graph=gt_graph,
            min_window_size=min_window_size,
            df_transforms=df_transforms,
            dict_transforms=dict_transforms,
            path=path,
            return_graph=return_graph,
            delete_masks=delete_masks,
            name=f"{path.parent.name}/{path.name}",
        )

        return obj

    @property
    def group(self) -> str:
        return self.name.split("/", maxsplit=1)[0]

    def attribute_dim(self, attr_key: str) -> int:
        attrs = self._graph.node_attrs(attr_keys=[attr_key])[attr_key]
        if attrs.dtype == pl.Array:
            return len(attrs[0])
        else:
            return 1


class GraphConcatDataset(ConcatDataset):
    @property
    def graph(self) -> td.graph.InMemoryGraph:
        return self.datasets[0].graph

    @property
    def gt_graph(self) -> td.graph.InMemoryGraph | None:
        return self.datasets[0].gt_graph

    @property
    def group(self) -> str:
        return self.datasets[0].group
