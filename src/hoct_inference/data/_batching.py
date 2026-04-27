import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import polars as pl
import torch
import tracksdata as td
from tracksdata.graph._rustworkx_graph import RXFilter
from tracksdata.utils._dataframe import unpack_array_attrs

from hoct_inference._logging import LOG


@dataclass(frozen=True)
class DataKeys:
    # Node keys
    NODE_ID = "node_id"
    NODE_FEATS = "node_feats"
    NODE_POS = "node_pos"
    NODE_MASK = "node_mask"
    NODE_IS_DIV = "node_is_div"
    # Edge keys
    EDGE_ID = "edge_id"
    EDGE_BATCH_ID = "edge_batch_id"
    EDGE_MASK = "edge_mask"
    EDGE_TARGETS = "edge_targets"
    EDGE_POS = "edge_pos"
    EDGE_IS_DIV = "edge_is_div"
    # Temporal keys
    T = "t"
    DELTA_T = "delta_t"
    SOURCE_T = "source_t"
    # Graph keys
    GRAPH = "graph"
    GT_GRAPH = "gt_graph"


class DataItem(TypedDict):
    t: torch.Tensor
    node_id: torch.Tensor
    node_pos: torch.Tensor
    node_feats: torch.Tensor
    edge_id: torch.Tensor
    edge_pos: torch.Tensor
    edge_batch_id: torch.Tensor
    delta_t: torch.Tensor
    source_t: torch.Tensor
    node_is_div: torch.Tensor | None
    edge_targets: torch.Tensor | None
    edge_is_div: torch.Tensor | None
    graph: td.graph.InMemoryGraph | None
    gt_graph: td.graph.InMemoryGraph | None


_EDGE_KEYS = {
    DataKeys.EDGE_ID,
    DataKeys.EDGE_BATCH_ID,
    DataKeys.EDGE_TARGETS,
    DataKeys.EDGE_POS,
    DataKeys.EDGE_IS_DIV,
    DataKeys.SOURCE_T,
    DataKeys.DELTA_T,
}


def item_from_filter(
    sp_filter: RXFilter,
    spatial_cols: list[str],
    properties: list[str],
    df_transforms: list[Callable[[pl.DataFrame], pl.DataFrame]],
    dict_transforms: list[Callable[[DataItem], DataItem]],
    extra_edge_attrs: list[str] = (),
) -> DataItem:
    """Load an item from a spatial filter.

    Parameters
    ----------
    sp_filter : RXFilter
        The spatial filter.
    spatial_cols : list[str]
        The spatial columns.
    properties : list[str]
        The properties.
    df_transforms : list[Callable[[pl.DataFrame], pl.DataFrame]]
        The dataframe transforms.
    dict_transforms : list[Callable[[DataItem], DataItem]]
        The dictionary transforms.
    extra_edge_attrs : list[str]
        The extra edge attributes to load for additional edge attributes.
        IMPORTANT: This modifies the global variable _EDGE_KEYS to identify edge attributes.

    Returns
    -------
    DataItem
        The item.
    """
    attrs = [
        td.DEFAULT_ATTR_KEYS.T,
        *spatial_cols,
        *properties,
        td.DEFAULT_ATTR_KEYS.NODE_ID,
    ]
    if "is_div" in sp_filter._graph.node_attr_keys():
        attrs.append("is_div")

    node_attrs = sp_filter.node_attrs(attr_keys=attrs)
    if "inertia_tensor" in node_attrs.columns:
        # FIXME: this might be unnecessary in the future
        node_attrs = node_attrs.with_columns(
            pl.col("inertia_tensor").cast(pl.Array(pl.Float32, (3, 3))).alias("inertia_tensor"),
        )

    if node_attrs.shape[0] == 0:
        return None

    # TODO: crop transform could be optimized and applied during slicing
    for transform in df_transforms:
        LOG.debug("applying attr transform %s", transform)
        node_attrs = transform(node_attrs)

    edge_attrs = sp_filter.edge_attrs()
    imploded_node_ids = node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].implode()
    edge_attrs = edge_attrs.filter(
        edge_attrs[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].is_in(imploded_node_ids),
        edge_attrs[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].is_in(imploded_node_ids),
    )

    node_map = {node_id: i for i, node_id in enumerate(node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list())}

    # mapping to 0-indexed node ids (batch space)
    edge_attrs = edge_attrs.with_columns(
        pl.col(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE)
        .map_elements(node_map.__getitem__, return_dtype=pl.Int64)
        .alias("batch_source"),
        pl.col(td.DEFAULT_ATTR_KEYS.EDGE_TARGET)
        .map_elements(node_map.__getitem__, return_dtype=pl.Int64)
        .alias("batch_target"),
    )
    data = {
        DataKeys.GT_GRAPH: None,
        DataKeys.GRAPH: None,
    }
    data[DataKeys.EDGE_ID] = edge_attrs[td.DEFAULT_ATTR_KEYS.EDGE_ID].to_torch()
    data[DataKeys.EDGE_BATCH_ID] = edge_attrs.select("batch_source", "batch_target").to_torch(dtype=pl.Int64)

    if LOG.isEnabledFor(logging.DEBUG):
        LOG.debug("roi node time points: %s", node_attrs[td.DEFAULT_ATTR_KEYS.T].unique().to_list())
        LOG.debug("roi node shape: %s", node_attrs.shape)
        LOG.debug("roi node attributes: %s", node_attrs.columns)

    for edge_attr in extra_edge_attrs:
        _EDGE_KEYS.add(edge_attr)
        data[edge_attr] = edge_attrs[edge_attr].to_torch()

    if "edge_is_gt" in edge_attrs.columns:
        data[DataKeys.EDGE_TARGETS] = edge_attrs["edge_is_gt"].to_torch()[:, None]
    else:
        data[DataKeys.EDGE_TARGETS] = None

    if "is_div" in edge_attrs.columns:
        data[DataKeys.EDGE_IS_DIV] = edge_attrs["is_div"].cast(pl.Float32).to_torch()[:, None]
        data[DataKeys.NODE_IS_DIV] = node_attrs["is_div"].cast(pl.Float32).to_torch()[:, None]
    else:
        data[DataKeys.EDGE_IS_DIV] = None
        data[DataKeys.NODE_IS_DIV] = None

    data[DataKeys.SOURCE_T] = edge_attrs.join(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.T),
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    )[td.DEFAULT_ATTR_KEYS.T].to_torch()

    data.update(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.T).to_torch(
            return_type="dict", dtype=pl.Int64
        )
    )

    # adding node positions after transformations
    edge_attrs = td.functional.join_node_attrs_to_edges(
        node_attrs.select(td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.T, *spatial_cols),
        edge_attrs,
    ).with_columns(
        *[((pl.col(f"source_{col}") + pl.col(f"target_{col}")) * 0.5).alias(col) for col in spatial_cols],
    )

    data[DataKeys.NODE_POS] = node_attrs.select(*spatial_cols).to_torch(dtype=pl.Float32)
    data[DataKeys.EDGE_POS] = edge_attrs.select(*spatial_cols).to_torch(dtype=pl.Float32)
    data[DataKeys.DELTA_T] = (edge_attrs["target_t"] - edge_attrs["source_t"]).cast(pl.Float32).to_torch()

    node_attrs = node_attrs.drop(td.DEFAULT_ATTR_KEYS.NODE_ID, "is_div", strict=False)  # not needed anymore
    node_attrs = node_attrs.select(
        td.DEFAULT_ATTR_KEYS.T,
        *spatial_cols,
        *properties,
    )
    node_attrs = unpack_array_attrs(node_attrs)

    for col in node_attrs.columns:
        for type_of_check, check_func in [
            ("nulls", pl.Series.is_null),
            ("nans", pl.Series.is_nan),
        ]:
            n_problems = check_func(node_attrs[col]).sum()
            if n_problems > 0:
                LOG.warning(f"Column {col} has '{type_of_check}' {n_problems} out of {len(node_attrs)}")
                node_attrs = node_attrs.with_columns(
                    pl.col(col).fill_null(0).fill_nan(0),
                )

    data[DataKeys.NODE_FEATS] = node_attrs.to_torch(dtype=pl.Float32)

    for transform in dict_transforms:
        LOG.debug("applying dict transform %s", transform)
        data = transform(data)

    return DataItem(**data)


def _pad_tensor(
    tensor: torch.Tensor,
    n_samples: int,
    n_slots: int | None = None,
) -> torch.Tensor:
    """Pad a tensor to the given number of samples and slots.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to pad.
    n_samples : int
        The number of samples to pad to (dimension 0).
    n_slots : int
        The number of slots to pad to (dimension 1).

    Returns
    -------
    torch.Tensor
        The padded tensor.
    """
    padding = []
    if tensor.ndim > 2:
        padding.extend(
            [0] * (2 * (tensor.ndim - 2)),
        )

    if tensor.ndim > 1:
        if n_slots is not None:
            padding.extend(
                [0, n_slots - tensor.shape[1]],
            )
        else:
            padding.extend([0, 0])

    padding.extend(
        [0, n_samples - tensor.shape[0]],
    )

    return torch.nn.functional.pad(tensor, padding, value=0)


def collate_varying_length(
    batches: list[dict[str, int | torch.Tensor | td.graph.InMemoryGraph]],
) -> dict[str, torch.Tensor | td.graph.InMemoryGraph]:
    """
    Collate a list of dictionaries with items of varying length.

    Adds a mask tensor to the output to indicate the valid elements in the collated tensor.

    Parameters
    ----------
    batches : list[dict[str, int | torch.Tensor | td.graph.InMemoryGraph]]
        The list of dictionaries to collate.

    Returns
    -------
    dict[str, torch.Tensor | td.graph.InMemoryGraph]
        The collated dictionary.
    """
    # removing empty batches
    batches = [b for b in batches if b is not None]

    keys = list(batches[0].keys())

    n_samples = [len(b[DataKeys.NODE_ID]) for b in batches]
    n_batch = len(batches)

    # node mask
    node_mask = torch.zeros(n_batch, max(n_samples), dtype=torch.bool)
    for i, n_sample in enumerate(n_samples):
        node_mask[i, :n_sample] = batches[i].get(DataKeys.NODE_MASK, True)

    # edge mask
    n_edges = [len(b.get(DataKeys.EDGE_BATCH_ID, [])) for b in batches]
    edge_mask = torch.zeros(n_batch, max(n_edges), dtype=torch.bool)
    for i, n_edge in enumerate(n_edges):
        edge_mask[i, :n_edge] = batches[i].get(DataKeys.EDGE_MASK, True)

    output = {
        DataKeys.NODE_MASK: node_mask,
        DataKeys.EDGE_MASK: edge_mask,
    }

    # masks in batch were already used before
    keys = [k for k in keys if k not in output.keys()]  # masks

    for key in keys:
        item_list = []
        for batch in batches:
            item = batch[key]
            if isinstance(item, torch.Tensor):
                # we are only padding on the sample dimension
                if key in _EDGE_KEYS:
                    item = _pad_tensor(item, edge_mask.shape[1], n_slots=None)
                else:
                    # pad by samples
                    item = _pad_tensor(item, node_mask.shape[1], n_slots=None)
            item_list.append(item)

        if isinstance(item_list[0], torch.Tensor | np.ndarray):
            output[key] = torch.stack(item_list)
        elif isinstance(item_list[0], int):
            output[key] = torch.tensor(item_list)
        elif isinstance(item_list[0], td.graph.InMemoryGraph | None | str):
            output[key] = item_list
        else:
            raise ValueError(f"Unsupported type: {type(item_list[0])}")

    return output
