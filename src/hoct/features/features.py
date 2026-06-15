import numpy as np
import polars as pl
import tracksdata as td
from numpy.typing import NDArray
from toolz import curry


def _border_dist_nd(
    coords: NDArray,
    shape: tuple[int, ...],
    cutoff: int = 5,
) -> NDArray:
    """
    Compute the distance to the border of a ND image.
    1 if at the border, 0 if at least a cutoff distance from the border.

    Reference:
        https://github.com/weigertlab/trackastra/blob/8b13f5445b29c6129a7e15a546720cd450c3b4a7/trackastra/data/wrfeat.py#L56

    coords : NDArray
        The coordinates of the nodes.
    shape : tuple[int, ...]
        The shape of the image.
    cutoff : int, optional
        The cutoff distance from the border.

    Returns
    -------
    NDArray
        The distance to the border of the nodes.
    """
    shape = np.asarray(shape)[None, :]
    distance = np.minimum(coords, shape - coords).min(axis=1)
    inv_dist = 1 - np.minimum(1, distance / cutoff)

    return inv_dist.tolist()


@curry
def border_dist_3d(z: NDArray, y: NDArray, x: NDArray, shape: tuple[int, int, int]) -> NDArray:
    """
    Compute the distance to the border of a 3D image.
    """
    return _border_dist_nd(np.stack([z, y, x], axis=1), shape)


@curry
def border_dist_2d(y: NDArray, x: NDArray, shape: tuple[int, int]) -> NDArray:
    """
    Compute the distance to the border of a 2D image.
    """
    return _border_dist_nd(np.stack([y, x], axis=1), shape)


def normalize_image(
    image: NDArray,
    clip: bool = False,
    uq: float = 0.999,
) -> NDArray:
    """
    Normalize an image to [0, 1] using the quantile.
    """
    image = np.asarray(image, dtype=np.float32)

    image_min = image.min()
    image_max = np.quantile(image, uq)
    image = (image - image_min) / (image_max - image_min + 1e-7)

    if clip:
        np.clip(image, 0, 1, out=image)

    return image


def add_is_div(graph: td.graph.BaseGraph, gt_graph: td.graph.BaseGraph) -> None:
    """
    Add "is_div" to nodes and edges in the graph using the ground truth graph.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to add the "is_div" node attribute to.
    gt_graph : td.graph.BaseGraph
        The ground truth graph to use to compute the "is_div" node attribute.
    """
    node_attrs = graph.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
    ).filter(pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) >= 0)

    gt_degree = gt_graph.out_degree(node_ids=node_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list())

    key = "is_div"
    node_attrs = node_attrs.with_columns(
        pl.Series(key, np.asarray(gt_degree) > 1),
    )

    edge_attrs = graph.edge_attrs(attr_keys=[])
    edge_attrs = edge_attrs.join(
        node_attrs,
        left_on=td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
        right_on=td.DEFAULT_ATTR_KEYS.NODE_ID,
        how="left",
    ).fill_null(False)

    graph.add_node_attr_key(key, pl.Boolean, False)
    graph.add_edge_attr_key(key, pl.Boolean, False)

    graph.update_edge_attrs(
        attrs={key: edge_attrs[key]},
        edge_ids=edge_attrs[td.DEFAULT_ATTR_KEYS.EDGE_ID].to_list(),
    )

    graph.update_node_attrs(
        attrs={key: node_attrs[key]},
        node_ids=node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
    )


def add_delta_t(graph: td.graph.BaseGraph) -> None:
    """
    Adds a "delta_t" edge attribute to the graph, which is the difference in time
    between the source and target nodes of the edge.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to add the "delta_t" edge attribute to.
    """
    graph.add_edge_attr_key("delta_t", pl.Float32, 0.0)
    node_attrs = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.T])
    edge_attrs = graph.edge_attrs(attr_keys=[])
    edge_attrs = td.functional.join_node_attrs_to_edges(node_attrs, edge_attrs)

    edge_attrs = edge_attrs.with_columns(
        (pl.col("target_t") - pl.col("source_t")).alias("delta_t"),
    )
    graph.update_edge_attrs(
        edge_ids=edge_attrs[td.DEFAULT_ATTR_KEYS.EDGE_ID].to_list(),
        attrs={"delta_t": edge_attrs["delta_t"].to_list()},
    )


def add_border_dist(
    graph: td.graph.BaseGraph,
    shape: tuple[int, ...],
    was_2d: bool,
) -> None:
    """
    Adds a "border_dist" node attribute to the graph, which is the distance to the
    border of the image.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to add the "border_dist" node attribute to.
    shape : tuple[int, ...]
        The shape of the image.
    was_2d : bool
        Whether the image was 2D.
    """

    if was_2d:
        attr_keys = ["y", "x"]
        _border_func = border_dist_2d
    else:
        attr_keys = ["z", "y", "x"]
        _border_func = border_dist_3d

    graph.add_node_attr_key("border_dist", pl.Float32, 0.0)

    td.nodes.GenericFuncNodeAttrs(
        func=_border_func(shape=shape[-len(attr_keys) :]),
        output_key="border_dist",
        attr_keys=attr_keys,
        batch_size=2**32,  # as many as possible
    ).add_node_attrs(graph)
