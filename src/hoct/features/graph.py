from functools import partial

import dask.array as da
import numpy as np
import polars as pl
import tracksdata as td
from numpy.typing import ArrayLike
from tracksdata.nodes._mask import Mask
from tracksdata.utils._multiprocessing import multiprocessing_apply

from hoct.features.constants import EDGE_GT_KEY, REGIONPROPS
from hoct.features.features import add_border_dist, add_delta_t, add_is_div


def convert_to_3d(graph: td.graph.RustWorkXGraph) -> None:
    """
    Converts a 2D graph to a 3D graph in place.

    Parameters
    ----------
    graph : td.graph.RustWorkXGraph
        The graph to convert.
    """
    # unsafe private api interaction
    graph.add_node_attr_key("z", pl.Float32, 0.0)

    # unsafe private api interaction
    for node_attr in graph.rx_graph.nodes():
        # updating bbox
        bbox = node_attr[td.DEFAULT_ATTR_KEYS.BBOX]
        bbox = np.asarray([0, *bbox[:2], 1, *bbox[2:]])
        node_attr[td.DEFAULT_ATTR_KEYS.BBOX] = bbox

        # mask updated by reference
        mask = node_attr[td.DEFAULT_ATTR_KEYS.MASK]
        node_attr[td.DEFAULT_ATTR_KEYS.MASK] = Mask(mask.mask[None, ...], bbox)

    shape = graph.metadata["shape"]
    graph.metadata.update(shape=(shape[0], 1, *shape[1:]), was_2d=True)


def convert_to_2d(graph: td.graph.RustWorkXGraph) -> None:
    """
    Converts a 3D graph to a 2D graph in place.

    Parameters
    ----------
    graph : td.graph.RustWorkXGraph
        The graph to convert.
    """
    # unsafe private api interaction
    graph._node_attr_keys.remove("z")

    # unsafe private api interaction
    for node_attr in graph.rx_graph.nodes():
        del node_attr["z"]
        mask = node_attr[td.DEFAULT_ATTR_KEYS.MASK]
        bbox = mask.bbox
        bbox = np.asarray([bbox[1], bbox[2], bbox[4], bbox[5]])
        node_attr[td.DEFAULT_ATTR_KEYS.BBOX] = bbox
        node_attr[td.DEFAULT_ATTR_KEYS.MASK] = Mask(mask.mask[0, ...], bbox)


def create_graph(
    labels: ArrayLike,
    *,
    distance_threshold: float,
    n_neighbors: int,
    delta_t: float,
    scale: tuple[float, ...] | None = None,
    images: ArrayLike | None = None,
    gt_graph: td.graph.BaseGraph | None = None,
    out_graph: td.graph.BaseGraph | None = None,
) -> td.graph.InMemoryGraph:
    """
    Creates a graph from segmentation labels and optional intensity images.

    This function supports both training (with ground truth) and inference (without ground truth) modes.

    Parameters
    ----------
    labels : ArrayLike
        Segmentation labels of shape (T, [Z,] Y, X) where T is time.
    distance_threshold : float, default=200.0
        Maximum distance for creating candidate edges.
    n_neighbors : int
        Number of nearest neighbors to connect per node.
    delta_t : float
        Maximum temporal gap for edges.
    scale : tuple[float, ...]
        Physical spacing (t, [z,] y, x). If None, inferred from data dimensions.
    images : ArrayLike | None
        Optional intensity images of shape (T, [Z,] Y, X).
        If None, only geometric features are computed.
    gt_graph : td.graph.BaseGraph | None
        Optional ground truth graph for training. If provided, adds ground truth edge labels.
        If None (inference mode), skips ground truth-related features.
    out_graph : td.graph.BaseGraph | None
        Optional output graph to write to, modified in place if provided.
        If None, a new in-memory graph is created.

    Returns
    -------
    td.graph.InMemoryGraph
        The candidate tracking graph with nodes and edges.
        If gt_graph provided, includes ground truth edge labels.

    Notes
    -----
    - For 2D+t data (T, Y, X), automatically adds a singleton Z dimension
    - Scale is inferred as (1, 1, 1, 1) if not provided
    - Ground truth features are only added if gt_graph is provided
    """
    if out_graph is None:
        graph = td.graph.InMemoryGraph()
    else:
        graph = out_graph

    # Convert to dask arrays if needed
    if not isinstance(labels, da.Array):
        labels = da.from_array(labels)

    if images is not None and not isinstance(images, da.Array):
        images = da.from_array(images)

    if scale is None:
        scale = (1.0,) * labels.ndim

    # Handle 2D vs 3D data
    if labels.ndim == 3:
        labels = da.expand_dims(labels, axis=1)
        was_2d = True
        # 2D+t case
        if images is not None:
            images = da.expand_dims(images, axis=1)

        scale = (scale[0], 1.0, *scale[1:])

    elif labels.ndim == 4:
        # 3D+t case
        was_2d = False

    else:
        raise ValueError(f"Labels must be 3D (T, Y, X) or 4D (T, Z, Y, X), got shape: {labels.shape}")

    assert len(scale) == 4, f"Scale must have 4 elements (t, z, y, x), got {len(scale)}"

    # Add nodes from regionprops
    # Only request intensity properties if images are provided
    if images is not None:
        extra_properties = REGIONPROPS.copy()
    else:
        # Filter out intensity-requiring properties when no images provided
        extra_properties = [prop for prop in REGIONPROPS if not prop.startswith("intensity_")]

    if "border_dist" in extra_properties:
        extra_properties.remove("border_dist")

    td.nodes.RegionPropsNodes(
        extra_properties=extra_properties,
    ).add_nodes(graph, labels=labels, intensity_image=images)

    # Add scaled position attributes
    cols = [td.DEFAULT_ATTR_KEYS.T, "z", "y", "x"]
    node_attrs = graph.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, *cols])
    node_attrs = node_attrs.with_columns([(pl.col(c) * scale[i]).alias(f"scaled_{c}") for i, c in enumerate(cols)])

    if images is None:
        for prop in REGIONPROPS:
            if "intensity" in prop:
                graph.add_node_attr_key(prop, pl.Float32, 0.0)

    # Add candidate edges
    with td.options.Options(n_workers=1):
        td.edges.DistanceEdges(
            distance_threshold=distance_threshold,
            n_neighbors=n_neighbors,
            delta_t=delta_t,
            neighbors_per_frame=True,
        ).add_edges(graph)

        # Add required features
        add_delta_t(graph)
        add_border_dist(graph, labels.shape, was_2d=was_2d)

    # Add ground truth features if GT graph provided
    if gt_graph is not None:
        graph.match(gt_graph)
        add_is_div(graph, gt_graph)

        gt_edge_ids = td.functional.ancestral_connected_edges(graph, gt_graph, match=False)
        graph.add_edge_attr_key(EDGE_GT_KEY, pl.Boolean, False)
        graph.update_edge_attrs(attrs={EDGE_GT_KEY: True}, edge_ids=gt_edge_ids)

    # Store metadata
    graph.metadata.update(
        distance_threshold=distance_threshold,
        n_neighbors=n_neighbors,
        delta_t=delta_t,
        scale=scale,
        was_2d=was_2d,
    )

    return graph


def _features_from_single_frame(
    t: int,
    graph: td.graph.BaseGraph,
    missing_features: list[str],
    images: ArrayLike | None,
) -> dict[str, list[float]]:
    new_attrs = {f: [] for f in missing_features}

    if images is not None:
        frame = np.asarray(images[t])
    else:
        frame = None

    node_attrs = graph.filter(td.NodeAttr(td.DEFAULT_ATTR_KEYS.T) == t).node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MASK]
    )

    for mask in node_attrs[td.DEFAULT_ATTR_KEYS.MASK]:
        if frame is not None:
            crop = mask.crop(frame)
        else:
            crop = None
        prop = mask.regionprops(intensity_image=crop)
        for f in missing_features:
            value = getattr(prop, f)
            new_attrs[f].append(value)

    new_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID] = node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()

    return new_attrs


def add_features(
    graph: td.graph.BaseGraph,
    images: ArrayLike | None = None,
    shape: tuple[int, ...] | None = None,
) -> None:
    """
    Adds features to a graph with existing nodes.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph with existing nodes.
    images : ArrayLike
        The images to add to the graph.
    shape : tuple[int, ...] | None
        The shape of the images.
    """
    missing_features = set(REGIONPROPS) - set(graph.node_attr_keys())

    add_delta_t(graph)

    if "border_dist" in missing_features:
        shape = images.shape if images is not None else shape
        if shape is None:
            raise ValueError("`shape` is required when `images` are not provided")
        add_border_dist(graph, shape, was_2d=False)
        missing_features.remove("border_dist")

    for feature in list(missing_features):
        if feature == "inertia_tensor":
            graph.add_node_attr_key(feature, pl.Array(pl.Float32, (3, 3)), np.zeros((3, 3), dtype=np.float32))
        else:
            graph.add_node_attr_key(feature, pl.Float32, 0.0)

        if "intensity" in feature and images is None:
            missing_features.remove(feature)

    for new_node_attrs in multiprocessing_apply(
        func=partial(
            _features_from_single_frame,
            graph=graph,
            missing_features=missing_features,
            images=images,
        ),
        sequence=graph.time_points(),
        desc="Processing frames",
    ):
        indices = new_node_attrs.pop(td.DEFAULT_ATTR_KEYS.NODE_ID)
        graph.update_node_attrs(
            attrs=new_node_attrs,
            node_ids=indices,
        )
