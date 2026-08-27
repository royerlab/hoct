"""Stable array-based representations of HOCT tracking solutions."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import tracksdata as td
from numpy.typing import NDArray

from hoct.features.graph import LABEL_ID_KEY

__all__ = ["TrackingResult", "solution_to_tracking_result"]


def _readonly_array(value: Any, *, dtype: np.dtype, columns: int, name: str) -> NDArray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N, {columns}), got {array.shape}")
    if np.issubdtype(dtype, np.integer) and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers, got dtype {array.dtype}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numbers, got dtype {array.dtype}")
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_similarities(value: Any) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"similarities must have shape (N,), got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"similarities must contain numbers, got dtype {array.dtype}")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TrackingResult:
    """A label-stable tracking solution suitable for language bridges.

    Attributes
    ----------
    detections : numpy.ndarray
        Integer ``(N, 2)`` array whose columns are ``[t, label_id]``.
    links : numpy.ndarray
        Integer ``(M, 4)`` array whose columns are
        ``[source_t, source_label, target_t, target_label]``.
    similarities : numpy.ndarray
        Floating-point ``(M,)`` array containing the model similarity for each
        corresponding link.
    """

    detections: NDArray[np.int64]
    links: NDArray[np.int64]
    similarities: NDArray[np.float64]

    def __post_init__(self) -> None:
        detections = _readonly_array(self.detections, dtype=np.dtype(np.int64), columns=2, name="detections")
        links = _readonly_array(self.links, dtype=np.dtype(np.int64), columns=4, name="links")
        similarities = _readonly_similarities(self.similarities)

        if links.shape[0] != similarities.shape[0]:
            raise ValueError(
                f"links and similarities must have the same number of rows, got {links.shape[0]} and "
                f"{similarities.shape[0]}"
            )

        detection_ids = [tuple(row) for row in detections.tolist()]
        if len(detection_ids) != len(set(detection_ids)):
            raise ValueError("detections must contain unique (t, label_id) pairs")

        known_detections = set(detection_ids)
        missing_endpoints = {
            endpoint
            for row in links.tolist()
            for endpoint in (tuple(row[:2]), tuple(row[2:]))
            if endpoint not in known_detections
        }
        if missing_endpoints:
            raise ValueError(f"link endpoints are missing from detections: {sorted(missing_endpoints)}")

        object.__setattr__(self, "detections", detections)
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "similarities", similarities)

    def link_table(self) -> NDArray[np.float64]:
        """Return links and similarities as a homogeneous ``(M, 5)`` array."""
        table = np.empty((self.links.shape[0], 5), dtype=np.float64)
        table[:, :4] = self.links
        table[:, 4] = self.similarities
        table.setflags(write=False)
        return table


def _validate_solution_schema(graph: td.graph.BaseGraph) -> bool:
    solution_key = td.DEFAULT_ATTR_KEYS.SOLUTION
    has_node_solution = solution_key in graph.node_attr_keys()
    has_edge_solution = solution_key in graph.edge_attr_keys()
    if has_node_solution != has_edge_solution:
        raise ValueError("solution must be present on both nodes and edges, or on neither")
    return has_node_solution


def solution_to_tracking_result(graph: td.graph.BaseGraph) -> TrackingResult:
    """Convert a HOCT solution graph into a stable, array-based result.

    A full candidate graph is filtered using its boolean ``solution`` node and
    edge attributes. An already-filtered graph without these attributes is
    interpreted as containing only selected nodes and links.

    Parameters
    ----------
    graph : tracksdata.graph.BaseGraph
        Full or already-filtered tracking solution. Nodes must carry ``t`` and
        ``label_id``. Selected edges must carry ``similarity``.

    Returns
    -------
    TrackingResult
        Deterministically sorted detections, links, and link similarities.
    """
    required_node_keys = {td.DEFAULT_ATTR_KEYS.T, LABEL_ID_KEY}
    missing_node_keys = required_node_keys - set(graph.node_attr_keys())
    if missing_node_keys:
        raise ValueError(f"solution graph is missing node attributes: {sorted(missing_node_keys)}")

    has_solution = _validate_solution_schema(graph)
    solution_key = td.DEFAULT_ATTR_KEYS.SOLUTION
    node_id_key = td.DEFAULT_ATTR_KEYS.NODE_ID
    node_keys = [node_id_key, td.DEFAULT_ATTR_KEYS.T, LABEL_ID_KEY]
    if has_solution:
        node_keys.append(solution_key)
    nodes = graph.node_attrs(attr_keys=node_keys)
    if has_solution:
        if nodes.schema[solution_key] != pl.Boolean:
            raise TypeError(f"node '{solution_key}' attribute must be boolean")
        nodes = nodes.filter(pl.col(solution_key))

    selected_nodes = {
        int(node_id): (int(t), int(label))
        for node_id, t, label in nodes.select(node_id_key, td.DEFAULT_ATTR_KEYS.T, LABEL_ID_KEY).iter_rows()
    }
    detections = np.asarray(sorted(selected_nodes.values()), dtype=np.int64).reshape((-1, 2))

    edge_id_key = td.DEFAULT_ATTR_KEYS.EDGE_ID
    source_key = td.DEFAULT_ATTR_KEYS.EDGE_SOURCE
    target_key = td.DEFAULT_ATTR_KEYS.EDGE_TARGET
    edge_keys = [edge_id_key, source_key, target_key]
    if has_solution:
        edge_keys.append(solution_key)
    edges = graph.edge_attrs(attr_keys=edge_keys)
    if has_solution:
        if edges.schema[solution_key] != pl.Boolean:
            raise TypeError(f"edge '{solution_key}' attribute must be boolean")
        edges = edges.filter(pl.col(solution_key))

    if edges.height == 0:
        return TrackingResult(
            detections=detections,
            links=np.empty((0, 4), dtype=np.int64),
            similarities=np.empty((0,), dtype=np.float64),
        )

    if "similarity" not in graph.edge_attr_keys():
        raise ValueError("selected solution edges require a 'similarity' attribute")

    selected_edge_ids = edges[edge_id_key].implode()
    selected_edges = graph.edge_attrs(attr_keys=[edge_id_key, source_key, target_key, "similarity"]).filter(
        pl.col(edge_id_key).is_in(selected_edge_ids)
    )

    records: list[tuple[int, int, int, int, float, int]] = []
    missing_endpoints: set[int] = set()
    for edge_id, similarity, source_id, target_id in selected_edges.select(
        edge_id_key, "similarity", source_key, target_key
    ).iter_rows():
        source = selected_nodes.get(int(source_id))
        target = selected_nodes.get(int(target_id))
        if source is None:
            missing_endpoints.add(int(source_id))
        if target is None:
            missing_endpoints.add(int(target_id))
        if source is not None and target is not None:
            records.append((*source, *target, float(similarity), int(edge_id)))

    if missing_endpoints:
        raise ValueError(f"selected link endpoints are not selected detections: {sorted(missing_endpoints)}")

    records.sort(key=lambda row: (*row[:4], row[5]))
    links = np.asarray([row[:4] for row in records], dtype=np.int64).reshape((-1, 4))
    similarities = np.asarray([row[4] for row in records], dtype=np.float64)
    return TrackingResult(detections=detections, links=links, similarities=similarities)
