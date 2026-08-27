"""Tests for label-stable array interop results."""

import numpy as np
import polars as pl
import pytest
import tracksdata as td

from hoct import TrackingResult, solution_to_tracking_result


def _tracking_graph(*, solution_attrs: bool, similarities: bool = True) -> td.graph.InMemoryGraph:
    graph = td.graph.InMemoryGraph()
    graph.add_node_attr_key("label_id", pl.Int64, 0)
    if solution_attrs:
        graph.add_node_attr_key("solution", pl.Boolean, False)
    if similarities:
        graph.add_edge_attr_key("similarity", pl.Float64, 0.0)
    if solution_attrs:
        graph.add_edge_attr_key("solution", pl.Boolean, False)

    node_specs = [
        (1, 42, True),
        (0, 42, False),
        (0, 7, True),
        (1, 7, True),
    ]
    node_ids = [
        graph.add_node(
            {
                "t": t,
                "label_id": label,
                **({"solution": selected} if solution_attrs else {}),
            }
        )
        for t, label, selected in node_specs
    ]

    edge_specs = [
        (node_ids[2], node_ids[0], 0.6, True),
        (node_ids[1], node_ids[0], 0.9, False),
        (node_ids[2], node_ids[3], 0.8, True),
    ]
    for source, target, similarity, selected in edge_specs:
        graph.add_edge(
            source,
            target,
            {
                **({"similarity": similarity} if similarities else {}),
                **({"solution": selected} if solution_attrs else {}),
            },
        )
    return graph


def test_solution_graph_is_filtered_and_sorted_deterministically():
    result = solution_to_tracking_result(_tracking_graph(solution_attrs=True))

    np.testing.assert_array_equal(result.detections, [[0, 7], [1, 7], [1, 42]])
    np.testing.assert_array_equal(result.links, [[0, 7, 1, 7], [0, 7, 1, 42]])
    np.testing.assert_allclose(result.similarities, [0.8, 0.6])
    np.testing.assert_allclose(
        result.link_table(),
        [[0.0, 7.0, 1.0, 7.0, 0.8], [0.0, 7.0, 1.0, 42.0, 0.6]],
    )


def test_already_filtered_graph_uses_every_node_and_edge():
    graph = _tracking_graph(solution_attrs=False)

    result = solution_to_tracking_result(graph)

    np.testing.assert_array_equal(result.detections, [[0, 7], [0, 42], [1, 7], [1, 42]])
    np.testing.assert_array_equal(
        result.links,
        [[0, 7, 1, 7], [0, 7, 1, 42], [0, 42, 1, 42]],
    )
    np.testing.assert_allclose(result.similarities, [0.8, 0.6, 0.9])


def test_result_defensively_copies_and_makes_arrays_read_only():
    detections = np.array([[0, 7], [1, 42]], dtype=np.int64)
    links = np.array([[0, 7, 1, 42]], dtype=np.int64)
    similarities = np.array([0.75], dtype=np.float64)

    result = TrackingResult(detections, links, similarities)
    detections[0, 1] = 99
    links[0, 1] = 99
    similarities[0] = 0.0

    np.testing.assert_array_equal(result.detections, [[0, 7], [1, 42]])
    np.testing.assert_array_equal(result.links, [[0, 7, 1, 42]])
    np.testing.assert_allclose(result.similarities, [0.75])
    assert not result.detections.flags.writeable
    assert not result.links.flags.writeable
    assert not result.similarities.flags.writeable
    assert not result.link_table().flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.links[0, 0] = 1


def test_empty_result_has_exact_shapes_without_similarity_schema():
    graph = td.graph.InMemoryGraph()
    graph.add_node_attr_key("label_id", pl.Int64, 0)

    result = solution_to_tracking_result(graph)

    assert result.detections.shape == (0, 2)
    assert result.detections.dtype == np.int64
    assert result.links.shape == (0, 4)
    assert result.links.dtype == np.int64
    assert result.similarities.shape == (0,)
    assert result.similarities.dtype == np.float64
    assert result.link_table().shape == (0, 5)


def test_selected_edge_requires_similarity():
    graph = _tracking_graph(solution_attrs=False, similarities=False)

    with pytest.raises(ValueError, match="similarity"):
        solution_to_tracking_result(graph)


def test_selected_edge_requires_selected_endpoints():
    graph = _tracking_graph(solution_attrs=True)
    edge_ids = graph.edge_attrs()["edge_id"].to_list()
    graph.update_edge_attrs(attrs={"solution": True}, edge_ids=[edge_ids[1]])

    with pytest.raises(ValueError, match="not selected detections"):
        solution_to_tracking_result(graph)


def test_label_id_is_required():
    graph = td.graph.InMemoryGraph()

    with pytest.raises(ValueError, match="label_id"):
        solution_to_tracking_result(graph)


def test_solution_attributes_must_be_present_on_nodes_and_edges():
    graph = td.graph.InMemoryGraph()
    graph.add_node_attr_key("label_id", pl.Int64, 0)
    graph.add_node_attr_key("solution", pl.Boolean, False)

    with pytest.raises(ValueError, match="both nodes and edges"):
        solution_to_tracking_result(graph)


def test_solution_attributes_must_be_boolean():
    graph = td.graph.InMemoryGraph()
    graph.add_node_attr_key("label_id", pl.Int64, 0)
    graph.add_node_attr_key("solution", pl.Int32, 0)
    graph.add_edge_attr_key("solution", pl.Int32, 0)

    with pytest.raises(TypeError, match="must be boolean"):
        solution_to_tracking_result(graph)


def test_tracking_result_validates_shapes_rows_and_endpoints():
    with pytest.raises(ValueError, match="shape"):
        TrackingResult(np.array([0, 7]), np.empty((0, 4), dtype=np.int64), np.empty((0,)))
    with pytest.raises(ValueError, match="same number"):
        TrackingResult(np.array([[0, 7]]), np.empty((0, 4), dtype=np.int64), np.array([0.5]))
    with pytest.raises(ValueError, match="missing from detections"):
        TrackingResult(np.array([[0, 7]]), np.array([[0, 7, 1, 7]]), np.array([0.5]))
    with pytest.raises(ValueError, match="unique"):
        TrackingResult(np.array([[0, 7], [0, 7]]), np.empty((0, 4), dtype=np.int64), np.empty((0,)))
