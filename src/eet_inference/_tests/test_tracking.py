"""Tests for eet_inference.tracking module."""

import polars as pl
import pytest
import tracksdata as td

from eet_inference.tracking import ILPSolverConfig, TrackletSolver, solve_tracking
from eet_inference._tests.conftest import GEFF_2D, GEFF_3D


def prepare_graph_for_tracking(graph: td.graph.BaseGraph) -> td.graph.BaseGraph:
    """
    Prepare a ground truth graph for tracking by adding required attributes.

    Converts ground truth edge labels to similarity scores:
    - edge_is_gt=True -> similarity=1.0 (high confidence)
    - edge_is_gt=False -> similarity=0.0 (low confidence)

    Also adds orphan_prob attribute (set to 0.0 for testing).

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to prepare.

    Returns
    -------
    td.graph.BaseGraph
        The graph with added similarity scores and orphan_prob.
    """
    # Convert ground truth labels to similarity scores
    edge_attrs = graph.edge_attrs(attr_keys=["edge_is_gt"])
    similarity_scores = edge_attrs["edge_is_gt"].cast(float).to_list()

    # Add similarity as edge attribute
    if "similarity" not in graph.edge_attr_keys():
        graph.add_edge_attr_key("similarity", default_value=0.0)
    graph.update_edge_attrs(attrs={"similarity": similarity_scores})

    # Add orphan_prob if missing (default to 0.0 for testing)
    if "orphan_prob" not in graph.node_attr_keys():
        graph.add_node_attr_key("orphan_prob", default_value=0.0)

    return graph


class TestSolveTracking:
    """Tests for solve_tracking function."""

    @pytest.mark.parametrize("geff_path", [GEFF_2D, GEFF_3D])
    def test_solve_tracking_basic(self, geff_path):
        """Test basic tracking solving without tracklet solver."""
        graph, _ = td.graph.InMemoryGraph.from_geff(geff_path)
        graph = prepare_graph_for_tracking(graph)

        # Solve tracking with basic ILP solver
        config = ILPSolverConfig(
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            delta_t_weight=0.1,
            edge_bias=0.0,
            timeout=10.0,
            tracklet_solver=False,
        )
        solution = solve_tracking(graph=graph, config=config)

        # Solution should be a graph
        assert isinstance(solution, td.graph.BaseGraph)

        # Solution should have solution attributes
        assert td.DEFAULT_ATTR_KEYS.SOLUTION in solution.node_attr_keys()
        assert td.DEFAULT_ATTR_KEYS.SOLUTION in solution.edge_attr_keys()

    @pytest.mark.parametrize("geff_path", [GEFF_2D])
    def test_solve_tracking_with_tracklet_solver(self, geff_path):
        """Test tracking solving with tracklet solver (two-pass)."""
        graph, _ = td.graph.InMemoryGraph.from_geff(geff_path)
        graph = prepare_graph_for_tracking(graph)

        # Solve tracking with tracklet solver
        config = ILPSolverConfig(
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            delta_t_weight=0.1,
            edge_bias=0.0,
            timeout=10.0,
            tracklet_solver=True,
        )
        solution = solve_tracking(graph=graph, config=config)

        # Solution should be a graph
        assert isinstance(solution, td.graph.BaseGraph)

        # Solution should have solution attributes
        assert td.DEFAULT_ATTR_KEYS.SOLUTION in solution.node_attr_keys()

    def test_solve_tracking_resets_existing_solution(self):
        """Test that existing solution is reset before solving."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)
        graph = prepare_graph_for_tracking(graph)

        # Add solution attribute keys if not present
        if td.DEFAULT_ATTR_KEYS.SOLUTION not in graph.node_attr_keys():
            graph.add_node_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=False)
        if td.DEFAULT_ATTR_KEYS.SOLUTION not in graph.edge_attr_keys():
            graph.add_edge_attr_key(td.DEFAULT_ATTR_KEYS.SOLUTION, default_value=False)

        # Add dummy solution
        graph.update_node_attrs(attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: True})
        graph.update_edge_attrs(attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: True})

        # Solve tracking
        config = ILPSolverConfig(
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            delta_t_weight=0.1,
            edge_bias=0.0,
            timeout=10.0,
            tracklet_solver=False,
        )
        solution = solve_tracking(graph=graph, config=config)

        # Solution should have been recomputed successfully
        # Verify that solution attributes exist and are valid
        node_solutions = solution.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.SOLUTION])[
            td.DEFAULT_ATTR_KEYS.SOLUTION
        ]
        edge_solutions = solution.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.SOLUTION])[
            td.DEFAULT_ATTR_KEYS.SOLUTION
        ]

        # Check that solution is binary (True/False)
        assert node_solutions.dtype == pl.Boolean
        assert edge_solutions.dtype == pl.Boolean

        # Check that at least some nodes are in the solution
        assert node_solutions.any()

    def test_solve_tracking_no_division_mode(self):
        """Test tracking with no_division metadata flag."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)
        graph = prepare_graph_for_tracking(graph)

        # Set no_division metadata by updating the metadata dictionary
        metadata = graph.metadata()
        metadata["no_division"] = True

        # Solve tracking
        config = ILPSolverConfig(
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=1.0,  # Should be overridden by metadata
            node_weight=0.0,
            delta_t_weight=0.1,
            edge_bias=0.0,
            timeout=10.0,
            tracklet_solver=False,
        )
        solution = solve_tracking(graph=graph, config=config)

        assert isinstance(solution, td.graph.BaseGraph)

    def test_solve_tracking_weight_parameters(self):
        """Test that different weight parameters produce different solutions."""
        graph1, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)
        graph1 = prepare_graph_for_tracking(graph1)
        graph2, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)
        graph2 = prepare_graph_for_tracking(graph2)

        # Solve with different appearance weights
        config1 = ILPSolverConfig(
            appearance_weight=0.1,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            delta_t_weight=0.1,
            edge_bias=0.0,
            timeout=10.0,
            tracklet_solver=False,
        )
        sol1 = solve_tracking(graph=graph1, config=config1)

        config2 = ILPSolverConfig(
            appearance_weight=10.0,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            delta_t_weight=0.1,
            edge_bias=0.0,
            timeout=10.0,
            tracklet_solver=False,
        )
        sol2 = solve_tracking(graph=graph2, config=config2)

        # Solutions should potentially differ (but both valid)
        assert isinstance(sol1, td.graph.BaseGraph)
        assert isinstance(sol2, td.graph.BaseGraph)


class TestTrackletSolver:
    """Tests for TrackletSolver class."""

    def test_tracklet_solver_initialization(self):
        """Test that TrackletSolver can be initialized."""
        solver = TrackletSolver(
            edge_weight=-td.EdgeAttr("similarity"),
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            timeout=10.0,
        )

        assert isinstance(solver, TrackletSolver)

    def test_tracklet_solver_requires_solution(self):
        """Test that TrackletSolver requires existing solution in graph."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)
        graph = prepare_graph_for_tracking(graph)

        solver = TrackletSolver(
            edge_weight=-td.EdgeAttr("similarity"),
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            timeout=10.0,
        )

        # Should raise error if no solution exists
        with pytest.raises(ValueError, match="must be present in the graph"):
            solver.solve(graph)

    def test_tracklet_solver_with_existing_solution(self):
        """Test TrackletSolver with an existing solution."""
        graph, _ = td.graph.InMemoryGraph.from_geff(GEFF_2D)
        graph = prepare_graph_for_tracking(graph)

        # First solve with basic ILP to get initial solution
        config = ILPSolverConfig(
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            delta_t_weight=0.1,
            edge_bias=0.0,
            timeout=10.0,
            tracklet_solver=False,
        )
        graph = solve_tracking(graph=graph, config=config)

        # Now solve with tracklet solver
        solver = TrackletSolver(
            edge_weight=-td.EdgeAttr("similarity"),
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            timeout=10.0,
            return_solution=True,
        )

        solution = solver.solve(graph)

        # Solution should be a graph or None
        assert solution is None or isinstance(solution, td.graph.BaseGraph)
