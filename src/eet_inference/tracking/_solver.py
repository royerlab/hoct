import math
from typing import ClassVar

import tracksdata as td

from eet_inference._logging import LOG
from eet_inference.tracking._tracklet_solver import TrackletSolver



def solve_tracking(
    graph: td.graph.BaseGraph,
    appearance_weight: float,
    disappearance_weight: float,
    division_weight: float,
    node_weight: float,
    delta_t_weight: float,
    edge_bias: float,
    timeout: float,
    tracklet_solver: bool,
) -> td.graph.InMemoryGraph:
    """
    Solve the tracking problem and score the solution if a ground truth graph is provided.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to solve the tracking problem for.
    solver_config : ILPSolverConfig
        The configuration for the ILPSolver.
    gt_graph : td.graph.BaseGraph | None, optional
        The ground truth graph to score the solution against.

    Returns
    -------
    TrackingSolution
        The solution to the tracking problem.
    """
    if td.DEFAULT_ATTR_KEYS.SOLUTION in graph.node_attr_keys():
        graph.update_node_attrs(attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: False})
        graph.update_edge_attrs(attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: False})

    sim_weight = -td.EdgeAttr("similarity") + edge_bias
    # delta_t_weight = (sim_weight.sign() * solver_config.delta_t_weight * (td.EdgeAttr("delta_t").abs() - 1)).exp()
    delta_t_weight = (-delta_t_weight * (td.EdgeAttr("delta_t").abs() - 1)).exp()
    edge_weight = sim_weight * delta_t_weight

    # edge_df = graph.edge_attrs()
    # edge_weight_series = edge_weight.evaluate(edge_df)
    # edge_df = edge_df.with_columns(
    #     sim_weight.evaluate(edge_df).alias("sim_weight"),
    #     delta_t_weight.evaluate(edge_df).alias("delta_t_weight"),
    #     edge_weight_series.alias("edge_weight"),
    # ).select("delta_t", "similarity", "sim_weight", "delta_t_weight", "edge_weight")
    # print(edge_df)

    not_first_frame = td.NodeAttr(td.DEFAULT_ATTR_KEYS.T) != td.NodeAttr(td.DEFAULT_ATTR_KEYS.T).min()
    not_last_frame = td.NodeAttr(td.DEFAULT_ATTR_KEYS.T) != td.NodeAttr(td.DEFAULT_ATTR_KEYS.T).max()

    LOG.info("Metadata: %s", graph.metadata())
    no_division = graph.metadata().get("no_division", False)

    kwargs = {
        "appearance_weight": appearance_weight * (1 - td.NodeAttr("orphan_prob")) * not_first_frame,
        "disappearance_weight": disappearance_weight * not_last_frame,
        "division_weight": 1_000_000 if no_division else division_weight,
        "node_weight": node_weight,
        "timeout": timeout,
    }

    # two pass solving
    if tracklet_solver:
        td.solvers.ILPSolver(
            edge_weight=-td.EdgeAttr("similarity") + edge_bias + (td.EdgeAttr("delta_t") > 1) * math.inf,
            **kwargs,
        ).solve(graph)

        solution_graph = TrackletSolver(
            edge_weight=edge_weight,
            # edge_weight=-td.EdgeAttr("similarity") + solver_config.edge_bias,
            **kwargs,
        ).solve(graph)

    else:
        solution_graph = td.solvers.ILPSolver(
            edge_weight=edge_weight,
            **kwargs,
        ).solve(graph)

    return solution_graph

