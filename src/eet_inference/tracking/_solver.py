import math

import tracksdata as td
from pydantic import BaseModel, Field

from eet_inference._logging import LOG
from eet_inference.tracking._tracklet_solver import TrackletSolver


class ILPSolverConfig(BaseModel):
    """Configuration for the ILP tracking solver.

    Parameters
    ----------
    appearance_weight : float, default=1.0
        Weight for appearance edges (nodes appearing in the first frame or orphans).
        Higher values make the solver prefer nodes to appear rather than be linked.
    disappearance_weight : float, default=1.0
        Weight for disappearance edges (nodes disappearing in the last frame or becoming orphans).
        Higher values make the solver prefer nodes to disappear rather than continue tracking.
    division_weight : float, default=1.0
        Weight for cell division edges. Set to a very high value (e.g., 1e6) to effectively
        disable divisions in the tracking solution.
    node_weight : float, default=1.0
        Weight for node selection in the tracking graph.
        Higher values encourage more nodes to be selected in the solution.
    delta_t_weight : float, default=0.0
        Penalty weight for edges spanning multiple time frames.
        Positive values discourage long temporal gaps between linked nodes.
        The penalty grows exponentially: exp(-delta_t_weight * (delta_t - 1))
    edge_bias : float, default=0.0
        Constant bias added to all edge weights before solving.
        Positive values favor edge creation, negative values discourage it.
    timeout : float, default=600.0
        Maximum time in seconds for the ILP solver to run.
        The solver will return the best solution found within this time limit.
    tracklet_solver : bool, default=False
        Whether to use a two-pass tracklet solver.
        First pass: solve for tracklets (short tracks)
        Second pass: link tracklets together
        This can improve performance on large graphs but may affect solution quality.

    Examples
    --------
    >>> # Default configuration
    >>> config = ILPSolverConfig()

    >>> # Disable divisions by setting very high weight
    >>> config = ILPSolverConfig(division_weight=1e6)

    >>> # Penalize temporal gaps
    >>> config = ILPSolverConfig(delta_t_weight=0.5)

    >>> # Two-pass tracklet solving with custom weights
    >>> config = ILPSolverConfig(
    ...     appearance_weight=2.0,
    ...     disappearance_weight=2.0,
    ...     tracklet_solver=True
    ... )
    """

    appearance_weight: float = Field(default=1.0, ge=0.0, description="Weight for appearance edges")
    disappearance_weight: float = Field(default=1.0, ge=0.0, description="Weight for disappearance edges")
    division_weight: float = Field(default=1.0, ge=0.0, description="Weight for division edges")
    node_weight: float = Field(default=1.0, description="Weight for node selection")
    delta_t_weight: float = Field(default=0.0, ge=0.0, description="Penalty for edges spanning multiple frames")
    edge_bias: float = Field(default=0.0, description="Bias added to edge weights")
    timeout: float = Field(default=600.0, gt=0.0, description="Solver timeout in seconds")
    tracklet_solver: bool = Field(default=False, description="Use two-pass tracklet solver")

    model_config = {"frozen": True}  # Make immutable


def solve_tracking(
    graph: td.graph.BaseGraph,
    config: ILPSolverConfig,
) -> td.graph.InMemoryGraph:
    """
    Solve the tracking problem using ILP optimization.

    This function applies an Integer Linear Programming (ILP) solver to find the optimal
    tracking solution given edge similarity scores and orphan probabilities in the graph.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to solve the tracking problem for. Must contain:
        - 'similarity' edge attribute: edge probabilities/scores
        - 'orphan_prob' node attribute: orphan probabilities for nodes
    config : ILPSolverConfig
        Configuration parameters for the ILP solver (weights, timeout, etc.).

    Returns
    -------
    td.graph.InMemoryGraph
        The solved tracking graph with 'solution' attributes set on nodes and edges.

    Notes
    -----
    - Resets any existing solution in the graph before solving
    - Respects 'no_division' metadata flag to disable cell divisions
    - Appearance weights are modulated by orphan probabilities
    - If tracklet_solver=True, uses two-pass solving (tracklets then linkage)
    """
    if td.DEFAULT_ATTR_KEYS.SOLUTION in graph.node_attr_keys():
        graph.update_node_attrs(attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: False})
        graph.update_edge_attrs(attrs={td.DEFAULT_ATTR_KEYS.SOLUTION: False})

    # Compute edge weights
    sim_weight = -td.EdgeAttr("similarity") + config.edge_bias
    delta_t_penalty = (-config.delta_t_weight * (td.EdgeAttr("delta_t").abs() - 1)).exp()
    edge_weight = sim_weight * delta_t_penalty

    # Compute node filters
    not_first_frame = td.NodeAttr(td.DEFAULT_ATTR_KEYS.T) != td.NodeAttr(td.DEFAULT_ATTR_KEYS.T).min()
    not_last_frame = td.NodeAttr(td.DEFAULT_ATTR_KEYS.T) != td.NodeAttr(td.DEFAULT_ATTR_KEYS.T).max()

    LOG.info("Metadata: %s", graph.metadata())
    no_division = graph.metadata().get("no_division", False)

    kwargs = {
        "appearance_weight": config.appearance_weight * (1 - td.NodeAttr("orphan_prob")) * not_first_frame,
        "disappearance_weight": config.disappearance_weight * not_last_frame,
        "division_weight": 1_000_000 if no_division else config.division_weight,
        "node_weight": config.node_weight,
        "timeout": config.timeout,
    }

    # Two-pass solving: first tracklets, then linkage
    if config.tracklet_solver:
        td.solvers.ILPSolver(
            edge_weight=-td.EdgeAttr("similarity") + config.edge_bias + (td.EdgeAttr("delta_t") > 1) * math.inf,
            **kwargs,
        ).solve(graph)

        solution_graph = TrackletSolver(
            edge_weight=edge_weight,
            **kwargs,
        ).solve(graph)

    else:
        solution_graph = td.solvers.ILPSolver(
            edge_weight=edge_weight,
            **kwargs,
        ).solve(graph)

    return solution_graph

