"""High-level prediction API for EET inference from raw images and labels."""

import polars as pl
import tracksdata as td
from eet_features.constants import REGIONPROPS
from eet_features.graph import create_graph
from numpy.typing import ArrayLike
from tracksdata.functional import TilingScheme

from eet_inference._logging import LOG
from eet_inference.data import FrameDataset, GraphConcatDataset, TiledRoiDataset
from eet_inference.data._transforms import Affine, Flip, Standardize
from eet_inference.inference import EdgeModel, model_predict
from eet_inference.tracking import ILPSolverConfig

__all__ = ["create_graph_from_points", "predict"]


_MEAN = [
    4.6326e02,
    2.9380e00,
    3.5649e02,
    3.4491e02,
    1.1521e01,
    2.7600e-01,
    9.6600e-01,
    5.7400e-01,
    1.6200e-01,
    1.6781e02,
    -2.7000e-02,
    5.0000e-02,
    -2.7000e-02,
    8.7012e01,
    -1.4010e00,
    5.0000e-02,
    -1.4010e00,
    8.3695e01,
    9.0000e-03,
]

_STD = [
    5.5578e02,
    7.6000e00,
    1.9588e02,
    2.2610e02,
    8.1990e00,
    2.1600e-01,
    2.8100e-01,
    1.9300e-01,
    6.9000e-02,
    6.7845e02,
    3.1670e00,
    2.8750e00,
    3.1670e00,
    5.1292e02,
    1.8274e02,
    2.8750e00,
    1.8274e02,
    3.0608e02,
    7.8000e-02,
]


def create_graph_from_points(
    points: pl.DataFrame,
    distance_threshold: float = 200.0,
    n_neighbors: int = 5,
    max_delta_t: int = 3,
    scale: tuple[float, ...] | None = None,
    shape: tuple[int, ...] | None = None,
) -> td.graph.InMemoryGraph:
    """
    Create a candidate tracking graph from point coordinates only.

    This function creates a minimal graph with only spatial and temporal coordinates,
    without computing regionprops or intensity features. It's useful for tracking
    pre-detected points from other detection methods.

    Parameters
    ----------
    points : pl.DataFrame
        DataFrame with columns: 't', 'z', 'y', 'x' (and optionally 'node_id').
        For 2D data, 'z' column can be omitted or set to 0.
    distance_threshold : float, default=200.0
        Maximum distance for creating candidate edges.
    n_neighbors : int, default=5
        Maximum number of neighbors to connect per node.
    max_delta_t : int, default=3
        Maximum temporal gap for edges.
    scale : tuple[float, ...] | None, default=None
        Physical spacing (t, [z,] y, x). If None, uses isotropic spacing.
    shape : tuple[int, ...] | None, default=None
        Shape of the volume (T, [Z,] Y, X) for border distance calculation.
        If None, border distances are not computed.

    Returns
    -------
    td.graph.InMemoryGraph
        Candidate tracking graph with nodes and edges.

    Examples
    --------
    >>> import polars as pl
    >>> from eet_inference import create_graph_from_points
    >>>
    >>> # Create 2D points
    >>> points = pl.DataFrame(
    ...     {
    ...         "t": [0, 0, 1, 1],
    ...         "y": [10, 20, 15, 25],
    ...         "x": [10, 20, 12, 22],
    ...     }
    ... )
    >>> graph = create_graph_from_points(points)
    >>>
    >>> # Create 3D points with node_id
    >>> points = pl.DataFrame(
    ...     {
    ...         "node_id": [1, 2, 3, 4],
    ...         "t": [0, 0, 1, 1],
    ...         "z": [5, 5, 6, 6],
    ...         "y": [10, 20, 15, 25],
    ...         "x": [10, 20, 12, 22],
    ...     }
    ... )
    >>> graph = create_graph_from_points(points, scale=(1, 0.5, 1, 1))

    Notes
    -----
    - Automatically detects 2D vs 3D based on presence of 'z' column
    - If 'node_id' column exists, it's used as the node identifier
    - Only adds t, z, y, x coordinates - no regionprops or intensity features
    """
    # TODO: Implement using tracksdata - waiting for user guidance
    pass


def predict(
    model: EdgeModel,
    *,
    graph: td.graph.BaseGraph | None = None,
    labels: ArrayLike | None = None,
    images: ArrayLike | None = None,
    solver_config: ILPSolverConfig | None = None,
    distance_threshold: float = 300.0,
    n_neighbors: int = 5,
    max_delta_t: int = 3,
    scale: tuple[float, ...] | None = None,
    window_size: int = 5,
    tiling_scheme: TilingScheme | None = None,
    test_time_augs: int = 0,
    return_solution: bool = True,
) -> td.graph.InMemoryGraph | None:
    """
    Run end-to-end cell tracking prediction from raw data.

    This is the main high-level API for EET inference. It takes raw segmentation
    labels (and optionally intensity images), creates a candidate tracking graph,
    runs the neural network to predict edge probabilities and orphan probabilities,
    and solves the tracking problem using ILP optimization.

    Parameters
    ----------
    model : EdgeModel
        Trained EET edge prediction model (PyTorch JIT or regular model).
    graph : td.graph.BaseGraph | None
        Graph with features ready for prediction.
    labels : ArrayLike | None
        Segmentation labels of shape (T, [Z,] Y, X)
    images : ArrayLike | None
        Optional intensity images of shape (T, [Z,] Y, X).
    solver_config : ILPSolverConfig | None, default=None
        Configuration for the ILP tracking solver. If None, uses defaults.
    distance_threshold : float, default=200.0
        Maximum distance for creating candidate edges.
    n_neighbors : int, default=5
        Maximum number of neighbors to connect per node.
    max_delta_t : int, default=3
        Maximum temporal gap for edges.
    scale : tuple[float, ...] | None, default=None
        Physical spacing (t, [z,] y, x). If None, uses isotropic spacing.
    window_size : int, default=3
        Temporal window size for the frame dataset. Only used if tiling_scheme is None.
    tiling_scheme : TilingScheme | None, default=None
        Optional tiling scheme for spatially tiled inference. If provided, uses TiledRoiDataset
        instead of FrameDataset. Useful for large volumes that don't fit in memory.
    test_time_augs : int, default=0
        Number of test time augmentations to apply. If 0, no augmentations are applied.
        If > 0, a random augmentation is applied for each augmentation.
    return_solution : bool
        Whether to return the solution graph or not.

    Returns
    -------
    td.graph.InMemoryGraph
        Solved tracking graph with 'solution' attributes on nodes and edges.
        Nodes and edges with solution=True form the final tracking result.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from eet_inference import predict
    >>> from eet_inference.tracking import ILPSolverConfig
    >>>
    >>> # Load model
    >>> model = torch.jit.load("eet_model.pt")
    >>>
    >>> # Create synthetic labels
    >>> labels = np.random.randint(0, 20, size=(10, 256, 256))
    >>>
    >>> # Run prediction with default settings
    >>> graph = predict(model, labels=labels)
    >>>
    >>> # Access solution
    >>> solution_nodes = graph.node_attrs(["solution"])
    >>> solution_edges = graph.edge_attrs(["solution"])
    >>>
    >>> # Run with custom solver config
    >>> config = ILPSolverConfig(
    ...     appearance_weight=2.0,
    ...     division_weight=1e6,  # disable divisions
    ...     delta_t_weight=0.5,
    ... )
    >>> graph = predict(model, labels=labels, solver_config=config)

    Notes
    -----
    This function performs the following steps:
    1. Creates candidate tracking graph from labels using eet_features.graph.create_graph
    2. Creates dataset (FrameDataset or TiledRoiDataset depending on tiling_scheme)
    3. Runs model inference to predict edge similarities and orphan probabilities
    4. Solves tracking using ILP optimization
    5. Returns graph with solution attributes

    When tiling_scheme is provided, the function uses TiledRoiDataset for spatially tiled
    inference, which is useful for large volumes that cannot fit in GPU memory at once.
    """
    if solver_config is None:
        solver_config = ILPSolverConfig.default()

    LOG.info("Starting EET prediction pipeline")

    if graph is None:
        LOG.info("Creating candidate tracking graph")
        graph = create_graph(
            labels=labels,
            images=images,
            gt_graph=None,  # Inference mode - no ground truth
            distance_threshold=distance_threshold,
            n_neighbors=n_neighbors,
            delta_t=max_delta_t,
            scale=scale,
        )
    else:
        if labels is not None:
            LOG.warning("`graph` and `labels` are not `None`, `labels` are ignored")
        if images is not None:
            LOG.warning("`graph` and `images` are not `None`, `images` are ignored")

    LOG.info(f"Created graph with {graph.num_nodes()} nodes and {graph.num_edges()} edges")

    if test_time_augs > 0:
        df_transforms = [
            Flip(columns=["z", "y", "x"], p=0.5),
            Affine(
                degree_range=(-180, 180),
                scale_range=(1, 1),
                shear_range=((0, 0), (0, 0)),
            ),
        ]
    else:
        df_transforms = []

    if tiling_scheme is not None:
        LOG.info("Creating tiled ROI dataset")
        dataset = TiledRoiDataset(
            graph=graph,
            properties=REGIONPROPS,
            tiling_scheme=tiling_scheme,
            df_transforms=df_transforms,
            dict_transforms=[Standardize(mean=_MEAN, std=_STD)],
        )
    else:
        LOG.info("Creating frame dataset with window_size=%d", window_size)
        dataset = FrameDataset(
            graph=graph,
            min_window_size=window_size,
            properties=REGIONPROPS,
            df_transforms=df_transforms,
            dict_transforms=[Standardize(mean=_MEAN, std=_STD)],
        )

    if test_time_augs > 0:
        dataset = GraphConcatDataset([dataset] * test_time_augs)

    LOG.info("Running model inference and solving tracking")
    solution_graph = model_predict(model, dataset, solver_config=solver_config, return_solution=return_solution)

    if solution_graph is not None:
        LOG.info(f"Solution: {solution_graph.num_nodes()} nodes, {solution_graph.num_edges()} edges")

    return solution_graph
