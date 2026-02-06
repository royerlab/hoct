"""Model prediction and inference utilities for EET."""

from contextlib import nullcontext
from typing import NamedTuple

import polars as pl
import torch
import tracksdata as td
from torch.utils.data import DataLoader

from eet_inference._logging import LOG
from eet_inference.data import DataKeys, FrameDataset, TiledRoiDataset
from eet_inference.tracking import ILPSolverConfig, solve_tracking


class ModelPrediction(NamedTuple):
    """Output from EdgeModel forward pass.

    Attributes
    ----------
    edge_logits : torch.Tensor
        Edge classification logits of shape (B, E, C) where C is n_classes.
    node_features : torch.Tensor
        Updated node features of shape (B, N, hidden_size).
    edge_features : torch.Tensor
        Edge features of shape (B, E, hidden_size).
    orphan_logits : torch.Tensor
        Orphan logits for nodes of shape (B, N, 1).
    """

    edge_logits: torch.Tensor
    node_features: torch.Tensor
    edge_features: torch.Tensor
    orphan_logits: torch.Tensor


class EdgeModel(torch.nn.Module):
    """Abstract base class for edge prediction models.

    Subclasses should implement the forward method to perform edge prediction.
    """

    def forward(
        self,
        input_batch: torch.Tensor,
        node_pos: torch.Tensor,
        edge_pos: torch.Tensor,
        edge_indices: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> ModelPrediction:
        """Forward pass for edge prediction.

        Parameters
        ----------
        input_batch : torch.Tensor
            Input features of shape (B, N, C).
        node_pos : torch.Tensor
            Node positions of shape (B, N, D).
        edge_pos : torch.Tensor
            Edge positions of shape (B, E, D).
        edge_indices : torch.Tensor
            Edge indices of shape (B, E, 2).
        node_mask : torch.Tensor
            Node mask of shape (B, N).
        edge_mask : torch.Tensor
            Edge mask of shape (B, E).

        Returns
        -------
        ModelPrediction
            Model predictions containing edge logits, node features, edge features, and orphan logits.
        """
        ...


@torch.inference_mode()
def model_predict(
    model: EdgeModel,
    ds: FrameDataset | TiledRoiDataset | DataLoader,
    solver_config: ILPSolverConfig | None = None,
) -> td.graph.InMemoryGraph:
    """
    Run model prediction on a dataset and solve tracking.

    This function:
    1. Runs the model on all items in the dataset
    2. Aggregates predictions (similarity and orphan probabilities)
    3. Updates the graph with predictions
    4. Solves the tracking problem using ILP solver

    Parameters
    ----------
    model : EdgeModel
        The trained edge prediction model.
    ds : FrameDataset | TiledRoiDataset | DataLoader
        Dataset or DataLoader to run inference on.
    solver_config : ILPSolverConfig | None, default=None
        Configuration for the ILP tracking solver. If None, uses default configuration.
        See ILPSolverConfig for available parameters (weights, timeout, etc.).

    Notes
    -----
    The function modifies the dataset's graph in-place by adding:
    - 'similarity' edge attribute: normalized edge probabilities
    - 'orphan_prob' node attribute: probability of node being an orphan

    Examples
    --------
    >>> from eet_inference.data import FrameDataset
    >>> from eet_inference.inference import model_predict, EdgeModel
    >>> from eet_inference.tracking import ILPSolverConfig
    >>>
    >>> # Load dataset and model
    >>> ds = FrameDataset(graph_path="data.geff", window_size=3)
    >>> model = EdgeModel.load("model.pt")
    >>>
    >>> # Run inference with default config
    >>> model_predict(model, ds)
    >>>
    >>> # Run inference with custom config
    >>> config = ILPSolverConfig(
    ...     appearance_weight=2.0,
    ...     division_weight=1e6,  # Disable divisions
    ...     tracklet_solver=True,
    ... )
    >>> model_predict(model, ds, solver_config=config)
    """
    if solver_config is None:
        solver_config = ILPSolverConfig()
    model.eval()
    device = next(model.parameters()).device

    if str(device) == "cpu":
        LOG.warning("Model is on CPU, use `cuda` or `mps` to speed up inference")

    edge_ids = []
    delta_t = []
    sim_exp = []
    orphan_exp = []
    node_ids = []

    # Handle both Dataset and DataLoader
    if not isinstance(ds, DataLoader):

        def _ds_iterator():
            for i in range(len(ds)):
                yield ds[i]

        def _expand_dims(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.unsqueeze(0).to(device)

    else:

        def _ds_iterator():
            return ds

        def _expand_dims(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device)

    # disabling recompilation
    torch._C._jit_set_bailout_depth(0)

    # Run model inference
    with (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    ):
        for batch in _ds_iterator():
            input_batch = _expand_dims(batch[DataKeys.NODE_FEATS])
            edges = _expand_dims(batch[DataKeys.EDGE_BATCH_ID])
            node_mask = _expand_dims(batch.get(DataKeys.NODE_MASK, None))
            edge_mask = _expand_dims(batch.get(DataKeys.EDGE_MASK, None))
            e_id = _expand_dims(batch[DataKeys.EDGE_ID])
            n_id = _expand_dims(batch[DataKeys.NODE_ID])
            d_t = _expand_dims(batch[DataKeys.DELTA_T])
            node_pos = _expand_dims(batch[DataKeys.NODE_POS])
            edge_pos = _expand_dims(batch[DataKeys.EDGE_POS])

            if node_mask is None:
                node_mask = torch.ones(input_batch.shape[:2], dtype=torch.bool, device=device)

            if edge_mask is None:
                edge_mask = torch.ones(edges.shape[:2], dtype=torch.bool, device=device)

            model_output = model.forward(input_batch, node_pos, edge_pos, edges, node_mask, edge_mask)
            pred, _, _, oph_logits = model_output

            if edge_mask is not None:
                pred = pred[edge_mask]
                e_id = e_id[edge_mask]
                d_t = d_t[edge_mask]

            edge_ids.append(e_id.cpu().ravel())
            sim_exp.append(pred.float().clamp(max=20).exp().cpu().ravel())
            delta_t.append(d_t.cpu().ravel())

            if node_mask is not None:
                oph_logits = oph_logits[node_mask]
                n_id = n_id[node_mask]

            orphan_exp.append(oph_logits.float().clamp(max=20).exp().cpu().ravel())
            node_ids.append(n_id.cpu().ravel())

    # Concatenate all predictions
    edge_ids = torch.cat(edge_ids, dim=0)
    sim_exp = torch.cat(sim_exp, dim=0)
    delta_t = torch.cat(delta_t, dim=0)
    node_ids = torch.cat(node_ids, dim=0)
    orphan_exp = torch.cat(orphan_exp, dim=0)

    # Validate shapes
    if edge_ids.shape != sim_exp.shape:
        raise ValueError(f"'edge_ids' and 'similarity' have different shapes: {edge_ids.shape} != {sim_exp.shape}")

    if edge_ids.shape != delta_t.shape:
        raise ValueError(f"'edge_ids' and 'delta_t' have different shapes: {edge_ids.shape} != {delta_t.shape}")

    # Extract dataset from DataLoader if needed
    if isinstance(ds, DataLoader):
        ds = ds.dataset

    # Aggregate node predictions (median over all windows)
    node_df = (
        pl.DataFrame(
            {
                DataKeys.NODE_ID: node_ids.tolist(),
                "orphan_exp": orphan_exp.tolist(),
            }
        )
        .group_by(DataKeys.NODE_ID)
        .median()
    )

    # Aggregate edge predictions (median similarity, first delta_t)
    edge_df = (
        pl.DataFrame(
            {
                DataKeys.EDGE_ID: edge_ids.tolist(),
                "sim_exp": sim_exp.tolist(),
                "delta_t": delta_t.tolist(),
            }
        )
        .group_by(DataKeys.EDGE_ID)
        .agg(pl.col("sim_exp").median(), pl.col("delta_t").first())
    )

    # Compute parental softmax normalization
    # Join edge data with source/target node IDs
    edge_df = edge_df.join(
        ds.graph.edge_attrs(attr_keys=[]),
        on=DataKeys.EDGE_ID,
    )

    # Join with orphan probabilities for target nodes
    edge_df = edge_df.join(node_df, left_on="target_id", right_on=DataKeys.NODE_ID, validate="m:1")

    # Normalize: p(edge | target) = exp(edge_logit) / (sum(exp(edge_logits to target)) + exp(orphan_logit))
    edge_df = edge_df.with_columns(
        (pl.col("sim_exp") / (pl.col("sim_exp").sum().over("target_id", "delta_t") + pl.col("orphan_exp")))
        .fill_nan(0.0)
        .alias("similarity"),
    )

    # Compute orphan probabilities with parental normalization
    denom_df = edge_df.group_by("target_id", "delta_t").agg(pl.col("sim_exp").sum().alias("denom"))

    node_df = node_df.join(denom_df, left_on=DataKeys.NODE_ID, right_on="target_id")
    node_df = node_df.with_columns(pl.col(pl.Float64, pl.Float32).fill_null(0.0))
    node_df = node_df.with_columns(
        (pl.col("orphan_exp") / (pl.col("denom") + pl.col("orphan_exp"))).fill_nan(0.0).alias("orphan_prob"),
        (pl.col("delta_t").max().over(DataKeys.NODE_ID) - pl.col("delta_t") + 1).alias("delta_t_weighted"),
    )

    # Weighted average over all delta_t (give more weight to smaller delta_t)
    node_df = (
        node_df.with_columns((pl.col("orphan_prob") * pl.col("delta_t_weighted")).alias("orphan_prob_weighted"))
        .group_by(DataKeys.NODE_ID)
        .agg((pl.col("orphan_prob_weighted").sum() / pl.col("delta_t_weighted").sum()).alias("orphan_prob"))
    )

    # Update graph with predictions
    if "similarity" not in ds.graph.edge_attr_keys():
        ds.graph.add_edge_attr_key("similarity", pl.Float32, -1.0)

    ds.graph.update_edge_attrs(
        attrs={
            "similarity": edge_df["similarity"].to_list(),
        },
        edge_ids=edge_df[DataKeys.EDGE_ID].to_list(),
    )

    if "orphan_prob" not in ds.graph.node_attr_keys():
        ds.graph.add_node_attr_key("orphan_prob", pl.Float32, 1.0)

    ds.graph.update_node_attrs(
        attrs={
            "orphan_prob": node_df["orphan_prob"].to_list(),
        },
        node_ids=node_df[DataKeys.NODE_ID].to_list(),
    )

    # Solve tracking
    solution_graph = solve_tracking(
        graph=ds.graph,
        config=solver_config,
    )

    return solution_graph
