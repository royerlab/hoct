"""Model prediction and inference utilities for HOCT."""

import os
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from typing import Any, NamedTuple

# this is to avoid OOM errors when using large tiling schemes
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # type: ignore

import polars as pl
import torch
import tracksdata as td
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from hoct_inference._logging import LOG
from hoct_inference.data import DataKeys, FrameDataset, TiledRoiDataset
from hoct_inference.tracking import ILPSolverConfig, solve_tracking


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


def _make_iterator(
    ds: FrameDataset | TiledRoiDataset | DataLoader,
    device: torch.device,
) -> tuple[Callable[[], Iterator], Callable[[torch.Tensor | None], torch.Tensor | None]]:
    """Return (iterator_fn, expand_dims_fn) adapted for Dataset or DataLoader input."""
    if not isinstance(ds, DataLoader):

        def _ds_iterator() -> Iterator:
            for i in tqdm(range(len(ds)), desc="Model inference"):
                yield ds[i]

        def _expand_dims(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.unsqueeze(0).to(device)

    else:

        def _ds_iterator() -> Iterator:
            return ds

        def _expand_dims(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(device)

    return _ds_iterator, _expand_dims


def _autocast_ctx(device: torch.device) -> AbstractContextManager:
    """Return bfloat16 autocast context for CUDA, or a no-op for other devices."""
    if device.type == "cuda":
        return torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def _prefetch_iterator(
    ds: Dataset | DataLoader,
    device: torch.device,
    prefetch: bool | None = None,
) -> Generator[Any, None, None]:
    """Iterate over dataset / dataloader items with optional prefetch + H2D transfer.

    When ``prefetch=True`` a background thread fetches the next item and moves
    its tensors to ``device`` while the caller processes the current item,
    hiding both I/O and the H2D copy behind compute. For map-style ``Dataset``
    inputs a leading batch dim is added via ``unsqueeze(0)`` before transfer;
    ``DataLoader`` batches are assumed to already be batched.

    Non-tensor fields pass through unchanged. ``None`` items are yielded as-is
    so the caller can decide how to handle empty batches.

    Parameters
    ----------
    ds : Dataset | DataLoader
        Map-style dataset (supports ``len(ds)`` and ``ds[i]``) or a DataLoader.
    device : torch.device
        Target device for the H2D transfer.
    prefetch : bool | None, default=None
        Whether to fetch + transfer the next item in a background thread. When
        ``None``, defaults to ``True`` for map-style ``Dataset`` inputs and
        ``False`` for ``DataLoader`` (which has its own worker-based prefetching).

    Yields
    ------
    Any
        The fetched item with tensor fields moved to ``device``.
    """
    is_loader = isinstance(ds, DataLoader)
    is_iterable = isinstance(ds, IterableDataset)
    if prefetch is None:
        prefetch = not is_loader and not is_iterable
    LOG.info(f"Prefetching: {prefetch}")

    needs_unsqueeze = not is_loader

    def _transfer(item: Any) -> Any:
        if item is None:
            return None
        return {
            k: (
                (v.unsqueeze(0) if needs_unsqueeze else v).to(device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in item.items()
        }

    stop = object()
    total = len(ds) if hasattr(ds, "__len__") else None

    if is_loader or is_iterable:
        source = iter(ds)

        def _fetch() -> Any:
            try:
                return _transfer(next(source))
            except StopIteration:
                return stop
    else:
        indices = iter(range(total))

        def _fetch() -> Any:
            try:
                i = next(indices)
            except StopIteration:
                return stop
            return _transfer(ds[i])

    pbar = tqdm(total=total, desc="Model inference")
    try:
        if not prefetch:
            while True:
                item = _fetch()
                if item is stop:
                    return
                pbar.update()
                yield item

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_fetch)
            while True:
                item = future.result()
                if item is stop:
                    return
                future = pool.submit(_fetch)
                pbar.update()
                yield item
    finally:
        pbar.close()


@torch.inference_mode()
def model_predict(
    model: EdgeModel,
    ds: FrameDataset | TiledRoiDataset | IterableDataset | DataLoader,
    solver_config: ILPSolverConfig | None = None,
    return_solution: bool = True,
    prefetch: bool | None = None,
) -> td.graph.InMemoryGraph | None:
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
    return_solution : bool, default=True
        Whether to return the solved tracking graph.
    prefetch : bool | None, default=None
        Whether to fetch the next item and move its tensors to the model's device
        in a background thread while the model processes the current one. When
        ``None``, defaults to ``True`` for map-style ``Dataset`` inputs and
        ``False`` for ``DataLoader`` (which already prefetches via its own workers).

    Notes
    -----
    The function modifies the dataset's graph in-place by adding:
    - 'similarity' edge attribute: normalized edge probabilities
    - 'orphan_prob' node attribute: probability of node being an orphan

    Examples
    --------
    >>> from hoct_inference.data import FrameDataset
    >>> from hoct_inference.inference import model_predict, EdgeModel
    >>> from hoct_inference.tracking import ILPSolverConfig
    >>>
    >>> # Load dataset and model
    >>> graph, _ = td.graph.InMemoryGraph.from_geff("data.geff")
    >>> ds = FrameDataset(graph=graph, min_window_size=3)
    >>> model = torch.jit.load("model.pt")
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
    LOG.info("Starting model prediction pipeline")
    if solver_config is None:
        solver_config = ILPSolverConfig.default()
    model.eval()
    device = next(model.parameters()).device
    LOG.info(f"Model loaded on device: {device}")

    if device.type == "cpu":
        LOG.warning("Model is on CPU, use `cuda` or `mps` to speed up inference")

    edge_ids = []
    delta_t = []
    sim_exp = []
    orphan_exp = []
    node_ids = []

    # disabling recompilation
    torch._C._jit_set_bailout_depth(0)
    # torch.jit.set_fusion_strategy([])  # this doesn't work

    LOG.info("Starting model inference loop")
    # Run model inference
    with _autocast_ctx(device):
        batch_idx = 0
        for batch in _prefetch_iterator(ds, device, prefetch=prefetch):
            LOG.debug("Processing batch %d", batch_idx)
            if batch is None:
                LOG.debug("Batch %d is None, skipping", batch_idx)
                continue

            input_batch = batch[DataKeys.NODE_FEATS]
            edges = batch[DataKeys.EDGE_BATCH_ID]
            node_mask = batch.get(DataKeys.NODE_MASK)
            edge_mask = batch.get(DataKeys.EDGE_MASK)
            e_id = batch[DataKeys.EDGE_ID]
            n_id = batch[DataKeys.NODE_ID]
            d_t = batch[DataKeys.DELTA_T]
            node_pos = batch[DataKeys.NODE_POS]
            edge_pos = batch[DataKeys.EDGE_POS]

            # e_id.shape[1] is the number of edges in the batch
            if e_id.shape[1] <= 1 or (edge_mask is not None and edge_mask.sum() == 0):
                continue

            if node_mask is None:
                node_mask = torch.ones(input_batch.shape[:2], dtype=torch.bool, device=device)

            if edge_mask is None:
                edge_mask = torch.ones(edges.shape[:2], dtype=torch.bool, device=device)

            LOG.debug("Running model forward pass for batch %d", batch_idx)
            model_output = model.forward(input_batch, node_pos, edge_pos, edges, node_mask, edge_mask)
            pred, _, _, oph_logits = model_output
            LOG.debug("Model forward pass completed for batch %d", batch_idx)

            pred = pred[edge_mask]
            e_id = e_id[edge_mask]
            d_t = d_t[edge_mask]

            edge_ids.append(e_id.cpu().ravel())
            sim_exp.append(pred.float().clamp(max=20).exp().cpu().ravel())
            delta_t.append(d_t.cpu().ravel())

            oph_logits = oph_logits[node_mask]
            n_id = n_id[node_mask]

            orphan_exp.append(oph_logits.float().clamp(max=20).exp().cpu().ravel())
            node_ids.append(n_id.cpu().ravel())
            LOG.debug("Batch %d processed successfully", batch_idx)
            batch_idx += 1

    LOG.info("Completed inference on %d batches", batch_idx)
    LOG.info("Concatenating predictions")
    # Concatenate all predictions
    edge_ids = torch.cat(edge_ids, dim=0)
    sim_exp = torch.cat(sim_exp, dim=0)
    delta_t = torch.cat(delta_t, dim=0)
    node_ids = torch.cat(node_ids, dim=0)
    orphan_exp = torch.cat(orphan_exp, dim=0)
    LOG.info("Concatenated predictions - edges: %d, nodes: %d", edge_ids.shape[0], node_ids.shape[0])

    # Validate shapes
    if edge_ids.shape != sim_exp.shape:
        raise ValueError(f"'edge_ids' and 'similarity' have different shapes: {edge_ids.shape} != {sim_exp.shape}")

    if edge_ids.shape != delta_t.shape:
        raise ValueError(f"'edge_ids' and 'delta_t' have different shapes: {edge_ids.shape} != {delta_t.shape}")
    LOG.info("Shape validation passed")

    # Extract dataset from DataLoader if needed
    if isinstance(ds, DataLoader):
        ds = ds.dataset

    LOG.info("Aggregating node predictions")
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
    LOG.info("Node predictions aggregated: %d unique nodes", len(node_df))

    LOG.info("Aggregating edge predictions")
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
    LOG.info("Edge predictions aggregated: %d unique edges", len(edge_df))

    LOG.info("Computing parental softmax normalization")
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
        (-solver_config.delta_t_weight * (pl.col("delta_t").abs() - 1)).exp().alias("delta_t_weighted"),
    )

    # Weighted average over all delta_t (give more weight to smaller delta_t)
    node_df = (
        node_df.with_columns((pl.col("orphan_prob") * pl.col("delta_t_weighted")).alias("orphan_prob_weighted"))
        .group_by(DataKeys.NODE_ID)
        .agg((pl.col("orphan_prob_weighted").sum() / pl.col("delta_t_weighted").sum()).alias("orphan_prob"))
    )
    LOG.info("Parental softmax normalization completed")

    LOG.info("Updating graph with predictions")
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
        ds.graph.add_node_attr_key("orphan_prob", pl.Float32, 0.0)

    ds.graph.update_node_attrs(
        attrs={
            "orphan_prob": node_df["orphan_prob"].to_list(),
        },
        node_ids=node_df[DataKeys.NODE_ID].to_list(),
    )
    LOG.info("Graph updated with edge similarities and node orphan probabilities")

    LOG.info("Starting ILP tracking solver")
    # Solve tracking
    solution_graph = solve_tracking(
        graph=ds.graph,
        config=solver_config,
        return_solution=return_solution,
    )
    LOG.info("Tracking solver completed successfully")

    return solution_graph


@torch.inference_mode()
def extract_edge_features(
    model: EdgeModel,
    ds: FrameDataset | TiledRoiDataset | DataLoader,
    edge_filter_key: str | None = None,
) -> pl.DataFrame:
    """
    Run model feature extraction on a dataset.

    Parameters
    ----------
    model : EdgeModel
        The trained edge prediction model.
    ds : FrameDataset | TiledRoiDataset | DataLoader
        Dataset or DataLoader to run inference on.
    edge_filter_key : str | None, default=None
        Key used to select edges after model prediction.

    Returns
    -------
    pl.DataFrame
        DataFrame with model features for each edge and node.
    """
    device = next(model.parameters()).device
    LOG.info("Model loaded on device: %s", device)
    model.eval()

    _ds_iterator, _expand_dims = _make_iterator(ds, device)
    # disabling recompilation
    torch._C._jit_set_bailout_depth(0)
    # torch.jit.set_fusion_strategy([])  # this doesn't work

    LOG.info("Starting model inference loop")
    edge_ids = []
    edge_features = []

    # Run model inference
    with _autocast_ctx(device):
        batch_idx = 0
        for batch in _ds_iterator():
            LOG.debug("Processing batch %d", batch_idx)
            if batch is None:
                LOG.debug("Batch %d is None, skipping", batch_idx)
                continue

            input_batch = _expand_dims(batch[DataKeys.NODE_FEATS])
            edges = _expand_dims(batch[DataKeys.EDGE_BATCH_ID])
            node_mask = _expand_dims(batch.get(DataKeys.NODE_MASK, None))
            edge_mask = _expand_dims(batch.get(DataKeys.EDGE_MASK, None))
            e_id = _expand_dims(batch[DataKeys.EDGE_ID])
            node_pos = _expand_dims(batch[DataKeys.NODE_POS])
            edge_pos = _expand_dims(batch[DataKeys.EDGE_POS])

            # e_id.shape[1] is the number of edges in the batch
            if e_id.shape[1] <= 1 or (edge_mask is not None and edge_mask.sum() == 0):
                continue

            if node_mask is None:
                node_mask = torch.ones(input_batch.shape[:2], dtype=torch.bool, device=device)

            if edge_mask is None:
                edge_mask = torch.ones(edges.shape[:2], dtype=torch.bool, device=device)

            LOG.debug("Running model forward pass for batch %d", batch_idx)
            model_output = model.forward(input_batch, node_pos, edge_pos, edges, node_mask, edge_mask)
            _, _, e_feats, _ = model_output
            LOG.debug("Model forward pass completed for batch %d", batch_idx)

            if edge_filter_key is not None:
                edge_filter_mask = _expand_dims(batch[edge_filter_key].bool())
                if edge_filter_mask.sum() == 0:
                    raise ValueError(f"No edges found after filtering with key '{edge_filter_key}'")
                e_id = e_id[edge_filter_mask]
                e_feats = e_feats[edge_filter_mask]

            edge_ids.append(e_id.cpu().ravel())
            edge_features.append(e_feats.cpu())
            batch_idx += 1

    LOG.info("Completed inference on %d batches", batch_idx)
    LOG.info("Concatenating predictions")
    edge_ids = torch.cat(edge_ids, dim=0)
    edge_features = torch.cat(edge_features, dim=1).squeeze(0)
    LOG.info("Concatenated predictions - edges: %d", edge_ids.shape[0])

    edge_df = pl.DataFrame(
        {
            DataKeys.EDGE_ID: edge_ids,
            "edge_features": edge_features,
        }
    )

    return edge_df
