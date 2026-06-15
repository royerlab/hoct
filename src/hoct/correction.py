import logging

import numpy as np
import polars as pl
import torch
import tracksdata as td
from torch import nn
from tracksdata.functional import TilingScheme

from hoct._api import _create_dataset
from hoct._logging import LOG
from hoct.data import LabeledDataset
from hoct.inference import EdgeModel, ModelPrediction, extract_edge_features

__all__ = ["ProbedModel", "fit_from_labels", "label_edge"]


def label_edge(
    graph: td.graph.BaseGraph,
    source_id: int,
    target_id: int,
    attr_key: str,
    value: bool,
) -> None:
    """
    Set the label of an edge in the graph.

    When value is True, the chosen edge is marked True and all other edges
    sharing the same target node are marked False (mutual exclusion).
    When value is False, only the specified edge is marked False.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to update.
    source_id : int
        The source node id of the edge to label.
    target_id : int
        The target node id of the edge to label.
    attr_key : str
        The edge attribute key to write the label into.
    value : bool
        The label value.
    """
    LOG.debug("Labeling edge %d -> %d: %s = %s", source_id, target_id, attr_key, value)

    if value:
        # FIXME: this query needs to be optimized, we shouldn't need to load all edges
        edges = (
            graph.edge_attrs(
                attr_keys=[
                    td.DEFAULT_ATTR_KEYS.EDGE_ID,
                    td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                    td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
                ]
            )
            .filter(pl.col(td.DEFAULT_ATTR_KEYS.EDGE_TARGET) == target_id)
            .with_columns((pl.col(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE) == source_id).alias(attr_key))
        )
        graph.update_edge_attrs(
            edge_ids=edges[td.DEFAULT_ATTR_KEYS.EDGE_ID].to_list(), attrs={attr_key: edges[attr_key].to_list()}
        )

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("Updated edges: %s", edges.to_dicts())

    else:
        edge_id = graph.edge_id(source_id, target_id)
        graph.update_edge_attrs(edge_ids=[edge_id], attrs={attr_key: False})


class ProbedModel(EdgeModel):
    """EdgeModel with a learned linear probe replacing the original classification head.

    The backbone is kept frozen; only the linear head is trained from user corrections.

    Parameters
    ----------
    edge_model : EdgeModel
        The pretrained backbone model.
    coeffs : np.ndarray
        Linear head weight coefficients; any shape is accepted (raveled to 1D).
    bias : float
        Logistic regression intercept (scalar).
    """

    def __init__(self, edge_model: EdgeModel, coeffs: np.ndarray, bias: float) -> None:
        super().__init__()
        self._edge_model = edge_model
        coeffs_1d = np.asarray(coeffs, dtype=np.float32).ravel()
        self._head = nn.Linear(len(coeffs_1d), 1)
        self._head.weight.data = torch.from_numpy(coeffs_1d).unsqueeze(0)
        self._head.bias.data = torch.tensor([float(bias)])

    def forward(self, *args, **kwargs) -> ModelPrediction:
        _, node_features, edge_features, orphan_logits = self._edge_model(*args, **kwargs)
        new_logits = self._head(edge_features)
        return ModelPrediction(
            edge_logits=new_logits,
            node_features=node_features,
            edge_features=edge_features,
            orphan_logits=orphan_logits,
        )


def fit_from_labels(
    graph: td.graph.BaseGraph,
    model: EdgeModel,
    label_mask_key: str,
    label_key: str,
    tiling_scheme: TilingScheme | None = None,
    window_size: int = 5,
    test_time_augs: int = 5,
    n_steps: int = 500,
    lr: float = 0.1,
    l2_weight: float = 1.0,
    consistency_weight: float = 0.25,
) -> ProbedModel:
    """
    Fit a linear probe from sparse edge corrections and return an adapted model.

    Extracts backbone features for labeled edges, fits a logistic regression
    head via full-batch L-BFGS and returns a ProbedModel wrapping the original
    backbone. The head is always initialised from the backbone's own head layer
    (no warm-start across rounds).

    Two regularization terms prevent the probe from disrupting the global solution:

    - **Adaptive L2** (``l2_weight``): penalizes large head weights, scaled inversely
      with the number of labeled edges so that early rounds are strongly regularized.
    - **Consistency loss** (``consistency_weight``): anchors the probe's predictions on
      *unlabeled* edges to the current ILP solution (0/1), preventing local corrections
      from propagating destructively across the rest of the graph. Scaled the same way.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph containing labeled edges.
    model : EdgeModel
        The pretrained backbone model (or a ProbedModel from a previous round).
    label_mask_key : str
        Edge attribute key whose boolean values indicate which edges are labeled.
    label_key : str
        Edge attribute key holding the label values (True = correct edge).
    tiling_scheme : TilingScheme | None
        Spatial tiling scheme; if None, uses temporal windowing.
    window_size : int
        Temporal window size when not tiling.
    test_time_augs : int
        Number of test-time augmentations.
    n_steps : int
        Maximum number of L-BFGS iterations (full-batch).
    lr : float
        L-BFGS learning rate.
    l2_weight : float
        L2 regularization weight on head.weight (not bias). Effective weight is
        scaled as ``l2_weight x n_features / n_labels``. Default: 1.0.
    consistency_weight : float
        Weight on the ILP-consistency loss for unlabeled edges. Effective weight
        is scaled as ``consistency_weight x n_features / n_labels``. Default: 0.25.

    Returns
    -------
    ProbedModel
        The backbone with a fitted linear probe as its classification head.
    """
    backbone = model._edge_model if isinstance(model, ProbedModel) else model
    device = next(backbone.parameters()).device

    dataset = _create_dataset(graph, tiling_scheme, window_size, test_time_augs)
    labeled_dataset = LabeledDataset(dataset, label_mask_key)

    # No dedup: same edge in multiple windows → multiple feature rows (windowing augmentation).
    all_features_df = extract_edge_features(backbone, labeled_dataset, edge_filter_key=None)

    # Only include the ILP solution column when it exists on the graph; otherwise the
    # consistency loss is skipped (e.g. before any ILP solve has populated it).
    has_solution = td.DEFAULT_ATTR_KEYS.SOLUTION in graph.edge_attr_keys()
    attr_keys = [td.DEFAULT_ATTR_KEYS.EDGE_ID, label_mask_key, label_key]
    if has_solution:
        attr_keys.append(td.DEFAULT_ATTR_KEYS.SOLUTION)

    all_edge_attrs = graph.edge_attrs(attr_keys=attr_keys)
    labeled_attrs = all_edge_attrs.filter(pl.col(label_mask_key)).join(
        all_features_df, on=td.DEFAULT_ATTR_KEYS.EDGE_ID, how="inner"
    )
    unlabeled_attrs = all_edge_attrs.filter(~pl.col(label_mask_key)).join(
        all_features_df, on=td.DEFAULT_ATTR_KEYS.EDGE_ID, how="inner"
    )

    X = labeled_attrs["edge_features"].to_torch().float().to(device)
    y = labeled_attrs[label_key].to_torch().float().to(device)

    # Consistency target: ILP solution (0/1) on unlabeled edges.
    X_unlab: torch.Tensor | None = None
    y_unlab: torch.Tensor | None = None
    if consistency_weight > 0 and has_solution and len(unlabeled_attrs) > 0:
        X_unlab = unlabeled_attrs["edge_features"].to_torch().float().to(device)
        y_unlab = unlabeled_attrs[td.DEFAULT_ATTR_KEYS.SOLUTION].to_torch().float().to(device)
    elif consistency_weight > 0 and not has_solution:
        LOG.warning(
            "Consistency loss requested but graph has no '%s' attribute; skipping it.",
            td.DEFAULT_ATTR_KEYS.SOLUTION,
        )

    n_features = X.shape[1]
    head = nn.Linear(n_features, 1)
    # Always init from backbone head — never warm-start from previous probe.
    backbone_params = dict(backbone.named_parameters())
    w = backbone_params.get("head.weight")
    b = backbone_params.get("head.bias")
    if w is not None and w.shape == torch.Size([1, n_features]) and b is not None:
        head.weight.data.copy_(w.detach().cpu())
        head.bias.data.copy_(b.detach().cpu())
    else:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        LOG.warning("Backbone 'head' not found or shape mismatch — using zero initialization")
    head = head.to(device)

    n_pos = y.sum().clamp(min=1)
    n_neg = (y.numel() - y.sum()).clamp(min=1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=n_neg / n_pos)
    optimizer = torch.optim.LBFGS(head.parameters(), lr=lr, max_iter=n_steps)

    # Adaptive scaling: both terms weaken as more labels arrive.
    n_labeled = int(graph.edge_attrs(attr_keys=[label_mask_key]).filter(pl.col(label_mask_key)).shape[0])
    scale = n_features / max(n_labeled, 1)
    effective_l2 = l2_weight * scale
    effective_consistency = consistency_weight * scale

    head.train()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(head(X).squeeze(1), y)
        if effective_l2 > 0:
            loss = loss + effective_l2 * (head.weight**2).sum()
        if effective_consistency > 0 and X_unlab is not None and y_unlab is not None:
            _eps = 1e-6
            loss = loss + effective_consistency * nn.functional.binary_cross_entropy_with_logits(
                head(X_unlab).squeeze(1), y_unlab.clamp(_eps, 1.0 - _eps)
            )
        loss.backward()
        return loss

    LOG.info("Fitting probe: %d labels, %d features, %d steps", len(y), n_features, n_steps)
    optimizer.step(closure)
    head.eval()

    probed_model = ProbedModel(backbone, head.weight.data.cpu().numpy()[0], float(head.bias.data.cpu().item()))
    probed_model.to(device)
    return probed_model
