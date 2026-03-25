import logging

import numpy as np
import polars as pl
import torch
import tracksdata as td
from torch import nn
from tracksdata.functional import TilingScheme

from eet_inference._api import _create_dataset
from eet_inference._logging import LOG
from eet_inference.data import LabeledDataset
from eet_inference.inference._predict import EdgeModel, ModelPrediction, extract_edge_features


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
        edges = graph.filter(td.EdgeAttr(td.DEFAULT_ATTR_KEYS.EDGE_TARGET) == target_id).edge_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_ID, td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]
        )
        edges = edges.with_columns((pl.col(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE) == source_id).alias(attr_key))
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
        Logistic regression coefficients of shape (1, hidden_size).
    bias : float
        Logistic regression intercept.
    """

    def __init__(self, edge_model: EdgeModel, coeffs: np.ndarray, bias: float) -> None:
        super().__init__()
        self._edge_model = edge_model
        self._head = nn.Linear(len(coeffs), 1)
        self._head.weight.data = torch.from_numpy(coeffs)
        self._head.bias.data = torch.tensor(bias)

    def forward(self, *args, **kwargs) -> ModelPrediction:
        prediction = self._edge_model(*args, **kwargs)
        new_logits = self._head(prediction.edge_features)
        return ModelPrediction(
            edge_logits=new_logits,
            node_features=prediction.node_features,
            edge_features=prediction.edge_features,
            orphan_logits=prediction.orphan_logits,
        )


def fit_from_labels(
    graph: td.graph.BaseGraph,
    model: EdgeModel,
    label_mask_key: str,
    label_key: str,
    tiling_scheme: TilingScheme | None = None,
    window_size: int = 5,
    test_time_augs: int = 0,
    logistic_kwargs: dict[str, float] | None = None,
) -> ProbedModel:
    """
    Fit a linear probe from sparse edge corrections and return an adapted model.

    Extracts backbone features for all labeled edges, fits a logistic regression
    head on those features, and returns a ProbedModel wrapping the original backbone.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph containing labeled edges.
    model : EdgeModel
        The pretrained backbone model to probe.
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
    logistic_kwargs : dict[str, float] | None
        Extra keyword arguments forwarded to LogisticRegression.

    Returns
    -------
    ProbedModel
        The backbone with a fitted linear probe as its classification head.
    """
    from sklearn.linear_model import LogisticRegression

    device = next(model.parameters()).device
    dataset = _create_dataset(graph, tiling_scheme, window_size, test_time_augs)
    labeled_dataset = LabeledDataset(dataset, label_mask_key, label_key)
    edge_features = extract_edge_features(model, labeled_dataset, edge_filter_key=label_mask_key)
    edge_attrs = graph.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_ID, label_mask_key, label_key]).filter(
        pl.col(label_mask_key)
    )

    edge_attrs = edge_attrs.join(edge_features, on=td.DEFAULT_ATTR_KEYS.EDGE_ID, how="inner")

    X = edge_attrs["edge_features"].to_numpy()
    y = edge_attrs[label_key].to_numpy()

    if logistic_kwargs is None:
        logistic_kwargs = {}

    LOG.info("Fitting logistic regression probe with kwargs: %s", logistic_kwargs)
    linear_model = LogisticRegression(**logistic_kwargs)
    linear_model.fit(X, y)
    LOG.info("Logistic regression probe fitted")

    probed_model = ProbedModel(model, linear_model.coef_, linear_model.intercept_)
    probed_model.to(device)

    return probed_model
