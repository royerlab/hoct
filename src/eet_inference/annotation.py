import logging

import numpy as np
import polars as pl
import torch
import tracksdata as td
from torch import nn
from tracksdata.functional import TilingScheme

from eet_inference._api import _create_dataset
from eet_inference._logging import LOG
from eet_inference.data import AnnotatedDataset
from eet_inference.inference._predict import EdgeModel, ModelPrediction, model_feature_predict


def add_edge_label(
    graph: td.graph.BaseGraph,
    source_id: int,
    target_id: int,
    attr_key: str,
    value: bool,
) -> None:
    """
    Adds a label to an edge in the graph.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to add the label to.
    source_id : int
        The source node id.
    target_id : int
        The target node id.
    attr_key : str
        The attribute key to add the label to.
    value : bool
        The value of the label.
    """

    LOG.debug("Adding label '%s' = %s to edge %d -> %d", attr_key, value, source_id, target_id)

    if value:
        edges = graph.filter(td.EdgeAttr(td.DEFAULT_ATTR_KEYS.EDGE_TARGET) == target_id).edge_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_ID, td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]
        )
        edges = edges.with_columns((pl.col(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE) == source_id).alias(attr_key))
        graph.update_edge_attrs(
            edge_ids=edges[td.DEFAULT_ATTR_KEYS.EDGE_ID].to_list(), attrs={attr_key: edges[attr_key].to_list()}
        )

        if LOG.isEnabledFor(logging.DEBUG):
            LOG.debug("Edges: %s", edges.to_dicts())

    else:
        edge_id = graph.edge_id(source_id, target_id)
        graph.update_edge_attrs(edge_ids=[edge_id], attrs={attr_key: False})


class WrappedModel(EdgeModel):
    def __init__(self, edge_model: EdgeModel, coeffs: np.ndarray, bias: float) -> None:
        super().__init__()
        self._edge_model = edge_model
        self._head = nn.Linear(len(coeffs), 1)
        self._head.weight.data = torch.from_numpy(coeffs)
        self._head.bias.data = torch.tensor(bias)

    def forward(self, *args, **kwargs) -> ModelPrediction:
        prediction = self._edge_model(*args, **kwargs)
        new_prediction = self._head(prediction.edge_features)
        new_prediction = ModelPrediction(
            edge_logits=new_prediction,
            node_features=prediction.node_features,
            edge_features=prediction.edge_features,
            orphan_logits=prediction.orphan_logits,
        )
        return new_prediction


def fit_from_sparse_labels(
    graph: td.graph.BaseGraph,
    model: EdgeModel,
    annotated_key: str,
    label_key: str,
    tiling_scheme: TilingScheme | None = None,
    window_size: int = 5,
    test_time_augs: int = 0,
    logistic_kwargs: dict[str, float] | None = None,
) -> WrappedModel:
    """
    Fits a model from annotated edges.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to fit the model from.
    model : EdgeModel
        The model to fit.
    annotated_key : str
        Key indicating if the edge is annotated.
    label_key : str
        Key indicating the label of the edge.
    tiling_scheme : TilingScheme | None
        The tiling scheme to use for the dataset.
    window_size : int
        The window size to use for the dataset.
    test_time_augs : int
        The number of test time augmentations to use.
    logistic_kwargs : dict[str, float] | None
        The keyword arguments to pass to the logistic regression model.

    Returns
    -------
    WrappedModel
        The wrapped model that can be used to predict edge labels.
    """
    from sklearn.linear_model import LogisticRegression

    device = next(model.parameters()).device
    dataset = _create_dataset(graph, tiling_scheme, window_size, test_time_augs)
    annotated_dataset = AnnotatedDataset(dataset, annotated_key, label_key)
    edge_features = model_feature_predict(model, annotated_dataset, edge_filter_key=annotated_key)
    edge_attrs = graph.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_ID, annotated_key, label_key]).filter(
        pl.col(annotated_key)
    )

    edge_attrs = edge_attrs.join(edge_features, on=td.DEFAULT_ATTR_KEYS.EDGE_ID, how="inner")

    X = edge_attrs["edge_features"].to_numpy()
    y = edge_attrs[label_key].to_numpy()

    # fit a logistic regression model
    LOG.info("Fitting logistic regression model")
    if logistic_kwargs is None:
        logistic_kwargs = {}

    LOG.info("Fitting logistic regression model with kwargs: %s", logistic_kwargs)

    linear_model = LogisticRegression(**logistic_kwargs)
    linear_model.fit(X, y)
    LOG.info("Logistic regression model fitted")

    wrapped_model = WrappedModel(model, linear_model.coef_, linear_model.intercept_)
    wrapped_model.to(device)

    return wrapped_model
