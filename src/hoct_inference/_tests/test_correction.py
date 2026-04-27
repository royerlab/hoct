"""Tests for hoct_inference.correction module."""

import numpy as np
import polars as pl
import pytest
import torch
import tracksdata as td
from hoct_features.constants import REGIONPROPS
from hoct_features.graph import create_graph
from torch import nn

from hoct_inference.correction import ProbedModel, fit_from_labels, label_edge
from hoct_inference.data import FrameDataset, LabeledDataset
from hoct_inference.inference import EdgeModel, ModelPrediction, model_predict
from hoct_inference.tracking import ILPSolverConfig

_HIDDEN_SIZE = 4


class FakeEdgeModel(EdgeModel):
    """Minimal EdgeModel returning zero tensors of the correct shape."""

    def __init__(self, hidden_size: int = _HIDDEN_SIZE) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self._dummy = nn.Parameter(torch.zeros(1))  # required for .device access

    def forward(
        self,
        input_batch: torch.Tensor,
        node_pos: torch.Tensor,
        edge_pos: torch.Tensor,
        edge_indices: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> ModelPrediction:
        B, N, _ = input_batch.shape
        E = edge_indices.shape[1]
        device = input_batch.device
        return ModelPrediction(
            edge_logits=torch.zeros(B, E, 1, device=device),
            node_features=torch.zeros(B, N, self.hidden_size, device=device),
            edge_features=torch.zeros(B, E, self.hidden_size, device=device),
            orphan_logits=torch.zeros(B, N, 1, device=device),
        )


@pytest.fixture
def synthetic_graph():
    """Two cells tracked over 3 frames, close enough to produce cross-cell candidate edges."""
    labels = np.zeros((3, 64, 64), dtype=np.int32)
    labels[0, 10:20, 10:20] = 1
    labels[0, 40:50, 40:50] = 2
    labels[1, 12:22, 12:22] = 1
    labels[1, 38:48, 38:48] = 2
    labels[2, 14:24, 14:24] = 1
    labels[2, 36:46, 36:46] = 2
    return create_graph(labels=labels, distance_threshold=300.0, n_neighbors=5, delta_t=2)


@pytest.fixture
def fake_model():
    return FakeEdgeModel()


@pytest.fixture
def labeled_graph(synthetic_graph):
    """Graph with a 'is_labeled' mask and 'is_correct' label on a subset of edges."""
    graph = synthetic_graph
    graph.add_edge_attr_key("is_labeled", pl.Boolean, False)
    graph.add_edge_attr_key("is_correct", pl.Boolean, False)

    # Label 3 edges: 2 correct, 1 incorrect — LR needs both classes
    all_edges = graph.edge_attrs(attr_keys=[])
    assert len(all_edges) >= 3, "Synthetic graph too sparse for tests"
    labeled_ids = all_edges[td.DEFAULT_ATTR_KEYS.EDGE_ID][:3].to_list()
    graph.update_edge_attrs(
        edge_ids=labeled_ids,
        attrs={"is_labeled": [True, True, True], "is_correct": [True, True, False]},
    )
    return graph


# ---------------------------------------------------------------------------
# label_edge
# ---------------------------------------------------------------------------


class TestLabelEdge:
    def test_positive_marks_chosen_edge_true_others_false(self, synthetic_graph):
        """Positive label sets the chosen edge True and all sibling edges False."""
        graph = synthetic_graph
        graph.add_edge_attr_key("is_correct", pl.Boolean, False)

        edges = graph.edge_attrs(attr_keys=[])
        target_counts = edges.group_by(td.DEFAULT_ATTR_KEYS.EDGE_TARGET).len()
        multi_targets = target_counts.filter(pl.col("len") > 1)

        if len(multi_targets) == 0:
            pytest.skip("No target with multiple candidate edges in this graph")

        target_id = int(multi_targets[td.DEFAULT_ATTR_KEYS.EDGE_TARGET][0])
        incoming = edges.filter(pl.col(td.DEFAULT_ATTR_KEYS.EDGE_TARGET) == target_id)
        source_id = int(incoming[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE][0])

        label_edge(graph, source_id, target_id, "is_correct", value=True)

        result = graph.edge_attrs(
            attr_keys=[
                td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
                "is_correct",
            ]
        ).filter(pl.col(td.DEFAULT_ATTR_KEYS.EDGE_TARGET) == target_id)

        chosen = result.filter(pl.col(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE) == source_id)
        others = result.filter(pl.col(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE) != source_id)

        assert chosen["is_correct"][0] is True
        assert all(v is False for v in others["is_correct"].to_list())

    def test_negative_marks_only_the_specified_edge_false(self, synthetic_graph):
        """Negative label updates only the specified edge, leaving others untouched."""
        graph = synthetic_graph
        graph.add_edge_attr_key("is_correct", pl.Boolean, True)  # all start True

        edges = graph.edge_attrs(attr_keys=[])
        source_id = int(edges[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE][0])
        target_id = int(edges[td.DEFAULT_ATTR_KEYS.EDGE_TARGET][0])
        edge_id = graph.edge_id(source_id, target_id)

        label_edge(graph, source_id, target_id, "is_correct", value=False)

        result = graph.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_ID, "is_correct"])
        updated = result.filter(pl.col(td.DEFAULT_ATTR_KEYS.EDGE_ID) == edge_id)
        untouched = result.filter(pl.col(td.DEFAULT_ATTR_KEYS.EDGE_ID) != edge_id)

        assert updated["is_correct"][0] is False
        assert all(v is True for v in untouched["is_correct"].to_list())


# ---------------------------------------------------------------------------
# LabeledDataset
# ---------------------------------------------------------------------------


class TestLabeledDataset:
    def test_unlabeled_items_return_none(self, synthetic_graph):
        """All items return None when no edges are labeled."""
        graph = synthetic_graph
        graph.add_edge_attr_key("is_labeled", pl.Boolean, False)
        graph.add_edge_attr_key("is_correct", pl.Boolean, False)

        ds = FrameDataset(graph=graph, min_window_size=2, properties=REGIONPROPS)
        labeled_ds = LabeledDataset(ds, "is_labeled")

        assert all(labeled_ds[i] is None for i in range(len(labeled_ds)))

    def test_labeled_items_include_mask_key(self, labeled_graph):
        """Items with at least one labeled edge return a DataItem containing the mask key."""
        ds = FrameDataset(graph=labeled_graph, min_window_size=2, properties=REGIONPROPS)
        labeled_ds = LabeledDataset(ds, "is_labeled")

        non_none = [labeled_ds[i] for i in range(len(labeled_ds)) if labeled_ds[i] is not None]
        assert len(non_none) > 0, "Expected at least one window with labeled edges"
        for item in non_none:
            assert "is_labeled" in item

    def test_len_matches_base_dataset(self, labeled_graph):
        ds = FrameDataset(graph=labeled_graph, min_window_size=2, properties=REGIONPROPS)
        labeled_ds = LabeledDataset(ds, "is_labeled")
        assert len(labeled_ds) == len(ds)


# ---------------------------------------------------------------------------
# ProbedModel
# ---------------------------------------------------------------------------


class TestProbedModel:
    @pytest.fixture
    def probed(self, fake_model):
        coeffs = np.ones(_HIDDEN_SIZE, dtype=np.float32)
        return ProbedModel(fake_model, coeffs, bias=0.5)

    def _forward_args(self, B=1, N=4, E=6, C=7):
        return (
            torch.zeros(B, N, C),
            torch.zeros(B, N, 3),
            torch.zeros(B, E, 3),
            torch.zeros(B, E, 2, dtype=torch.long),
            torch.ones(B, N, dtype=torch.bool),
            torch.ones(B, E, dtype=torch.bool),
        )

    def test_forward_returns_model_prediction(self, probed):
        out = probed.forward(*self._forward_args())
        assert isinstance(out, ModelPrediction)

    def test_edge_logits_shape(self, probed):
        B, N, E, C = 1, 4, 6, 7
        out = probed.forward(*self._forward_args(B, N, E, C))
        assert out.edge_logits.shape == (B, E, 1)

    def test_probe_replaces_logits_preserves_rest(self, fake_model, probed):
        """ProbedModel changes edge_logits via the head; other outputs come straight from backbone."""
        args = self._forward_args()
        backbone_out = fake_model.forward(*args)
        probed_out = probed.forward(*args)

        # logits come from the linear head (non-zero bias → different from backbone zeros)
        assert not torch.equal(probed_out.edge_logits, backbone_out.edge_logits)
        assert torch.equal(probed_out.node_features, backbone_out.node_features)
        assert torch.equal(probed_out.edge_features, backbone_out.edge_features)
        assert torch.equal(probed_out.orphan_logits, backbone_out.orphan_logits)

    def test_head_is_registered_submodule(self, probed):
        """super().__init__() is called so nn.Linear is tracked in parameters()."""
        param_names = [name for name, _ in probed.named_parameters()]
        assert any("_head" in name for name in param_names)

    def test_head_input_size_matches_hidden(self, probed):
        assert probed._head.in_features == _HIDDEN_SIZE


# ---------------------------------------------------------------------------
# fit_from_labels
# ---------------------------------------------------------------------------


class TestFitFromLabels:
    def test_returns_probed_model(self, labeled_graph, fake_model):
        result = fit_from_labels(
            graph=labeled_graph,
            model=fake_model,
            label_mask_key="is_labeled",
            label_key="is_correct",
        )
        assert isinstance(result, ProbedModel)
        assert result._head.in_features == _HIDDEN_SIZE
        assert result._head.out_features == 1
        model_device = next(fake_model.parameters()).device
        head_device = next(result._head.parameters()).device
        assert head_device == model_device


# ---------------------------------------------------------------------------
# Integration: fit ProbedModel → run standard prediction pipeline
# ---------------------------------------------------------------------------


class TestFitAndPredict:
    def test_probed_model_runs_full_prediction_pipeline(self, labeled_graph, fake_model):
        """Fit a ProbedModel from sparse corrections, then run it through model_predict."""
        probed = fit_from_labels(
            graph=labeled_graph,
            model=fake_model,
            label_mask_key="is_labeled",
            label_key="is_correct",
        )

        config = ILPSolverConfig(
            appearance_weight=1.0,
            disappearance_weight=1.0,
            division_weight=1.0,
            node_weight=0.0,
            delta_t_weight=0.0,
            edge_bias=0.5,
            timeout=10.0,
            tracklet_solver=False,
        )
        ds = FrameDataset(graph=labeled_graph, min_window_size=2, properties=REGIONPROPS)
        solution = model_predict(probed, ds, solver_config=config)

        assert isinstance(solution, td.graph.BaseGraph)
        assert td.DEFAULT_ATTR_KEYS.SOLUTION in solution.node_attr_keys()
        assert td.DEFAULT_ATTR_KEYS.SOLUTION in solution.edge_attr_keys()
