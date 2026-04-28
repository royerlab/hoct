"""Tests for hoct_inference.inference._predict helpers."""

import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from hoct_inference.inference._predict import _make_iterator


class _FakeIterableDataset(IterableDataset):
    """Pure IterableDataset (no __len__ / __getitem__) — mirrors TiledRoiDataset / _GraphChainDataset."""

    def __init__(self, items: list[dict]):
        super().__init__()
        self._items = items

    def __iter__(self):
        yield from self._items


class TestMakeIterator:
    """Regression tests for ``_make_iterator``.

    Previously ``_make_iterator`` only handled map-style datasets and ``DataLoader``
    via ``len(ds)`` + ``ds[i]``, so any pure ``IterableDataset`` (e.g. ``TiledRoiDataset``
    or ``GraphConcatDataset`` of iterable datasets) raised ``TypeError`` /
    ``NotImplementedError`` when used with ``extract_edge_features``.
    """

    def test_iterable_dataset_is_iterated_with_unsqueeze(self):
        items = [{"x": torch.tensor([1.0, 2.0])}, {"x": torch.tensor([3.0, 4.0])}]
        ds = _FakeIterableDataset(items)

        ds_iter, expand = _make_iterator(ds, device=torch.device("cpu"))
        out = list(ds_iter())

        assert len(out) == 2
        # _expand_dims must add a leading batch dim, like for map-style Datasets.
        first = expand(out[0]["x"])
        assert first.shape == (1, 2)
        assert torch.equal(first, items[0]["x"].unsqueeze(0))

    def test_map_style_dataset_uses_indexing(self):
        x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        ds = TensorDataset(x)

        ds_iter, expand = _make_iterator(ds, device=torch.device("cpu"))
        out = list(ds_iter())

        assert len(out) == 3
        assert expand(out[0][0]).shape == (1, 2)

    def test_dataloader_is_passed_through_without_unsqueeze(self):
        x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        loader = DataLoader(TensorDataset(x), batch_size=2)

        ds_iter, expand = _make_iterator(loader, device=torch.device("cpu"))
        out = list(ds_iter())

        # DataLoader already provides a batch dim — _expand_dims must not add one.
        assert out[0][0].shape == (2, 2)
        assert expand(out[0][0]).shape == (2, 2)
