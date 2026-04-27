from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset, IterableDataset

from hoct_inference._logging import LOG

if TYPE_CHECKING:
    from hoct_inference.data import DataItem, FrameDataset, GraphConcatDataset, TiledRoiDataset


class LabeledDataset(IterableDataset, Dataset):
    """
    Dataset that filters to items containing at least one labeled edge.

    Works with both map-style (``FrameDataset``) and iterable-style
    (``TiledRoiDataset``, chained ``GraphConcatDataset``) bases:

    * If the wrapped dataset has ``__getitem__`` and ``__len__``, ``LabeledDataset``
      exposes the same map-style protocol (forwarding ``extra_edge_attrs``).
    * It also always supports iteration via ``__iter__`` / ``iter_items``, which
      delegates to ``dataset.iter_items(extra_edge_attrs=(label_mask_key,))``
      when available.

    Items where no edge has the label mask attribute set to True are skipped:
    iteration drops them, and ``__getitem__`` returns ``None``.

    Parameters
    ----------
    dataset : FrameDataset | TiledRoiDataset | GraphConcatDataset
        The base dataset to wrap.
    label_mask_key : str
        Edge attribute key whose boolean values indicate which edges are labeled.
    """

    def __init__(
        self,
        dataset: "FrameDataset | TiledRoiDataset | GraphConcatDataset",
        label_mask_key: str,
    ):
        super().__init__()
        self._dataset = dataset
        self._label_mask_key = label_mask_key

    # ----- map-style protocol (when supported by the underlying dataset) -----

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> "DataItem | None":
        data = self._dataset.__getitem__(index, extra_edge_attrs=(self._label_mask_key,))
        if not data[self._label_mask_key].any():
            LOG.debug("Item %d has no labeled edges for key %s", index, self._label_mask_key)
            return None
        LOG.debug("Item %d has labeled edges for key %s", index, self._label_mask_key)
        return data

    # ----- iterable protocol (always available) -----

    def iter_items(self, **kwargs: Any):
        """Yield only items containing at least one labeled edge."""
        extra = tuple(kwargs.pop("extra_edge_attrs", ()))
        if self._label_mask_key not in extra:
            extra = (*extra, self._label_mask_key)

        if hasattr(self._dataset, "iter_items"):
            source = self._dataset.iter_items(extra_edge_attrs=extra, **kwargs)
        else:
            # Fall back to map-style traversal.
            source = (
                self._dataset.__getitem__(i, extra_edge_attrs=extra, **kwargs)
                for i in range(len(self._dataset))
            )

        for data in source:
            if data[self._label_mask_key].any():
                yield data

    def __iter__(self):
        yield from self.iter_items()
