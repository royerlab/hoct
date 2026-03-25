from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from eet_inference._logging import LOG

if TYPE_CHECKING:
    from eet_inference.data import DataItem, FrameDataset, GraphConcatDataset, TiledRoiDataset


class LabeledDataset(Dataset):
    """
    Dataset that filters to items containing at least one labeled edge.

    Parameters
    ----------
    dataset : FrameDataset | TiledRoiDataset | GraphConcatDataset
        The base dataset to wrap.
    label_mask_key : str
        Edge attribute key whose boolean values indicate which edges are labeled.
    label_key : str
        Edge attribute key holding the actual label values.
    """

    def __init__(
        self,
        dataset: "FrameDataset | TiledRoiDataset | GraphConcatDataset",
        label_mask_key: str,
        label_key: str,
    ):
        super().__init__()
        self._dataset = dataset
        self._label_mask_key = label_mask_key
        self._label_key = label_key

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> "DataItem | None":
        data = self._dataset.__getitem__(index, extra_edge_attrs=(self._label_mask_key,))
        if not data[self._label_mask_key].any():
            LOG.debug("Item %d has no labeled edges for key %s", index, self._label_mask_key)
            return None
        LOG.debug("Item %d has labeled edges for key %s", index, self._label_mask_key)
        return data
