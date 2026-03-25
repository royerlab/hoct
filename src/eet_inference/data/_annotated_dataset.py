from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from eet_inference._logging import LOG

if TYPE_CHECKING:
    from eet_inference.data import DataItem, FrameDataset, GraphConcatDataset, TiledRoiDataset


class AnnotatedDataset(Dataset):
    """
    Dataset that returns only items with a non-empty annotated key and additional edge attributes.

    Parameters
    ----------
    dataset : FrameDataset | TiledRoiDataset | GraphConcatDataset
        The dataset to annotate.
    annotated_key : str
        Key indicating if the edge is annotated.
    label_key : str
        Key indicating the label of the edge.
    """

    def __init__(
        self,
        dataset: "FrameDataset | TiledRoiDataset | GraphConcatDataset",
        annotated_key: str,
        label_key: str,
    ):
        super().__init__()
        self._dataset = dataset
        self._annotated_key = annotated_key
        self._label_key = label_key

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> "DataItem | None":
        data = self._dataset.__getitem__(index, extra_edge_attrs=(self._annotated_key,))
        if not data[self._annotated_key].any():
            LOG.debug("Item %d has no non-empty annotated key %s", index, self._annotated_key)
            return None
        LOG.debug("Item %d has non-empty annotated key %s", index, self._annotated_key)
        return data
