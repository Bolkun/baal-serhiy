import warnings
from copy import deepcopy
from itertools import zip_longest
from typing import Union, Optional, Callable, Any, Dict, List

import numpy as np

from torch.utils.data import Dataset


from baal.active import ActiveLearningDataset


def _identity(x):
    return x


class ExtendedActiveLearningDataset(ActiveLearningDataset):
    """A dataset that allows for active learning and working with augmentations.
    Args:
        dataset: The baseline dataset.
        labelled: An array that acts as a mask which is greater than 1 for every
            data point that is labelled, and 0 for every data point that is not
            labelled.
        make_unlabelled: The function that returns an
            unlabelled version of a datum so that it can still be used in the DataLoader.
        random_state: Set the random seed for label_randomly().
        pool_specifics: Attributes to set when creating the pool.
                                         Useful to remove data augmentation.
        last_active_steps: If specified, will iterate over the last active steps
                            instead of the full dataset. Useful when doing partial finetuning.
        augmented_map: A map that indicates which data points are augmented. 0 is for original unaugmented images. Non Zero will reference the pool index of the original image
    Notes:
        n_augmented_images_labelled: This value describes how often an augmented image was recommended by the active learning algorithm to be labelled.
                                     If you label an image with the standard label function it will also label the original label but only increase the count for unlabelled.
    """

    def __init__(
        self,
        dataset: Dataset,
        labelled: Optional[np.ndarray] = None,
        make_unlabelled: Callable = _identity,
        random_state=None,
        pool_specifics: Optional[dict] = None,
        last_active_steps: int = -1,
        augmented_map: Optional[np.ndarray] = None,
    ) -> None:
        if augmented_map is None:
            self.augmented_map = np.zeros(len(dataset), dtype=int)
        self.unaugmented_pool_size = len(dataset)
        self.n_augmented_images_labelled = 0
        self.n_unaugmented_images_labelled = 0
        self.augmented_n_times = 0
        super().__init__(
            dataset=dataset,
            labelled=labelled,
            make_unlabelled=make_unlabelled,
            random_state=random_state,
            pool_specifics=pool_specifics,
            last_active_steps=last_active_steps,
        )

    def can_augment(self) -> bool:
        return True

    @property
    def n_unaugmented(self):
        """The number of unaugmented data points."""
        return (~self.augmented).sum()

    @property
    def n_augmented(self):
        """The number of augmented data points."""
        return self.augmented.sum()

    @property
    def augmented(self):
        """An array that acts as a boolean mask which is True for every
        data point that is labelled, and False for every data point that is not
        labelled."""
        orig_len = self.unaugmented_pool_size
        print("orig len" + str(orig_len))
        print("augmented n times" + str(self.augmented_n_times))
        return np.concatenate(
            (
                np.zeros(orig_len),
                np.ones(orig_len * self.augmented_n_times),
            )
        ).astype(bool)

    def augment_n_times(self, n, augmented_dataset=None) -> None:
        """Augment the every image in the dataset n times and append those augmented images to the end of the dataset
        Currently only works if an augmented version of the dataset is already present and n=1"""
        labelled_map_copy = deepcopy(self.labelled_map)
        augmented_map_extender = np.arange(len(self.augmented_map))
        if self.n_augmented != 0:
            raise ValueError("The dataset has already been augmented.")
        self.augmented_n_times = n
        if augmented_dataset == None:
            dataset_copy = deepcopy(self._dataset)
        else:
            dataset_copy = augmented_dataset
        while n > 0:
            # print("type before"+str(type(self._dataset)))
            self.labelled_map = np.concatenate((self.labelled_map, labelled_map_copy))
            self.augmented_map = np.concatenate(
                (self.augmented_map, augmented_map_extender)
            )
            self._dataset = self._dataset.__add__(dataset_copy)
            # print(len(dataset_copy))
            n -= 1

    def get_augmented_ids_of_image(self, idx):
        if self.is_augmentation(idx):
            raise ValueError(
                "The idx given responds to an augmented image, please specify an id that responds to an unaugmented image!"
            )
        augmented_ids = np.where(self.augmented_map == idx)
        # print(type(augmented_ids))
        return augmented_ids

    def is_augmentation(self, idx) -> bool:
        """Check if the idx is an augmentation.
        NOTE: this function is currently bugged, in it's current way it is supposed to work with list that contain only a single element, the function will need to be rewritten for multiple elements
        """
        # print("idx"+str(idx)+" augMapVal:"+str(self.augmented_map[idx])+" "+"augMapType:"+str(type(self.augmented_map[idx])))
        if not (isinstance(idx, int) or isinstance(idx, np.int64) or isinstance(idx, np.int32)):
            # print(type(idx))
            # We were provided only the index, we make a list.
            idx = int(idx[0])
        if self.augmented_map[idx] == 0:
            return False
        else:
            return True

    def label(self, idx):
        """
        Overriding the label function of ActiveLearningDataset.
        Use this function if you want to automatically label all augmentations and the original image.
        Use label after the dataset has been augmented
        Args:
            index: one or many indices to label, relative to the pool index and not the dataset index.
        Raises:
            ValueError if the indices do not match the values or
             if no `value` is provided and `can_label` is True.
        """
        oracle_id_list = self._pool_to_oracle_index(idx)
        for oracle_idx in oracle_id_list:
            if self.is_augmentation(oracle_idx):
                self.n_augmented_images_labelled += 1
                oracle_idx = self.augmented_map[oracle_idx]
            else:
                self.n_unaugmented_images_labelled += 1
            augmented_ids = self.get_augmented_ids_of_image(oracle_idx)

            for id in augmented_ids:
                if len(id) == 0:
                    break
                super().label(self._oracle_to_pool_index(id))
            super().label(self._oracle_to_pool_index(int(oracle_idx)))

    def label2(self, index, value: Optional[Any] = None):
        if isinstance(index, int):
            # We were provided only the index, we make a list.
            index_lst = [index]
            value_lst: List[Any] = [value]
        else:
            index_lst = index
            if value is None:
                value_lst = [value]
            else:
                value_lst = value

        if value_lst[0] is not None and len(index_lst) != len(value_lst):
            raise ValueError(
                "Expected `index` and `value` to be of same length when `value` is provided."
                f"Got index={len(index_lst)} and value={len(value_lst)}"
            )
        indexes = self._pool_to_oracle_index(index_lst)
        active_step = self.current_al_step + 1
        for idx, val in zip_longest(indexes, value_lst, fillvalue=None):
            if self.can_label and val is not None:
                self._dataset.label2(idx, val)  # type: ignore
                self.labelled_map[idx] = active_step
            elif self.can_label and val is None:
                raise ValueError(
                    """The dataset is able to label data, but no label was provided.
                                 If this is a research setting, please set the
                                  `ActiveLearningDataset.can_label` to `False`.
                                  """
                )
            else:
                # Regular research usecase.
                self.labelled_map[idx] = active_step
                if val is not None:
                    warnings.warn(
                        "We will consider the original label of this datasample : {}, {}.".format(
                            self._dataset[idx][0], self._dataset[idx][1]
                        ),
                        UserWarning,
                    )

    def label_just_this_id(self, idx):
        """
        Can be used to label just the id provided and not the augmentations.
        It is recommended to not mix using this label function with the normal label function.
        """
        if self.is_augmentation(idx):
            self.n_augmented_images_labelled += 1
        else:
            self.n_unaugmented_images_labelled += 1
        super().label(idx)