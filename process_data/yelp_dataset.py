r"""Yelp Dataset Processor for SeqRecBenchmark.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import ast
import os

import pandas as pd

from process_data.base_dataset import BaseDatasetProcessor

__all__ = [
    "YelpDatasetProcessor",
]


def read_json(file_path: str, selected_cols: list[str] | None = None) -> pd.DataFrame:
    r"""Read a JSON file and return a DataFrame. Note that only the columns
    specified in ``selected_cols`` will be selected. If ``selected_cols`` is
    ``None``, all columns will be selected. The rows with blank values in the
    selected columns will be dropped.

    Args:
        file_path (str):
            The path of the JSON file.
        selected_cols (list[str], optional, default=None):
            The columns to select from the JSON file. If ``None``, all
            columns will be selected. The default value is ``None``.
            Note that if the selected columns remain blank in a row, this
            row will be dropped.

    Returns:
        The DataFrame containing the data from the JSON file.
    """
    df = pd.read_json(file_path, lines=True)
    if selected_cols is not None:
        df = df[selected_cols]
    df = df.dropna()
    return df


class YelpDatasetProcessor(BaseDatasetProcessor):
    r"""Processor for the Yelp dataset (2018 & 2022 versions).

    Each Yelp dataset contains five files:
    - ``yelp_academic_dataset_business.json``
    - ``yelp_academic_dataset_checkin.json``
    - ``yelp_academic_dataset_review.json``
    - ``yelp_academic_dataset_tip.json``
    - ``yelp_academic_dataset_user.json``

    Here we only use two files:
    - ``yelp_academic_dataset_review.json``: contains user-item interactions.
        Each json object in this file represents a review, where the
        ``user_id`` (22 character unique user id, renamed to ``UserID``) and
        ``business_id`` (22 character unique item id, renamed to ``ItemID``)
        are the user and item ids, respectively. The ``date`` field, e.g.,
        `2016-03-09`, is also used as the timestamp (renamed to ``Timestamp``).
    - ``yelp_academic_dataset_business.json``: contains business metadata.
        Each json object in this file represents a business, where the
        ``business_id`` (22 character unique item id, renamed to ``ItemID``)
        is the item id. The ``name`` field is used as the item title (renamed
        to ``Title``).
    """

    def __init__(
        self,
        dataset_dir: str,
        meta_available: bool = False,
        k_core: int = 5,
        sample_user_size: int = None,
    ) -> None:
        r"""Initialize the Yelp dataset processor.

        Args:
            dataset_dir (str):
                The directory of the dataset, e.g., ``/path/to/yelp2018``.
                The last directory name should be the dataset name, e.g.,
                ``yelp2018``.
            meta_available (bool, optional, default=False):
                Whether the meta data is available. If ``True``, we will save the
                item titles in the ``dataset_dir/proc/item2title.pkl`` file, which
                is a Pandas DataFrame object containing the following columns:
                - ``ItemID``: numeric ID of the item.
                - ``Title``: title of the item.
                The default value is ``False``.
            k_core (int, optional, default=5):
                The K-core value for filtering out the non-hot users and items.
                The default value is 5, which means that we will keep the users
                and items that have at least 5 interactions.
            sample_user_size (int, optional, default=None):
                The number of users to sample from the dataset. If ``None``, we
                will use all the users in the dataset. This argument is used to
                reduce the size of the dataset for LLM-based recommendation. The
                default value is ``None``.
        """
        super().__init__(dataset_dir, meta_available, k_core, sample_user_size)

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        r"""Load the raw data from the raw data files. The following data
        are required to be loaded:
        - ``interactions`` (pd.DataFrame):
            The user-item interaction data, which is a Pandas DataFrame object.
            Each row represents an interaction ``(UserID, ItemID, Timestamp)``,
            with types ``(int | str, int | str, int | str)``.
        - ``item2title`` (pd.DataFrame | None):
            The item titles. If ``meta_available`` is ``False``, ``item2title``
            will be ``None``. If ``meta_available`` is ``True``, it will be a
            Pandas DataFrame object, where each row represents a mapping
            ``(ItemID, Title)``, with types ``(int | str, str)``.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]:
                The first element is the user-item interaction data, and the
                second element is the item titles. If ``meta_available`` is
                ``False``, the second element will be ``None``.
        """
        raw_dir = os.path.join(self.dataset_dir, "raw")
        interactions = read_json(
            os.path.join(raw_dir, "yelp_academic_dataset_review.json"),
            selected_cols=["user_id", "business_id", "date"],
        )
        interactions.columns = ["UserID", "ItemID", "Timestamp"]
        interactions["Timestamp"] = pd.to_datetime(
            interactions["Timestamp"], format="%Y-%m-%d"
        )
        interactions["Timestamp"] = interactions["Timestamp"].astype("int64") // 10**9
        if self.meta_available:
            item2title = read_json(
                os.path.join(raw_dir, "yelp_academic_dataset_business.json"),
                selected_cols=["business_id", "name"],
            )
            item2title.columns = ["ItemID", "Title"]
            item2title = item2title[
                item2title["Title"].notna() & item2title["Title"] != "nan"
            ]
        return interactions, item2title
