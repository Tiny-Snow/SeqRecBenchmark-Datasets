r"""Base processor for SeqRecBenchmark datasets.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import json
import os
import random
from abc import ABC, abstractmethod
from typing import Final

import numpy as np
import pandas as pd
import tiktoken

__all__ = [
    "BaseDatasetProcessor",
]


class BaseDatasetProcessor(ABC):
    r"""Base processor for SeqRecBenchmark datasets.

    Each dataset processor will process one type of dataset. All dataset
    processors require the dataset directory path ``dataset_dir``, e.g.,
    ``path/to/amazon2014-book``, where the last part is the dataset name, e.g.,
    ``amazon2014-book``. The raw dataset files are supposed to be in the
    ``dataset_dir/raw`` directory, and the processed dataset will be saved in
    the ``dataset_dir/proc`` directory.

    Currently, the following dataset types are supported to be processed:
    - **SeqRecDataset**: dataset for sequential recommendation. Each user has a
        sequence of interaction items sorted by time (from the earliest to the
        latest). We assign a numerical ID to each user and item, i.e., the
        ``UserID`` and ``ItemID``. The K-core strategy is used to filter the
        inactive users and items. The processed dataset will be saved in the
        ``dataset_dir/proc/user2item.pkl``, which is a Pandas DataFrame object
        containing the following columns:
        - ``UserID``: numeric ID of the user.
        - ``ItemID``: a list of numeric IDs of the items that the user has
            interacted with, sorted by time (from the earliest to the latest).
        - ``Timestamp``: a list of the corresponding timestamps of the interacted
            items, sorted by time (from the earliest to the latest). The
            timestamps are in the unix time format.
    - **LLMRecDataset**: dataset for LLM-based recommendation. If the meta data
        is available, i.e., ``meta_available`` is ``True``, we will save the
        item tiles in the ``dataset_dir/proc/item2title.pkl`` file, which is
        a Pandas DataFrame object containing the following columns:
        - ``ItemID``: numeric ID of the item.
        - ``Title``: title of the item.

    In addition, we will also save the statistics of the dataset in the
    ``summary.json`` file, including:

    - ``user_size``: the number of users in the dataset.
    - ``item_size``: the number of items in the dataset.
    - ``interactions``: the number of interactions in the dataset.
    - ``density``: the density of the dataset, which is defined as the ratio of
        the number of interactions to the number of users multiplied by the
        number of items.
    - ``interaction_length``: the minimum, maximum, and average length of the
        interaction sequence of users.
    - ``title_token``: the minimum, maximum, and average number of tokens in the
        item title. The tokenization is based on the ``tiktoken`` library for
        ``gpt-4`` model.

    .. note::
        The numeric IDs of users and items start from 100, since the first 100
        IDs are reserved for the special tokens in the model (e.g., <PAD>).

    .. note::
        Currently, we only support the item title as the input of LLM. The
        additional information, e.g., item description and review text, will be
        considered in the future.

    .. note::
        When ``sample_user_size`` is not ``None``, the processed data will
        be saved in the ``dataset_dir_{sample_user_size}/proc`` directory.
    """

    START_ID: Final[int] = 100

    def __init__(
        self,
        dataset_dir: str,
        meta_available: bool = False,
        k_core: int = 5,
        sample_user_size: int | None = None,
    ) -> None:
        r"""Initialize the base dataset processor.

        Args:
            dataset_dir (str):
                The directory of the dataset, e.g., ``/path/to/amazon2014-book``.
                The last directory name should be the dataset name, e.g.,
                ``amazon2014-book``.
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
        self.dataset_dir = dataset_dir
        self.meta_available = meta_available
        self.k_core = k_core
        self.sample_user_size = sample_user_size
        self.dataset_name = os.path.basename(dataset_dir)

    def process(self) -> None:
        r"""Process the dataset and save the processed data. After calling
        this method, the processed data will be saved in the ``dataset_dir/proc``
        directory.
        """
        # load the interactions and item2title raw data
        interactions, item2title = self._load_data()
        # filter the users and items to exclude the invalid item titles
        interactions, item2title = self._filter_item_title(interactions, item2title)
        # drop the duplicate users/items
        interactions, item2title = self._drop_duplicates(interactions, item2title)
        # sample the users if needed
        interactions = self._sample_users(interactions)
        # apply K-core filtering
        interactions, item2title = self._filter_k_core(interactions, item2title)
        # sorting the items by time, and filter the users with unseen test item
        user2item, item2title = self._group_interactions(interactions, item2title)
        # apply the consecutive numeric ID mapping
        user2item, item2title = self._apply_id_mapping(user2item, item2title)
        # save the processed data and statistics
        self._save_processed_data(user2item, item2title)

    @abstractmethod
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
        pass

    def _drop_duplicates(
        self, interactions: pd.DataFrame, item2title: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        r"""Drop the duplicate interactions and items from the dataset. The duplicates
        are defined as the users/items that have the same ID.

        Args:
            interactions (pd.DataFrame):
                The user-item interaction data.
            item2title (pd.DataFrame | None):
                The item titles.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]:
                The first element is the user-item interaction data, and the
                second element is the item titles. If ``meta_available`` is
                ``False``, the second element will be ``None``.
        """
        interactions = interactions.drop_duplicates(
            subset=["UserID", "ItemID", "Timestamp"], keep="first"
        )
        interactions = interactions.reset_index(drop=True)
        if self.meta_available:
            item2title = item2title.drop_duplicates(subset=["ItemID"], keep="first")
            item2title = item2title.reset_index(drop=True)
        return interactions, item2title

    def _sample_users(self, interactions: pd.DataFrame) -> pd.DataFrame:
        r"""Sample users from the dataset. If ``sample_user_size`` is not
        ``None``, we will sample ``sample_user_size`` users randomly.

        Args:
            interactions (pd.DataFrame):
                The user-item interaction data.

        Returns:
            pd.DataFrame:
                The sampled user-item interaction data.
        """
        if self.sample_user_size is None:
            return interactions
        users = interactions["UserID"].unique()
        sample_size = min(self.sample_user_size, len(users))
        sampled_users = random.sample(list(users), sample_size)
        interactions = interactions[interactions["UserID"].isin(sampled_users)]
        interactions = interactions.reset_index(drop=True)
        return interactions

    def _filter_item_title(
        self, interactions: pd.DataFrame, item2title: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        r"""Filter the users and items to exclude the invalid item titles.
        By default, we filter out the items that have empty title (if the
        ``meta_available`` is ``True``). You can override this method to
        customize the filtering strategy.

        Args:
            interactions (pd.DataFrame):
                The user-item interaction data.
            item2title (pd.DataFrame | None):
                The item titles.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]:
                The first element is the filtered user-item interaction data,
                and the second element is the filtered item titles.
        """
        if self.meta_available:
            valid_items = item2title[
                item2title["Title"].notna()
                & item2title["Title"].astype(str).str.strip().astype(bool)
            ]["ItemID"]
            valid_items = np.intersect1d(
                interactions["ItemID"].unique(),
                valid_items.unique(),
                assume_unique=True,
            )
            interactions = interactions[interactions["ItemID"].isin(valid_items)]
            interactions = interactions.reset_index(drop=True)
            item2title = item2title[item2title["ItemID"].isin(valid_items)]
            item2title = item2title.reset_index(drop=True)
        return interactions, item2title

    def _filter_k_core(
        self, interactions: pd.DataFrame, item2title: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        r"""Filter the users and items to exclude the inactive users and items
        based on the K-core strategy, i.e., every user and item should have at
        least ``k_core`` interactions.

        Args:
            interactions (pd.DataFrame):
                The user-item interaction data.
            item2title (pd.DataFrame | None):
                The item titles.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]:
                The first element is the filtered user-item interaction data,
                and the second element is the filtered item titles.
        """
        while True:
            user_cnt = interactions["UserID"].value_counts()
            item_cnt = interactions["ItemID"].value_counts()
            deleted_users = user_cnt[user_cnt < self.k_core].index
            deleted_items = item_cnt[item_cnt < self.k_core].index
            if deleted_users.empty and deleted_items.empty:
                break
            interactions = interactions[
                ~interactions["UserID"].isin(deleted_users)
                & ~interactions["ItemID"].isin(deleted_items)
            ]
        interactions = interactions.reset_index(drop=True)
        if self.meta_available:
            item2title = item2title[item2title["ItemID"].isin(interactions["ItemID"])]
            item2title = item2title.reset_index(drop=True)
        return interactions, item2title

    def _group_interactions(
        self, interactions: pd.DataFrame, item2title: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        r"""Group the interactions by user and sort the items by time.

        Args:
            interactions (pd.DataFrame):
                The user-item interaction data.
            item2title (pd.DataFrame | None):
                The item titles.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]:
                The first element is the grouped user-item interaction data,
                where each row ``(UserID, ItemID, Timestamp)`` represents the
                interactions of a user, ``UserID`` is the user ID, ``ItemID``
                is a list of item IDs, and ``Timestamp`` is a list of the
                corresponding timestamps. The second element is the item
                titles after filtering.
        """
        user2item = (
            interactions.sort_values(by=["UserID", "Timestamp"])
            .groupby("UserID")
            .agg({"ItemID": list, "Timestamp": list})
            .reset_index()
        )
        return user2item, item2title

    def _apply_id_mapping(
        self, user2item: pd.DataFrame, item2title: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        r"""Apply the consecutive numeric ID mapping to the ``user2item`` and
        ``item2title`` data. The original IDs are re-mapped to consecutive
        integers starting from ``START_ID``.

        Args:
            user2item (pd.DataFrame):
                The grouped user-item interaction data.
            item2title (pd.DataFrame | None):
                The item titles.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]:
                The remapped ``user2item`` and ``item2title`` data.
        """
        user_ids = user2item["UserID"].unique()
        item_ids = np.unique(np.concatenate(user2item["ItemID"].values))
        user_map = {user: idx for idx, user in enumerate(user_ids, start=self.START_ID)}
        item_map = {item: idx for idx, item in enumerate(item_ids, start=self.START_ID)}
        user2item["UserID"] = user2item["UserID"].map(user_map)
        user2item["ItemID"] = user2item["ItemID"].apply(
            lambda x: [item_map[item] for item in x]
        )
        user2item = user2item.sort_values(by=["UserID"]).reset_index(drop=True)
        if self.meta_available:
            item2title["ItemID"] = item2title["ItemID"].map(item_map)
            item2title = item2title.sort_values(by=["ItemID"]).reset_index(drop=True)
        return user2item, item2title

    def _calculate_statistics(
        self, user2item: pd.DataFrame, item2title: pd.DataFrame | None
    ) -> dict[str, float | int | dict[str, float | int]]:
        r"""Calculate the statistics of the dataset. The following statistics
        will be calculated:
        - ``user_size``: the number of users in the dataset.
        - ``item_size``: the number of items in the dataset.
        - ``interactions``: the number of interactions in the dataset.
        - ``density``: the density of the dataset, which is defined as the
            ratio of the number of interactions to the number of users
            multiplied by the number of items.
        - ``interaction_length``: the minimum, maximum, and average length of
            the interaction sequence of users.
        - ``title_token``: the minimum, maximum, and average number of tokens
            in the item title. The tokenization is based on ``tiktoken``
            library for ``gpt-4`` model.

        Args:
            user2item (pd.DataFrame):
                The processed user-item interaction data.
            item2title (pd.DataFrame | None):
                The processed item titles.

        Returns:
            dict[str, float | int | dict[str, float | int]]:
                The statistics of the dataset.
        """
        user_size = user2item["UserID"].nunique()
        item_size = user2item["ItemID"].explode().nunique()
        interaction_lengths = user2item["ItemID"].apply(len)
        interactions = interaction_lengths.sum()
        density = interactions / (user_size * item_size)
        if self.meta_available:
            title_lengths = item2title["Title"].apply(self.tokenize_length)
        statistics = {
            "user_size": int(user_size),
            "item_size": int(item_size),
            "interactions": int(interactions),
            "density": round(float(density), 6),
            "interaction_length": {
                "min": round(float(interaction_lengths.min()), 6),
                "max": round(float(interaction_lengths.max()), 6),
                "avg": round(float(interaction_lengths.mean()), 6),
            },
        }
        if self.meta_available:
            statistics["title_token"] = {
                "min": round(float(title_lengths.min()), 6),
                "max": round(float(title_lengths.max()), 6),
                "avg": round(float(title_lengths.mean()), 6),
            }
        return statistics

    def _save_processed_data(
        self, user2item: pd.DataFrame, item2title: pd.DataFrame | None
    ) -> None:
        r"""Save the processed data and the statistics of the dataset. The
        following files will be saved in the ``dataset_dir/proc`` directory:
        - ``user2item.pkl``: the processed user-item interaction data, each
            row represents ``(UserID, ItemID, Timestamp)``, where ``ItemID``
            and ``Timestamp`` are lists of item IDs and timestamps,
            respectively.
        - ``item2title.pkl``: the processed item titles, each row represents
            ``(ItemID, Title)``.
        - ``summary.json``: the statistics of the dataset.

        .. note::
            When ``sample_user_size`` is not ``None``, the processed data will
            be saved in the ``dataset_dir_{sample_user_size}/proc`` directory.

        Args:
            user2item (pd.DataFrame):
                The processed user-item interaction data.
            item2title (pd.DataFrame | None):
                The processed item titles.
        """
        save_dir = (
            os.path.join(self.dataset_dir, "proc")
            if self.sample_user_size is None
            else os.path.join(self.dataset_dir + f"_{self.sample_user_size}", "proc")
        )
        os.makedirs(save_dir, exist_ok=True)
        user2item.to_pickle(os.path.join(save_dir, "user2item.pkl"))
        if self.meta_available:
            item2title.to_pickle(os.path.join(save_dir, "item2title.pkl"))
        statistics = self._calculate_statistics(user2item, item2title)
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(statistics, f, indent=4)

    def tokenize_length(self, text: str, model_name: str = "gpt-4") -> int:
        r"""Tokenize the text and return the number of tokens.

        Args:
            text (str):
                The text to tokenize.
            model_name (str, optional, default="gpt-4"):
                The name of the model. The default value is "gpt-4".

        Returns:
            int:
                The number of tokens in the text.
        """
        enc = tiktoken.encoding_for_model(model_name)
        return len(enc.encode(text))
