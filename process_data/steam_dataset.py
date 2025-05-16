r"""Steam Dataset Processor for SeqRecBenchmark.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import ast
import os

import pandas as pd

from process_data.base_dataset import BaseDatasetProcessor

__all__ = [
    "SteamDatasetProcessor",
]


def read_json(
    file_path: str, selected_cols: list[str] | None = None, standard: bool = True
) -> pd.DataFrame:
    r"""Read a JSON file and return a DataFrame. Note that only the columns
    specified in ``selected_cols`` will be selected. If ``selected_cols`` is
    ``None``, all columns will be selected. The rows with blank values in the
    selected columns will be dropped.

    .. warning::
        If the ``selected_cols`` contains a column that is not in the JSON
        file, an ``KeyError`` will be raised.

    Args:
        file_path (str):
            The path of the JSON file.
        selected_cols (list[str], optional, default=None):
            The columns to select from the JSON file. If ``None``, all
            columns will be selected. The default value is ``None``.
            Note that if the selected columns remain blank in a row, this
            row will be dropped.
        standard (bool, optional, default=True):
            Whether the file is in the standard JSON format. If ``False``,
            we assume that each row is a Python literal (e.g., a dictionary).
            The default value is ``True``.

    Returns:
        The DataFrame containing the data from the JSON file.
    """
    if standard:
        df = pd.read_json(file_path, lines=True)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(ast.literal_eval(line))
            except Exception as error:
                print(f"Error parsing line: {line}")
                print(f"Error message: {error}")
        df = pd.DataFrame(data)
    if selected_cols is not None:
        df = df[selected_cols]
    df = df.dropna()
    return df


class SteamDatasetProcessor(BaseDatasetProcessor):
    r"""Processor for the Steam dataset.

    Each Steam dataset contains two files:
    - ``steam_reviews.json``: contains user-item interactions. Each json object
        in this file represents a review, where the ``username`` (string),
        ``product_id`` (int/string), and ``date`` (e.g., `2017-12-17`) fields
        are used as ``UserID``, ``ItemID``, and ``Timestamp``, respectively.
    - ``steam_games.json``: contains game metadata. Each json object in this
        file represents a game, where the ``id`` (int/string) field is used as
        ``ItemID``, and the ``title`` (string) field is used as the item title
        (renamed to ``ItemTitle``).

    .. note::
        The above two files are not in standard JSON format.
    """

    def __init__(
        self,
        dataset_dir: str,
        meta_available: bool = False,
        k_core: int = 5,
        sample_user_size: int = None,
    ) -> None:
        r"""Initialize the Steam dataset processor.

        Args:
            dataset_dir (str):
                The directory of the dataset, e.g., ``/path/to/steam``.
                The last directory name should be the dataset name, e.g.,
                ``steam``.
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
            os.path.join(raw_dir, "steam_reviews.json"),
            selected_cols=["username", "product_id", "date"],
            standard=False,
        )
        interactions.columns = ["UserID", "ItemID", "Timestamp"]
        interactions["Timestamp"] = pd.to_datetime(
            interactions["Timestamp"], format="%Y-%m-%d"
        )
        interactions["Timestamp"] = interactions["Timestamp"].astype("int64") // 10**9
        interactions["ItemID"] = interactions["ItemID"].astype("str")
        if self.meta_available:
            item2title = read_json(
                os.path.join(raw_dir, "steam_games.json"),
                selected_cols=["id", "title"],
                standard=False,
            )
            item2title.columns = ["ItemID", "Title"]
            item2title["ItemID"] = item2title["ItemID"].astype("str")
            item2title = item2title[
                item2title["Title"].notna() & item2title["Title"] != "nan"
            ]
        else:
            item2title = None
        return interactions, item2title
