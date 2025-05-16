r"""Food Dataset Processor for SeqRecBenchmark.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import os
from typing import Any

import pandas as pd

from process_data.base_dataset import BaseDatasetProcessor

__all__ = [
    "FoodDatasetProcessor",
]


def read_csv(
    file_path: str,
    delimiter: str,
    sel_cols: list[int] | list[str] | None = None,
    columns: list[str] | None = None,
    types: list[Any] | None = None,
    header: int | None = None,
) -> pd.DataFrame:
    r"""Read a CSV file and return a DataFrame. Only the columns specified in
    ``select_cols`` will be read. The columns are renamed to the specified
    names ``columns``, which should be a list of strings. Each column will be
    converted to the specified type in ``types``. If ``types`` is ``None``,
    the columns will be automatically inferred by Pandas. In addition, the
    columns with blank values will be dropped.

    Args:
        file_path (str):
            The path of the CSV file.
        delimiter (str):
            The delimiter of the CSV file.
        select_cols (list[int] | list[str] | None, optional, default=None):
            The columns to select from the CSV file. If ``None``, all columns
            will be selected. The default value is ``None``.
        columns (list[str] | None, optional, default=None):
            The column names to rename the columns to. If ``None``, no renaming
            will be performed. The default value is ``None``.
        types (list[Any], optional, default=None):
            The types to convert the columns to. If ``None``, the columns
            will be automatically inferred by Pandas. The default value is
            ``None``.
        header (int | None, optional, default=None):
            The row number to use as the column names. If ``None``, no row
            will be used as the column names. The default value is ``None``.

    Returns:
        The DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(
        file_path,
        sep=delimiter,
        header=header,
        usecols=sel_cols,
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
    )
    if columns is not None:
        df.columns = columns
    if types is not None:
        types = {col: typ for col, typ in zip(columns, types)}
        df = df.astype(types)
    df = df.dropna()
    return df


class FoodDatasetProcessor(BaseDatasetProcessor):
    r"""Processor for the Food dataset.

    The Food dataset contains the raw interaction data ``RAW_interactions.csv``
    and meta data ``RAW_recipes.csv`` we will process:
    - ``RAW_interactions.csv``: contains the interaction data, including:
        - ``user_id``: The ID of the user, which is renamed to ``UserID``.
        - ``recipe_id``: The ID of the recipe, which is renamed to ``ItemID``.
        - ``date``: The date of the interaction, e.g., 2003-02-17, which is
            renamed to ``Timestamp``.
        - ``rating``: The rating of the interaction (not used).
        - ``review``: The review text of the interaction (not used).
    - ``RAW_recipes.csv``: contains the meta data of the recipes, including:
        - ``name``: The name of the recipe, which is renamed to ``Title``.
            Note that the text may have multiple spaces, which are cleaned
            to a single space in the processing.
        - ``id``: The ID of the recipe, which is renamed to ``ItemID``.
        - The other columns are not used.
    """

    def __init__(
        self,
        dataset_dir: str,
        meta_available: bool = False,
        k_core: int = 5,
        sample_user_size: int = None,
    ) -> None:
        r"""Initialize the Food dataset processor.

        Args:
            dataset_dir (str):
                The directory of the dataset, e.g., ``/path/to/food``.
                The last directory name should be the dataset name, e.g.,
                ``food``.
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
        interactions = read_csv(
            os.path.join(raw_dir, "RAW_interactions.csv"),
            delimiter=",",
            sel_cols=[0, 1, 2],
            columns=["UserID", "ItemID", "Timestamp"],
            types=[int, int, str],
            header=0,
        )
        interactions["Timestamp"] = pd.to_datetime(
            interactions["Timestamp"], format="%Y-%m-%d"
        )
        interactions["Timestamp"] = interactions["Timestamp"].astype("int64") // 10**9
        if self.meta_available:
            item2title = read_csv(
                os.path.join(raw_dir, "RAW_recipes.csv"),
                delimiter=",",
                sel_cols=[0, 1],
                columns=["Title", "ItemID"],
                types=[str, int],
                header=0,
            )
            item2title = item2title[["ItemID", "Title"]]
            item2title = item2title[
                item2title["Title"].notna() & item2title["Title"] != "nan"
            ]
            item2title["Title"] = (
                item2title["Title"].str.strip().str.replace(r"\s+", " ", regex=True)
            )
        else:
            item2title = None
        return interactions, item2title
