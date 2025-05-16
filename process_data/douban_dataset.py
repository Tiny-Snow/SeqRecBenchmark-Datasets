r"""Douban Dataset Processor for SeqRecBenchmark.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import os
from typing import Any

import pandas as pd

from process_data.base_dataset import BaseDatasetProcessor

__all__ = [
    "DoubanDatasetProcessor",
]


def read_csv(
    file_path: str,
    delimiter: str,
    columns: list[str] | None = None,
    types: list[Any] | None = None,
    header: int | None = None,
) -> pd.DataFrame:
    r"""Read a CSV file and return a DataFrame. If ``columns`` is not ``None``,
    the columns will be renamed to the specified names ``columns``. Each
    column will be converted to the specified type in ``types``. If ``types``
    is ``None``, the columns will be automatically inferred by Pandas. In
    addition, the columns with blank values will be dropped.

    Args:
        file_path (str):
            The path of the CSV file.
        delimiter (str):
            The delimiter of the CSV file.
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
    if types is not None:
        types = {col: typ for col, typ in zip(columns, types)}
    df = pd.read_csv(
        file_path,
        sep=delimiter,
        header=header,
        names=columns,
        dtype=types,
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
    )
    df = df.dropna()
    return df


class DoubanDatasetProcessor(BaseDatasetProcessor):
    r"""Processor for the Douban datasets.

    Each Douban dataset contains a ``dataset_name.tsv`` (e.g., ``douban-book.tsv``)
    file with the following (renamed) columns:
    - ``UserID``: contains the user ID.
    - ``ItemID``: contains the item ID.
    - ``Rating``: contains the rating given by the user to the item (not used).
    - ``Timestamp``: contains the timestamp of the rating (float unix time).

    .. note::
        The original Douban dataset only contains the rating data, while the
        metadata (e.g., item title) is not included.
    """

    def __init__(
        self,
        dataset_dir: str,
        k_core: int = 5,
        sample_user_size: int = None,
    ) -> None:
        r"""Initialize the Douban dataset processor.

        Args:
            dataset_dir (str):
                The directory of the dataset, e.g., ``/path/to/douban-book``.
                The last directory name should be the dataset name, e.g.,
                ``douban-book``.
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
        super().__init__(dataset_dir, False, k_core, sample_user_size)

    def _load_data(self) -> tuple[pd.DataFrame, None]:
        r"""Load the raw data from the raw data files. The following data
        are required to be loaded:
        - ``interactions`` (pd.DataFrame):
            The user-item interaction data, which is a Pandas DataFrame object.
            Each row represents an interaction ``(UserID, ItemID, Timestamp)``,
            with types ``(int | str, int | str, int | str)``.
        - ``item2title`` (None):
            The item titles, which is ``None`` for the Douban dataset.

        Returns:
            tuple[pd.DataFrame, None]:
                The first element is the user-item interaction data, and the
                second element is the item titles (``None``).
        """
        raw_dir = os.path.join(self.dataset_dir, "raw")
        interactions = read_csv(
            os.path.join(raw_dir, f"{self.dataset_name}.tsv"),
            delimiter="\t",
            columns=["UserID", "ItemID", "Rating", "Timestamp"],
            types=[int, int, float, float],
            header=0,
        )
        interactions["Timestamp"] = interactions["Timestamp"].astype("int64")
        interactions = interactions[["UserID", "ItemID", "Timestamp"]]
        return interactions, None
