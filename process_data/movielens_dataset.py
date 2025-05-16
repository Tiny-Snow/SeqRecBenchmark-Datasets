r"""Movielens dataset processor.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import os
from typing import Any

import pandas as pd

from process_data.base_dataset import BaseDatasetProcessor

__all__ = [
    "MovielensDatasetProcessor",
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


class MovielensDatasetProcessor(BaseDatasetProcessor):
    r"""Processor for the Movielens datasets (1M & 10M & 20M & 25M & 32M).

    Each Movielens dataset contains two files we need to process:
    - ``ratings.dat/.csv``: contains the ratings data. Each line contains the
        ``UserID``, ``MovieID``, ``Rating``, and ``Timestamp`` (unix time). We
        do not use the ``Rating``.
    - ``movies.dat/.csv``: contains the movie meta data. Each line contains the
        ``MovieID``, ``Title``, and ``Genres``. We do not use the ``Genres``.

    .. note::
        The Movielens-1M/10M dataset's ``.dat`` files use ``::`` as the
        delimiter and have no header, while the Movielens-20M/25M/32M dataset's
        ``.csv`` files use ``,`` as the delimiter and have a header.

    .. note::
        All movie titles are in the format of ``Title (Year)``, where ``Year`` is a
        4-digit number, e.g., ``Toy Story (1995)``. For LLM-based recommendation,
        the `Year` is not useful, so we remove it.
    """

    def __init__(
        self,
        dataset_dir: str,
        meta_available: bool = False,
        k_core: int = 5,
        sample_user_size: int = None,
    ) -> None:
        r"""Initialize the Movielens dataset processor.

        Args:
            dataset_dir (str):
                The directory of the dataset, e.g., ``/path/to/movielens-1m``.
                The last directory name should be the dataset name, e.g.,
                ``movielens-1m``.
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
        if self.dataset_name in ["movielens-1m", "movielens-10m"]:
            self.delimiter, self.file_suffix, self.header = "::", ".dat", None
        elif self.dataset_name in ["movielens-20m", "movielens-25m", "movielens-32m"]:
            self.delimiter, self.file_suffix, self.header = ",", ".csv", 0
        else:
            raise ValueError(
                f"Invalid dataset name: {self.dataset_name}. You should "
                f"specify the version of the Movielens dataset (1m, 10m, 20m, "
                f"25m, or 32m)."
            )

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
            os.path.join(raw_dir, f"ratings{self.file_suffix}"),
            delimiter=self.delimiter,
            columns=["UserID", "ItemID", "Rating", "Timestamp"],
            types=[int, int, int, str],
            header=self.header,
        )
        interactions = interactions[["UserID", "ItemID", "Timestamp"]]
        interactions["Timestamp"] = interactions["Timestamp"].astype("int64")
        if self.meta_available:
            item2title = read_csv(
                os.path.join(raw_dir, f"movies{self.file_suffix}"),
                delimiter=self.delimiter,
                columns=["ItemID", "Title", "Genres"],
                types=[int, str, str],
                header=self.header,
            )
            item2title = item2title[["ItemID", "Title"]]
            item2title = item2title[
                item2title["Title"].notna() & item2title["Title"] != "nan"
            ]
            item2title["Title"] = item2title["Title"].str.replace(
                r"\s*\(\d{4}\)", "", regex=True
            )
        else:
            item2title = None
        return interactions, item2title
