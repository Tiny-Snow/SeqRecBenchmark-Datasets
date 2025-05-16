r"""KuaiRec Dataset Processor for SeqRecBenchmark.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import os
from typing import Any

import pandas as pd

from process_data.base_dataset import BaseDatasetProcessor

__all__ = [
    "KuaiRecDatasetProcessor",
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


class KuaiRecDatasetProcessor(BaseDatasetProcessor):
    r"""Processor for the KuaiRec dataset.

    The KuaiRec dataset contains two interaction files: ``big_matrix.csv` and
    ``small_matrix.csv``. The ``small_matrix.csv`` is a fully-observed dataset,
    while the ``big_matrix.csv`` is a sparser dataset excluding the
    ``small_matrix.csv``. We merge the two datasets, each row represents a
    user-item interaction, with the following columns we will use:
    - ``user_id`` (int64): The ID of the user (renamed to ``UserID``).
    - ``video_id`` (int64): The ID of the video (renamed to ``ItemID``).
    - ``timestamp`` (float64, e.g., `1593878903.438`): The timestamp of the
        interaction (renamed to ``Timestamp``).
    - ``watch_ratio`` (float64, e.g., `1.273397`): The ratio of the
        cumulative watching time to the video duration, used to filter out
        negative interactions (if `< 2.0`).

    .. note::
        Although the original dataset provide the video metadata, we do not
        process it, as the titles are not appropriate for recommendation tasks.
    """

    def __init__(
        self,
        dataset_dir: str,
        k_core: int = 5,
        sample_user_size: int = None,
    ) -> None:
        r"""Initialize the KuaiRec dataset processor.

        Args:
            dataset_dir (str):
                The directory of the dataset, e.g., ``/path/to/kuairec``.
                The last directory name should be the dataset name, e.g.,
                ``kuairec``.
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
        interactions_big = read_csv(
            os.path.join(raw_dir, "big_matrix.csv"),
            delimiter=",",
            sel_cols=["user_id", "video_id", "timestamp", "watch_ratio"],
            columns=["UserID", "ItemID", "Timestamp", "WatchRatio"],
            types=[int, int, float, float],
            header=0,
        )
        interactions_small = read_csv(
            os.path.join(raw_dir, "small_matrix.csv"),
            delimiter=",",
            sel_cols=["user_id", "video_id", "timestamp", "watch_ratio"],
            columns=["UserID", "ItemID", "Timestamp", "WatchRatio"],
            types=[int, int, float, float],
            header=0,
        )
        interactions = pd.concat([interactions_big, interactions_small], axis=0)
        print(interactions)
        interactions = interactions[interactions["WatchRatio"] >= 2.0]
        interactions = interactions.drop(columns=["WatchRatio"])
        # NOTE: Not convert to second timestamp, remain as milliseconds
        # interactions["Timestamp"] = interactions["Timestamp"].astype("int64")
        return interactions, None
