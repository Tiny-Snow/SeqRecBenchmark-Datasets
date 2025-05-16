r"""Gowalla dataset processor.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import os
from typing import Any

import pandas as pd

from process_data.base_dataset import BaseDatasetProcessor

__all__ = [
    "GowallaDatasetProcessor",
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


class GowallaDatasetProcessor(BaseDatasetProcessor):
    r"""Processor for the Gowalla dataset.

    The Gowalla dataset contains one file ``loc-gowalla_totalCheckins.txt.gz``
    that we need to process. This file is essentially a CSV file with `\t`
    as the delimiter. Each row is a check-in record with the following columns:
    - `UserID`: The user ID.
    - `Timestamp`: The timestamp of the check-in. Note that the timestamp
        is in ISO 8601 format (e.g., 2010-10-19T23:55:27Z).
    - `Latitude`: The latitude of the check-in location (not used).
    - `Longitude`: The longitude of the check-in location (not used).
    - `ItemID`: The item ID (location ID).

    .. note::
        Since Gowalla is a location-based social network dataset, the item
        title metadata is not available.
    """

    def __init__(
        self,
        dataset_dir: str,
        k_core: int = 5,
        sample_user_size: int = None,
    ) -> None:
        r"""Initialize the Gowalla dataset processor.

        Args:
            dataset_dir (str):
                The directory of the dataset, e.g., ``/path/to/gowalla``.
                The last directory name should be the dataset name, e.g.,
                ``gowalla``.
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
            The item titles, which is ``None`` for the Gowalla dataset.

        Returns:
            tuple[pd.DataFrame, None]:
                The first element is the user-item interaction data, and the
                second element is the item titles (``None``).
        """
        raw_dir = os.path.join(self.dataset_dir, "raw")
        interactions = read_csv(
            os.path.join(raw_dir, "loc-gowalla_totalCheckins.txt.gz"),
            delimiter="\t",
            columns=["UserID", "Timestamp", "Latitude", "Longitude", "ItemID"],
            types=[int, str, float, float, int],
            header=None,
        )
        interactions = interactions[["UserID", "ItemID", "Timestamp"]]
        interactions["Timestamp"] = pd.to_datetime(
            interactions["Timestamp"], format="%Y-%m-%dT%H:%M:%SZ"
        )
        interactions["Timestamp"] = interactions["Timestamp"].astype("int64") // 10**9
        return interactions, None
