r"""Check the processed data for SeqRecBenchmark datasets.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd


def check_data(dataset_dir: str) -> None:
    r"""Check the processed data for SeqRecBenchmark datasets.

    Specifically, this function will check the ``user2item.pkl``,
    ``item2title.pkl``, and ``summary.json`` files in ``dataset_dir/proc``
    directory, if they exist.

    Args:
        dataset_dir (str):
            The directory of the dataset, e.g., ``/path/to/amazon2014-book``.
            The last directory name should be the dataset name, e.g.,
            ``amazon2014-book``.
    """
    # check user2item.pkl
    user2item_path = os.path.join(dataset_dir, "proc", "user2item.pkl")
    if os.path.exists(user2item_path):
        with open(user2item_path, "rb") as f:
            user2item = pickle.load(f)
        print(f"Loaded {user2item_path} successfully, the user2item data is:")
        print(user2item)
        print("The first 5 rows of user2item data:")
        print(user2item.head(5).to_string(index=False))
        user_ids = user2item["UserID"]
        if user_ids.isnull().any():
            print("ERROR: user2item data contains NA / NaN values.")
            return
        if len(user_ids) != len(np.unique(user_ids)):
            print("ERROR: user2item data contains duplicate user IDs.")
            return
        item_ids = user2item["ItemID"].explode().unique()
        if np.any(pd.isnull(item_ids)):
            print("ERROR: user2item data contains NA / NaN values.")
            return
        print(f"Number of unique user IDs: {len(user_ids)}")
        print(f"Number of unique item IDs: {len(item_ids)}")
    else:
        print(f"{user2item_path} does not exist.")
        return
    # check item2title.pkl
    item2title_path = os.path.join(dataset_dir, "proc", "item2title.pkl")
    if os.path.exists(item2title_path):
        with open(item2title_path, "rb") as f:
            item2title = pickle.load(f)
        print(f"Loaded {item2title_path} successfully, the item2title data is:")
        print(item2title)
        print("The first 5 rows of item2title data:")
        print(item2title.head(5).to_string(index=False))
        item_titles_ids = item2title["ItemID"]
        if item_titles_ids.isnull().any():
            print("ERROR: item2title data contains NA / NaN values.")
            return
        if len(item_titles_ids) != len(np.unique(item_titles_ids)):
            print("ERROR: item2title data contains duplicate item IDs.")
            return
        if not np.array_equal(np.sort(item_ids), np.sort(item_titles_ids)):
            print("ERROR: item2title data does not match user2item data.")
            return
    else:
        print(f"{item2title_path} does not exist.")
    # check summary.json
    summary_path = os.path.join(dataset_dir, "proc", "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        print(f"Loaded {summary_path} successfully, the summary data is:")
        print(summary)
    else:
        print(f"{summary_path} does not exist.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_data.py <dataset_dir>")
        sys.exit(1)
    dataset_dir = sys.argv[1]
    check_data(dataset_dir)
