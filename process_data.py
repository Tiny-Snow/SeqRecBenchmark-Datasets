r"""Dataset processing for SeqRecBenchmark.

Copyright (c) 2025 Weiqin Yang (Tiny Snow) & Yue Pan @ Zhejiang University
"""

import argparse
import random

import numpy as np
import process_data


def set_seed(seed: int) -> None:
    r"""Set the random seed for reproducibility.
    This function sets the random seed for the built-in ``random`` and
    ``numpy`` libraries.

    Args:
        seed (int):
            The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser(description="Process SeqRec datasets.")
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=[
            "amazon",
            "douban",
            "food",
            "gowalla",
            "kuairec",
            "movielens",
            "retailrocket",
            "steam",
            "yelp",
            "yoochoose",
        ],
        required=True,
        help="The type of the dataset to process.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="The directory of the dataset.",
    )
    parser.add_argument(
        "--k-core",
        type=int,
        default=5,
        help="The K-core value for filtering out inactive users and items.",
    )
    parser.add_argument(
        "--sample-user-size",
        type=int,
        default=None,
        help="The number of users to sample from the dataset.",
    )
    args = parser.parse_args()
    if args.dataset_type == "amazon":
        processor = process_data.AmazonDatasetProcessor(
            dataset_dir=args.dataset_dir,
            meta_available=True,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    elif args.dataset_type == "douban":
        processor = process_data.DoubanDatasetProcessor(
            dataset_dir=args.dataset_dir,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    elif args.dataset_type == "food":
        processor = process_data.FoodDatasetProcessor(
            dataset_dir=args.dataset_dir,
            meta_available=True,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    elif args.dataset_type == "gowalla":
        processor = process_data.GowallaDatasetProcessor(
            dataset_dir=args.dataset_dir,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    elif args.dataset_type == "kuairec":
        processor = process_data.KuaiRecDatasetProcessor(
            dataset_dir=args.dataset_dir,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    elif args.dataset_type == "movielens":
        processor = process_data.MovielensDatasetProcessor(
            dataset_dir=args.dataset_dir,
            meta_available=True,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    elif args.dataset_type == "retailrocket":
        processor = process_data.RetailRocketDatasetProcessor(
            dataset_dir=args.dataset_dir,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    elif args.dataset_type == "steam":
        processor = process_data.SteamDatasetProcessor(
            dataset_dir=args.dataset_dir,
            meta_available=True,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    elif args.dataset_type == "yelp":
        processor = process_data.YelpDatasetProcessor(
            dataset_dir=args.dataset_dir,
            meta_available=True,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    elif args.dataset_type == "yoochoose":
        processor = process_data.YooChooseDatasetProcessor(
            dataset_dir=args.dataset_dir,
            k_core=args.k_core,
            sample_user_size=args.sample_user_size,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")
    processor.process()
