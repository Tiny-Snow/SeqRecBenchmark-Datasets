# SeqRec Benchmark Datasets

## Introduction

Our **SeqRec Benchmark** provides a series of standardized datasets for sequential recommendation. These datasets are crawled from various sources, including Amazon, Douban, Gowalla, MovieLens, Yelp, etc. For those review datasets where text metadata is available, we also process the item title data for LLM-based recommendation.

### Dataset Format

The processed datasets contain the following files:

- `user2item.pkl`: A Pandas DataFrame with three columns: `UserID`, `ItemID` and `Timestamp`. Each row represents a user and the items they have interacted with, along with the corresponding timestamps. The `UserID` column contains the unique and sorted user IDs. The `ItemID` and `Timestamp` columns are lists of item IDs and timestamps, respectively. Note that the `UserID` and `ItemID` are both starting from 100, while the 0-99 IDs are reserved for the special tokens (e.g. `<PAD>`), and the `Timestamp` is in Unix time format.
- `item2title.pkl`: A Pandas DataFrame with two columns: `ItemID` and `Title`. Each row represents an item and its corresponding title. The `ItemID` column contains the unique and sorted item IDs, and the `Title` column contains the item titles. Note that `item2title.pkl` is only available for those datasets with text metadata.
- `summary.json`: A JSON file that contains the dataset statistics.

Let's take `amazon2014-book` dataset as an example:

- `amazon2014-book/proc/user2item.pkl`:

```
        UserID                                             ItemID                                          Timestamp
0          100  [46164, 132129, 198911, 205467, 206349, 209419...  [1353369600, 1353369600, 1353369600, 135336960...
1          101    [78991, 265544, 265550, 265548, 265543, 265545]  [1353196800, 1354147200, 1354320000, 135466560...
2          102         [99459, 99460, 12688, 67549, 29220, 18387]  [1358380800, 1359072000, 1368835200, 138965760...
3          103           [100195, 220952, 273328, 274192, 276757]  [1402012800, 1402012800, 1402012800, 140201280...
4          104  [275457, 255948, 146955, 255124, 128362, 20828...  [1362614400, 1373328000, 1376611200, 137730240...
...        ...                                                ...                                                ...
509329  509429            [95342, 202158, 159155, 262965, 254531]  [1334620800, 1358208000, 1358640000, 136918080...
509330  509430  [225184, 168367, 170812, 160865, 191281, 23610...  [1269475200, 1269993600, 1269993600, 127059840...
509331  509431  [10661, 11429, 39779, 40382, 44603, 79624, 108...  [1375747200, 1375747200, 1375747200, 137574720...
509332  509432         [18175, 78602, 9676, 209146, 99293, 70297]  [1264896000, 1264896000, 1356307200, 135630720...
509333  509433  [160054, 160056, 160057, 160058, 160059, 16006...  [973987200, 1044057600, 1094688000, 1105488000...

[509334 rows x 3 columns]
```

- `amazon2014-book/proc/item2title.pkl`:

```
        ItemID                                              Title
0          100                                        The Prophet
1          101                                     Master Georgie
2          102                             The Book of Revelation
3          103  The Greatest Book on &quot;Dispensational Trut...
4          104                          Rightly Dividing the Word
...        ...                                                ...
280492  280592                                  Tales of Honor #1
280493  280593           Newsweek Special Issue - Michael Jackson
280494  280594  The Berenstain Bears Keep the Faith (Berenstai...
280495  280595  We Are All Completely Beside Ourselves: A Nove...
280496  280596      Samantha Sanderson On The Scene (FaithGirlz!)

[280497 rows x 2 columns]
```

- `amazon2014-book/proc/summary.json`:
```json
{
    "user_size": 509334,
    "item_size": 280497,
    "interactions": 7109843,
    "density": 5e-05,
    "interaction_length": {
        "min": 5.0,
        "max": 22942.0,
        "avg": 13.959098
    },
    "title_token": {
        "min": 1.0,
        "max": 114.0,
        "avg": 11.149659
    }
}
```

### Dataset Processing Methods

The dataset processing methods are provided in the [process_data/base_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/base_dataset.py). Basically, we will process the dataset into the following steps:

- **Load the raw data**: implemented in the `DatasetProcessor._load_data()` method. In this step, two Pandas DataFrames are returned: `interactions` with each row as an interaction and three columns: `(UserID, ItemID, Timestamp)`, and `item2title` with each row as an item and two columns: `(ItemID, Title)`. This virtual method should be overridden in the specific DatasetProcessor subclass. Just load the data from the raw files, and no need to do any processing here.
- **Filter the invalid item titles**: optionally implemented in the `DatasetProcessor._filter_item_title()` method. By default, we only filter the items with empty titles. You may override this method in the specific DatasetProcessor subclass to specify the filtering rules.
- **Drop duplicate users/items**: implemented in the `DatasetProcessor._drop_duplicates()` method. This step is to drop the users or items with duplicate IDs.
- **Sample users**: implemented in the `DatasetProcessor._sample_users()` method. If the dataset is too large (especially for the LLM-based recommendation), we may sample the users to reduce the dataset size. Note that the final dataset usually has smaller user size than the number specified in this step, since some users may be filtered out in the later steps (e.g., $K$-core filtering).
- **Apply $K$-core filtering**: implemented in the `DatasetProcessor._filter_k_core()` method. This step is to filter the users and items with less than $K$ interactions. The default value of $K$ is 5.
- **Group the interactions**: implemented in the `DatasetProcessor._group_interactions()` method. In this step, the interactions with the same user are grouped together, and the items are sorted by the timestamp (from the earliest to the latest).
- **Apply consecutive numeric ID mapping**: implemented in the `DatasetProcessor._apply_id_mapping()` method. In this step, we will apply the consecutive numeric ID mapping for the users and items, and the final IDs start from 100. The final DataFrames `user2item` and `item2title` are sorted by `UserID` and `ItemID`, respectively.
- **Save the processed data and statistics**: implemented in the `DatasetProcessor._save_processed_data()` method. In this step, we will save the processed data into the `dataset_name[_sample_user_size]/proc` directory. The processed data includes `user2item.pkl`, `item2title.pkl` and `summary.json`. If the dataset is sampled to `sample_user_size` users, the `_sample_user_size` suffix will be added to the dataset name.

**Reducing Dataset Size.** For those extremely large datasets, we randomly sample some users to reduce the dataset size. For instance, the `Amazon-2018-Book` dataset has over 27M interactions, we provide the 1M users sampled version `Amazon-2018-Book-1M`. Even though the original dataset is listed here, we may not provide its processed version due to the limited storage space.

**Customization.** You can modify and run [run_process_data.sh](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/run_process_data.sh) to customize your dataset processing.

**Dependencies.** The processed datasets are generated by Python 3.12.10 with `pickle protocol 5`. Note that Python 3.8+ is required to read the processed `.pkl` files.

---

## Processed Datasets

### Dataset Statistics

The following table summarizes the statistics of the currently supported datasets. The statistics include the number of users, items, interactions, density, average interactions per user (i.e., mean sequence length), and average title tokens per item (based on GPT-4 tokenizer). The details of the datasets are provided in the following sections.

| Dataset | Users | Items | Interactions | Density | Avg. Interactions | Avg. Title Tokens |
| ------- | ----: | ----: | -----------: | ------: | ----------------------: | ----------------: |
| [Amazon-2014-Beauty](#amazon) | 22,332 | 12,086 | 198,215 | 0.00073 | 8.88 | 19.57 |
| [Amazon-2014-Book](#amazon) | 509,334 | 280,497 | 7,109,843 | 0.00005 | 13.96 | 11.15 |
| [Amazon-2014-Book-1M](#amazon) | 38,234 | 38,519 | 517,167 | 0.00035 | 13.53 | 9.81 |
| [Amazon-2014-CD](#amazon) | 51,573 | 46,373 | 795,961 | 0.00033 | 15.43 | 6.51 |
| [Amazon-2014-Clothing](#amazon) | 39,230 | 22,948 | 277,534 | 0.00031 | 7.07 | 14.19 |
| [Amazon-2014-Electronic](#amazon) | 183,699 | 61,028 | 1,608,404 | 0.00014 | 8.76 | 25.26 |
| [Amazon-2014-Electronic-1M](#amazon) | 28,566 | 15,743 | 241,198 | 0.00054 | 8.44 | 24.78 |
| [Amazon-2014-Health](#amazon) | 38,329 | 18,427 | 343,831 | 0.00049 | 8.97 | 19.56 |
| [Amazon-2014-Movie](#amazon) | 58,695 | 27,378 | 889,819 | 0.00055 | 15.16 | 8.74 |
| [Amazon-2014-Toy](#amazon) | 19,124 | 11,758 | 165,247 | 0.00074 | 8.64 | 11.32 |
| [Amazon-2018-Book](#amazon) | 1,846,955 | 701,221 | 27,004,767 | 0.00002 | 14.62 | 12.05 |
| [Amazon-2018-Book-1M](#amazon) | 71,461 | 66,903 | 973,051 | 0.00020 | 13.62 | 10.74 |
| [Amazon-2018-CD](#amazon) | 93,996 | 64,424 | 1,188,235 | 0.00020 | 12.64 | 7.14 |
| [Amazon-2018-Clothing](#amazon) | 1,164,450 | 372,536 | 10,709,754 | 0.00003 | 9.20 | 189.85 |
| [Amazon-2018-Clothing-1M](#amazon) | 35,606 | 20,404 | 294,343 | 0.00041 | 8.27 | 312.02 |
| [Amazon-2018-Electronic](#amazon) | 695,201 | 157,619 | 6,395,592 | 0.00006 | 9.20 | 30.44 |
| [Amazon-2018-Electronic-1M](#amazon) | 43,462 | 23,133 | 374,989 | 0.00037 | 8.63 | 29.35 |
| [Amazon-2018-Game](#amazon) | 50,597 | 16,874 | 453,637 | 0.00053 | 8.97 | 11.06 |
| [Amazon-2018-Movie](#amazon) | 281,662 | 59,193 | 3,225,818 | 0.00019 | 11.45 | 8.08 |
| [Amazon-2018-Toy](#amazon) | 192,197 | 75,820 | 1,686,227 | 0.00012 | 8.77 | 16.03 |
| [Douban-Book](#douban) | 31,956 | 38,969 | 1,616,164 | 0.00130 | 50.57 | N/A |
| [Douban-Movie](#douban) | 70,358 | 40,148 | 11,548,122 | 0.00409 | 164.13 | N/A |
| [Douban-Music](#douban) | 23,985 | 38,511 | 1,560,777 | 0.00169 | 65.07 | N/A |
| [Food](#food) | 17,813 | 41,240 | 555,618 | 0.00076 | 31.19 | 6.38 |
| [Gowalla](#gowalla) | 76,894 | 304,421 | 4,616,090 | 0.00020 | 60.03 | N/A |
| [Gowalla-50K](#gowalla) | 33,573 | 135,729 | 1,827,255 | 0.00040 | 54.43 | N/A |
| [KuaiRec](#kuairec) | 7,174 | 7,228 | 1,071,111 | 0.02066 | 149.30 | N/A |
| [MovieLens-1M](#movielens) | 6,040 | 3,416 | 999,611 | 0.04845 | 165.50 | 4.73 |
| [MovieLens-10M](#movielens) | 69,878 | 10,196 | 9,998,816 | 0.01403 | 143.09 | 5.42 |
| [MovieLens-20M](#movielens) | 138,493 | 18,345 | 19,984,024 | 0.00787 | 144.30 | 6.01 |
| [MovieLens-25M](#movielens) | 162,541 | 32,720 | 24,945,870 | 0.00469 | 153.47 | 5.48 |
| [MovieLens-32M](#movielens) | 200,948 | 43,883 | 31,921,432 | 0.00362 | 158.85 | 5.27 |
| [RetailRocket](#retailrocket) | 60,109 | 34,183 | 693,333 | 0.00034 | 11.53 | N/A |
| [Steam](#steam) | 281,460 | 11,961 | 3,555,275 | 0.00106 | 12.63 | 5.05 |
| [Steam-100K](#steam) | 10,369 | 3,593 | 126,824 | 0.00340 | 12.23 | 5.19 |
| [Steam-1M](#steam) | 109,294 | 9,266 | 1,367,400 | 0.00135 | 12.51 | 5.07 |
| [Yelp2018](#yelp) | 213,170 | 94,304 | 3,277,932 | 0.00016 | 15.38 | 4.50 |
| [Yelp2018-500K](#yelp) | 72,500 | 49,474 | 1,094,295 | 0.00030 | 15.09 | 4.52 |
| [Yelp2022](#yelp) | 277,628 | 112,392 | 4,250,457 | 0.00014 | 15.31 | 4.63 |
| [Yelp2022-500K](#yelp) | 59,866 | 45,881 | 900,041 | 0.00033 | 15.03 | 4.66 |
| [YooChoose-Buys](#yoochoose) | 43,107 | 4,659 | 292,230 | 0.00146 | 6.78 | N/A |
| [YooChoose-Clicks](#yoochoose) | 1,878,772 | 30,480 | 16,002,569 | 0.00028 | 8.52 | N/A |
| [YooChoose-Clicks-100K](#yoochoose) | 17,744 | 5,630 | 145,716 | 0.00146 | 8.21 | N/A |
| [YooChoose-Clicks-500K](#yoochoose) | 98,726 | 13,989 | 830,196 | 0.00060 | 8.41 | N/A |

### Amazon

#### Description

The Amazon dataset is a large crawl of product reviews from [Amazon](https://www.amazon.com/), including reviews, metadata and graphs. We process two versions of the Amazon dataset: [Amazon-2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) and [Amazon-2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/). The latest [Amazon-2023](https://amazon-reviews-2023.github.io/) dataset can be similarly processed, though the researchers rarely adopt it in the papers. Due to the large size and high sparsity of the Amazon dataset, it is usually divided into specific domains, including `Amazon-Beauty`, `Amazon-Book` (Books), `Amazon-CD` (CDs and Vinyl), `Amazon-Clothing` (Clothing, Shoes and Jewelry), `Amazon-Electronic` (Electronics), `Amazon-Health` (Health and Personal Care), `Amazon-Movie` (Movies and TV), `Amazon-Toy` (Toys and Games), `Amazon-Game` (Video Games), etc. The item titles are accessible in all Amazon datasets except for `Amazon-2014-Game` (thus we do not provide this dataset). For those extremely large datasets (> 5M interactions, e.g., `Amazon-2018-Book`), we sample 1M users from the original dataset to reduce the dataset size (e.g., `Amazon-2018-Book-1M`). For more processing details, please refer to the [process_data/amazon_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/amazon_dataset.py).

#### References

```bibtex
@inproceedings{he2016ups,
  title={Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering},
  author={He, Ruining and McAuley, Julian},
  booktitle={proceedings of the 25th international conference on world wide web},
  pages={507--517},
  year={2016}
}
@inproceedings{mcauley2015image,
  title={Image-based recommendations on styles and substitutes},
  author={McAuley, Julian and Targett, Christopher and Shi, Qinfeng and Van Den Hengel, Anton},
  booktitle={Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval},
  pages={43--52},
  year={2015}
}
@inproceedings{ni2019justifying,
  title={Justifying recommendations using distantly-labeled reviews and fine-grained aspects},
  author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
  booktitle={Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP)},
  pages={188--197},
  year={2019}
}
@article{hou2024bridging,
  title={Bridging language and items for retrieval and recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

### Douban

#### Description

The [Douban](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec) dataset is crawled from a popular Chinese social networking service platform [Douban](http://www.douban.com/), including ratings in three domains: `Douban-Book`, `Douban-Movie` and `Douban-Music`. Note that the item metadata is not available in the original dataset, and we only provide the user-item interaction data `user2item.pkl`. For more processing details, please refer to the [process_data/douban_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/douban_dataset.py).

#### References

```bibtex
@inproceedings{song2019session,
  title={Session-based social recommendation via dynamic graph attention networks},
  author={Song, Weiping and Xiao, Zhiping and Wang, Yifan and Charlin, Laurent and Zhang, Ming and Tang, Jian},
  booktitle={Proceedings of the Twelfth ACM international conference on web search and data mining},
  pages={555--563},
  year={2019}
}
```

### Food

#### Description

The [Food](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset consists of recipes and reviews from [Food.com](https://www.food.com/). The review text and various recipe metadata are also provided. We take the name of recipe as the item title. For more processing details, please refer to the [process_data/food_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/food_dataset.py).

```bibtex
@article{majumder2019generating,
  title={Generating personalized recipes from historical user preferences},
  author={Majumder, Bodhisattwa Prasad and Li, Shuyang and Ni, Jianmo and McAuley, Julian},
  journal={arXiv preprint arXiv:1909.00105},
  year={2019}
}
```

### Gowalla

#### Description

The [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html) dataset is a check-in dataset collected from the location-based social network [Gowalla](https://en.wikipedia.org/wiki/Gowalla), which is widely used in collaborative filtering and sequential recommendation research. Note that the item metadata is not available in the original dataset, and we only provide the user-item interaction data `user2item.pkl`. The processed dataset has over 4.6M interactions, which is somewhat large for research purposes. Thus we also provide a smaller version `Gowalla-50K` with 1.8M interactions, which samples 50K users from the original dataset. For more processing details, please refer to the [process_data/gowalla_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/gowalla_dataset.py).

#### References

```bibtex
@inproceedings{cho2011friendship,
  title={Friendship and mobility: user movement in location-based social networks},
  author={Cho, Eunjoon and Myers, Seth A and Leskovec, Jure},
  booktitle={Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={1082--1090},
  year={2011}
}
```

### KuaiRec

#### Description

The [KuaiRec](https://kuairec.com/) dataset (version 2.0) is a real-world dataset collected from the recommendation logs of the video-sharing mobile app [Kuaishou](https://www.kuaishou.com/cn). The dataset contains two types of interaction data: `big_matrix` and `small_matrix`, where the `small_matrix` is a fully-observed dataset, and the `big_matrix` is a sparser peripheral dataset that excludes the user-video interactions in the `small_matrix`. To construct a proper sequential recommendation dataset based on KuaiRec, as suggested in the [original paper](https://arxiv.org/pdf/2202.10842), we merge the `big_matrix` and `small_matrix` datasets, and then filter the "negative" interactions where the user's cumulative watching time is less than twice the video duration. We do not provide the item titles in the processed dataset, as the video titles here are not appropriate for LLM-based recommendation -- they are full of emoticons and keyword tags. For more processing details, please refer to the [process_data/kuairec_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/kuairec_dataset.py).

#### References

```bibtex
@inproceedings{gao2022kuairec,
  title={KuaiRec: A fully-observed dataset and insights for evaluating recommender systems},
  author={Gao, Chongming and Li, Shijun and Lei, Wenqiang and Chen, Jiawei and Li, Biao and Jiang, Peng and He, Xiangnan and Mao, Jiaxin and Chua, Tat-Seng},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={540--550},
  year={2022}
}
```

### MovieLens

#### Description

The MovieLens (ML) dataset is crawled from the [MovieLens](https://movielens.org) website, consisting of the ratings and metadata of user-movie interactions. The MovieLens dataset is widely used in recommender systems research due to its dense and rich interaction data. The dataset is available in several versions, including [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/), [MovieLens-10M](https://grouplens.org/datasets/movielens/10m/), [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/), [MovieLens-25M](https://grouplens.org/datasets/movielens/25m/), and the latest [MovieLens-32M](https://grouplens.org/datasets/movielens/32m/) datasets. We process all these versions, and note that the MovieLens-1M and MovieLens-20M (usually sampled) are the most frequently used datasets in the research papers. For more processing details, please refer to the [process_data/movielens_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/movielens_dataset.py).

#### References

```bibtex
@article{harper2015movielens,
  title={The movielens datasets: History and context},
  author={Harper, F Maxwell and Konstan, Joseph A},
  journal={Acm transactions on interactive intelligent systems (tiis)},
  volume={5},
  number={4},
  pages={1--19},
  year={2015},
  publisher={Acm New York, NY, USA}
}
```

### RetailRocket

#### Description

The [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) dataset is a real-world dataset collected from the [RetailRocket](https://retailrocket.net/) e-commerce platform. The dataset contains three types of user-item interactions: `view`, `addtocart`, and `transaction`. Following the [previous work](https://dl.acm.org/doi/abs/10.1145/3539618.3591624), we only keep the `view` interactions for sequential recommendation. Note that the item titles are not available in the original dataset, and we only provide the user-item interaction data `user2item.pkl`. For more processing details, please refer to the [process_data/retailrocket_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/retailrocket_dataset.py).

#### References

```bibtex
@misc{roman_zykov_noskov_artem_anokhin_alexander_2022,
	title={Retailrocket recommender system dataset},
	url={https://www.kaggle.com/dsv/4471234},
	DOI={10.34740/KAGGLE/DSV/4471234},
	publisher={Kaggle},
	author={Roman Zykov and Noskov Artem and Anokhin Alexander},
	year={2022}
}
```

### Steam

#### Description

The [Steam](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) dataset (version 2) contains the video game reviews and game metadata crawled from the [Steam](https://store.steampowered.com/) platform. This dataset is widely used in sequential recommendation research. To facilitate the research, we also provide two smaller versions: `Steam-1M` and `Steam-100K`, which sample 1M and 100K users from the original dataset, respectively. The item titles are accessible in both datasets. For more processing details, please refer to the [process_data/steam_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/steam_dataset.py).

#### References

```bibtex
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE international conference on data mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
@inproceedings{wan2018item,
  title={Item recommendation on monotonic behavior chains},
  author={Wan, Mengting and McAuley, Julian},
  booktitle={Proceedings of the 12th ACM conference on recommender systems},
  pages={86--94},
  year={2018}
}
@inproceedings{pathak2017generating,
  title={Generating and personalizing bundle recommendations on steam},
  author={Pathak, Apurva and Gupta, Kshitiz and McAuley, Julian},
  booktitle={Proceedings of the 40th international ACM SIGIR conference on research and development in information retrieval},
  pages={1073--1076},
  year={2017}
}
```

### YooChoose

#### Description

The [YooChoose](https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015) dataset is originally provided by [YooChoose](http://www.yoochoose.com/) for the [RecSys Challenge 2015](https://recsys.acm.org/recsys15/challenge/). The dataset contains a collection of sessions from a retailer, where each session is encapsulating the click events that the user performed in the session. The data was collected during several months in the year of 2014, reflecting the clicks and purchases performed by the users of an on-line retailer in Europe. The dataset is divided into two parts: `YooChoose-Clicks` and `YooChoose-Buys`, containing the click and purchase events, respectively. Both the two datasets are provided with labels `Session ID` (i.e., `UserID`), `Item ID` (i.e., `ItemID`) and `Timestamp`. However, the item metadata is not available in the original dataset, and we only provide the user-item interaction data `user2item.pkl`. Additionally, as the `YooChoose-Clicks` dataset is extremely large (16M interactions), we randomly sample 100K/500K users to reduce the dataset size, resulting in `YooChoose-Clicks-100K` and `YooChoose-Clicks-500K`. Note that we drop the test data `yoochoose-test.dat` to make the training paradigm of the two datasets consistent. For more processing details, please refer to the [process_data/yoochoose_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/yoochoose_dataset.py).

#### References

```bibtex
@inproceedings{ben2015recsys,
  title={Recsys challenge 2015 and the yoochoose dataset},
  author={Ben-Shimon, David and Tsikinovsky, Alexander and Friedmann, Michael and Shapira, Bracha and Rokach, Lior and Hoerle, Johannes},
  booktitle={Proceedings of the 9th ACM Conference on Recommender Systems},
  pages={357--358},
  year={2015}
}
```


### Yelp

#### Description

The [Yelp](https://business.yelp.com/data/resources/open-dataset) dataset is a subset of [Yelp](https://www.yelp.com/)'s businesses, reviews, and user data. It was originally put together for the Yelp Dataset Challenge, which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. The complete Yelp dataset contains five files: `business.json`, `checkin.json`, `review.json`, `tip.json`, and `user.json`. For sequential recommendation, we process `review.json` to get the user-business interactions, and the `business.json` file to get the business metadata. This results in the processed [Yelp2018](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/versions/1) and [Yelp2022](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/versions/4) datasets. In addition, we also provide the smaller versions `Yelp2018-500K` and `Yelp2022-500K` for research purposes, which sample 500K users from the original datasets. The lastest version of the Yelp dataset is available in [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset). For historical versions, please refer to [Kaggle Yelp](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset). For more processing details, please refer to the [process_data/yelp_dataset.py](https://github.com/Tiny-Snow/SeqRecBenchmark-Datasets/tree/main/process_data/yelp_dataset.py).

#### References

```bibtex
@article{asghar2016yelp,
  title={Yelp dataset challenge: Review rating prediction},
  author={Asghar, Nabiha},
  journal={arXiv preprint arXiv:1605.05362},
  year={2016}
}
```

---

## Useful Links

- [RSPD](https://cseweb.ucsd.edu/~jmcauley/datasets.html): Recommender Systems and Personalization Datasets, a collection of famous datasets for recommendation systems used in Julian McAuley's lab, UCSD.
- [SNAP](https://snap.stanford.edu/): Stanford Network Analysis Project, a general purpose network analysis and graph mining library. 
- [RecBole](https://github.com/RUCAIBox/RecBole): A unified and comprehensive recommendation library, including various recommendation models and datasets. This library collects many recommendation datasets and can be found in [RecBole/DatasetList](https://recbole.io/dataset_list.html) and [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets). 
- [RecZoo](https://huggingface.co/reczoo#datasets): A collection of recommendation datasets in Hugging Face.
- [RSTasks](https://paperswithcode.com/task/recommendation-systems): A collection of recommendation system benchmarks in Papers with Code.

---

## Citation

We are pleased to see that the carefully curated dataset above could have a positive impact on the recommendation community. If you use the above data, please cite the following reference:

```bibtex
@inproceedings{yang2024psl,
  author = {Yang, Weiqin and Chen, Jiawei and Xin, Xin and Zhou, Sheng and Hu, Binbin and Feng, Yan and Chen, Chun and Wang, Can},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {120974--121006},
  publisher = {Curran Associates, Inc.},
  title = {PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/db1d5c63576587fc1d40d33a75190c71-Paper-Conference.pdf},
  volume = {37},
  year = {2024}
}
```
