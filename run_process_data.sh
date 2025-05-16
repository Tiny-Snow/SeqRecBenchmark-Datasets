#!/bin/bash

DATASET_ROOT="/path/to/data"
cd /path/to/data

DATASET_TYPE="amazon"           # must be a valid dataset type, e.g., amazon, douban, etc.
DATASET_NAME="amazon2018-book"  # must match the dataset (dir) name
SAMPLE=1000000                  # 0 means no sampling for user size, if not 0, the dataset name will be appended with _SAMPLE

if [ ${SAMPLE} != 0 ]; then
    python process_data.py --dataset-type=${DATASET_TYPE} --dataset-dir=${DATASET_ROOT}/${DATASET_NAME} --sample-user-size=${SAMPLE}
else
    python process_data.py --dataset-type=${DATASET_TYPE} --dataset-dir=${DATASET_ROOT}/${DATASET_NAME}
fi

if [ ${SAMPLE} != 0 ]; then
    python check_data.py ${DATASET_ROOT}/${DATASET_NAME}_${SAMPLE}
else
    python check_data.py ${DATASET_ROOT}/${DATASET_NAME}
fi
