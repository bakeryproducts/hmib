#!/usr/bin/bash
set -e
SCALE=$1
NAME=$2

python3 src/preprocess.py --src input/hmib_folders/VALID_SPLIT_0 --pr $SCALE --lu $SCALE --co $SCALE --ki $SCALE --dst input/preprocessed/SCALE_$NAME
python3 src/preprocess.py --src input/hmib_folders/VALID_SPLIT_1 --pr $SCALE --lu $SCALE --co $SCALE --ki $SCALE --dst input/preprocessed/SCALE_$NAME
python3 src/preprocess.py --src input/hmib_folders/VALID_SPLIT_2 --pr $SCALE --lu $SCALE --co $SCALE --ki $SCALE --dst input/preprocessed/SCALE_$NAME
python3 src/preprocess.py --src input/hmib_folders/VALID_SPLIT_3 --pr $SCALE --lu $SCALE --co $SCALE --ki $SCALE --dst input/preprocessed/SCALE_$NAME

python3 src/preprocess.py --src input/hmib_folders/TRAIN_SPLIT_0 --pr $SCALE --lu $SCALE --co $SCALE --ki $SCALE --dst input/preprocessed/SCALE_$NAME
python3 src/preprocess.py --src input/hmib_folders/TRAIN_SPLIT_1 --pr $SCALE --lu $SCALE --co $SCALE --ki $SCALE --dst input/preprocessed/SCALE_$NAME
python3 src/preprocess.py --src input/hmib_folders/TRAIN_SPLIT_2 --pr $SCALE --lu $SCALE --co $SCALE --ki $SCALE --dst input/preprocessed/SCALE_$NAME
python3 src/preprocess.py --src input/hmib_folders/TRAIN_SPLIT_3 --pr $SCALE --lu $SCALE --co $SCALE --ki $SCALE --dst input/preprocessed/SCALE_$NAME
