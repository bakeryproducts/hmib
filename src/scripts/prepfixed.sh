#!/usr/bin/bash
set -e
SCALE=$1
DST=$2
CROP=1024
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v0/lung/ --dst temp/$DST/train_hpa/lung/v0 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive --ext tiff --HACKHPA
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v1/lung/ --dst temp/$DST/train_hpa/lung/v1 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive --ext tiff --HACKHPA
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v2/lung/ --dst temp/$DST/train_hpa/lung/v2 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive --ext tiff --HACKHPA
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v3/lung/ --dst temp/$DST/train_hpa/lung/v3 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive --ext tiff --HACKHPA

python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v0/spleen/ --dst temp/$DST/train_hpa/spleen/v0 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive  --ext tiff --HACKHPA
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v1/spleen/ --dst temp/$DST/train_hpa/spleen/v1 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive  --ext tiff --HACKHPA
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v2/spleen/ --dst temp/$DST/train_hpa/spleen/v2 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive  --ext tiff --HACKHPA
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v3/spleen/ --dst temp/$DST/train_hpa/spleen/v3 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive  --ext tiff --HACKHPA

python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v0/prostate/ --dst temp/$DST/train_hpa/prostate/v0 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive  --ext tiff --HACKHPA
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v1/prostate/ --dst temp/$DST/train_hpa/prostate/v1 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive  --ext tiff --HACKHPA
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v2/prostate/ --dst temp/$DST/train_hpa/prostate/v2 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive  --ext tiff --HACKHPA
python3 hsrc/data_gen.py --src input/FIXED_TRAIN/splits/v3/prostate/ --dst temp/$DST/train_hpa/prostate/v3 --src_scale .4 --dst_scale $SCALE --cropsize $CROP --recursive  --ext tiff --HACKHPA
