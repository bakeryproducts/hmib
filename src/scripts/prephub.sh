#!/usr/bin/bash
set -e
SCALE=$1
DST=$2
CROP=1024
python3 hsrc/data_gen.py --src input/extra/hubmap_colon/train --dst temp/$DST/hubmap_colon_$SCALE/colon --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode grid
python3 hsrc/data_gen.py --src input/extra/hubmap_kidney/train --dst temp/$DST/hubmap_kidney_$SCALE/kidney --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode inst --total 15
