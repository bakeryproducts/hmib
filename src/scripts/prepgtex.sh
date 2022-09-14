#!/usr/bin/bash
set -e
SCALE=$1
DST=$2
CROP=1024
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/prostate --dst temp/$DST/gtex_$SCALE/prostate --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode grid &
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/kidney --dst temp/$DST/gtex_$SCALE/kidney --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode grid &
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/spleen --dst temp/$DST/gtex_$SCALE/spleen --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode inst &
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/colon --dst temp/$DST/gtex_$SCALE/colon --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode grid 

python3 hsrc/data_gen.py --src input/extra/gtex/pieces/prostate_test --dst temp/$DST/gtex_${SCALE}_test/prostate --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode grid &
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/kidney_test --dst temp/$DST/gtex_${SCALE}_test/kidney --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode grid &
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/spleen_test --dst temp/$DST/gtex_${SCALE}_test/spleen --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode grid &
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/colon_test --dst temp/$DST/gtex_${SCALE}_test/colon --src_scale .5 --dst_scale $SCALE --cropsize $CROP --recursive --mode grid
