#!/usr/bin/bash
set -e
SCALE=$1
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/prostate --dst temp/gtex_$SCALE/prostate --src_scale .5 --dst_scale $SCALE --cropsize 1024 --recursive --mode grid
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/kidney --dst temp/gtex_$SCALE/kidney --src_scale .5 --dst_scale $SCALE --cropsize 1024 --recursive --mode grid
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/spleen --dst temp/gtex_$SCALE/spleen --src_scale .5 --dst_scale $SCALE --cropsize 1024 --recursive --mode inst
python3 hsrc/data_gen.py --src input/extra/gtex/pieces/colon --dst temp/gtex_$SCALE/colon --src_scale .5 --dst_scale $SCALE --cropsize 1024 --recursive --mode grid
