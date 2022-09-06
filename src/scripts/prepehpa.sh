#!/usr/bin/bash
set -e
SCALE=$1
python3 hsrc/data_gen.py --src input/extra/hpa/images/colon --dst temp/hpa_$SCALE/colon --src_scale .4 --dst_scale $SCALE --recursive --ext jpg --HACKHPA
python3 hsrc/data_gen.py --src input/extra/hpa/images/spleen --dst temp/hpa_$SCALE/spleen --src_scale .4 --dst_scale $SCALE --recursive --ext jpg --HACKHPA
python3 hsrc/data_gen.py --src input/extra/hpa/images/kidney --dst temp/hpa_$SCALE/kidney --src_scale .4 --dst_scale $SCALE --recursive --ext jpg --HACKHPA
python3 hsrc/data_gen.py --src input/extra/hpa/images/lung --dst temp/hpa_$SCALE/lung --src_scale .4 --dst_scale $SCALE --recursive --ext jpg --HACKHPA

