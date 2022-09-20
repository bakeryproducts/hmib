#!/usr/bin/bash
set -e
MODEL=$1
SCALE=$2
# ORGAN='prostate'
ROOT="$(dirname $(dirname $(dirname "$MODEL")))"
DEVICE=${3:-0}
export CUDA_VISIBLE_DEVICES=$DEVICE
TTA=${4:-"none"}

ORGANS='prostate kidney spleen colon'
for ORGAN in $ORGANS; do
    python3 src/predict.py \
        --model_file $MODEL \
        --images_dir input/extra/gtex/pieces/${ORGAN}_test/  \
        --output_dir ${ORGAN}_test \
        --image_meter_scale 0.5 \
        --network_scale $SCALE \
        --device 0  \
        --base_block_size 768 \
        --pad 128  \
        --batch_size 16 \
        --organ $ORGAN\
        --ext tiff  \
        --mode whole_blocks \
        --scale_block False \
        --use_mp True \
        --tta $TTA
    python3 src/evaluate.py \
        --pred_masks_dir $ROOT/${TTA}_predicts/${ORGAN}_test/ \
        --true_masks_dir input/CUTS_$SCALE/gtex_test/${ORGAN}/bigmasks \
        --thr 10 \
        --thr_max 250 \
        --loff \
        --thr_total 10 \
        --csv dices
done


# python3 src/predict.py \
#     --model_file $MODEL \
#     --images_dir input/extra/gtex/pieces/kidney_test/  \ 
#     --output_dir kidney_test \
#     --image_meter_scale 0.5 \ 
#     --device 0  \ 
#     --base_block_size 512  \ 
#     --pad 128  \ 
#     --batch_size 8  \ 
#     --network_scale $SCALE \
#     --organ kidney  \ 
#     --ext tiff  \ 
#     --mode whole_blocks \ 
#     --scale_block False



