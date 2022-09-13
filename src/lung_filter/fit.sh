export CUDA_VISIBLE_DEVICES=0,

source ../../../.hubmap_venv/bin/activate
which python

WORK_DIR=output
IMAGES_DIR=../../input/hmib/lung_filter

BATCH=64
EPOCHS=50
WORKERS=16
INPUT_SIZE=224
LR=0.001
RATE=50

MODEL=resnet18
EXPERIMENT=${MODEL}_${INPUT_SIZE}_${BATCH}b${EPOCHS}e${LR}lr

export PYTHONPATH=.:${PYTHONPATH}

nohup python train.py \
    --model_name=${MODEL} \
    --images_dir=${IMAGES_DIR} \
    --batch_size=${BATCH} \
    --num_epochs=${EPOCHS} \
    --num_workers=${WORKERS} \
    --input_size=${INPUT_SIZE} \
    --lr=${LR} \
    --experiment=${EXPERIMENT} > ${WORK_DIR}/logs/${EXPERIMENT}_train.log 2>&1 &
