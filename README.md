# HuBMAP + HPA - Hacking the Human Body 
https://www.kaggle.com/competitions/hubmap-organ-segmentation/discussion


### "The code is provided as is, it has not been rewritten. Given competitions are done in a hurry, code may not meet usual open source standard." CPMP
### 8th place, Gleb & Alex
# Usage

data structure:
 - input (all input data)
    - hmib (original kgl data)
    - preprocessed
    - ...
 - output (logs, models)
 - src
    - tv train-validation callbacks, main stuff
    - train - fitter init, all callbacks init
    - network - NN init, Unet, SSL, etc
    - build_data - dataloaders
    - data - datasets
    - metrics - loss, metrics
 - notebooks


## start

`python3 -m torch.distributed.launch --use_env --nproc_per_node=4 src/main.py --config-name=u`


## hydra start:

4 consecutive runs, changing split param
`python3 starter.py -m +SPLIT=0,1,2,3  +nproc=4`


# Old hubmap data cuts:

src folder : input/extra/hubmap/

there should be unpacked kaggle comp data: train, test, train.csv, etc..

`python3 hsrc/data_gen.py`

Creates input/extra/hubmap/preprocessed folder with :
 - bigmasks - json to tiff
 - CUTS - glomeruli cut from images, SCALE, WH, N in data_gen.py
 - SPLITS - copy paste cuts into 4 train val folds
 


# TODO

## Basic
- public .78 just resizes into 768???
- stride trick
- TTA stain
- loss only from 1channel?

- ~~rle masks, check diff vs polygon masks on lungs~~ rle is better
- ~~multilabel , looks the same~~
- ~~Proper val, whole image~~ ~~big center crop for now~~ separate datasets, merge them
- ~~build scale regressor, do not rely on kaggle scales~~ test.csv should contain scale, but still nice to have?
- ~~deep supervision~~
- more blur augs
- boundary loss weight-in
- ~~repeated aug~~
- finetuning with frequent val epoch, every N step -> inside train cb
- proper regularization, stoch depth, LayerScale? dropouts
- fix seg head, check for heavier heads, CBA -> Nxtimes
- SAM
- ~~upernet, aspp fuse?~~
- nfnets?
- uptrain helps.
- ~~decouple WD for head/body (https://arxiv.org/pdf/2106.04560.pdf)~~

## Transformers
- segformer?
- inject with class token
- HiViT?

## Shift
- HUBMAP validation images for each class, can be without GT
- SSL, mae
- https://github.com/BMIRDS/deepslide - lung WSI dataset!
- !!! overlapped, PAIRED dataset !!! https://www.nature.com/articles/s41598-018-37257-4.pdf
        https://zenodo.org/record/5137717
- !!! paired domain-shift https://bci.grand-challenge.org/ https://github.com/bupt-ai-cz/BCI
- domain-shift dataset ? https://imig.science/midog/the-challgenge/
- a LOT of challenges https://grand-challenge.org/challenges/

## Labeling problems
 - pseudo?
 - lung mask -> band-like polygons?  terminal bronchiole? 

## Random

- nonblocking cuda call
- memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


mount  -t tmpfs -o size=20g  tmpfs /userhome/memory_data

# dupes
## prostate
6730 - 32741
30294 - 1229
26982 - 30414
10666 - 16659

## kidney
24097 - 9470

## colon
351 - 26780
## lung
10488 - 20563 - 14388
1878 - 11629


# TEST

## gtex:

colon:

GTEX-1JMPY-0426
0_10645_8853_14908_16156.tiff  1_15028_25841_18556_12381.tiff  2_31025_6677_8158_7198.tiff

kidney
GTEX-144GL-1926
0_2262_4788_10748_11581.tiff   2_32749_2709_11388_12349.tiff   4_22032_22257_9085_11005.tiff 1_17489_3445_11420_11613.tiff  3_35980_19409_11068_11581.tiff  5_4661_25136_10012_10781.tiff

spleen
GTEX-1N2DV-0726
0_5271_8052_20187_21947.tiff  1_35184_6324_21467_19515.tiff
GTEX-1L5NE-0926
0_3574_4787_19611_21627.tiff  1_25137_6546_24154_22363.tiff 



## hubmap colon

CL_HandE_1234_B004_bottomleft_000000.png  HandE_B005_CL_b_RGB_bottomleft_000002.png
CL_HandE_1234_B004_bottomleft_000001.png  HandE_B005_CL_b_RGB_bottomleft_000003.png
CL_HandE_1234_B004_bottomleft_000003.png

## hub kidney

1e2425f28_000000 - 11.png
2f6ecfcdf_000000 - 11.png
4ef6695ce_000000 - 11.png







