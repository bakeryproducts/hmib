# HM2

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
- ~~rle masks, check diff vs polygon masks on lungs~~ rle is better
- ~~multilabel , looks the same~~
- ~~Proper val, whole image~~ ~~big center crop for now~~ separate datasets, merge them
- ~~build scale regressor, do not rely on kaggle scales~~ test.csv should contain scale, but still nice to have?
- ~~deep supervision~~
- tissue / non tissue mask? white background depends on scanner
- boundary loss weight-in
- layer-wise LR decay
- trivial aug w/ gt
- repeated aug
- finetuning with frequent val epoch, every N step -> inside train cb
- proper regularization, stoch depth, LayerScale? dropouts
- fix seg head, check for heavier heads, CBA -> Nxtimes
- SAM
- upernet, aspp fuse? 
- nfnets?
- swin bb
- uptrain helps.
- maybe cls test on top left all white crop ?
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

# Probing:


## Hubmap
### Kidney ✔️
 - R in 154-158
 1) 195, 140, scoring, in 140-158
 2) 154, 144, note, in 154-158
 - G in 98-103
 1) 150, 80, scoring, in 80-103
 2) 98, 84, note, in 98 - 103
 - B in 170-173
 1) 160, 100, note, in 160,255
 2) 220, 175, pass, in 160-175
 3) 173, 163, res, in 170-173

### prostate
- R in 186-189
1) 180, 120, notebook => in 180-255
2) 240, 195, pass => in 180-195
3) 192, 183, not found => in 186-189
- G in ?
1) 130, 70, notebook => in 130-255
2) 180,142, pass, in 130-142 
- B in ?


### spleen
- R in ?
1) 200, 50, res, in 150-200
2) 190, 160, score, in 160-170
3) TODO
- G in 90-94
1) 190, 110, pass, in 0-110
2) 90, 10, note, in 90-110
3) 105, 94, pass, in 90-94
- B in 126-130
1) 195, 110, scoring, in 110-138
2) 130, 118, res, in 126-130 


### largeintestine ✔️
- R in 128-132
1) 180, 120, scoring => in 120-140
2) 136, 124, not found => in 128-132
- G in 114-118
1) 130, 70, res => in 110-130
2) 126, 114, scoring => in 114-118
- B in 136-140
1) 160, 100, not found => in 120-140
2) 136, 120, notebook => in 136-140
