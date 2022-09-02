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
- more blur augs
- harder regularization 
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
- Count 79
1) 130, 70, scoring => in 70-90
2) 86, 74, not found => in 78-82
3) 81, 78, scoring => 79

- Size mean
1) 4499 4000 3000 2000, pass, in 0-2000
2) 1900, 1000, note, in 1900-2000
- Size std
1) 200, 50, 5, 1, pass, in 0-1

- R in 154-158
3) 195, 140, scoring, in 140-158
4) 154, 144, note, in 154-158
- G in 98-103
5) 150, 80, scoring, in 80-103
6) 98, 84, note, in 98 - 103
- B in 170-173
7) 160, 100, note, in 160,255
8) 220, 175, pass, in 160-175
9) 173, 163, res, in 170-173


### prostate ✔️
- Count 98
1) 130, 70, not found => in 90-110
2) 106, 94, scoring => 94-98
3) 97, 94, notebook => 98
- Size mean ~ 160
1) 832, 328, pass => in 160-328
2) 294, 192, pass => in 160-192
3) 185, 164, pass => in 160-164
- Size std ~ 0
1) 3, 1, pass => in 0-1
- R in 186-189
1) 180, 120, notebook => in 180-255
2) 240, 195, pass => in 180-195
3) 192, 183, not found => in 186-189
- G in 139-142
1) 130, 70, notebook => in 130-255
2) 180,142, pass, in 130-142 
3) 139, 133, notebook => in 139-142
- B in 192-196
1) 180, 120, notebook => in 180-255
2) 196, 184, res => in 192-196


### spleen ✔️
- Count 113
1) 110-60, note, in 110-448
2) 140-120, pass, in 110-120
3) 118-112, score, 113
- Size mean
1) 4000, 500, csv, in 2834 - 1666
2) 2600, 1900, score, in 1900-2133
3) custom, in 2010-2050
- Size std
1) 300, 50, pass, in 0-50
2) 40, 5, pass, in 0-5

- Whiteness
1) .2, .10, .05, .01, score, in .05, .01

- R in 166-168
1) 200, 50, res, in 150-200
2) 190, 160, score, in 160-170
3) 168, 162, res, in 166-168
- G in 90-94
1) 190, 110, pass, in 0-110
2) 90, 10, note, in 90-110
3) 105, 94, pass, in 90-94
- B in 126-130
1) 195, 110, scoring, in 110-138
2) 130, 118, res, in 126-130 


### largeintestine ✔️
- Count 43
1) 110, 50, pass => in 0-50
2) 46, 34, res => in 42-46
3) 45, 42, scoring => 43
- Size mean
1) 4499, 4000, 2500, 100, res => in 4000-4499
2) 4400, 4100, res => in 4300-4400
3) 4380, 4320, res => in 4360-4380
4) 4376, 4364, scoring => in 4364-4368
- R in 128-132
5) 180, 120, scoring => in 120-140
6) 136, 124, not found => in 128-132
- G in 114-118
7) 130, 70, res => in 110-130
8) 126, 114, scoring => in 114-118
- B in 136-140
9) 160, 100, not found => in 120-140
10) 136, 120, notebook => in 136-140

### lung ✔️
- Count 115
1) 110, 70, note, in 110-448
2) 150, 115, pass, in 110 - 115
3) 114, 111, note, 115
- Size mean
1) 4499 4000 3000 2000, pass, in 0-2000
2) 2001, 1950, 1500, 1400, pass, in 0-1400
3) 1300, 500, note, in 1300-1400

- Whiteness
1) .7, .. .2 , pass, in 0-.2
2) .15, .10, .05, .02, score, in .02 - .05

- Gradientness
1) .2, .01, note, in .2 - 1
2) .7 .3, , in .65 - .70
- Holesiziness
1) 500, 300, 100, 0, sub, in 100-300
2) 250, 220, 180, 150, pass, 100-150

- R in 197-200
1) 150, 80, note, in 150-255
2) 200, 160, res, in 187-200
3) 197, 190, note, in 197-200
- G in 185-191
1) 140, 90, note, in 140-255 
2) 180, 150, note, in 180-255
3) 205, 185, score, in 185-191
- B in 205-211
1) 200, 100, , note, in 200-255
2) 225, 205,  score, in 205-211
