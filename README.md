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
### Kidney
 - R:
 1) test 180-130, csv, in 164-148
 2) test 160-152, csv, in 158-156
 - G
 1) 180-130, pass, in 0-130
 2) 110-65, pass, in 0-65
 3) 55-30, pass , in 0-30



