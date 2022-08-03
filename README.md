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
- build scale regressor, do not rely on kaggle scales
- dice per class
- instance norm? domain shift! normalize stat
- progressive ema momentum
- boundary loss weight-in
- layer-wise LR decay
- check if FINE_LR is working
- trivial aug
- repeated aug
- finetuning with frequent val epoch, every N step -> inside train cb
- ema & swa combo?
- proper regularization, stoch depth, dropouts
- fix seg head, check for heavier heads, CBA -> Nxtimes
- SAM
- upernet
- nfnets?
- swin bb

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
