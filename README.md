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
git

increasing momentum
nonblocking cuda call
memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
normalize stat

tmpfs
mount  -t tmpfs -o size=20g  tmpfs /userhome/memory_data
