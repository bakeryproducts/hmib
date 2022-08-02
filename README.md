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

# TODO
git

increasing momentum
nonblocking cuda call
memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
normalize stat

tmpfs
mount  -t tmpfs -o size=20g  tmpfs /userhome/memory_data
