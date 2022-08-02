import os
import shutil
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
import torch

from config import cfg_init
from logger import logger, log_init
from train import start

import warnings


def set_gpus(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        gpus = ','.join([str(g) for g in cfg.TRAIN.GPUS])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    else:
        logger.warning(f'WARNING, GPUS already set in env: CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES")}')


def parallel_init(cfg):
    cfg.PARALLEL.LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    cfg.PARALLEL.IS_MASTER = cfg.PARALLEL.LOCAL_RANK == 0
    torch.cuda.set_device(cfg.PARALLEL.LOCAL_RANK)
    if cfg.PARALLEL.DDP:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cfg.PARALLEL.WORLD_SIZE = torch.distributed.get_world_size()
    else:
        cfg.PARALLEL.WORLD_SIZE = 1
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    MAX_SEED = 1000000
    seed = torch.randint(0,MAX_SEED, (1,)).cuda()
    if cfg.PARALLEL.DDP: torch.distributed.broadcast(seed, 0)
    cfg.TRAIN.SEED = int(seed.cpu())


def copy_src(cfg, src, dst):
    if cfg.PARALLEL.LOCAL_RANK == 0:
        shutil.copytree(src / 'src', str(dst / 'src'))
        #if os.path.exists('/mnt/src'): shutil.copytree('/mnt/src', str(output_folder/'src'))


def warnings_init(cfg):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    if not cfg.PARALLEL.IS_MASTER:
        warnings.filterwarnings("ignore")
    logger.warning('ALL WARNING MESSAGES ON NON-MASTER PROCESSES ARE SUPPRESSED')


@hydra.main(config_path="configs", config_name="u", version_base=None)
def main(cfg):
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    # return
    set_gpus(cfg)
    parallel_init(cfg)
    warnings_init(cfg)

    src_folder = Path(os.getcwd())
    output_folder = Path(HydraConfig.get().run.dir)
    copy_src(cfg, src=src_folder, dst=output_folder)
    log_init(cfg, output_folder)
    start(cfg, output_folder)


if __name__ == "__main__":
    """
    config select : --config-name=**name** / -cn=**name**
        python3 src/main.py
        python3 src/main.py -cn base

        DDP:
        python3 -m torch.distributed.launch --use_env --nproc_per_node=4 src/main.py -cn=gleb
        python3 -m torch.distributed.launch --use_env --nproc_per_node=4 src/main.py
    """
    cfg_init()
    main()
