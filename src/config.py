from typing import List
from dataclasses import dataclass, field

import hydra
from omegaconf import OmegaConf

from shallow import configer as shc



@dataclass
class Train(shc.Train):
    START_VAL: int = 1
    BATCH_NUM_INSTANCES: int = 0


@dataclass
class _mixup():
    PROB: float = 0.0
    ALPHA: float = 0.5
    REPEAT: int = 1


@dataclass
class _msr():
    scale_down: float = 0.5
    scale_up: float = 1.5


@dataclass
class _copy_paste():
    PROB: float = 0.0
    ALPHA: float = 0.5
    REPEAT: int = 1
    SCALE: float = 0
    MEAN: float = 1
    MIN_SIZE: int = 20


@dataclass
class _cutmix():
    PROB: float = 0.0
    BETA: float = 0.0
    ALPHA: float = 0.5
    REPEAT: int = 1


@dataclass
class _sam():
    RHO: float = 0.0
    REDUCE: float = 0.5


@dataclass
class _fmix():
    PROB: float = 0.0
    REPEAT: int = 1
    ALPHA: float = 1


def cfg_init():
    nodes = [
        shc._generate_node(group='TRANSFORMERS', name="_transformers", node_class=shc.Transformers),
        shc._generate_node(group='DATA', name="_data", node_class=shc.Data),
        shc._generate_node(group='PARALLEL', name="_parallel", node_class=shc.Parallel),
        shc._generate_node(group='TRAIN', name="_train", node_class=Train),
        shc._generate_node(group='VALID', name="_valid", node_class=shc.Valid),
        shc._generate_node(group='TEST', name="_test", node_class=shc.Test),
        shc._generate_node(group='FEATURES.SAM', name="_sam", node_class=_sam),
        shc._generate_node(group='AUGS.MIXUP', name="_mixup", node_class=_mixup),
    ]
    shc.cfg_init(lambda: nodes)
