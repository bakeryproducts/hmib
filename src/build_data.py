from pathlib import Path
from functools import partial

from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.data.dataset import ConcatDataset

import data


def get_by_name(path, glob, filter):
    return [p for p in path.rglob(glob) if filter(p)]


class DatasetsGen:
    def __init__(self, cfg):
        self.cfg = cfg
        DATA_DIR = Path(cfg.INPUTS).absolute()
        if not DATA_DIR.exists(): raise Exception(DATA_DIR)

        img_loader = eval(cfg.DATA.img_loader)
        ann_loader = eval(cfg.DATA.ann_loader)
        general_args = dict(cfg=cfg, ImgLoader=img_loader, AnnLoader=ann_loader,)

        self.dataset_args = dict()
        for dataset_cfg in cfg.DATA.datasets:
            dataset_cfg = dict(dataset_cfg)
            name = dataset_cfg.pop('name')
            organ = name.split('_')[-1]
            data_source = name.split('_')[0]
            root = DATA_DIR / dataset_cfg.pop('root')
            base_dataset = data.MainDatasetv2 # can pop from cfg

            args = dict(
                root=root/'images',
                ann_path=root/'masks',
                organ=organ,
                data_source=data_source,
                **general_args,
                **dataset_cfg,
            )
            ds = partial(base_dataset, **args)
            self.dataset_args[name] = dict(ds=ds)


    def generate_by_key(self, key):
        # well, initialization of Datast object takes time, so...
        assert key in self.dataset_args, (key, self.dataset_args.keys())
        dsa = self.dataset_args[key]
        ds = dsa['ds']
        args = dsa.get('args', ())
        kwargs = dsa.get('kwargs', {})
        return ds(*args, **kwargs)

    def __getitem__(self, key):
        return self.generate_by_key(key)


def init_datasets(cfg, dataset_generator, dataset_types):
    # dataset_types : ['TRAIN', 'VALID']
    # dataset_generator: InitDatasets
    converted_datasets = {}
    for dataset_type in dataset_types:
        data_field = cfg.DATA[dataset_type]
        datasets = [dataset_generator[ds] for ds in data_field.DATASETS]
        ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
        converted_datasets[dataset_type] = ds
    return converted_datasets


def build_dataloaders(cfg, datasets, **all_kwargs):
    dls = {}
    for kind, dataset in datasets.items():
        kwargs = all_kwargs.copy()
        kwargs['num_workers'] = cfg.TRAIN.NUM_WORKERS

        if kind == 'VALID':
            kwargs['batch_size'] = cfg[kind]['BATCH_SIZE']
            kwargs['pin_memory'] = True
            # kwargs['shuffle'] = False
            kwargs['drop_last'] = False
            # kwargs['sampler'] = None
            kwargs['sampler'] = DistributedSampler(dataset,
                                                   num_replicas=cfg.PARALLEL.WORLD_SIZE,
                                                   rank=cfg.PARALLEL.LOCAL_RANK,
                                                   shuffle=True,
                                                   seed=cfg.TRAIN.SEED)
        elif kind == 'VALID_HUB':
            kwargs['batch_size'] = cfg[kind]['BATCH_SIZE']
            kwargs['pin_memory'] = True
            # kwargs['shuffle'] = False
            kwargs['drop_last'] = False
            # kwargs['sampler'] = None
            kwargs['sampler'] = DistributedSampler(dataset,
                                                   num_replicas=cfg.PARALLEL.WORLD_SIZE,
                                                   rank=cfg.PARALLEL.LOCAL_RANK,
                                                   shuffle=True,
                                                   seed=cfg.TRAIN.SEED)
        elif kind == 'VALID_GTEX':
            kwargs['batch_size'] = cfg[kind]['BATCH_SIZE']
            kwargs['pin_memory'] = True
            # kwargs['shuffle'] = False
            kwargs['drop_last'] = False
            # kwargs['sampler'] = None
            kwargs['sampler'] = DistributedSampler(dataset,
                                                   num_replicas=cfg.PARALLEL.WORLD_SIZE,
                                                   rank=cfg.PARALLEL.LOCAL_RANK,
                                                   shuffle=True,
                                                   seed=cfg.TRAIN.SEED)

        elif kind == 'TRAIN':
            kwargs['batch_size'] = cfg[kind]['BATCH_SIZE']
            kwargs['pin_memory'] = True
            # kwargs['shuffle'] = True
            kwargs['drop_last'] = True
            # kwargs['sampler'] = None
            kwargs['sampler'] = DistributedSampler(dataset,
                                                   num_replicas=cfg.PARALLEL.WORLD_SIZE,
                                                   rank=cfg.PARALLEL.LOCAL_RANK,
                                                   shuffle=True,
                                                   seed=cfg.TRAIN.SEED)

        dls[kind] = DataLoader(dataset, **kwargs)
    return dls
