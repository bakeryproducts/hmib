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

        train_ds_args = dict(self.cfg.DATA.TRAIN.DATASET_ARGS)
        val_ds_args = dict(self.cfg.DATA.VALID.DATASET_ARGS)
        root = DATA_DIR / cfg.DATA.root
        ann_path = DATA_DIR / cfg.DATA.ann_root
        base_path = DATA_DIR / 'hmib/train.csv'
        img_loader = eval(cfg.DATA.img_loader)
        ann_loader = eval(cfg.DATA.ann_loader)

        ext_train = partial(data.MainDataset,
                            cfg=cfg,
                            root=root,
                            ann_path=ann_path,
                            base_path=base_path,
                            ImgLoader=img_loader,
                            AnnLoader=ann_loader,
                            train=True,
                            **train_ds_args)

        ext_val = partial(data.MainDataset,
                          cfg=cfg,
                          root=root,
                          ann_path=ann_path,
                          base_path=base_path,
                          ImgLoader=img_loader,
                          AnnLoader=ann_loader,
                          train=False,
                          **val_ds_args)

        split_path = Path(DATA_DIR / 'splits')
        # (t0, v0), (t1,v1),(t2,v2),(t3,v3) = [[f'train_{i}.csv', f'valid_{i}.csv'] for i in [0,1,2,3]]
        f0, f1, f2, f3 = [split_path / f'{i}.csv' for i in [0,1,2,3]]

        #s0
        train_0 = [split_path / p for p in [f1, f2, f3]]
        valid_0 = [split_path / p for p in [f0]]


        self.dataset_args = dict(
            train_0=dict(ds=ext_train, kwargs={'index_paths':train_0}),
            valid_0=dict(ds=ext_val,   kwargs={'index_paths':valid_0,}),

        )

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
            kwargs['drop_last'] = True
            # kwargs['sampler'] = None
            kwargs['sampler'] = DistributedSampler(dataset,
                                                   num_replicas=cfg.PARALLEL.WORLD_SIZE,
                                                   rank=cfg.PARALLEL.LOCAL_RANK,
                                                   shuffle=False,
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
