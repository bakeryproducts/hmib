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

        train_imgs = DATA_DIR / cfg.DATA.train_imgs
        train_anns = DATA_DIR / cfg.DATA.train_anns

        valid_imgs = DATA_DIR / cfg.DATA.valid_imgs
        valid_anns = DATA_DIR / cfg.DATA.valid_anns

        base_path = DATA_DIR / 'hmib/train.csv'
        img_loader = eval(cfg.DATA.img_loader)
        ann_loader = eval(cfg.DATA.ann_loader)

        ext_train = partial(data.MainDataset,
                            cfg=cfg,
                            root=train_imgs,
                            ann_path=train_anns,
                            base_path=base_path,
                            ImgLoader=img_loader,
                            AnnLoader=ann_loader,
                            train=True,
                            **train_ds_args)

        ext_val = partial(data.MainDataset,
                          cfg=cfg,
                          root=valid_imgs,
                          ann_path=valid_anns,
                          base_path=base_path,
                          ImgLoader=img_loader,
                          AnnLoader=ann_loader,
                          train=False,
                          **val_ds_args)

        ext_val_hkid = partial(data.ExtraValDataset,
                               cfg=cfg,
                               root=DATA_DIR / 'extra/hubmap_kidney/preprocessed/SPLITS/2.131_1024/S0/train/images',
                               ann_path=DATA_DIR / 'extra/hubmap_kidney/preprocessed/SPLITS/2.131_1024/S0/train/masks',
                               base_path=base_path,
                               ImgLoader=img_loader,
                               AnnLoader=ann_loader,
                               organ='kidney',
                               **val_ds_args)

        ext_val_hcol = partial(data.ExtraValDataset,
                               cfg=cfg,
                               root=DATA_DIR / 'extra/hubmap_colon/preprocessed/CUTS/2.344_1024/images',
                               ann_path=DATA_DIR / 'extra/hubmap_colon/preprocessed/CUTS/2.344_1024/masks',
                               base_path=base_path,
                               ImgLoader=img_loader,
                               AnnLoader=ann_loader,
                               organ='largeintestine',
                               **val_ds_args)

        ext_val_gcol = partial(data.ExtraValDataset,
                               cfg=cfg,
                               root=DATA_DIR / 'extra/gtex/CUTS/colon/preprocessed/CUTS/2.344_1024/images',
                               ann_path=DATA_DIR / 'extra/gtex/CUTS/colon/preprocessed/CUTS/2.344_1024/masks',
                               base_path=base_path,
                               ImgLoader=img_loader,
                               AnnLoader=ann_loader,
                               organ='largeintestine',
                               **val_ds_args)
        ext_val_gspl = partial(data.ExtraValDataset,
                               cfg=cfg,
                               root=DATA_DIR / 'extra/gtex/CUTS/spleen/preprocessed/CUTS/2.344_1024/images',
                               ann_path=DATA_DIR / 'extra/gtex/CUTS/spleen/preprocessed/CUTS/2.344_1024/masks',
                               base_path=base_path,
                               ImgLoader=img_loader,
                               AnnLoader=ann_loader,
                               organ='spleen',
                               **val_ds_args)
        ext_val_gpro = partial(data.ExtraValDataset,
                               cfg=cfg,
                               root=DATA_DIR / 'extra/gtex/CUTS/prostate/preprocessed/CUTS/2.344_1024/images',
                               ann_path=DATA_DIR / 'extra/gtex/CUTS/prostate/preprocessed/CUTS/2.344_1024/masks',
                               base_path=base_path,
                               ImgLoader=img_loader,
                               AnnLoader=ann_loader,
                               organ='prostate',
                               **val_ds_args)

        split_path = Path(DATA_DIR / 'splits')
        # (t0, v0), (t1,v1),(t2,v2),(t3,v3) = [[f'train_{i}.csv', f'valid_{i}.csv'] for i in [0,1,2,3]]
        f0, f1, f2, f3 = [split_path / f'{i}.csv' for i in [0,1,2,3]]

        #s0
        train_0 = [split_path / p for p in [f1, f2, f3]]
        valid_0 = [split_path / p for p in [f0]]

        train_1 = [split_path / p for p in [f0, f2, f3]]
        valid_1 = [split_path / p for p in [f1]]

        train_2 = [split_path / p for p in [f1, f0, f3]]
        valid_2 = [split_path / p for p in [f2]]

        train_3 = [split_path / p for p in [f1, f2, f0]]
        valid_3 = [split_path / p for p in [f3]]


        self.dataset_args = dict(
            train_0=dict(ds=ext_train, kwargs={'index_paths':train_0}),
            valid_0=dict(ds=ext_val,   kwargs={'index_paths':valid_0,}),

            train_1=dict(ds=ext_train, kwargs={'index_paths':train_1}),
            valid_1=dict(ds=ext_val,   kwargs={'index_paths':valid_1,}),

            train_2=dict(ds=ext_train, kwargs={'index_paths':train_2}),
            valid_2=dict(ds=ext_val,   kwargs={'index_paths':valid_2,}),

            train_3=dict(ds=ext_train, kwargs={'index_paths':train_3}),
            valid_3=dict(ds=ext_val,   kwargs={'index_paths':valid_3,}),

            hkid=dict(ds=ext_val_hkid),
            hcol=dict(ds=ext_val_hcol),

            gcol=dict(ds=ext_val_gcol),
            gspl=dict(ds=ext_val_gspl),
            gpro=dict(ds=ext_val_gpro),

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
            kwargs['drop_last'] = False
            # kwargs['sampler'] = None
            kwargs['sampler'] = DistributedSampler(dataset,
                                                   num_replicas=cfg.PARALLEL.WORLD_SIZE,
                                                   rank=cfg.PARALLEL.LOCAL_RANK,
                                                   shuffle=False,
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
