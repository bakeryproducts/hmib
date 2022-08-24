from pathlib import Path
import random

from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIGenericIterator


def random_flip(images, probability, seed):
    flip = fn.random.coin_flip(probability=1-probability, seed=seed)
    return fn.flip(images, horizontal=flip)


@pipeline_def
def segmentation_pipe(input_files, crop_size):
    image_files, mask_files = input_files
    seed = random.randint(1, 0x7fffffff)  # get a common seed
    device = 'mixed'
    random.Random(seed).shuffle(image_files)
    random.Random(seed).shuffle(mask_files)

    enc_images, _ = fn.readers.file(files=image_files, seed=seed)
    enc_masks, _ = fn.readers.file(files=mask_files, seed=seed)
    images = fn.decoders.image(enc_images, device=device)
    masks = fn.decoders.image(enc_masks, device=device)

    # crop = ops.Crop(crop=crop_size, output_dtype=types.FLOAT)
    # uniform = ops.random.Uniform(range=(0.0, 1.0), seed=seed)

    # images = crop(images, crop_pos_x=uniform(), crop_pos_y=uniform(), seed=seed)
    # masks = crop(masks, crop_pos_x=uniform(), crop_pos_y=uniform(), seed=seed)

    #return images, masks

    area_min = area_max = 192/512
    images = fn.random_resized_crop(
        images,
        size=crop_size,
        random_area=[area_min, area_max],
        random_aspect_ratio=[0.9, 1.1],
        seed=seed)  # the seed
    masks = fn.random_resized_crop(
        masks,
        size=crop_size,
        interp_type = types.INTERP_NN,
        random_area=[area_min, area_max],
        random_aspect_ratio=[0.9, 1.1],
        seed=seed)  # same seed

    # images = random_flip(images, .5, seed)
    # masks = random_flip(masks, .5, seed)

    return images, masks


def read_names(ds):
    l = set()
    ii, mm = [], []
    for i in ds:
        ifn = i['x']
        mfn = i['y']
        name = Path(ifn).stem
        # if name not in l:
        ii.append(ifn)
        mm.append(mfn)
        # else:
        #     break
        l.add(name)
    return ii, mm


def build_dataloaders(cfg, datasets, **all_kwargs):
    dls = {}
    for kind, dataset in datasets.items():
        train = kind == 'TRAIN'

        imgs, masks = read_names(dataset)
        device_id = cfg.PARALLEL.LOCAL_RANK
        num_threads = cfg.TRAIN.NUM_WORKERS
        batch_size = cfg[kind]['BATCH_SIZE']
        crop_size = cfg.AUGS.CROP[0] if train else cfg.AUGS.VALCROP[0]

        pipe = segmentation_pipe((imgs, masks),
                                 crop_size=crop_size,
                                 batch_size=batch_size,
                                 num_threads=num_threads,
                                 device_id=device_id,
                                 )
        loader = DALIGenericIterator(pipe, ['x', 'y'], auto_reset=True, size=len(imgs))

        # kwargs = all_kwargs.copy()
        # kwargs['num_workers'] = cfg.TRAIN.NUM_WORKERS

        # if kind == 'VALID':
        # elif kind == 'TRAIN':


        dls[kind] = loader

    return dls
