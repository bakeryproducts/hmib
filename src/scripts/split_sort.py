from pathlib import Path

import fire
import shutil


def copy_files(files, dst):
    for f in files:
        d = (dst / f.parent.name)
        d.mkdir(exist_ok=True)
        shutil.copy(str(f), str(d))


def start(src, dst, postfix=None):
    """
    read from structure:
        src
            split1
                images
                masks
            split2
                images
                masks
            ...
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(exist_ok=True)

    splits = []
    # for split in src.glob("*"):
    for i in range(4):
        split = src / f'v{i}'
        scale_folder = [f for f in split.glob("*") if f.name != 'bigmasks'][0]
        imgs_folder = scale_folder / 'images'
        mask_folder = scale_folder / 'masks'
        imgs = list(imgs_folder.glob("*"))
        masks = list(mask_folder.glob("*"))
        splits.append([imgs, masks])


    organ = src.name
    train = lambda x: f'f_TRAIN_SPLIT_{x}'
    valid = lambda x: f'f_VALID_SPLIT_{x}'


    for IDX in range(4): # 4 split
        #print(splits[IDX])
        d = dst / valid(IDX) / organ # dst / f_TRAIN_SPLIT_0 / lung_3 /
        if postfix is not None: d = d.parent / (d.name + postfix)
        d.mkdir(exist_ok=True, parents=True)
        [copy_files(s, d) for s in splits[IDX]] # d / images; d / masks

        d = dst / train(IDX) / organ
        if postfix is not None: d = d.parent / (d.name + postfix)
        d.mkdir(exist_ok=True, parents=True)
        for SIDX in range(4):
            if SIDX == IDX: continue
            [copy_files(s, d) for s in splits[SIDX]]


if __name__ == '__main__':
    fire.Fire(start)
