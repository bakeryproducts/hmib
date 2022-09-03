import shutil
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import fire


def start(src, dst, split_csv=None, in_split=False, organ=None):
    src = Path(src)
    dst = Path(dst)
    csv = src / 'trv2.csv'
    df = pd.read_csv(str(csv))

    prefix = 'ALL'
    if split_csv is not None:
        split_csv = Path(split_csv)
        ind_df = pd.read_csv(split_csv)
        split_mask = df.index.isin(ind_df.values.flatten())
        if not in_split:
            split_mask = ~split_mask
        df = df[split_mask]
        prefix = 'VALID' if in_split else 'TRAIN'
        dst = dst / f'{prefix}_SPLIT_{split_csv.stem}'

    dst.mkdir(exist_ok=True, parents=True)
    df.to_csv(str(dst / 'data')+'.csv', index=None)
    #shutil.copy(csv, dst)

    imdst = dst / 'train_images'
    imdst.mkdir(parents=True, exist_ok=True)


    for i, row in tqdm(df.iterrows()):
        name = f'{row.id}.tiff'
        if organ is not None:
            if organ != row.organ: continue
        fsrc = src / 'train_images' / name
        fdst = imdst / row.organ
        fdst.mkdir(exist_ok=True)
        shutil.copy(fsrc, fdst)


if __name__ == '__main__':
    fire.Fire(start)
