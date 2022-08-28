import os.path as osp
from pathlib import Path

import cv2
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from mask_utils import convex2mask, rle_decode, rle_encode


def read_new_df(new_csv_folder):
    return pd.concat([
        pd.read_csv(csv_file) for csv_file in Path(new_csv_folder).rglob("*.csv")
    ])


def main(ori_csv_file, new_csv_dir, result_csv_file, new_size=1024):
    ori_df = pd.read_csv(ori_csv_file)
    new_df = read_new_df(new_csv_dir)
    result_df = ori_df.copy()

    # Strip extension and convert to int
    new_df.id = new_df.id.apply(lambda filename: int(osp.splitext(filename)[0]))

    # Merge
    image_ids = new_df.id.unique()
    for image_id in tqdm(image_ids, desc=f"Updating rle from {new_csv_dir}"):
        # Check occurence in df
        n_occurrences = (ori_df.id == image_id).sum()
        if n_occurrences == 0:
            raise IndexError(f"No id {image_id} in original csv file {ori_csv_file}")
        if n_occurrences > 1:
            raise IndexError(f"Non unique occurence {image_id} in original csv file {ori_csv_file}")

        # Build original mask
        ori_index = ori_df[ori_df.id == image_id].index[0]
        ori_row = ori_df.loc[ori_index]
        ori_mask = rle_decode(ori_row.rle, (ori_row.img_height, ori_row.img_width))

        # Build new mask
        new_mask = np.zeros((new_size, new_size), dtype=np.uint8)
        for new_row in new_df[new_df.id == image_id].itertuples():
            poly_mask = convex2mask(new_row.convex, (new_size, new_size))
            new_mask = np.maximum(new_mask, poly_mask)

        # Resize new mask to original size
        new_mask = cv2.resize(new_mask, ori_mask.shape[::-1], interpolation=cv2.INTER_CUBIC)

        # Merge masks
        result_mask = np.maximum(ori_mask, new_mask)
        result_df.loc[ori_index, "rle"] = rle_encode(result_mask)

    # Save the result
    result_df.to_csv(result_csv_file, index=False)


if __name__ == "__main__":
    fire.Fire(main)
