import os.path as osp

import cv2
import fire
import numpy as np
import pandas as pd

from mask_utils import convex2mask, rle_decode, rle_encode


def main(ori_csv_file, new_csv_file, result_csv_file, ori_size=3000, new_size=1024):
    ori_df = pd.read_csv(ori_csv_file)
    new_df = pd.read_csv(new_csv_file)
    result_df = ori_df.copy()

    # Strip extension and convert to int
    new_df.id = new_df.id.apply(lambda filename: int(osp.splitext(filename)[0]))

    # Merge
    for new_row in new_df.itertuples():
        # Check occurence in df
        n_occurrences = (ori_df.id == new_row.id).sum()
        if n_occurrences == 0:
            raise IndexError(f"No id {new_row.id} in original csv file {ori_csv_file}")
        if n_occurrences > 1:
            raise IndexError(f"Non unique occurence {new_row.id} in original csv file {ori_csv_file}")

        ori_index = ori_df[ori_df.id == new_row.id].index[0]
        ori_row = ori_df.loc[ori_index]

        # Build masks
        ori_mask = rle_decode(ori_row.rle, (ori_size, ori_size))
        new_mask = convex2mask(new_row.convex, (new_size, new_size))

        if new_mask.shape != ori_mask.shape:
            new_mask = cv2.resize(new_mask, ori_mask.shape, interpolation=cv2.INTER_CUBIC)

        # Merge masks
        result_mask = np.maximum(ori_mask, new_mask)
        result_df.loc[ori_index, "rle"] = rle_encode(result_mask)

    # Save the result
    result_df.to_csv(result_csv_file, index=False)


if __name__ == "__main__":
    fire.Fire(main)
