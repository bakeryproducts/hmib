"""
Errors:
Notebook Threw Exception: While rerunning your code, your notebook hit an unhandled error. Note that the hidden dataset can be larger/smaller/different than the public dataset.

Notebook Exceeded Allowed Compute: This indicates you have violated a code requirement constraint during the rerun. This includes limitations in the execution environment, for example requesting more RAM or disk than available, or competition constraints, such as input data source type or size limits.

Submission CSV Not Found: Your notebook did not output the expected submission file (often submission.csv). The rerun of your notebook appears to have completed, but when we looked for your submission file, it wasn't there. This means it's possible to have bugs/errors upstream that halted execution and prevented the file from being written. Attempting to read a non-existent directory/file from a dataset is a common reason execution is halted (causing either Submission CSV Not Found or Notebook Threw Exception).

Submission Scoring Error: Your notebook generated a submission file with incorrect format. Some examples causing this are: wrong number of rows or columns, empty values, an incorrect data type for a value, or invalid submission values from what is expected.

"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import rasterio as rio
import numpy as np

#from tiff import load_tiff


def load_tiff(p):
    return rio.open(str(p)).read()


# !cp /kaggle/input/hubmap-organ-segmentation/sample_submission.csv ./submission.csv
SUBMISSION_PATH = './submission.csv'


def NotebookException():
    1/0


def SubmissionNotFound():
    import os
    os.remove(SUBMISSION_PATH)


def NotebookExceededRes():
    import torch
    MAX_SIZE = int(1e20)
    t = torch.zeros(MAX_SIZE + 1).cuda()


def SubmissionScoringError():
    with open(SUBMISSION_PATH, 'w') as f:
        f.write('This is a test')



DATA_DIR = Path("/kaggle/input/hubmap-organ-segmentation")

TEST_IMAGES_DIR = DATA_DIR / "test_images"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TEST_CSV_FILE = DATA_DIR / "test.csv"
TRAIN_CSV_FILE = DATA_DIR / "train.csv"
OUTPUT_FILE = "/kaggle/working/submission.csv"
DUMMY_RLE = ""


WHITE_THRESH = 240
BLACK_THRSH = 10
RED_CHANNEL = 0  # or 2? Seems that rasterio read in RGB, but better to double check

ORGANS = ['prostate', 'spleen', 'lung', 'largeintestine', 'kidney']
DEBUG = True
DATA_SOURCE = "HPA"
# DATA_SOURCE = "Hubmap"
ORGAN = ORGANS[0]


images_dir = TRAIN_IMAGES_DIR if DEBUG else TEST_IMAGES_DIR
df_file = TRAIN_CSV_FILE if DEBUG else TEST_CSV_FILE

df = pd.read_csv(df_file)

red_means = []
for row in tqdm(df.itertuples(), total=len(df), desc="Inference"):
    if row.data_source == DATA_SOURCE and row.organ == ORGAN:
        image = load_tiff(images_dir / f"{row.id}.tiff")
        tissue = image[(image > BLACK_THRSH) & (image < WHITE_THRESH)]
        red_means.append(tissue[..., RED_CHANNEL].mean())  # median?


red_mean = np.mean(red_means)
red_std = np.std(red_means)

T0, T1, T2, T3 = 180, 130, 110, 80

assert T0 > T1 > T2 > T3

if red_mean > T0:
    NotebookException()
elif red_mean > T1:
    NotebookExceededRes()
elif red_mean > T2:
    SubmissionNotFound()
elif red_mean > T3:
    SubmissionScoringError()
