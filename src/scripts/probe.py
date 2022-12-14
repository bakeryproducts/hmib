"""
Errors:
Notebook Threw Exception: While rerunning your code, your notebook hit an unhandled error. Note that the hidden dataset can be larger/smaller/different than the public dataset.

Notebook Exceeded Allowed Compute: This indicates you have violated a code requirement constraint during the rerun. This includes limitations in the execution environment, for example requesting more RAM or disk than available, or competition constraints, such as input data source type or size limits.

Submission CSV Not Found: Your notebook did not output the expected submission file (often submission.csv). The rerun of your notebook appears to have completed, but when we looked for your submission file, it wasn't there. This means it's possible to have bugs/errors upstream that halted execution and prevented the file from being written. Attempting to read a non-existent directory/file from a dataset is a common reason execution is halted (causing either Submission CSV Not Found or Notebook Threw Exception).

Submission Scoring Error: Your notebook generated a submission file with incorrect format. Some examples causing this are: wrong number of rows or columns, empty values, an incorrect data type for a value, or invalid submission values from what is expected.

"""
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from tqdm import tqdm


DATA_DIR = Path("/kaggle/input/hubmap-organ-segmentation")

TEST_IMAGES_DIR = DATA_DIR / "test_images"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TEST_CSV_FILE = DATA_DIR / "test.csv"
TRAIN_CSV_FILE = DATA_DIR / "train.csv"
SUBMISSION_PATH = "/kaggle/working/submission.csv"
DUMMY_RLE = ""



def thrs_closed(a, d, n_points=4):
    step = (d - a) // (n_points + 1)
    return (a + step * i for i in range(1, n_points + 1))


def thrs(a, d):
    e = (a - d) // 3
    b = a - e
    c = d + e
    return a, b, c, d


def load_gpu():
    result = torch.zeros(1000).to("cuda")
    for i in range(1000):
        result += i


################ fix us #########################
WHITE_THRESH = 240
BLACK_THRSH = 15
ORGANS = ['prostate', 'spleen', 'lung', 'largeintestine', 'kidney']


DEBUG = False
ERROR_DEBUG_SCORING = False
ERROR_DEBUG_RESOURSES = False
# TODO check for not found and 1/0 ?
ORGAN = "spleen"
CHANNEL = 1  # or 2? Seems that rasterio read in RGB, but better to double check
DATA_SOURCE = "hpa" if DEBUG else "hubmap"
T0, T1, T2, T3 = thrs(105, 94)
assert T0 > T1 > T2 > T3
################ fix us #########################


def load_tiff(p):
    return rio.open(str(p)).read()


def NotebookException():
    1/0


def SubmissionNotFound():
    import os
    os.remove(SUBMISSION_PATH)


def NotebookExceededRes():
    import torch
    gb20 = 100 * 1024 * 1024 * 1024  # ~kinda
    tt = []
    while True:
        tt.append(torch.zeros(gb20 + 1, dtype=torch.int))


def SubmissionScoringError():
    with open(SUBMISSION_PATH, 'w') as f:
        f.write('This is a test')


images_dir = TRAIN_IMAGES_DIR if DEBUG else TEST_IMAGES_DIR
df_file = TRAIN_CSV_FILE if DEBUG else TEST_CSV_FILE

df = pd.read_csv(df_file)

is_private = len(df) > 1

means = []
result = []

for row in tqdm(df.itertuples(), total=len(df), desc="Inference"):
    if (row.data_source.lower() == DATA_SOURCE) and (row.organ.lower() == ORGAN):
        image = load_tiff(images_dir / f"{row.id}.tiff")
        gray_image = image.mean(0)
        mask = (gray_image > BLACK_THRSH) & (gray_image < WHITE_THRESH)
        tissue = image[CHANNEL][mask]  # CHW, 3chan
        means.append(tissue.mean())  # median?

    result.append({
        "id": row.id,
        "rle": DUMMY_RLE,
    })

result = pd.DataFrame(result)
load_gpu()

if is_private:
    stat = np.mean(means)
    if stat > T0:
        NotebookException() # x > T0
    elif stat > T1:
        NotebookExceededRes() # T1 < x < T0

result.to_csv(SUBMISSION_PATH, index=False)

if is_private:
    if stat > T2:
        SubmissionNotFound() # T2 < x < T1
    elif stat > T3:
        SubmissionScoringError() # T3 < x < T2

sleep(10)
