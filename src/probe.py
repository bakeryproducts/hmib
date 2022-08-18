"""
Errors:
Notebook Threw Exception: While rerunning your code, your notebook hit an unhandled error. Note that the hidden dataset can be larger/smaller/different than the public dataset.

Notebook Exceeded Allowed Compute: This indicates you have violated a code requirement constraint during the rerun. This includes limitations in the execution environment, for example requesting more RAM or disk than available, or competition constraints, such as input data source type or size limits.

Submission CSV Not Found: Your notebook did not output the expected submission file (often submission.csv). The rerun of your notebook appears to have completed, but when we looked for your submission file, it wasn't there. This means it's possible to have bugs/errors upstream that halted execution and prevented the file from being written. Attempting to read a non-existent directory/file from a dataset is a common reason execution is halted (causing either Submission CSV Not Found or Notebook Threw Exception).

Submission Scoring Error: Your notebook generated a submission file with incorrect format. Some examples causing this are: wrong number of rows or columns, empty values, an incorrect data type for a value, or invalid submission values from what is expected.

"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
from tqdm import tqdm


DATA_DIR = Path("/kaggle/input/hubmap-organ-segmentation")

TEST_IMAGES_DIR = DATA_DIR / "test_images"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TEST_CSV_FILE = DATA_DIR / "test.csv"
TRAIN_CSV_FILE = DATA_DIR / "train.csv"
SUBMISSION_PATH = "/kaggle/working/submission.csv"
DUMMY_RLE = ""

WHITE_THRESH = 230
BLACK_THRSH = 20
RED_CHANNEL = 0  # or 2? Seems that rasterio read in RGB, but better to double check
ORGANS = ['prostate', 'spleen', 'lung', 'largeintestine', 'kidney']



################ fix us #########################
DEBUG = False
ERROR_DEBUG_SCORING = False
ERROR_DEBUG_RESOURSES = False
# TODO check for not found and 1/0 ?
ORGAN = "kidney"
DATA_SOURCE = "HPA" if DEBUG else "Hubmap"
T0, T1, T2, T3 = 180, 130, 110, 80
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

means = []
result = []

cnt = 0
for row in tqdm(df.itertuples(), total=len(df), desc="Inference"):
    if row.data_source == DATA_SOURCE and row.organ == ORGAN:
        image = load_tiff(images_dir / f"{row.id}.tiff")
        gray_image = image.mean(0)
        mask = (gray_image > BLACK_THRSH) & (gray_image < WHITE_THRESH)
        tissue = image[RED_CHANNEL][mask]  # CHW, 3chan
        means.append(tissue.mean())  # median?

    if ERROR_DEBUG_RESOURSES and cnt > 5:
        # cnt to skip through saving with single test image
        NotebookExceededRes()

    cnt += 1
    result.append({
        "id": row.id,
        "rle": DUMMY_RLE,
    })

result = pd.DataFrame(result)
result.to_csv(SUBMISSION_PATH, index=False)

if ERROR_DEBUG_SCORING:
    if cnt > 5:
        # cnt to skip through saving with single test image
        SubmissionScoringError()

else:
    # can throw another exception, while we test for scoring, thou else
    stat = np.mean(means)

    if stat > T0:
        NotebookException() # x > T0
    elif stat > T1:
        NotebookExceededRes() # T1 < x < T0
    elif stat > T2:
        SubmissionNotFound() # T2 < x < T1
    elif stat > T3:
        SubmissionScoringError() # T3 < x < T2
    else:
        pass # x < T3
