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

from tiff import load_tiff


DATA_DIR = Path("/kaggle/input/hubmap-organ-segmentation")

TEST_IMAGES_DIR = DATA_DIR / "test_images"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TEST_CSV_FILE = DATA_DIR / "test.csv"
TRAIN_CSV_FILE = DATA_DIR / "train.csv"
OUTPUT_FILE = "/kaggle/working/submission.csv"
DUMMY_RLE = ""

DEBUG = False

WHITE_THRESH = 240
BLACK_THRSH = 10
RED_CHANNEL = 0  # or 2? Seems that rasterio read in RGB, but better to double check

DATA_SOURCES = ['Hubmap', 'HPA']
ORGANS = ['prostate', 'spleen', 'lung', 'largeintestine', 'kidney']


images_dir = TRAIN_IMAGES_DIR if DEBUG else TEST_IMAGES_DIR
df_file = TRAIN_CSV_FILE if DEBUG else TEST_CSV_FILE

df = pd.read_csv(df_file)

red_means = []
result = []
for row in tqdm(df.itertuples(), total=len(df), desc="Inference"):
    if row.data_source in DATA_SOURCES and row.organ in ORGANS:
        image = load_tiff(images_dir / f"{row.id}.tiff")
        tissue = image[(image > BLACK_THRSH) & (image < WHITE_THRESH)]
        red_means.append(tissue[..., RED_CHANNEL].mean())  # median?

    result.append({
        "id": row.id,
        "rle": DUMMY_RLE,
    })

result = pd.DataFrame(result)
result.to_csv(OUTPUT_FILE, index=False)


# images = [i for i in images if i.ishubmap() and i.organ == 'kidney']
#
# for i in images:
#     tissue = i[(i > BLACK_THRSH) & (i < WHITE_THRESH)]
#     red += tissue[red_channel].mean()# median?

red_mean = np.mean(red_means)

T0, T1, T2, T3 = 180, 80, 110, 130

if red_mean > T0:
    throw E0
elif red_mean > T3:
    throw E3
elif red_mean > T2:
    throw E2
elif red_mean > T1:
    throw E1



