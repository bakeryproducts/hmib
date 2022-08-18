"""
Errors:
Notebook Threw Exception: While rerunning your code, your notebook hit an unhandled error. Note that the hidden dataset can be larger/smaller/different than the public dataset.

Notebook Exceeded Allowed Compute: This indicates you have violated a code requirement constraint during the rerun. This includes limitations in the execution environment, for example requesting more RAM or disk than available, or competition constraints, such as input data source type or size limits.

Submission CSV Not Found: Your notebook did not output the expected submission file (often submission.csv). The rerun of your notebook appears to have completed, but when we looked for your submission file, it wasn't there. This means it's possible to have bugs/errors upstream that halted execution and prevented the file from being written. Attempting to read a non-existent directory/file from a dataset is a common reason execution is halted (causing either Submission CSV Not Found or Notebook Threw Exception).

Submission Scoring Error: Your notebook generated a submission file with incorrect format. Some examples causing this are: wrong number of rows or columns, empty values, an incorrect data type for a value, or invalid submission values from what is expected.

"""

WHITE_THRESH = 240
BLACK_THRSH = 10

for i in images:
    tissue = i[(i > BLACK_THRSH) & (i < WHITE_THRESH)]
    red += tissue[red_channel].mean()# median?

red_mean = [N]

T0, T1, T2, T3 = 180, 80, 110, 130

if red_mean > T0:
    throw E0
elif red_mean > T3:
    throw E3
elif red_mean > T2:
    throw E2
elif red_mean > T1:
    throw E3



