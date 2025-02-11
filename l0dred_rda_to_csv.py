import argparse
import os.path as op
from glob import glob
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", default="result_files", help="path with the csv result files"
    )
    return parser.parse_args()


# more expensive way to read as it creates dataframes:
# pandas2ri.PandasDataFrame(rdf_List[1][0][3]

# level0 predictions. 1 age prediction per parcel for each subject
# Here, 873 values per subject


def get_level0_pred(file):
    pandas2ri.activate()

    base = importr("base")
    base.load(file)
    rdf_List = base.mget(base.ls())
    try:
        data = np.hstack(
            (
                robjects.conversion.rpy2py(rdf_List[1][0][3]),
                robjects.conversion.rpy2py(rdf_List[1][0][4]).reshape(-1, 1),
            )
        )
    except IndexError:
        data = np.hstack(
            (
                robjects.conversion.rpy2py(rdf_List[0][0][3]),
                robjects.conversion.rpy2py(rdf_List[0][0][4]).reshape(-1, 1),
            )
        )

    base.rm(list=base.ls())
    return data


new_data = []
files = glob("/home/antogeo/codes/avg_vs_pred_stack/models/cache_level1" + "/*.rda")

for file in files:
    new_data = get_level0_pred(file)

    # get datasets used and add them in a column
    strip_name = op.splitext(op.basename(file))[0]
    train_sets = strip_name.split("_")[-1]

    new_data = np.hstack(
        (new_data, np.repeat(train_sets, new_data.shape[0]).reshape(-1, 1))
    )
    if "data" not in dir():
        data = np.empty([0, new_data.shape[1]])
    data = np.vstack((data, new_data))

column_names = [*range(1, data.shape[1] - 1), "age", "train"]
df = pd.DataFrame(data, columns=column_names)
df.to_csv(op.join("result_files", "analysis", "level0_pred.csv"))
