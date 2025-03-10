# Pranav Minasandra
# pminasandra.github.io
# 10 Mar 2025

import glob
import os
import os.path
from os.path import join as joinpaths

import pandas as pd

import config
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint

FEATURES = config.FEATURES_DIR

def load_feature_file(filepath):
    """
    Loads extracted features
    """

    df = pd.read_csv(filepath)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    return df

def load_features():
    """
    *GENERATOR* Loads all feature files
    """

    for f in glob.glob(joinpaths(FEATURES, "*_extracted_features.csv")):
        fname = os.path.basename(f)
        df = load_feature_file(f)

        yield fname, df
