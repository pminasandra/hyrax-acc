# Pranav Minasandra
# pminasandra.github.io
# Nov 28, 2023
# Yay, happy birthday to me!
# Ported to Hyrax Project from meerkat-acc on Mar 08, 2025

"""
Reads the HUGE csv files containing acc info.
"""

import glob
import os.path

import numpy as np
import pandas as pd

import config
import utilities

dtypes_dict_raw = {
    'Tag ID': str,
    'Timestamp': str,
    'X': np.float64,
    'Y': np.float64,
    'Z': np.float64,
    'Batt. V. (V)': str,
    'Metadata': str
}

if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint

def validate_acc_file(df, subset_cols=True):
    """
    Checks column labels and behavioral states in a given dataframe
    """
    if subset_cols:
        assert list(df.columns) == ["Timestamp", "X", "Y", "Z"] #labels as generated by tag

def load_acc_files(subset_cols=True):
    """
    GENERATOR!!
    Loads and validates dataframes from audit csvs.
    Args:
        list_of_dplments: which deployments to load
        subset_cols: whether to subset to specific cols
    Yields:
        3-tuple: dplment (str), accname (str), and csvfile (pd.DataFrame)
    Raises:
        AssertionError: if there are inappropriate csvfiles
    """
    tgtpath = os.path.join(config.ACC_DIR)
    for csvfilepath in glob.glob(os.path.join(tgtpath, "*.csv")):
        if subset_cols:
            csvfile = pd.read_csv(csvfilepath,
                                    usecols=["Timestamp", "X", "Y", "Z"],
                                    sep=config.ACC_FILE_SEP)
        else:
            try:
                csvfile = pd.read_csv(csvfilepath, 
                                        dtype=dtypes_dict_raw,
                                        sep=config.ACC_FILE_SEP)

            except Exception as e:
                print("dtype didn't work, trying again:", e)
                csvfile = pd.read_csv(csvfilepath)
        csvfile['Timestamp'] = pd.to_datetime(csvfile['Timestamp'],
                                    format='%Y/%m/%d %H:%M:%S.%f')
        validate_acc_file(csvfile, subset_cols=subset_cols)

        yield os.path.basename(csvfilepath)[:-len(".csv")], csvfile


if __name__ == "__main__":
    accgen = load_acc_files()
    for name, df in accgen:
        print("\n", name, "\n", df)
