# Pranav Minasandra
# pminasandra.github.io
# Nov 29, 2023
# Ported to Hyrax Project from meerkat-acc on Mar 08, 2025

"""
Extract features from ACC data
"""

import os.path

import pandas as pd

import accreading
import config
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint


ALL_FEATURES = {}
def feature(f):
    """
    DECORATOR!!
    Adds functions to a global list.
    Function f must run on a pandas frame view with
    columns ['Timestamp', 'X', 'Y', 'Z'] and must return
    a float.
    """

    global ALL_FEATURES
    assert f.__name__.startswith('_')

    ALL_FEATURES[f.__name__[1:]] = f
    return f


def each_unit_of_data(acc_df, unit='1s'):
    """
    GENERATOR!!
    """

    accreading.validate_acc_file(acc_df)

    acc_df['TimestampRounded'] = acc_df['Timestamp'].dt.round(unit)
    for time, frame in acc_df.groupby("TimestampRounded"):
        yield time, frame


##### FEATURES TO BE USED START HERE
# NOTE: Start all these functions with '_'

@feature
def _xmean(frame):
    return frame.X.mean()

@feature
def _ymean(frame):
    return frame.Y.mean()

@feature
def _zmean(frame):
    return frame.Z.mean()


@feature
def _xvar(frame):
    return frame.X.var()

@feature
def _yvar(frame):
    return frame.Y.var()

@feature
def _zvar(frame):
    return frame.Z.var()


@feature
def _xmin(frame):
    return frame.X.min()

@feature
def _ymin(frame):
    return frame.Y.min()

@feature
def _zmin(frame):
    return frame.Z.min()


@feature
def _xmax(frame):
    return frame.X.max()

@feature
def _ymax(frame):
    return frame.Y.max()

@feature
def _zmax(frame):
    return frame.Z.max()


@feature
def _mean_vedba(frame):
    dx = frame.X - frame.X.mean()
    dy = frame.Y - frame.Y.mean()
    dz = frame.Z - frame.Z.mean()

    return ((dx**2 + dy**2 + dz**2 )**0.5).mean()

##### FEATURES END HERE

def make_features_dir(unit='1s'):
    """
    Ensures that there exists a features dir
    """

    os.makedirs(os.path.join(config.DATA, f"Features_{unit}"), exist_ok=True)


def extract_all_features(accfile_generator, unit='1s'):
    """
    Extracts features from all available data
    Args:
        accfile_generator: a generator object, typically output
            from each_second_of_data(...)
    """

    for filename, df in accfile_generator:

        feature_df = {} # Not a df right now, but becomes one later. Minor hack to save mem
        feature_df["Timestamp"] = [] # timestamps will be stored here
        for fname in ALL_FEATURES:
            feature_df[fname] = [] #features will be stored here

        print(f"now working on {filename}.")
        unitwisewise_data_generator = each_unit_of_data(df, unit=unit)

        for time, frame in unitwise_data_generator:
            feature_df['Timestamp'].append(time)
            for fname, ffunc in ALL_FEATURES.items():
                fval = ffunc(frame)
                feature_df[fname].append(fval)

        tgtfilename = filename + "_extracted_features.csv"
        tgtfilepath = os.path.join(config.DATA, f"Features_{unit}", tgtfilename)

        feature_df = pd.DataFrame(feature_df)
        feature_df.to_csv(tgtfilepath, index=False)

if __name__ == "__main__":

    for tscale in config.timescales:
        make_features_dir(unit=tscale)

        accfilegen = accreading.load_acc_files()
        extract_all_features(accfilegen) #highly inefficient, data loaded many times. but meh.
