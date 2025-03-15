# Pranav Minasandra
# pminasandra.github.io
# Nov 30, 2023
# Ported to hyrax analyses from meerkat-acc on 15 Mar 2025

"""
Implement random forest classifier
"""

import glob
import os.path

import pandas as pd
import sklearn.ensemble

import auditreading
import config
import dataloading
import extractor
import utilities

ALL_FEATURES = list(extractor.ALL_FEATURES.keys())

if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint


def load_feature_data_for(individual):
    """
    Loads all available feature data for given deployment and individual.
    Args:
        individual (str), e.g., "JN" or "TD"
    Returns:
        pd.DataFrame
    Raises:
        AssertionError
    """

    assert individual in config.INDIVIDUALS

    tgtdirpath = os.path.join(config.DATA, "Features")
    tgtfiles = glob.glob(os.path.join(tgtdirpath, f"Axy*_{individual}_*.csv"))
    if len(tgtfiles) == 0:
        return pd.DataFrame()

    all_data = [dataloading.load_feature_file(file_) for file_ in tgtfiles]
    all_data = pd.concat(all_data)
# above concatenation is redundant, only one file per ind ideally

    all_data.sort_values(by='Timestamp', inplace=True)
    all_data.reset_index(inplace=True)

    return all_data


def load_all_training_data():
    """
    Loads all available feature and audit data to RAM.
    Returns: pd.DataFrame
    """

    all_training_data = []

    for fname, ft_data in dataloading.load_features():
        ind = fname.split("_")[1]
        audit_data = auditreading.load_audit_data_for(ind)

        cols_needed = list(ft_data.columns)
        cols_needed.append("behaviour_class")
        training_data = ft_data.join(audit_data.set_index('Timestamp'),
                                on='Timestamp',
                                how='inner',
                                lsuffix="_l",
                                rsuffix="_r"
                                )

        del ft_data
        del audit_data

        training_data = training_data[cols_needed]
        training_data.sort_values(by='Timestamp')
        training_data.reset_index(inplace=True)

        training_data['Individual'] = ind

        all_training_data.append(training_data)

    all_training_data = pd.concat(all_training_data)
    all_training_data.reset_index(inplace=True)
    return all_training_data


def train_random_forest(train_features, train_classes):
    """
    Creates and trains a random forest classifier.
    Args:
        train_features (list, np.ndarray, or pd.DataFrame)
        train_classes (list, np.ndarray, or pd.DataFrame)
    Returns:
        sklearn.ensemble.RandomForestClassifier
    """
    rfc = sklearn.ensemble.RandomForestClassifier(
            class_weight="balanced"
            )
    rfc.fit(train_features, train_classes)
    return rfc


if __name__=="__main__":
    data = load_all_training_data()
    data = data[["Timestamp", "Individual"]
                    + ALL_FEATURES + ["behaviour_class"]]
    os.makedirs(os.path.join(config.DATA, "ClassifierRelated"), exist_ok=True)
    data.to_csv(os.path.join(config.DATA,
            "ClassifierRelated",
            "all_trainable_data_for_classifier.csv"),
        index=False) 
