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
ALL_FEATURES = [f'{ft}_{tscale}s' for ft in ALL_FEATURES for tscale in config.timescales]

if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint


def load_feature_data_for(individual, timescales=config.timescales):
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
    ft_dfs = {}

    for tscale in timescales:
        tgtdirpath = os.path.join(config.DATA, f"Features_{tscale}s")
        tgtfiles = glob.glob(os.path.join(tgtdirpath, f"Axy*_{individual}_*.csv"))
        if len(tgtfiles) == 0:
            ft_dfs[tscale] = pd.DataFrame()
            continue

        all_data = [dataloading.load_feature_file(file_) for file_ in tgtfiles]
        all_data = pd.concat(all_data)
# above concatenation is redundant, only one file per ind ideally

        all_data.sort_values(by='Timestamp', inplace=True)
#        all_data.reset_index(inplace=True)
        

        ft_dfs[tscale] = all_data

    for tscale in ft_dfs:
        ft_dfs[tscale] = ft_dfs[tscale].set_index('Timestamp')
    df = pd.concat(ft_dfs, axis=1, join='inner')

    df.columns = [f'{col}_{tscale}s' for tscale, col in df.columns]
    df = df.reset_index()
    print(df)

    return df



def load_all_training_data():
    """
    Loads all available feature and audit data to RAM.
    Returns: pd.DataFrame
    """

    all_training_data = []
    audit_data_cache = {}

    for ind in config.INDIVIDUALS:
        audit_data = auditreading.load_audit_data_for(ind)
        if audit_data.empty:
            continue

        ft_data = load_feature_data_for(ind)

#    for fname, tscale, ft_data in dataloading.load_features():
#        ind = fname.split("_")[1]
#        if ind not in audit_data_cache:
#            audit_data = auditreading.load_audit_data_for(ind)
#            audit_data_cache[ind] = audit_data
#        else:
#            audit_data = audit_data_cache[ind]
#
        cols_needed = list(ft_data.columns)
        cols_needed.append("behaviour_class")
        training_data = ft_data.join(audit_data.set_index('Timestamp'),
                                on='Timestamp',
                                how='inner',
                                lsuffix="_l",
                                rsuffix="_r"
                                )
#
#        del ft_data
#        del audit_data
#
        training_data = training_data[cols_needed]
        training_data.sort_values(by='Timestamp')
        training_data.reset_index(inplace=True)

        training_data.loc[:, 'Individual'] = ind

        all_training_data.append(training_data)

    all_training_data = pd.concat(all_training_data)
    all_training_data.reset_index(inplace=True)
    print(all_training_data)
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
    data = data[~data["behaviour_class"].isin(config.DROP_BEHS)]
    data["behaviour_class"] = data["behaviour_class"].replace(config.MAP_BEHS)
    os.makedirs(os.path.join(config.DATA, "ClassifierRelated"), exist_ok=True)
    data.to_csv(os.path.join(config.DATA,
            "ClassifierRelated",
            "all_trainable_data_for_classifier.csv"),
        index=False) 
