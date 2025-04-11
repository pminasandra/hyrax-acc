# Pranav Minasandra
# pminasandra.github.io
# Nov 30, 2023
# Ported from meerkat-acc to hyrax-acc on 15 Mar 2025

"""
Analyse performance of random forest classifier
"""

import glob
import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

import config
import classifier
import extractor
import utilities

ALL_FEATURES = list(extractor.ALL_FEATURES.keys())

if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint


def _save_classifier_report(clfreport, clffilename):

    clfreport_save = pd.DataFrame(clfreport).transpose()
    clfreport_save.to_csv(
        os.path.join(config.DATA,
                        "ClassifierRelated",
                        clffilename + ".csv"
                    )
    )

    clfreport_save.to_latex(
        os.path.join(config.DATA,
                        "ClassifierRelated",
                        clffilename + ".tex"
                    ),
        float_format="{{:0.2f}}".format
    )


def _split_features_and_classes(data, features=ALL_FEATURES):
    d_features = data[features]
    d_classes = data["behaviour_class"]

    return d_features, d_classes


def trad_analyze_random_forest(data, timescales, n=100,
                                fig=None, ax=None,
                              ):
    """
    Implements traditional 85/15 analysis with available data
    Args:
        data (pd.DataFrame): all available training data.
        timescales (list-like): which timescales data should be used from
        n (int): how many times train-test data should be randomised, and
                    a new classifier trained.
        fig (plt.Figure)
        ax (matplotlib.axes._axes.Axes): Where to make plots. if None, additional
                           axes will be made.
    Returns:
        sklearn.metrics.classification_report (dict),
        fig,
        ax
    """

    features = [f'{ft}_{tscale}s' for ft in ALL_FEATURES for tscale in timescales]
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    data = data[features + ["Timestamp", "behaviour_class"]].copy()
    data_features, data_classes = _split_features_and_classes(data, features=features)
    y_true, y_pred = [], []
    # I haven't checked if you've created only one of these
    # I'm writing this code to fix something someone else should
    # have done properly at the end of last year. This isn't gonna
    # be code that is completely idiotproof. Be careful in using this.
    # Don't create a situation where you have just a fig or just an ax.
    for j in range(n):
        train_features, test_features, train_classes, test_classes =\
            sklearn.model_selection.train_test_split(data_features, data_classes,
            test_size=0.15, stratify=data_classes)

        if config.SCALE_DATA:
            scaler = StandardScaler()
            scaler.fit(train_features)

            train_features = scaler.transform(train_features)
            test_features = scaler.transform(test_features)

        clf = classifier.train_random_forest(train_features, train_classes)
        pred_classes = clf.predict(test_features)
        y_true.append(test_classes)
        y_pred.append(pred_classes)

    t_classes = np.hstack(y_true)
    pred_classes = np.hstack(y_pred)

    clfreport = sklearn.metrics.classification_report(t_classes,
                            pred_classes, output_dict=True)

    # right file naming
    if len(timescales) == 1:
        tscale_id = f'{timescales[0]}s_only'
    elif len(timescales) > 1:
        tscalestrs = [f'{tscale}s' for tscale in timescales]
        tscale_id = '_'.join(tscalestrs)

    _save_classifier_report(clfreport, f"classifier_report_randomized_{tscale_id}")

    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
                t_classes,
                pred_classes,
                normalize='true',
                ax=ax,
                cmap='Reds',
            )

    xlabels = ax.get_xticklabels()
    ax.set_xticklabels(xlabels, rotation=45)
    utilities.saveimg(fig, f"confusionmatrix_randomized_test_{tscale_id}")

    return clfreport, fig, ax


def indwise_analyze_random_forest(data, fig=None, ax=None):
    """
    Trains many random forest classifiers and tests on new individuals
    Args:
        data (pd.DataFrame): output from classifier.load...
    Returns:
        sklearn.metrics.classification_report (dict),
        fig,
        ax 
    """

    inds_available = data["Individual"].unique()
    true_classes = []
    pred_classes = []

    for ind in inds_available:
        print(f"testing classifier generalizability to {ind}.")
        data_train = data[data["Individual"] != ind].copy()
        data_test = data[data["Individual"] == ind].copy()

        train_features, train_classes = _split_features_and_classes(data_train)
        test_features, test_classes = _split_features_and_classes(data_test)

        if config.SCALE_DATA:
            scaler = StandardScaler()
            scaler.fit(train_features)

            train_features = scaler.transform(train_features)
            test_features = scaler.transform(test_features)

        rfc = classifier.train_random_forest(train_features, train_classes)
        preds = rfc.predict(test_features)

        true_classes.extend(list(test_classes))
        pred_classes.extend(list(preds))

    if fig is None and ax is None:
        fig, ax = plt.subplots()

    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
            true_classes,
            pred_classes,
            normalize='true',
            ax=ax,
            cmap='Reds',
        )

    utilities.saveimg(fig, "confusionmatrix_indwise_test")
    clfreport = sklearn.metrics.classification_report(true_classes,
                            pred_classes, output_dict=True)

    _save_classifier_report(clfreport, "classifier_report_indwise")

    return clfreport, fig, ax


def classify_all_available_data(rfc):
    """
    Predicts behavioral labels for all available ACC data.
    Args:
        rfc: a trained random forest classifier
    """
    dpldir = os.path.join(config.DATA, "Features")
    all_csv_files = glob.glob(os.path.join(dpldir, "*.csv"))

    for csvfile in all_csv_files:
        filename = os.path.basename(csvfile)
        print(f"now inferring sequences for {filename}")

        ind_data = pd.read_csv(csvfile)
        timestamps = ind_data['Timestamp']
        ind_data = ind_data[ALL_FEATURES]

        ind_preds = rfc.predict(ind_data)
        ind_preds = pd.DataFrame({'datetime': timestamps,
                                  'state': ind_preds})
        ind_name = filename.split("_")[1]
        tgtfilename = ind_name + "_predictions.csv"
        os.makedirs(os.path.join(config.DATA, "Predictions"), exist_ok=True)
        tgtfile = os.path.join(config.DATA, "Predictions",
                                tgtfilename)
        print("saving to", tgtfile)
        os.makedirs(os.path.dirname(tgtfile), exist_ok=True)
        ind_preds.to_csv(tgtfile, index=False)


if __name__ == "__main__":

    tscale_singletons = [[t] for t in config.timescales]
    tscale_pairs = [[1, tscale] for tscale in config.timescales]
    timescales = tscale_singletons + tscale_pairs
    datasource = os.path.join(config.DATA, "ClassifierRelated",
                        "all_trainable_data_for_classifier.csv")
    data = pd.read_csv(datasource)

    if config.LOG_TRANSFORM_VEDBA:
        for tscales in timescales:
            for tscale in tscales:
                vedba_col = f'mean_vedba_{tscale}s'
                data[vedba_col] += 1e-10
                data[vedba_col] = np.log(data[vedba_col])

    for tscales in timescales:
        print(f"Working on {tscales}")
        fig, ax = plt.subplots()
        trad_analyze_random_forest(data, tscales, n=100, fig=fig, ax=ax)
        plt.cla()
