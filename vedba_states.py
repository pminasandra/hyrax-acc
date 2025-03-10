# Pranav Minasandra
# pminasandra.github.io
# 10 Mar 2025

import os
import os.path
from os.path import join as joinpath

import numpy as np
import matplotlib.pyplot as plt

import config
import dataloading
import utilities

def generate_vedba_histograms():
    for fname, df in dataloading.load_features():
        tgtname = fname[:-len("_extracted_features.csv")]
        lvs = np.log(df["mean_vedba"] + 1e-8)
        lvs = lvs[(-6 < lvs) & (lvs < 2)] #standardisation looking at existing plots

        fig, ax = plt.subplots()
        ax.hist(lvs, 150, color="#eeeeee", edgecolor="black")
        ax.set_xlabel("Log VeDBA + $10^{-8}$")
        ax.set_ylabel("Frequency")
        ax.set_xlim((-6, 2)) #standardisation looking at existing plots

        tgtpath = joinpath(config.FIGURES, "VeDBA_hists")
        os.makedirs(tgtpath, exist_ok=True)
        utilities.saveimg(fig, tgtname, directory=tgtpath)

def extract_vedba_states():
    tgtdir = joinpath(config.DATA, "VeDBA_States")
    os.makedirs(tgtdir, exist_ok=True)

    for fname, df in dataloading.load_features():
        tgtname = fname[:-len("_extracted_features.csv")]

        dfs = df[["Timestamp", "mean_vedba"]].copy()
        dfs.loc[:, "state"] = "LOW"
        lvs = np.log(dfs["mean_vedba"] + 1e-8)
        dfs.loc[lvs > -3.75, "state"] = "HIGH"
        dfs.loc[:, "datetime"] = dfs["Timestamp"]

        dfs = dfs[["datetime", "state"]]

        dfs.to_csv(joinpath(tgtdir, tgtname + "_vedba_states.csv"), index=False)


if __name__ == "__main__":
    extract_vedba_states()
