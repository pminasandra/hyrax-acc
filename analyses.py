# Pranav Minasandra
# pminasandra.github.io
# 11 Mar 2025

import glob
import os
import os.path
from os.path import join as joinpath
import re

import matplotlib.pyplot as plt
import pandas as pd

import accreading
import auditreading
import config
import utilities

def plot_acc_beh(fig, ax, df_acc, df_beh):
    """
    Plot high-resolution accelerometer data (x, y, z) and color-code background by categorical states.
    
    Parameters:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        df_acc (pd.DataFrame): DataFrame with 'Timestamp', 'X', 'Y', and 'Z' columns.
        df_beh (pd.DataFrame): DataFrame with 'Timestamp' and 'behaviour_class' columns.
    """
    
    # Plot x, y, z from df_acc
    start_time = df_beh['Timestamp'].min()
    end_time = df_beh['Timestamp'].max()
    df_acc_subset = df_acc[(df_acc['Timestamp'] >= start_time) & (df_acc['Timestamp'] <= end_time)]

    ax.plot(df_acc_subset['Timestamp'], df_acc_subset['X'], alpha=0.8)
    ax.plot(df_acc_subset['Timestamp'], df_acc_subset['Y'], alpha=0.8)
    ax.plot(df_acc_subset['Timestamp'], df_acc_subset['Z'], alpha=0.8)
    
    # Color-code background using df_beh
    for i in range(len(df_beh) - 1):
        start = df_beh.iloc[i]['Timestamp']
        end = df_beh.iloc[i + 1]['Timestamp']
        state = df_beh.iloc[i]['behaviour_class']
        ax.axvspan(start, end, color=hash_color(state), alpha=0.3)
    
    # Set labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration')
    ax.set_title('Accelerometer Data with Behavioral States')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    return ax

def load_audit(auditname):
    """
    convenience function to load audit only from basename
    Args:
        auditname (str): e.g., 202307050704_JN.csv
    """

    auditname_sub = auditname[:-len(".csv")]
    ind = auditname_sub.split("_")[1]

    fpath = joinpath(config.AUDIT_DIR, f"*_{ind}_*", auditname)
    file_ = list(glob.glob(fpath))[0] #should be only one file
    df = auditreading.load_auditfile(file_)

    return df

def load_acc(ind):
    fpath = joinpath(config.ACC_DIR, f"*_{ind}_*.csv")
    file_ = list(glob.glob(fpath))[0] #should be only one file

    df = accreading.load_acc_file(file_)
    return df


def hash_color(label):
    """Generate a consistent color for each unique state label."""
    import hashlib
    hash_val = int(hashlib.md5(label.encode()).hexdigest(), 16)
    return plt.cm.tab10(hash_val % 10)

CACHED_ACC_FILES = {}

def interactive_sync_check(cache_acc=True):
    """
    Run an interactive audit visualization loop.
    Prompts for an audit file name, extracts the individual name,
    loads accelerometer and behavioral data, and visualizes it.
    """
    while True:
        audit_name = input("Enter audit filename (or 'q' to quit): ")
        if audit_name.lower() == 'q':
            break

        # Extract individual name using regex
        match = re.match(r'\d{12}_(\w+)\.csv', audit_name)
        if not match:
            print("Invalid audit filename format. Try again.")
            continue
        individual = match.group(1)

        print(f"Loading data for individual: {individual}")

        # Load data using provided functions
        if cache_acc:
            if individual not in CACHED_ACC_FILES:
                df_acc = load_acc(individual)
                CACHED_ACC_FILES[individual] = df_acc
            else:
                df_acc = CACHED_ACC_FILES[individual]
        else:
            df_acc = load_acc(individual)
        df_beh = load_audit(audit_name)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot data
        plot_acc_beh(fig, ax, df_acc, df_beh)

        # Improve interactivity
        plt.tight_layout()
        plt.show(block=True)  # Blocks execution until the window is closed

        # Clear figure to free memory before next iteration
        plt.close(fig)

if __name__ == "__main__":
    interactive_sync_check()
