# Pranav Minasandra
# pminasandra.github.io
# Nov 28, 2023
# Yay, happy birthday to me!
# Ported to Hyrax Project from meerkat-acc on Mar 08, 2025

"""
Reads and validates behavioral audits.
"""

from collections import defaultdict
import datetime as dt
import glob
import os.path

import numpy as np
import pandas as pd

import config
import conversions
import utilities

if not config.SUPPRESS_INFORMATIVE_PRINT:
    old_print = print
    print = utilities.sprint

sync_corrections = pd.read_csv(os.path.join(config.DATA, "audit_sync_corrs.csv"))

def expand_behaviour_states(filename, df):
    """
    Expand a read audit to second-by-second records
    """

    # Extract GPS start time from corresponding .MP4 filename
    video_filename = filename.replace(".csv", "")
    video_filename = os.path.basename(video_filename)
    gps_start_time = conversions.estimate_gps_start(video_filename)

    # this was the only place to skip ignoreable behaviours
    # once the df is expanded, it becomes hard.

    mask = df["behaviour_class"].isin(config.IGNORE_BEHS)
    df.loc[mask, "behaviour_class"] = df["behaviour_class"].shift(1)
    # replace those rows by preceding rows
    
    second_wise_data = defaultdict(lambda: (None, None, 0))  # {datetime: (class, specific, duration)}
    
    for _, row in df.iterrows():
        start_time = gps_start_time + dt.timedelta(seconds=row['start'])
        end_time = gps_start_time + dt.timedelta(seconds=row['end'])

        start_sec = round(start_time.timestamp())
        end_sec = round(end_time.timestamp())
        
        for sec in range(start_sec, end_sec + 1):
            sec_datetime = dt.datetime.fromtimestamp(sec) - dt.timedelta(hours=2)#POSIXTIME doesn't respect summer time, has to be a magic number
            duration = min(end_sec, sec + 1) - max(start_sec, sec)  # Duration within this second

            # Store behavior if it's longer within the second
            if duration >= second_wise_data[sec_datetime][2]:
                second_wise_data[sec_datetime] = (row['behaviour_class'], row['behaviour_specific'], duration)
    
    # Create expanded DataFrame
    expanded_data = [[dt, bc, bs] for dt, (bc, bs, _) in sorted(second_wise_data.items())]
    expanded_df = pd.DataFrame(expanded_data, columns=['Timestamp', 'behaviour_class', 'behaviour_specific'])
    expanded_df = expanded_df.dropna()
    
    return expanded_df

def validate_audit(df):
    """
    Checks column labels and behavioral states in a given dataframe,
    and performs other fixes.
    """
    assert list(df.columns) == ['Timestamp', 'behaviour_class', 'behaviour_specific']
#    for state in df['behaviour_class'].unique():
#        assert state in config.BEHAVIORS

    df["Timestamp"] += dt.timedelta(seconds=config.AUDIT_GPS_OFFSET)
    df["Timestamp"] -= dt.timedelta(hours=config.TIMEZONE)


def load_auditfile(csvfilepath):

    """
    Loads and validates dataframes from audit csv.
    Returns:
        csvfile (pd.DataFrame)
    Raises:
        AssertionError: if there are inappropriate csvfiles
    """
    csvfile = pd.read_csv(csvfilepath)
    csvfile = expand_behaviour_states(csvfilepath, csvfile)
    csvfile['Timestamp'] = pd.to_datetime(csvfile['Timestamp'])

    ftag = os.path.basename(csvfilepath)[:-len(".csv")]
    sync_corr = sync_corrections[sync_corrections["filename"] == ftag]

    sync_corr = sync_corr["sync_corr"].item()
    csvfile["Timestamp"] += pd.to_timedelta(sync_corr, unit='s')
    validate_audit(csvfile)

# TODO: talk to Vlad and get to all this
#        if config.DROP_MISSING:
#            csvfile = csvfile[csvfile['Behavior'] != "OOS"]
#        if config.COMBINE_BEHAVIORS:
#            csvfile['Behavior'] = csvfile['Behavior'].map(
#                    config.BEHAVIOR_SIMPLIFIER
#                    )
#        if config.DROP_OTHERS:
#            csvfile = csvfile[csvfile["Behavior"] != "Others"]

    return csvfile

def load_audits():
    """
    GENERATOR!!
    Loads and validates dataframes from audit csvs.
    Yields:
        2-tuple: auditname (str), and csvfile (pd.DataFrame)
    Raises:
        AssertionError: if there are inappropriate csvfiles
    """
    tgtpath = config.AUDIT_DIR
    for csvfilepath in glob.glob(os.path.join(tgtpath, "*", "*.csv")):
        csvfile = load_auditfile(csvfilepath)
        yield os.path.basename(csvfilepath)[:-len(".csv")], csvfile


def load_audit_data_for(individual, join_audits=True):
    """
    Loads all available audit data for specified individual from specified 
    deployment.
    Args:
        individual (str)
        join_audits (bool, def True): whether to concatenate all audit dataframes.
    Returns:
        pd.DataFrame
    Raises:
        AssertionError: when looking for incorrect deployments or individuals.
    """

    assert individual in config.INDIVIDUALS

    tgtdirpath = config.AUDIT_DIR
    tgtfiles = glob.glob(os.path.join(tgtdirpath, f"*_{individual}_*", "*.csv"))

    if len(tgtfiles) == 0:
        return pd.DataFrame({"Timestamp": [], "behaviour_class": [], "behaviour_specific": []})

    all_data = [load_auditfile(file_) for file_ in tgtfiles]
    for dat in all_data:
        dat.sort_values(by='Timestamp', inplace=True)
        dat.reset_index(inplace=True, drop=True)
        dat = dat[['Timestamp', 'behaviour_class', 'behaviour_specific']]
        validate_audit(dat)

    if join_audits:
        all_data = pd.concat(all_data, ignore_index=True)

    return all_data


if __name__ == "__main__":
    inds = ["JN", "TD", "F6", "PJ", "QM", "C6", "CF"]
    for ind in inds:
        audits = load_audit_data_for(ind, join_audits=False)
        k = 0
        for audit in audits:
            k += 1
            print(f"{ind}: #{k}: {', '.join(list(audit.behaviour_class.unique()))}")
