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

def expand_behaviour_states(filename, df):
    """
    Expand a read audit to second-by-second records
    """

    # Extract GPS start time from corresponding .MP4 filename
    video_filename = filename.replace(".csv", "")
    video_filename = os.path.basename(video_filename)
    gps_start_time = conversions.estimate_gps_start(video_filename)
    
    second_wise_data = defaultdict(lambda: (None, None, 0))  # {datetime: (class, specific, duration)}
    
    for _, row in df.iterrows():
        start_time = gps_start_time + dt.timedelta(seconds=row['start'])
        end_time = gps_start_time + dt.timedelta(seconds=row['end'])

        start_sec = round(start_time.timestamp())
        end_sec = round(end_time.timestamp())
        
        for sec in range(start_sec, end_sec + 1):
            sec_datetime = dt.datetime.fromtimestamp(sec)
            duration = min(end_sec, sec + 1) - max(start_sec, sec)  # Duration within this second

            # Store behavior if it's longer within the second
            if duration > second_wise_data[sec_datetime][2]:
                second_wise_data[sec_datetime] = (row['behaviour_class'], row['behaviour_specific'], duration)
    
    # Create expanded DataFrame
    expanded_data = [[dt, bc, bs] for dt, (bc, bs, _) in sorted(second_wise_data.items())]
    expanded_df = pd.DataFrame(expanded_data, columns=['Timestamp', 'behaviour_class', 'behaviour_specific'])
    
    return expanded_df

def validate_audit(df):
    """
    Checks column labels and behavioral states in a given dataframe,
    and performs other fixes.
    """
    assert list(df.columns) == ['Timestamp', 'behaviour_class', 'behaviour_specific']
#    for state in df['behaviour_class'].unique():
#        assert state in config.BEHAVIORS

    df["Timestamp"] += dt.timedelta(seconds=config.ACC_AUDIT_OFFSET)


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


def load_audit_data_for(individual):
    """
    Loads all available audit data for specified individual from specified 
    deployment.
    Args:
        individual (str)
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
    all_data = pd.concat(all_data, ignore_index=True)

    all_data.sort_values(by='Timestamp', inplace=True)
    all_data.reset_index(inplace=True)
    all_data = all_data[['Timestamp', 'behaviour_class', 'behaviour_specific']]
    validate_audit(all_data)

# TODO: talk to Vlad and get to all this
#    if config.DROP_MISSING:
#        all_data = all_data[all_data['Behavior'] != "No observation"]
#    if config.COMBINE_BEHAVIORS:
#        all_data['Behavior'] = all_data['Behavior'].map(config.BEHAVIOR_SIMPLIFIER)
#    if config.DROP_OTHERS:
#        all_data = all_data[all_data["Behavior"] != "Others"]

    return all_data


if __name__ == "__main__":
    for ind in config.INDIVIDUALS:
        print(ind)
        df = load_audit_data_for(ind)
        df = df[df.behaviour_class!="Out of Sight"]
        df = df[df.behaviour_class!=None]
        df = df[df.behaviour_class!="OOS"]
        df.reset_index(inplace=True)
        print(df)
        print(df.behaviour_class.unique())
    print()
