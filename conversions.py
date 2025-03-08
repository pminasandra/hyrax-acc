# Pranav Minasandra
# pminasandra.github.io
# March 5, 2025

"""
Conversions from given audit and acc formats to formats compatible with
existing behavioural classification software.
"""

from datetime import datetime, timedelta
import glob
import os.path
from os.path import join as joinpath

import numpy as np
import pandas as pd

import config

def convert_behaviour_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    # List to store transformed data
    output_data = []

    # Get timestamps
    timestamps = df['Time'].values

    for i, row in df.iterrows():
        current_time = row['Time']
        next_time = timestamps[i + 1] if i + 1 < len(timestamps) else np.nan

        for col in df.columns:
            if col == 'Time' or pd.isna(row[col]):
                continue  # Skip empty values

            behaviour_class = col
            behaviour_specific = row[col] if isinstance(row[col], str) else ""

            # Special handling for instantaneous events
            if behaviour_class in ['Time Synch', 'END']:
                next_time = current_time

            output_data.append([current_time, next_time, behaviour_class, behaviour_specific])

    # Convert to DataFrame and save
    output_df = pd.DataFrame(output_data, columns=['start', 'end',
                                        'behaviour_class', 'behaviour_specific'])
    output_df.to_csv(output_file, index=False)


def parse_timecode(timecode):
    """Convert Video_t timecode (hh:mm:ss:ff) to total seconds."""
    try:
        hh, mm, ss, ff = map(int, timecode.split(':'))
        return hh * 3600 + mm * 60 + ss + ff / 30  # Assuming 30 fps
    except:
        return np.nan

def estimate_gps_start(filename):
    """Estimate the GPS start time of a video file."""
    csv_file = joinpath(config.DATA, "timestamps.csv")
    df = pd.read_csv(csv_file)
    
    # Filter rows for the given filename
    file_rows = df[df['file'] == filename+".MP4"]
    if file_rows.empty:
        raise ValueError(f"Filename {filename} not found in the CSV.")
    
    # Drop rows where GPS_t is NA if other valid rows exist
    if file_rows['GPS_t'].isna().all():
        return 'nan'
    file_rows = file_rows.dropna(subset=['GPS_t'])
    
    # Convert columns to datetime and seconds
    file_rows['GPS_t'] = pd.to_datetime(file_rows['GPS_t'], errors='coerce')
    file_rows['Video_t_sec'] = file_rows['Video_t'].apply(parse_timecode)
    
    # Compute GPS start estimate (GPS_t - Video_t offset)
    file_rows['Estimated_Start'] = file_rows['GPS_t'] - pd.to_timedelta(file_rows['Video_t_sec'], unit='s')
    
    # Return mean of estimated starts
    return file_rows['Estimated_Start'].mean()

if __name__ == "__main__":
    files = glob.glob(joinpath(config.DATA, "Orig_Audits", "*", "*"))
    for file in files:
        fname = os.path.basename(file)
        pdir = os.path.basename(os.path.dirname(file))
        tgt_sub = os.path.sep.join([pdir, fname])

        tgt = joinpath(config.DATA, "Readable_Audits", tgt_sub)
        os.makedirs(os.path.dirname(tgt), exist_ok=True)

        print(file, "-->", tgt)
        convert_behaviour_csv(file, tgt)

    print()
    files = glob.glob(joinpath(config.DATA, "Readable_Audits", "*", "*.csv"))
    for file in files:
        inp = os.path.basename(file)[:-4]
        try:
            print(inp, ":", estimate_gps_start(inp))
        except ValueError as e:
            print(inp, ":", "nan", f"({e})")
