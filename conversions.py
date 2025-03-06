# Pranav Minasandra
# pminasandra.github.io
# March 5, 2025

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
    output_df = pd.DataFrame(output_data, columns=['start', 'end', 'behaviour_class', 'behaviour_specific'])
    output_df.to_csv(output_file, index=False)


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
