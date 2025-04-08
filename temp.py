import glob
import os.path

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import config
import accreading

def get_gps_start_time(video_filename):
    # Placeholder for the actual function that retrieves the GPS start time
    # Replace with the real implementation
    return datetime(2023, 7, 5, 7, 4, 0)  # Example fixed start time

def expand_behaviour_states(filename, df):
    # Extract GPS start time from corresponding .MP4 filename
    video_filename = filename.replace(".csv", ".MP4")
    gps_start_time = get_gps_start_time(video_filename)
    
    expanded_data = []
    
    for _, row in df.iterrows():
        start_time = gps_start_time + timedelta(seconds=row['start'])
        end_time = gps_start_time + timedelta(seconds=row['end'])
        
        # Generate second-by-second timestamps
        current_time = start_time
        while current_time <= end_time:
            expanded_data.append([current_time, row['behaviour_class'], row['behaviour_specific']])
            current_time += timedelta(seconds=1)
    
    # Create expanded DataFrame
    expanded_df = pd.DataFrame(expanded_data, columns=['datetime', 'behaviour_class', 'behaviour_specific'])
    
    return expanded_df

def make_all_parquet():
    acc_filegen = accreading.load_acc_files()
    for filename, df in acc_filegen:
        tgtpath = os.path.join(config.ACC_DIR, filename + ".parquet")
        df.to_parquet(tgtpath, index=False)

if __name__ == "__main__":
    make_all_parquet()
