# Pranav Minasandra
# pminasandra.github.io
# 10 Mar 2025

import os
import os.path
from os.path import join as joinpath

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import config
import dataloading
import utilities

def average_high_proportion(df, k):
    """
    Compute the average proportion of "HIGH" states for each k-minute interval of the day.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'datetime' (datetime64) and 'state' ("LOW" or "HIGH").
        k (int): Interval size in minutes.

    Returns:
        pd.DataFrame: A DataFrame with columns 'time_bin' and 'avg_high_proportion'.
    """
    # Ensure datetime is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract time of day in minutes
    df['time_bin'] = (df['datetime'].dt.hour * 60 + df['datetime'].dt.minute) // k * k

    # Compute proportion of HIGH states per k-minute interval
    high_proportion = (df.groupby('time_bin')['state']
                         .apply(lambda x: (x == 'HIGH').mean())
                         .reset_index(name='avg_high_proportion'))

    return high_proportion

def daily_activity_pattern_plot(props, fig, ax, **kwargs):
    propsc = props.copy()
    propsc["time_bin"] = pd.to_datetime("1970-01-01")\
                            + pd.to_timedelta(props["time_bin"], unit='m')
    ax.plot(propsc["time_bin"], propsc["avg_high_proportion"], **kwargs)


def add_shading(fig, ax):
# 1951 sunset 0538 sunrise in israel at utc+3
    shade1_start = pd.to_datetime("1970-01-01 00:00:00")
    shade2_start = pd.to_datetime("1970-01-01 02:00:00")
    shade3_start = pd.to_datetime("1970-01-01 02:30:00")
    shade4_start = pd.to_datetime("1970-01-01 16:30:00")
    shade5_start = pd.to_datetime("1970-01-01 17:00:00")
    shade5_end = pd.to_datetime("1970-01-01 23:59:59")

    night_colour = "#aaaaaa"
    twilight_colour = "#cccccc"
    ax.axvspan(shade1_start, shade2_start, color=night_colour)
    ax.axvspan(shade2_start, shade3_start, color=twilight_colour)
    ax.axvspan(shade4_start, shade5_start, color=twilight_colour)
    ax.axvspan(shade5_start, shade5_end, color=night_colour)

def create_activity_pattern_plot():
    fig, ax = plt.subplots()
    seq_dir = joinpath(config.DATA, "VeDBA_States")
    all_props = []
    for fname, df in dataloading.load_sequences(seq_dir, "vedba_states"):
        props = average_high_proportion(df, 15)
        all_props.append(props)
        daily_activity_pattern_plot(props, fig, ax, linewidth=0.5, alpha=0.7)

    all_props = sum(all_props)/len(all_props)
    daily_activity_pattern_plot(all_props, fig, ax, linewidth=1.0, color="black")

    add_shading(fig, ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    ax.set_xlabel("Time of day (UTC)")
    ax.set_ylabel("Proportion of time active")

    utilities.saveimg(fig, "hyrax_daily_activity_patterns")


if __name__ == "__main__":
    create_activity_pattern_plot()
