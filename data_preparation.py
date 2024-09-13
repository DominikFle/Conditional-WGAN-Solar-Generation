import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torchinfo import summary
import time


def load_data():
    paths_csv = []
    dfs = []
    df = None
    for year in range(2020, 2024):
        path = f"data/power_generation_and_consumption_{year}.csv"
        paths_csv.append(path)

        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates()
    df.head(-10)
    return df


def assert_no_data_gaps(series):
    len_hour_series = len(series)
    for i, hour in enumerate(series):
        if hour == 23:
            assert series[i + 1] == 0
        else:
            if len_hour_series - 1 == i:
                break
            assert (hour + 1) == series[
                i + 1
            ], f"at index {i} the val {hour} is followed by {series[i+1]}"


# create 24 shifted data
def create_24x_shifted_df(series: pd.Series, series_name=None):
    name = series.name if series_name is None else series_name
    series_list = []
    for i in range(24):
        s = series.copy()
        s.name = f"{name}_{i}"
        series_list.append(s)
    df = pd.concat(series_list, axis=1)
    for hours, col_name in enumerate(df.columns):
        df[col_name] = df[col_name].shift(periods=-hours)
    return df


def drop_days_with_gaps(
    hour_series: pd.Series, additional_drops: list[pd.Series], verbose=True
):
    """
    Dropping invalid days in place
    it also resets indices
    """
    # reset index:
    hour_series.reset_index(drop=True, inplace=True)
    len_hour_series = len(hour_series)
    indices_to_drop = []
    for i, hour in enumerate(hour_series):
        if hour == 23:
            if hour_series[i + 1] != 0:
                # go forward at next day (next day is broken)
                j = i + 1
                while hour_series[j] != 0:
                    indices_to_drop.append(j)
                    j = j + 1
        else:
            if len_hour_series - 1 == i:
                break
            if (hour + 1) != hour_series[i + 1]:
                print("hour", hour, "next", hour_series[i + 1], "index", i)
                # go backward (current day is broken)
                j = i
                while hour_series[j] != 23:
                    indices_to_drop.append(j)
                    j = j - 1
                # go forward (current day is broken)
                j = i + 1
                while hour_series[j] != 0:
                    indices_to_drop.append(j)
                    j = j + 1
    # drop the indices
    # hour_series.drop(indices_to_drop, inplace=True)
    for i, series in enumerate(additional_drops):
        # first reset the series as well
        series.reset_index(drop=True, inplace=True)
        # then drop the indices
        print(series.head(4))
        series.drop(indices_to_drop, inplace=True)
        series.reset_index(drop=True, inplace=True)


def cure_data(df):
    """
    Returns:
    solar_df: pd.DataFrame, wind_df: pd.DataFrame

    """
    date_id = df["date_id"]
    hour = pd.to_datetime(date_id).dt.hour
    solar = df["Solar"]
    wind = df["Wind onshore"] + df["Wind offshore"]

    drop_days_with_gaps(hour, [solar, wind, date_id])
    year = pd.to_datetime(date_id).dt.strftime("%Y")
    month = pd.to_datetime(date_id).dt.strftime("%b")
    month = pd.to_datetime(date_id).dt.month
    hour = pd.to_datetime(date_id).dt.hour
    assert_no_data_gaps(hour)

    solar_df = create_24x_shifted_df(solar, series_name="solar")
    wind_df = create_24x_shifted_df(wind, series_name="wind")
    wind_df["mean"] = wind_df.mean(axis=1)
    solar_df["mean"] = solar_df.mean(axis=1)

    wind_df["month"] = month
    solar_df["month"] = month

    # cutoff tail
    wind_df = wind_df.iloc[:-24]
    solar_df = solar_df.iloc[:-24]

    # resample every 24h
    wind_df = wind_df.iloc[::24].reset_index(drop=True, inplace=False)
    solar_df = solar_df.iloc[::24].reset_index(drop=True, inplace=False)
    return solar_df, wind_df
