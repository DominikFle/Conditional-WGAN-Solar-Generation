# Plotting targets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_24_point_series(df: pd.DataFrame, day_or_month: str, month: int, do_plot=True):
    """
    Get real data from the dataset
    """
    df_with_correct_month = df.loc[df["month"] == month]
    length = len(df_with_correct_month)
    # print(df_with_correct_month.head())
    ind = np.random.randint(length)
    # print(ind)
    if day_or_month == "day":
        series = df_with_correct_month.iloc[ind, :][:-2]
    elif day_or_month == "month":
        ind_total = df_with_correct_month.iloc[ind, :].name
        print(ind_total)
        month_series = df["mean"].iloc[ind_total : ind_total + 24]
        series = month_series
    else:
        raise ValueError(
            "Wrong day_or_month param (only 'day' and 'month are accepted)"
        )
    if do_plot:
        plt.plot(range(24), series.to_numpy())
    return series
