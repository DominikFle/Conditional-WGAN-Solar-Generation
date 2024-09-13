# Plotting targets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preparation import cure_data, load_data
from datamodule import Power_NORMALIZER, create_condition_vector
from model import WGAN


def get_24_point_series(
    df: pd.DataFrame, day_or_month: str, month: int, do_plot=True, to_numpy=True
):
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
    if to_numpy:
        return series.to_numpy()
    return series


def visualize_data_point(
    model: WGAN,
    day_or_month: int,
    month: int,
    solar_or_wind: int,
    title="Real",
    to_numpy=True,
):

    condition = create_condition_vector(
        solar_or_wind, day_or_month, month, return_2d=True
    )
    model.eval()
    out = model.generate(condition)
    out = out.squeeze().detach().cpu()
    out = out.numpy() * Power_NORMALIZER
    if to_numpy:
        return out
    plt.plot(np.arange(len(out)), out)
    plt.title(f"{title}  | in the month: {month}")
    plt.show()


def vis_real_fake_grid(appendix=""):
    model = WGAN(
        input_dim=32,
        condition_size=16,
        rand_size=16,
        output_dim=24,
        hidden_layers_size_Generator=[(32, 128), (128, 128), (128, 128)],
        hidden_layers_size_Critic=[(40, 128), (128, 256), (256, 64)],
        critic_to_gen_overtraining_factor=4,
        lr=0.0005,
        W_clip=0.01,
        batch_size=1,
        vis_every=100,
    )
    model = WGAN.load_from_checkpoint(
        "stored_models/WGAN-Electricity-Solar-Only-300.ckpt"
    )
    month_num_to_name = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    df = load_data()
    solar_df, wind_df = cure_data(df)
    real_solar = []
    fake_solar = []
    fig, axs = plt.subplots(2, 12)
    fig_size = (15, 7)
    fig.set_size_inches(fig_size)
    fig.tight_layout()
    for month in range(1, 13):
        ax_real = axs[0, month - 1]
        ax_fake = axs[1, month - 1]
        real_solar = get_24_point_series(solar_df, "day", month, do_plot=False)
        fake_solar = visualize_data_point(model, 0, month, 0, to_numpy=True)
        y_lim = [0, 45]

        # ax_real.set_ylabel("GW")
        # ax_fake.set_ylabel("GW")
        month_name = month_num_to_name[month]
        # ax_real.set_xlabel(month_name)
        ax_fake.set_xlabel(month_name)
        if month == 1:
            ax_real.set_ylabel("Real GW")
            ax_fake.set_ylabel("Fake GW")
        else:
            ax_real.set_yticklabels([])
            ax_fake.set_yticklabels([])
        ax_real.set_ylim(y_lim)
        ax_fake.set_ylim(y_lim)
        ax_real.plot(range(24), real_solar)
        ax_fake.plot(range(24), fake_solar)
    fig.savefig(f"real_fake_solar_{appendix}.png")


if __name__ == "__main__":
    for i in range(5):
        vis_real_fake_grid(appendix=f"run_{i}")
