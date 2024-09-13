import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def create_condition_vector(
    solar_or_wind: int = 0, day_or_month: int = 0, month: int = 1, return_2d=False
):
    """
    Args:
      solar_or_wind: enum --> 0=Solar, 1=Wind
      day_or_month: enum --> 0=Day, 1=Month
      month: [1-12]
    """
    assert month > 0 and month < 13, f" Month is not valid: {month}"
    month = torch.tensor(month - 1).to(torch.int64)  # -1 --> 0,11
    month_hot = F.one_hot(month, 12)
    solar_or_wind_hot = F.one_hot(torch.tensor(solar_or_wind).to(torch.int64), 2)
    day_or_month_hot = F.one_hot(torch.tensor(day_or_month).to(torch.int64), 2)
    condition = torch.cat([solar_or_wind_hot, day_or_month_hot, month_hot])
    if return_2d:
        return condition.unsqueeze(0)
    return condition


def reverse_condition(condition: torch.Tensor):
    """
    Returns
    solar_or_wind enum
    day_or_month enum
    and month in [1,12]
    """
    if len(condition.shape) > 1:
        condition = condition.unsqueeze(0)
    solar_or_wind = np.argmax(condition[:2])
    day_or_month = np.argmax(condition[2:4])
    month = np.argmax(condition[4:]) + 1
    return (solar_or_wind, day_or_month, month)


Power_NORMALIZER = 40


class IntermittentElectricityDataset(Dataset):
    def __init__(
        self,
        wind_df: pd.DataFrame,
        solar_df: pd.DataFrame,
        only_solar_day: bool = False,
    ):
        self.wind_df = wind_df
        self.solar_df = solar_df
        self.section_length = len(solar_df)
        self.only_solar_day = only_solar_day

    def __len__(self):
        len_solar_day = len(self.solar_df)
        len_wind_day = len(self.wind_df)
        assert len_solar_day == len_wind_day
        return int(
            4 * len_solar_day
        )  # four sections for 1. solar day 2. wind day 3. Solar Month 4. wind month

    def __getitem__(self, index):
        res_index = index % self.section_length
        section = int(index // self.section_length)
        if self.only_solar_day:
            section = 0
        # The Sections are the reason that a bimodal generator will be obtained
        if section == 0:
            # Solar Day -->
            series = self.solar_df.iloc[res_index, :]
            month = torch.tensor(series.iloc[-1])
            solar_or_wind = 0  # Solar = 0, Wind = 1
            day_or_month = 0  # Day=0
            condition = create_condition_vector(solar_or_wind, day_or_month, month)
            feature = torch.from_numpy(series[:24].values)

        elif section == 1:
            # Wind Day
            series = self.wind_df.iloc[res_index, :]
            month = torch.tensor(series.iloc[-1])
            solar_or_wind = 1  # Solar = 0, Wind = 1
            day_or_month = 0  # Day=0
            condition = create_condition_vector(solar_or_wind, day_or_month, month)
            feature = torch.from_numpy(series[:24].values)
        elif section == 2:
            # Solar Month
            if res_index + 24 > self.section_length:
                res_index = 0  # this is hacky --> avoiding the last entries of the df
            series = self.solar_df["mean"].iloc[res_index : res_index + 24]
            month = self.solar_df.iloc[res_index, -1]
            solar_or_wind = 0  # Solar = 0, Wind = 1
            day_or_month = 1  # Day=0
            condition = create_condition_vector(solar_or_wind, day_or_month, month)
            feature = torch.from_numpy(series[:24].values)

        elif section == 3:
            # Wind Month
            if res_index + 24 > self.section_length:
                res_index = 0  # this is hacky --> avoiding the last entries of the df
            series = self.wind_df["mean"].iloc[res_index : res_index + 24]
            month = self.wind_df.iloc[res_index, -1]
            solar_or_wind = 0  # Solar = 0, Wind = 1
            day_or_month = 1  # Day=0
            condition = create_condition_vector(solar_or_wind, day_or_month, month)
            feature = torch.from_numpy(series[:24].values)
        else:
            raise ValueError(f"No Section was applicable, section: {section}")
        feature = feature / Power_NORMALIZER
        return (feature.float(), condition.float())

    def get_feature_condtion():
        pass


class IntermittentElectricityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        wind_df: pd.DataFrame,
        solar_df: pd.DataFrame,
        batch_size: int = 32,
        shuffle=True,
        only_solar_day=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.wind_df = wind_df
        self.solar_df = solar_df
        self.shuffle = shuffle
        self.wind_df = wind_df
        self.solar_df = solar_df
        self.only_solar_day = only_solar_day

    def setup(self, stage: str):
        self.train_dataset = IntermittentElectricityDataset(
            self.wind_df, self.solar_df, self.only_solar_day
        )
        self.test_dataset = IntermittentElectricityDataset(
            self.wind_df, self.solar_df, self.only_solar_day
        )
        self.val_dataset = IntermittentElectricityDataset(
            self.wind_df, self.solar_df, self.only_solar_day
        )
        self.predict_dataset = IntermittentElectricityDataset(
            self.wind_df, self.solar_df, self.only_solar_day
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
