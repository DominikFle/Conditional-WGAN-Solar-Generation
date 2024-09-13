from data_preparation import cure_data, load_data
from model import WGAN
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from datamodule import IntermittentElectricityDataModule

df = load_data()
solar_df, wind_df = cure_data(df)
batch_size = 256
dm = IntermittentElectricityDataModule(
    wind_df=wind_df,
    solar_df=solar_df,
    batch_size=batch_size,
    shuffle=True,
    only_solar_day=True,
)
input_dim = 32
condition_size = 16
model = WGAN(
    input_dim=input_dim,
    condition_size=condition_size,
    rand_size=16,
    output_dim=24,
    hidden_layers_size_Generator=[(32, 128), (128, 128), (128, 128)],
    hidden_layers_size_Critic=[(40, 128), (128, 256), (256, 64)],
    critic_to_gen_overtraining_factor=4,
    lr=0.0005,
    W_clip=0.01,
    batch_size=batch_size,
    vis_every=100,
)
model_path = "/content/drive/MyDrive/AgoraData"
print(summary(model, input_size=(batch_size, condition_size)))
plt.rcParams["figure.figsize"] = (2, 2)
max_epochs = 140
trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=max_epochs,
    check_val_every_n_epoch=10,
)
ckpth = None
ckpth = f"/stored_models/WGAN-Electricity-Solar-Only-{max_epochs}.ckpt"
trainer.fit(model, dm, ckpt_path=ckpth)
trainer.save_checkpoint(
    filepath=f"/stored_models/WGAN-Electricity-Solar-Only-{max_epochs}.ckpt"
)
