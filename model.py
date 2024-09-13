from datamodule import create_condition_vector, reverse_condition
import torch.nn as nn
import torch
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, input_dim=32, output_dim=24, hidden_layers_size=[(32,64),(64,64),(64,128),(128,128)]):
        super().__init__()
        self.input_dim = input_dim # condition + random state

        self.gen_layers=nn.ModuleList([])
        for num_in,num_out in hidden_layers_size:
          self.gen_layers.append(nn.Linear(num_in, num_out))
          self.gen_layers.append(nn.BatchNorm1d(num_out))
          self.gen_layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.gen_layers.append(nn.Linear(hidden_layers_size[-1][-1], output_dim))

    def forward(self, z):
        for layer in self.gen_layers:
          z = layer(z)
        return z
    
class Critic(nn.Module):
    def __init__(self,layer_sizes = [(40,128),(128,128),(128,64)]):
        super().__init__()
        self.critic_layers=nn.ModuleList([])
        for num_in,num_out in layer_sizes:
          self.critic_layers.append(nn.Linear(num_in, num_out))
          self.critic_layers.append(nn.LayerNorm(num_out))
          self.critic_layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.critic_layers.append(nn.Linear(layer_sizes[-1][-1], 1))
       

    def forward(self, x):
        for layer in self.critic_layers:
          x = layer(x)
        return x
    
class WGAN(pl.LightningModule):
    def __init__(
        self,
        input_dim=32,
        condition_size=16,
        rand_size=16,
        output_dim=24,
        hidden_layers_size_Generator=[(32,64),(64,64),(64,128),(128,128)],
        hidden_layers_size_Critic=[(40,128),(128,128),(128,64)],
        critic_to_gen_overtraining_factor = 5,
        lr: float = 0.005,
        W_clip:float = 0.001,
        batch_size: int = 32,
        vis_every:int =50,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        assert rand_size+ condition_size == input_dim
        # networks
        self.generator = Generator(input_dim=input_dim, output_dim=output_dim, hidden_layers_size=hidden_layers_size_Generator)
        self.critic = Critic(layer_sizes=hidden_layers_size_Critic)
        
        validation_condition= create_condition_vector()
        validation_rand = self.get_noise_for_condition(validation_condition)

        self.validation_x=torch.cat([validation_condition,validation_rand ] , dim = -1)

        #self.example_input_array = torch.zeros(2, self.hparams.input_dim)

        self.counter_helper = 0 # hack sol for different amount of training for critic and generator
    def get_noise_for_condition(self, condition):
        if len(condition.shape) > 1:
          # there are batched conditions
          noise = torch.randn((condition.shape[0], self.hparams.rand_size))
        else:
          noise = torch.randn(self.hparams.rand_size)
        return noise
    def generate(self, condition):
        noise = self.get_noise_for_condition( condition)
        generator_in=torch.cat([condition,noise], dim = -1)
        return self.generator(generator_in)
    def forward(self, condition):
       
        noise = self.get_noise_for_condition( condition)
        generator_in=torch.cat([condition,noise], dim = -1)
        return self.generator(generator_in)

    def training_step(self, batch):
        feature, condition = batch

        optimizer_generator, optimizer_critic = self.optimizers()

        
        # Clamp parameters to a range [-c, c], c=self.W_clip
        for p in self.critic.parameters():
            p.data.clamp_(-self.hparams.W_clip, self.hparams.W_clip)
        # train generator
        # generate images
        if self.counter_helper % int(self.hparams.critic_to_gen_overtraining_factor-1) == 0:
          # sample noise
          noise = self.get_noise_for_condition( condition)
          z = torch.cat([condition,noise], dim = -1)
          self.toggle_optimizer(optimizer_generator)
          self.output_generator = self.generator(z)

          noise = self.get_noise_for_condition( condition)
          z=torch.cat([condition,noise], dim = -1)
          fake_output = self.generator(z)
         

          conditioned_fake_output = torch.cat([condition,fake_output] , dim = -1)
          g_loss = -self.critic(conditioned_fake_output)
          g_loss = g_loss.mean().mean(0).view(1)
          optimizer_generator.zero_grad()
          self.manual_backward(g_loss)
          g_cost = -g_loss
          optimizer_generator.step()
          self.untoggle_optimizer(optimizer_generator)

          self.log_dict({"generator_loss": g_loss}, prog_bar=True)
        # train Critic
        # Measure Critc's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_critic)
        self.critic.zero_grad()

        # get real and fake data
        noise = self.get_noise_for_condition( condition)
        z = torch.cat([condition,noise], dim = -1)
        fake_output = self.generator(z)
        fake_input = torch.cat([condition, fake_output] , dim = -1)
        real_input = torch.cat([condition,feature], dim = -1)
        #VIS Generator
        if self.counter_helper % self.hparams.vis_every == 0:
          self.visualize_data_point(condition[0,:], fake_output[0,:], title="Fake")
          self.visualize_data_point(condition[0,:], feature[0,:], title="Real")
        # Train Critic
        # Train with real images
        critic_loss_real = self.critic(real_input)
        critic_loss_real = critic_loss_real.mean(0).view(1)

        # Train with fake images
        critic_loss_fake = self.critic(fake_input)
        critic_loss_fake = critic_loss_fake.mean(0).view(1)

        # High Values of Critic --> Critic thinks its a real input
        # Low Values of Critic --> Critic thinks its a fake input
        critic_loss = critic_loss_fake - critic_loss_real
        optimizer_critic.zero_grad()
        self.manual_backward(critic_loss)
        optimizer_critic.step()
        self.untoggle_optimizer(optimizer_critic)

        Wasserstein_D = critic_loss_real - critic_loss_fake

        self.log_dict({"critic_loss": critic_loss,
                       "cl_fake": critic_loss_fake,
                       "cl_real": critic_loss_real}, prog_bar=True)

        #Update counter to keep training ratio between critic and generator 
        self.counter_helper = self.counter_helper +1
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_generator = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        opt_critic = torch.optim.RMSprop(self.critic.parameters(), lr=lr)
        return [opt_generator, opt_critic], []

    def visualize_data_point(self, condition, feature, title=""):
        condition = condition.detach().cpu().numpy()
        feature = feature.detach().cpu().numpy()
        solar_or_wind, day_or_month, month = reverse_condition(condition)
        solar_or_wind_str = "SOLAR" if solar_or_wind == 0 else "WIND"
        day_or_month_str = "DAY" if day_or_month == 0 else "MONTH"
        
        plt.plot(np.arange(len(feature)),feature)
        plt.title(f"{title}  | Energy: {solar_or_wind_str,} Measured along a: {day_or_month_str} in the month: {month}")
        plt.show()
    def on_validation_epoch_end(self):
        #z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        #sample_imgs = self(z)
        #grid = torchvision.utils.make_grid(sample_imgs)
        #self.logger.experiment.add_image("validation/generated_images", grid, self.current_epoch)

        #TODO print preconditioned sample with fixed random numbers
        pass