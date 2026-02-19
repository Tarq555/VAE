import torch
import torch.nn as nn

# تعريف نموذج المولد لـ GAN
class Generator(nn.Module):
  def __init__(self, latent_dim):
      super(Generator, self).__init__()
      self.model = nn.Sequential(
          nn.Linear(latent_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 256),
          nn.ReLU(),
          nn.Linear(256, 784),
          nn.Tanh()  # استخدام Tanh لتوليد قيم بين -1 و 1
      )

  def forward(self, z):
      return self.model(z)

# تعريف نموذج المميز لـ GAN
class Discriminator(nn.Module):
  def __init__(self):
      super(Discriminator, self).__init__()
      self.model = nn.Sequential(
          nn.Linear(784, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Linear(128, 1),
          nn.Sigmoid()  # استخدام Sigmoid لإخراج قيمة بين 0 و 1
      )

  def forward(self, x):
      return self.model(x)

# تعريف نموذج VAE
class VAE(nn.Module):
  def __init__(self, latent_dim):
      super(VAE, self).__init__()
      self.encoder = nn.Sequential(
          nn.Linear(784, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
      )
      self.fc_mu = nn.Linear(128, latent_dim)
      self.fc_logvar = nn.Linear(128, latent_dim)
      self.decoder = nn.Sequential(
          nn.Linear(latent_dim, 128),
          nn.ReLU(),
          nn.Linear(128, 256),
          nn.ReLU(),
          nn.Linear(256, 784),
          nn.Sigmoid()  # استخدام Sigmoid لإخراج قيم بين 0 و 1
      )


  def reparameterize(self, mu, logvar):
      std = torch.exp(0.5 * logvar)
      eps = torch.randn_like(std)
      return mu + eps * std

  def forward(self, x):
      h = self.encoder(x)
      mu = self.fc_mu(h)
      logvar = self.fc_logvar(h)
      z = self.reparameterize(mu, logvar)
      return self.decoder(z), mu, logvar