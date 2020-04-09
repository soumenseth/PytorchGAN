import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch

latent_dim = 100
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_epochs = 100
sample_interval = 10
img_size = 28
batch_size = 32

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels=1):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        self.img_shape = (self.channels, self.img_size, self.img_size)

        def block(features_in, features_out, normalize=True):
            layers = [nn.Linear(features_in, features_out)]
            if normalize:
                layers.append(nn.BatchNorm1d(features_out, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size, channels=1):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.img_shape = (self.channels, self.img_size, self.img_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

# Loss Function
loss = nn.BCELoss()

# initialize Generator and Discriminator
generator = Generator(latent_dim, img_size)
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    loss.cuda()

# Configure data loader
os.makedirs("./PytorchGAN/data/mnist", exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST(
        "./PytorchGAN/data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#--------------------#
#----  Training  ----#
#--------------------#

os.makedirs("./PytorchGAN/images/", exist_ok=True)
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truth
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad = False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad = False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # Train Generator
        optimizer_G.zero_grad()

        # Sample noise as Generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Generator loss
        g_loss = loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Discriminator loss
        real_loss = loss(discriminator(real_imgs), valid)
        fake_loss = loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            save_image(gen_imgs.data[:25], "./PytorchGAN/images/%d.png" % batches_done, nrow=5, normalize=True)