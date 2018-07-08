import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, g_dim=128, out_dim=3):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_dim * 8),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(g_dim * 8, g_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_dim * 4),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(g_dim * 4, g_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_dim * 2),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(g_dim * 2, g_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_dim),
            nn.ReLU(inplace=True)
        )

        # As stated in the paper, batchnorm is not applied to the gen. output layer.
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(g_dim, out_dim, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        # For educational purposes, I've expanded the connections that way.
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, d_dim=128, in_dim=3, leaky=0.2):
        super(Discriminator, self).__init__()
        # As stated in the paper, batchnorm is not applied to the disc. input layer.
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, d_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(leaky, inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(d_dim, d_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim * 2),
            nn.LeakyReLU(leaky, inplace=True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(d_dim * 2, d_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim * 4),
            nn.LeakyReLU(leaky, inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(d_dim * 4, d_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim * 8),
            nn.LeakyReLU(leaky, inplace=True)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(d_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # For educational purposes, I've expanded the connections that way.
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        return output
