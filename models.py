import torch
from torch import nn


class CNN_1(nn.Module):
    def __init__(self, nc, h, w, ndf):
        super(CNN_1, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=nc, out_channels=ndf//4, kernel_size=(2, 2), stride=1, padding=1)
        self.l_relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.ConvTranspose2d(in_channels=ndf//4, out_channels=ndf//2, kernel_size=(2, 2), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ndf//2)
        self.l_relu2 = nn.LeakyReLU(0.1)
        # self.conv3 = nn.ConvTranspose2d(in_channels=ndf//2, out_channels=ndf, kernel_size=(2, 2), stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(ndf)
        # self.l_relu3 = nn.LeakyReLU(0.1)
        self.conv5 = nn.ConvTranspose2d(in_channels=ndf//2, out_channels=ndf, kernel_size=(2, 2), stride=1, padding=1)
        self.linear = nn.Linear(921888, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.l_relu1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.l_relu2(x)
        # x = self.conv3(x)
        # x = self.bn2(x)
        # x = self.l_relu3(x)
        # x = self.conv4(x)
        # x = self.bn3(x)
        # x = self.l_relu4(x)
        x = self.conv5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return nn.Sigmoid()(x)