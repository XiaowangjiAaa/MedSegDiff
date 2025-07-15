import torch
import torch.nn as nn

class SimpleSegModel(nn.Module):
    """A very small convolutional network for DDPM experiments."""

    def __init__(self, in_channels: int, model_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, *args, **kwargs):
        # ignore timestep embedding and other arguments
        return self.net(x)
