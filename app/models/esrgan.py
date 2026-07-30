from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional


class ResidualDenseBlock(nn.Module):
    def __init__(self, features: int = 64, growth_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(features + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(features + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(features + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(features + 4 * growth_channels, features, 3, 1, 1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output1 = self.activation(self.conv1(tensor))
        output2 = self.activation(self.conv2(torch.cat((tensor, output1), 1)))
        output3 = self.activation(self.conv3(torch.cat((tensor, output1, output2), 1)))
        output4 = self.activation(
            self.conv4(torch.cat((tensor, output1, output2, output3), 1))
        )
        output5 = self.conv5(
            torch.cat((tensor, output1, output2, output3, output4), 1)
        )
        return output5 * 0.2 + tensor


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, features: int, growth_channels: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(features, growth_channels)
        self.rdb2 = ResidualDenseBlock(features, growth_channels)
        self.rdb3 = ResidualDenseBlock(features, growth_channels)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.rdb3(self.rdb2(self.rdb1(tensor))) * 0.2 + tensor


class RRDBNet(nn.Module):
    def __init__(
        self,
        features: int = 64,
        blocks: int = 23,
        growth_channels: int = 32,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(3, features, 3, 1, 1)
        self.body = nn.Sequential(
            *[
                ResidualInResidualDenseBlock(features, growth_channels)
                for _ in range(blocks)
            ]
        )
        self.conv_body = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_hr = nn.Conv2d(features, features, 3, 1, 1)
        self.conv_last = nn.Conv2d(features, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.conv_first(tensor)
        features = features + self.conv_body(self.body(features))
        features = self.lrelu(
            self.conv_up1(
                functional.interpolate(features, scale_factor=2, mode="nearest")
            )
        )
        features = self.lrelu(
            self.conv_up2(
                functional.interpolate(features, scale_factor=2, mode="nearest")
            )
        )
        return self.conv_last(self.lrelu(self.conv_hr(features)))


def load_esrgan_model(
    weights_path: Path,
    device: torch.device,
    blocks: int = 23,
) -> RRDBNet:
    if not weights_path.is_file():
        raise FileNotFoundError(f"ESRGAN weights not found: {weights_path}")

    model = RRDBNet(blocks=blocks)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    parameters = state.get("params_ema", state)
    model.load_state_dict(parameters)
    return model.eval().to(device)
