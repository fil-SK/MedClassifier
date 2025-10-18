import torch.nn as nn
import torch.nn.functional as F

class CustomNetMNIST(nn.Module):
    """
    Class for a custom model, provided by the MedMNIST repo.
    """
    def __init__(self, in_channels, num_classes):
        super(CustomNetMNIST, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ----------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution used inside the lightweight residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = F.relu(x, inplace=True)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        return x

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

class CustomNetMNIST(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomNetMNIST, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution used inside the lightweight residual blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = F.relu(x, inplace=True)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        return x


class SqueezeExcitation(nn.Module):
    """Channel attention block to help the network focus on informative features."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Residual building block that combines separable convolutions with SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels)
        self.se = SqueezeExcitation(out_channels, reduction=reduction)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class TissueNetLite(nn.Module):
    """
    Compact CNN tailored for 28x28 MedMNIST grayscale inputs.

    The architecture keeps the spatial resolution high in the early layers,
    mixes in depthwise separable convolutions to reduce parameter count, and
    applies squeeze-and-excitation attention so the network highlights the
    most discriminative channels before global pooling.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # With 28x28 inputs we keep the first stage at stride=1 to preserve
        # detail before downsampling in the later stages.
        self.stage1 = ResidualBlock(32, 64, stride=1)
        self.stage2 = ResidualBlock(64, 128, stride=2)
        self.stage3 = ResidualBlock(128, 160, stride=2)
        self.stage4 = ResidualBlock(160, 192, stride=1)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)