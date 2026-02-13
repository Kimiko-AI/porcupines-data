"""
CIFAR-adapted Spiking ResNet-18 using SpikingJelly.

The standard SpikingJelly spiking_resnet18 has an ImageNet-sized stem
(7×7 conv + maxpool) that over-downsamples 32×32 CIFAR images.  This
module replaces the stem with a single 3×3 conv (stride 1, no maxpool)
and sets input channels to 2 (ON/OFF event polarity from I2E).
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional, surrogate


# ---------------------------------------------------------------------------
# BasicBlock (identical to SpikingJelly's, kept here to avoid import issues)
# ---------------------------------------------------------------------------

def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return layer.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation,
    )


def _conv1x1(in_planes, out_planes, stride=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=1,
                        stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn2(out)
        return out


# ---------------------------------------------------------------------------
# CIFAR Spiking ResNet
# ---------------------------------------------------------------------------

class SpikingResNetCIFAR(nn.Module):
    """Spiking ResNet adapted for CIFAR-10 with event-stream input.

    Differences from the ImageNet variant:
    - ``conv1`` is 3×3/stride-1 (not 7×7/stride-2)
    - No ``maxpool`` layer
    - ``in_channels`` defaults to 2 (ON/OFF polarity channels from I2E)
    """

    def __init__(
        self,
        block=BasicBlock,
        layers: list[int] = None,
        num_classes: int = 10,
        in_channels: int = 2,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]  # ResNet-18
        if spiking_neuron is None:
            spiking_neuron = neuron.LIFNode

        self._norm_layer = layer.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1

        # CIFAR stem: 3×3/stride-1, no maxpool
        self.conv1 = layer.Conv2d(
            in_channels, self.inplanes,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))

        self.layer1 = self._make_layer(block, 64,  layers[0],
                                        spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        spiking_neuron=spiking_neuron, **kwargs)

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        # Weight init
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init last BN per block
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1,
                    spiking_neuron=None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers_ = [
            block(self.inplanes, planes, stride, downsample,
                  norm_layer=norm_layer, spiking_neuron=spiking_neuron,
                  **kwargs)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_.append(
                block(self.inplanes, planes,
                      norm_layer=norm_layer,
                      spiking_neuron=spiking_neuron, **kwargs)
            )
        return nn.Sequential(*layers_)

    def forward(self, x):
        """Forward pass for a single timestep.

        Parameters
        ----------
        x : Tensor
            ``(B, C, H, W)`` event frame for one timestep.

        Returns
        -------
        Tensor
            ``(B, num_classes)`` membrane potential / firing rate.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def spiking_resnet18_cifar(
    num_classes: int = 10,
    in_channels: int = 2,
    tau: float = 2.0,
    **kwargs,
) -> SpikingResNetCIFAR:
    """Create a CIFAR-adapted Spiking ResNet-18.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 10 for CIFAR-10).
    in_channels : int
        Input channels (default 2 for ON/OFF I2E events).
    tau : float
        LIF neuron time constant.
    **kwargs
        Extra kwargs forwarded to the spiking neuron.
    """
    return SpikingResNetCIFAR(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
        spiking_neuron=neuron.LIFNode,
        tau=tau,
        surrogate_function=surrogate.ATan(alpha=2.0),
        detach_reset=True,
        **kwargs,
    )
