#
# This file is a self-contained version of MONAI's ResNetFeatures.
# It includes all necessary dependencies from the MONAI library to run standalone,
# avoiding the need for a full MONAI installation and resolving Python 3.8 compatibility issues.
#
# Original source: https://github.com/Project-MONAI/MONAI
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0
#

from __future__ import annotations

import collections.abc
import logging
from functools import partial
from typing import Any, Callable, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _MaxPoolNd, _AvgPoolNd, _AdaptiveAvgPoolNd

# --- Start of code replicated from MONAI utilities ---

logger = logging.getLogger(__name__)

def ensure_tuple_rep(tup: Any, dim: int) -> tuple[Any, ...]:
    """
    Returns a tuple of `tup` with `dim` values.
    If `tup` is a single value, it is repeated `dim` times.
    """
    if isinstance(tup, collections.abc.Iterable):
        return tuple(tup)
    return (tup,) * dim

def look_up_option(opt: str, supported: set[str]):
    """
    Look up the option in the supported set.
    """
    if opt in supported:
        return opt
    raise ValueError(f"Unsupported option '{opt}', available options are {supported}")

# Replicated from monai.networks.layers.factories
class Conv:
    CONV = "conv"
    TRANSPOSE = "transpose"

    @classmethod
    def get_type(cls, name: str, spatial_dims: int) -> Type[_ConvNd]:
        types = {
            cls.CONV: (nn.Conv1d, nn.Conv2d, nn.Conv3d),
            cls.TRANSPOSE: (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d),
        }
        if name not in types:
            raise ValueError(f"Unsupported convolution type '{name}'")
        if 1 <= spatial_dims <= len(types[name]):
            return types[name][spatial_dims - 1]
        raise ValueError(f"Unsupported spatial_dims '{spatial_dims}'")

class Pool:
    MAX = "max"
    AVG = "avg"
    ADAPTIVEAVG = "adaptiveavg"

    @classmethod
    def get_type(cls, name: str, spatial_dims: int) -> Type[Union[_MaxPoolNd, _AvgPoolNd, _AdaptiveAvgPoolNd]]:
        types = {
            cls.MAX: (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d),
            cls.AVG: (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d),
            cls.ADAPTIVEAVG: (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d),
        }
        if name not in types:
            raise ValueError(f"Unsupported pooling type '{name}'")
        if 1 <= spatial_dims <= len(types[name]):
            return types[name][spatial_dims - 1]
        raise ValueError(f"Unsupported spatial_dims '{spatial_dims}'")

# Replicated from monai.networks.layers.utils
def get_norm_layer(name: Union[tuple, str], spatial_dims: int, channels: int):
    if name == "":
        return nn.Identity()
    if isinstance(name, str):
        name = name.lower()
        if name == "batch":
            layer = nn.BatchNorm1d if spatial_dims == 1 else nn.BatchNorm2d if spatial_dims == 2 else nn.BatchNorm3d
            return layer(channels)
        if name == "instance":
            layer = nn.InstanceNorm1d if spatial_dims == 1 else nn.InstanceNorm2d if spatial_dims == 2 else nn.InstanceNorm3d
            return layer(channels)
    raise ValueError(f"Unsupported normalization layer name: {name}")

def get_act_layer(name: Union[tuple, str]):
    if isinstance(name, str):
        name = name.lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if name == "prelu":
            return nn.PReLU()
        if name == "sigmoid":
            return nn.Sigmoid()
        if name == "tanh":
            return nn.Tanh()
    elif isinstance(name, tuple):
        act_name = name[0].lower()
        act_args = name[1]
        if act_name == "relu":
            return nn.ReLU(**act_args)
        if act_name == "leakyrelu":
            return nn.LeakyReLU(**act_args)
        if act_name == "prelu":
            return nn.PReLU(**act_args)
    raise ValueError(f"Unsupported activation layer name: {name}")

def get_pool_layer(name: Union[tuple, str], spatial_dims: int):
    if isinstance(name, str):
        name = name.lower()
        if name == "max":
            return Pool.get_type(Pool.MAX, spatial_dims)
        if name == "avg":
            return Pool.get_type(Pool.AVG, spatial_dims)
    elif isinstance(name, tuple):
        pool_name = name[0].lower()
        pool_args = name[1]
        if pool_name == "max":
            return Pool.get_type(Pool.MAX, spatial_dims)(**pool_args)
        if pool_name == "avg":
            return Pool.get_type(Pool.AVG, spatial_dims)(**pool_args)
    raise ValueError(f"Unsupported pooling layer name: {name}")


# --- End of code replicated from MONAI utilities ---

resnet_params = {
    # model_name: (block, layers, shortcut_type, bias_downsample)
    "resnet10": ("basic", [1, 1, 1, 1], "B", False),
    "resnet18": ("basic", [2, 2, 2, 2], "A", True),
    "resnet34": ("basic", [3, 4, 6, 3], "A", True),
    "resnet50": ("bottleneck", [3, 4, 6, 3], "B", False),
    "resnet101": ("bottleneck", [3, 4, 23, 3], "B", False),
    "resnet152": ("bottleneck", [3, 8, 36, 3], "B", False),
    "resnet200": ("bottleneck", [3, 24, 36, 3], "B", False),
}

def get_inplanes():
    return [64, 128, 256, 512]

def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Union[nn.Module, partial, None] = None,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        super().__init__()
        conv_type: Type[_ConvNd] = Conv.get_type(Conv.CONV, spatial_dims)

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.act = get_act_layer(name=act)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Union[nn.Module, partial, None] = None,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        super().__init__()
        conv_type: Type[_ConvNd] = Conv.get_type(Conv.CONV, spatial_dims)
        norm_layer = partial(get_norm_layer, name=norm, spatial_dims=spatial_dims)

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(channels=planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(channels=planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(channels=planes * self.expansion)
        self.act = get_act_layer(name=act)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Union[Type[ResNetBlock], Type[ResNetBottleneck], str],
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: Union[tuple[int], int] = 7,
        conv1_t_stride: Union[tuple[int], int] = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        bias_downsample: bool = True,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type = Conv.get_type(Conv.CONV, spatial_dims)
        pool_type = Pool.get_type(Pool.MAX, spatial_dims)
        avgp_type = Pool.get_type(Pool.ADAPTIVEAVG, spatial_dims)

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=tuple(k // 2 for k in conv1_kernel_size),
            bias=False,
        )

        norm_layer = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=self.in_planes)
        self.bn1 = norm_layer
        self.act = get_act_layer(name=act)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: Union[Type[ResNetBlock], Type[ResNetBottleneck]],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
        norm: Union[str, tuple] = "batch",
    ) -> nn.Sequential:
        conv_type = Conv.get_type(Conv.CONV, spatial_dims)
        downsample: Union[nn.Module, partial, None] = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
                    get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes,
                planes=planes,
                spatial_dims=spatial_dims,
                stride=stride,
                downsample=downsample,
                norm=norm,
            )
        ]
        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims, norm=norm))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)
        return x


class ResNetFeatures(ResNet):
    def __init__(self, model_name: str, pretrained: bool = False, spatial_dims: int = 3, in_channels: int = 1) -> None:
        if model_name not in resnet_params:
            model_name_string = ", ".join(resnet_params.keys())
            raise ValueError(f"invalid model_name {model_name} found, must be one of {model_name_string} ")

        if pretrained:
            logger.warning(
                "The 'pretrained' flag is ignored in this standalone version. "
                "Please load weights manually from a local checkpoint file."
            )

        block, layers, shortcut_type, bias_downsample = resnet_params[model_name]
        super().__init__(
            block=block,
            layers=layers,
            block_inplanes=get_inplanes(),
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            conv1_t_stride=2,
            shortcut_type=shortcut_type,
            feed_forward=False,
            bias_downsample=bias_downsample,
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)

        features = [x]
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        
        return features