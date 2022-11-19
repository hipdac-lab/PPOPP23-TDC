

import numpy as np
import torch
import pickle

import torchvision.models
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional, Tuple

from TTConv import TTConv2dM, TTConv2dR
from TKConv import TKConv2dM, TKConv2dR, TKConv2dC
from SVDConv import SVDConv2dC, SVDConv2dM, SVDConv2dR
from timm.models.registry import register_model
import timm

import utils


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def tt_conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1,
               conv: Type[Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR]] = TTConv2dR,
               hp_dict=None, name=None, dense_w: Optional = None):
    """3x3 convolution with padding"""
    return conv(in_planes, out_planes, kernel_size=3, stride=stride,
                padding=dilation, groups=groups, bias=False, dilation=dilation,
                hp_dict=hp_dict, name=name, dense_w=dense_w)


def tt_conv1x1(in_planes: int, out_planes: int, stride: int = 1,
               conv: Type[Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR]] = TTConv2dR,
               hp_dict=None, name=None, dense_w: Optional = None):
    """1x1 convolution"""
    if len(hp_dict.ranks[name]) == 1:
        return SVDConv2dC(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                          hp_dict=hp_dict, name=name, dense_w=dense_w)
    else:
        return conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                    hp_dict=hp_dict, name=name, dense_w=dense_w)


class TTBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            stage: int = 1,
            id: int = 0,
            conv: Type[Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR]] = TTConv2dR,
            hp_dict: Optional = None,
            dense_dict: Optional = None
    ) -> None:
        super(TTBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        layer = 'layer' + str(stage) + '.' + str(id) + '.conv1'
        w_name = layer + '.weight'
        if w_name in hp_dict.ranks:
            self.conv1 = tt_conv3x3(inplanes, planes, stride, conv=conv,
                                    hp_dict=hp_dict, name=w_name,
                                    dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        layer = 'layer' + str(stage) + '.' + str(id) + '.conv2'
        w_name = layer + '.weight'
        if w_name in hp_dict.ranks:
            self.conv2 = tt_conv3x3(planes, planes, conv=conv,
                                    hp_dict=hp_dict, name=w_name,
                                    dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

    def forward_flops(self, x, name):
        identity = x
        base_flops = 0
        compr_flops = 0

        print('>{}:'.format(name + 'conv1'))
        if isinstance(self.conv1, (TTConv2dM, TTConv2dR, TKConv2dC, TKConv2dM, TKConv2dR)):
            out, flops1, flops2 = self.conv1.forward_flops(x)
            base_flops += flops1
            compr_flops += flops2
        else:
            out = self.conv1(x)
            base_flops += out.shape[2] * out.shape[3] * self.conv1.weight.numel() / 1000 / 1000
            compr_flops += out.shape[2] * out.shape[3] * self.conv1.weight.numel() / 1000 / 1000
        out = self.bn1(out)
        out = self.relu(out)

        print('>{}:'.format(name + 'conv2'))
        if isinstance(self.conv2, (TTConv2dM, TTConv2dR, TKConv2dC, TKConv2dM, TKConv2dR)):
            out, flops1, flops2 = self.conv2.forward_flops(out)
            base_flops += flops1
            compr_flops += flops2
        else:
            out = self.conv2(out)
            base_flops += out.shape[2] * out.shape[3] * self.conv2.weight.numel() / 1000 / 1000
            compr_flops += out.shape[2] * out.shape[3] * self.conv2.weight.numel() / 1000 / 1000
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out, base_flops, compr_flops


class TTBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            stage: int = 1,
            id: int = 0,
            conv: Type[Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR]] = TTConv2dR,
            hp_dict: Optional = None,
            dense_dict: Optional = None
    ) -> None:
        super(TTBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        layer = 'layer' + str(stage) + '.' + str(id) + '.conv1'
        w_name = layer + '.weight'
        if w_name in hp_dict.ranks:
            self.conv1 = tt_conv1x1(inplanes, width, conv=conv,
                                    hp_dict=hp_dict, name=w_name,
                                    dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        layer = 'layer' + str(stage) + '.' + str(id) + '.conv2'
        w_name = layer + '.weight'
        if w_name in hp_dict.ranks:
            self.conv2 = tt_conv3x3(width, width, stride, groups, dilation,
                                    conv=conv, hp_dict=hp_dict, name=w_name,
                                    dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)

        self.bn2 = norm_layer(width)
        layer = 'layer' + str(stage) + '.' + str(id) + '.conv3'
        w_name = layer + '.weight'
        if w_name in hp_dict.ranks:
            self.conv3 = tt_conv1x1(width, planes * self.expansion,
                                    conv=conv, hp_dict=hp_dict, name=w_name,
                                    dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def forward_flops(self, x, name):
        identity = x
        base_flops = 0
        compr_flops = 0

        print('>{}:'.format(name + 'conv1'))
        if isinstance(self.conv1, (TTConv2dM, TKConv2dC, TKConv2dM, SVDConv2dC, SVDConv2dM)):
            out, flops1, flops2 = self.conv1.forward_flops(x)
            base_flops += flops1
            compr_flops += flops2
        else:
            out = self.conv1(x)
            base_flops += out.shape[2] * out.shape[3] * self.conv1.weight.numel() / 1000 / 1000
            compr_flops += out.shape[2] * out.shape[3] * self.conv1.weight.numel() / 1000 / 1000
        out = self.bn1(out)
        out = self.relu(out)

        print('>{}:'.format(name + 'conv2'))
        if isinstance(self.conv2, (TTConv2dM, TKConv2dC, TKConv2dM, SVDConv2dC, SVDConv2dM)):
            out, flops1, flops2 = self.conv2.forward_flops(out)
            base_flops += flops1
            compr_flops += flops2
        else:
            out = self.conv2(out)
            base_flops += out.shape[2] * out.shape[3] * self.conv2.weight.numel() / 1000 / 1000
            compr_flops += out.shape[2] * out.shape[3] * self.conv2.weight.numel() / 1000 / 1000
        out = self.bn2(out)
        out = self.relu(out)

        print('>{}:'.format(name + 'conv3'))
        if isinstance(self.conv3, (TTConv2dM, TKConv2dC, TKConv2dM, SVDConv2dC, SVDConv2dM)):
            out, flops1, flops2 = self.conv3.forward_flops(out)
            base_flops += flops1
            compr_flops += flops2
        else:
            out = self.conv3(out)
            base_flops += out.shape[2] * out.shape[3] * self.conv3.weight.numel() / 1000 / 1000
            compr_flops += out.shape[2] * out.shape[3] * self.conv3.weight.numel() / 1000 / 1000
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out, base_flops, compr_flops


class TTResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[TTBasicBlock, TTBottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            conv: Type[Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR]] = TTConv2dR,
            hp_dict: Optional = None,
            dense_dict: Optional = None
    ) -> None:
        super(TTResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stage=1, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       stage=2, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       stage=3, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       stage=4, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, TTBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, TTBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[TTBasicBlock, TTBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, stage: int = 1,
                    conv: Type[Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR]] = TTConv2dR,
                    hp_dict: Optional = None, dense_dict: Optional = None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            stage=stage, id=0, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict))
        self.inplanes = planes * block.expansion
        for id in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer,
                                stage=stage, id=id, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_flops(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        base_flops = 0
        compr_flops = 0

        for i, layer in enumerate(self.layer1):
            name = 'layer1.{}.'.format(str(i))
            x, flops1, flops2 = layer.forward_flops(x, name)
            base_flops += flops1
            compr_flops += flops2
        for i, layer in enumerate(self.layer2):
            name = 'layer2.{}.'.format(str(i))
            x, flops1, flops2 = layer.forward_flops(x, name)
            base_flops += flops1
            compr_flops += flops2
        for i, layer in enumerate(self.layer3):
            name = 'layer3.{}.'.format(str(i))
            x, flops1, flops2 = layer.forward_flops(x, name)
            base_flops += flops1
            compr_flops += flops2
        for i, layer in enumerate(self.layer4):
            name = 'layer4.{}.'.format(str(i))
            x, flops1, flops2 = layer.forward_flops(x, name)
            base_flops += flops1
            compr_flops += flops2
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, compr_flops, base_flops


def _tt_resnet(
        block: Type[Union[TTBasicBlock, TTBottleneck]],
        layers: List[int],
        conv: Type[Union[TTConv2dR, TTConv2dM, TKConv2dM, TKConv2dC, TKConv2dR]],
        hp_dict,
        dense_dict: Optional = None,
        **kwargs: Any
) -> TTResNet:
    model = TTResNet(block, layers, conv=conv, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if dense_dict is not None:
        tt_dict = model.state_dict()
        for key in tt_dict.keys():
            if key in dense_dict.keys():
                tt_dict[key] = dense_dict[key]
        model.load_state_dict(tt_dict)

    return model


@register_model
def ttr_resnet18(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet18', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBasicBlock, [2, 2, 2, 2], conv=TTConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def ttm_resnet18(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet18', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBasicBlock, [2, 2, 2, 2], conv=TTConv2dM, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkc_resnet18(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet18', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBasicBlock, [2, 2, 2, 2], conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkm_resnet18(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet18', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBasicBlock, [2, 2, 2, 2], conv=TKConv2dM, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def ttr_resnet34(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet34', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBasicBlock, [3, 4, 6, 3], conv=TTConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def ttr_resnet50(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet50', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBottleneck, [3, 4, 6, 3], conv=TTConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkr_resnet18(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet18', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBasicBlock, [2, 2, 2, 2], conv=TKConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkr_resnet34(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet34', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBasicBlock, [3, 4, 6, 3], conv=TKConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkr_resnet50(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet50', pretrained=True).state_dict()
            # dense_dict = torchvision.models.resnet50(pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBottleneck, [3, 4, 6, 3], conv=TKConv2dR, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkm_resnet50(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet50', pretrained=True).state_dict()
            # dense_dict = torchvision.models.resnet50(pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBottleneck, [3, 4, 6, 3], conv=TKConv2dM, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkc_resnet50(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet50', pretrained=True).state_dict()
            # dense_dict = torchvision.models.resnet50(pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _tt_resnet(TTBottleneck, [3, 4, 6, 3], conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    baseline = 'resnet50'
    model_name = 'tkc_' + baseline
    hp_dict = utils.get_hp_dict(model_name, ratio='3', tt_type='general')
    model = timm.create_model(model_name, hp_dict=hp_dict, decompose=True, pretrained=True)
    compr_params = 0
    for name, p in model.named_parameters():
        # if 'conv' in name or 'fc' in name:
        print(name, p.shape)
        if p.requires_grad:
            compr_params += int(np.prod(p.shape))

    x = torch.randn(1, 3, 224, 224)
    _ = model(x)
    print(compr_params)
    _, compr_flops, base_flops = model.forward_flops(x)
    base_params = 0
    model = timm.create_model(baseline)
    for name, p in model.named_parameters():
        # if 'conv' in name or 'fc' in name:
        # print(name, p.shape)
        if p.requires_grad:
            base_params += int(np.prod(p.shape))
    print('Baseline # parameters: {}'.format(base_params))
    print('Compressed # parameters: {}'.format(compr_params))
    print('Compression ratio: {:.3f}'.format(base_params / compr_params))
    print('Baseline # FLOPs: {:.2f}M'.format(base_flops))
    print('Compressed # FLOPs: {:.2f}M'.format(compr_flops))
    print('FLOPs ratio: {:.3f}'.format(base_flops / compr_flops))
