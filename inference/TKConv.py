

import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import math
import torch.nn as nn

from torch import Tensor
from torch.nn import Parameter, ParameterList
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple
from torch.nn import Conv2d

tl.set_backend('pytorch')


class TKConv2dC(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 hp_dict: Optional = None,
                 name: str = None,
                 dense_w: Tensor = None,
                 dense_b: Tensor = None,
                 ):
        if groups != 1:
            raise ValueError("groups must be 1 in this mode")
        if padding_mode != 'zeros':
            raise ValueError("padding_mode must be zero in this mode")
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(TKConv2dC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ranks = hp_dict.ranks[name]
        self.in_rank = self.ranks[1]
        self.out_rank = self.ranks[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode

        self.first_kernel = Parameter(Tensor(self.in_rank, self.in_channels, 1, 1))
        self.core_kernel = Parameter(Tensor(self.out_rank, self.in_rank, *kernel_size))
        self.last_kernel = Parameter(Tensor(self.out_channels, self.out_rank, 1, 1))

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            core_tensor, [last_factor, first_factor] = partial_tucker(dense_w, modes=[0, 1],
                                                                      rank=self.ranks, init='svd')
            self.first_kernel.data = torch.transpose(first_factor, 1, 0).unsqueeze(-1).unsqueeze(-1)
            self.last_kernel.data = last_factor.unsqueeze(-1).unsqueeze(-1)
            self.core_kernel.data = core_tensor

        else:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.first_kernel)
        init.xavier_uniform_(self.core_kernel)
        init.xavier_uniform_(self.last_kernel)

    def forward(self, x):
        out = F.conv2d(x, self.first_kernel)
        out = F.conv2d(out, self.core_kernel, None, self.stride,
                       self.padding, self.dilation, self.groups)
        out = F.conv2d(out, self.last_kernel, self.bias)
        return out

    def forward_features(self, x):
        features = []
        out = F.conv2d(x, self.first_kernel)
        features.append(out)
        out = F.conv2d(out, self.core_kernel, None, self.stride,
                       self.padding, self.dilation, self.groups)
        features.append(out)
        out = F.conv2d(out, self.last_kernel, self.bias)
        features.append(out)
        return out, features

    def forward_flops(self, x):
        compr_params = (self.first_kernel.numel() + self.core_kernel.numel() +
                        self.last_kernel.numel()) / 1000
        compr_flops = 0
        out = F.conv2d(x, self.first_kernel, None)
        _, _, height_, width_ = out.shape
        compr_flops += height_ * width_ * self.first_kernel.numel() / 1000 / 1000
        out = F.conv2d(out, self.core_kernel, None, self.stride,
                       self.padding, self.dilation, self.groups)
        _, _, height_, width_ = out.shape
        compr_flops += height_ * width_ * self.core_kernel.numel() / 1000 / 1000
        out = F.conv2d(out, self.last_kernel, self.bias)
        _, _, height_, width_ = out.shape
        compr_flops += height_ * width_ * self.last_kernel.numel() / 1000 / 1000

        base_params = self.kernel_size[0] * self.kernel_size[1] * self.in_channels * self.out_channels / 1000
        base_flops = height_ * width_ * self.kernel_size[0] * self.kernel_size[1] * \
                     self.in_channels * self.out_channels / 1000 / 1000

        print('baseline # params: {:.2f}K\t compressed # params: {:.2f}K\t '
              'baseline # flops: {:.2f}M\t compressed # flops: {:.2f}M'.format(base_params, compr_params, base_flops,
                                                                               compr_flops))

        return out, base_flops, compr_flops

    def extra_repr(self):
        s = 'first_conv(in={}, out={}, kernel_size=(1, 1), bias=False), ' \
            'core_conv(in={}, out={}, kernel_size={}, stride={}, padding={}, bias={}), ' \
            'last_conv(in={}, out={}, kernel_size=(1, 1), bias=False)' \
            .format(self.in_channels, self.in_rank,
                    self.in_rank, self.out_rank, self.kernel_size,
                    self.stride, self.padding, self.bias is None,
                    self.out_rank, self.out_channels)
        return s


class TKConv2dM(Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 hp_dict=None, name=str,
                 dense_w=None, dense_b=None):
        super().__init__()

        if groups != 1:
            raise ValueError("groups must be 1 in this mode")
        if padding_mode != 'zeros':
            raise ValueError("padding_mode must be zero in this mode")
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(TKConv2dM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ranks = hp_dict.ranks[name]
        self.in_rank = self.ranks[1]
        self.out_rank = self.ranks[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode

        self.first_factor = Parameter(torch.Tensor(self.in_rank, in_channels))
        self.core_kernel = Parameter(Tensor(self.out_rank, self.in_rank, *self.kernel_size))
        self.last_factor = Parameter(torch.Tensor(out_channels, self.out_rank))

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            core_tensor, (last_factor, first_factor) = partial_tucker(dense_w, modes=(0, 1),
                                                                      rank=(self.out_rank, self.in_rank), init='svd')
            self.first_factor.data = torch.transpose(first_factor, 1, 0)
            self.last_factor.data = last_factor
            self.core_kernel.data = core_tensor
        else:
            self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.first_factor)
        init.xavier_uniform_(self.last_factor)
        init.xavier_uniform_(self.core_kernel)

    def forward(self, x: Tensor) -> Tensor:
        # batch_size, channels, height, width = x.shape
        # out = self.first_factor.mm(x.permute(1, 0, 2, 3).reshape(channels, -1))
        # out = out.reshape(self.in_rank, batch_size, height, width).permute(1, 0, 2, 3)

        out = F.linear(x.permute(0, 2, 3, 1), self.first_factor).permute(0, 3, 1, 2)

        out = F.conv2d(out, self.core_kernel, None, self.stride,
                       self.padding, self.dilation, self.groups)
        out = F.linear(out.permute(0, 2, 3, 1), self.last_factor, self.bias).permute(0, 3, 1, 2)
        # _, _, height_, width_ = out.shape
        # out = self.last_factor.mm(out.permute(1, 0, 2, 3).reshape(self.out_rank, -1))
        # out = out.reshape(self.out_channels, batch_size, height_, width_).permute(1, 0, 2, 3)

        # if self.bias is not None:
        #     out += self.bias

        return out


class TKConv2dR(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 hp_dict: Optional = None,
                 name: str = None,
                 dense_w: Tensor = None,
                 dense_b: Tensor = None,
                 ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ranks = list(hp_dict.ranks[name])
        self.in_rank = self.ranks[1]
        self.out_rank = self.ranks[0]

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        self.kernel_shape = [out_channels, in_channels // groups, *kernel_size]

        self.filter_dim = int(self.kernel_shape[2] * self.kernel_shape[3])

        self.first_factor = Parameter(torch.Tensor(self.in_rank, in_channels))
        self.core_tensor = Parameter(
            torch.Tensor(self.out_rank, self.in_rank, self.kernel_shape[2], self.kernel_shape[3]))
        self.last_factor = Parameter(torch.Tensor(out_channels, self.out_rank))

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            core_tensor, (last_factor, first_factor) = partial_tucker(dense_w, modes=(0, 1),
                                                                      rank=(self.out_rank, self.in_rank), init='svd')
            self.first_factor.data = torch.transpose(first_factor, 1, 0)
            self.last_factor.data = last_factor
            self.core_tensor.data = core_tensor

        else:
            self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.first_factor)
        init.xavier_uniform_(self.core_tensor)
        init.xavier_uniform_(self.last_factor)
        weight = self._recover_weight()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _recover_weight(self):
        return tl.tucker_to_tensor((self.core_tensor, (self.last_factor, self.first_factor.t())))

    def _conv_forward(self, x, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, self._recover_weight())
