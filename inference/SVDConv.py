

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


class SVDConv2dR(Module):
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

        if kernel_size != 1:
            raise ValueError('kernel_size must be 1')
        if stride != 1:
            raise ValueError('stride must be 1')

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(SVDConv2dR, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ranks = hp_dict.ranks[name]
        self.rank = self.ranks if isinstance(self.ranks, int) else self.ranks[0]
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

        self.left_factor = Parameter(torch.Tensor(self.rank, self.in_channels))
        self.right_factor = Parameter(torch.Tensor(self.out_channels, self.rank))

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            u, s, v = np.linalg.svd(dense_w.detach().squeeze().cpu().numpy(), full_matrices=False)
            u = u[:, :self.rank]
            s = s[:self.rank]
            v = v[:self.rank, :]
            self.left_factor.data = torch.from_numpy(u)
            v = np.diag(s) @ v
            self.right_factor.data = torch.from_numpy(v)

        else:
            self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.left_factor)
        init.xavier_uniform_(self.right_factor)
        weight = self._recover_weight()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _recover_weight(self):
        return self.left_factor.mm(self.right_factor).unsqueeze(-1).unsqueeze(-1)

    def _conv_forward(self, x, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, self._recover_weight())


class SVDConv2dC(Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 hp_dict=None, name=str,
                 dense_w=None, dense_b=None):
        super(SVDConv2dC, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if padding_mode != 'zeros':
            raise ValueError("padding_mode must be zero in this mode")
        if groups != 1:
            raise ValueError("groups must be 1 in this mode")
        if kernel_size[0] * kernel_size[1] != 1:
            raise ValueError('kernel_size must be 1 in this mode')
        if stride[0] * stride[1] != 1:
            raise ValueError('stride must be 1')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ranks = hp_dict.ranks[name]
        self.rank = self.ranks if isinstance(self.ranks, int) else self.ranks[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode


        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        self.left_kernel = Parameter(torch.Tensor(self.rank, self.in_channels, *self.kernel_size))
        self.right_kernel = Parameter(torch.Tensor(self.out_channels, self.rank, *self.kernel_size))

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            u, s, v = np.linalg.svd(dense_w.detach().squeeze().cpu().numpy(), full_matrices=False)
            u = u[:, :self.rank]
            s = s[:self.rank]
            v = v[:self.rank, :]
            self.right_kernel.data = torch.from_numpy(u).unsqueeze(-1).unsqueeze(-1)
            self.left_kernel.data = torch.from_numpy(np.diag(s) @ v).unsqueeze(-1).unsqueeze(-1)

        else:
            self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.left_kernel)
        init.xavier_uniform_(self.right_kernel)

    def forward(self, x):
        out = F.conv2d(x, self.left_kernel, None)
        out = F.conv2d(out, self.right_kernel, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def forward_flops(self, x):
        compr_params = (self.left_kernel.numel() + self.right_kernel.numel()) / 1000
        compr_flops = 0
        out = F.conv2d(x, self.left_kernel, None)
        _, _, height_, width_ = out.shape
        compr_flops += height_ * width_ * self.left_kernel.numel() / 1000 / 1000

        out = F.conv2d(out, self.right_kernel, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        _, _, height_, width_ = out.shape
        compr_flops += height_ * width_ * self.right_kernel.numel() / 1000 / 1000

        base_params = self.kernel_size[0] * self.kernel_size[1] * self.in_channels * self.out_channels / 1000
        base_flops = height_ * width_ * self.kernel_size[0] * self.kernel_size[1] * \
                     self.in_channels * self.out_channels / 1000 / 1000

        print('baseline # params: {:.2f}K\t compressed # params: {:.2f}K\t '
              'baseline # flops: {:.2f}M\t compressed # flops: {:.2f}M'.format(base_params, compr_params, base_flops,
                                                                               compr_flops))

        return out, base_flops, compr_flops

    def extra_repr(self) -> str:
        s = 'left_conv(in={}, out={}, kernel_size=(1, 1), bias=False), ' \
            'right_conv(in={}, out={}, kernel_size={}, stride={}, padding={}, bias={}), ' \
            .format(self.in_channels, self.rank,
                    self.rank, self.out_channels, self.kernel_size,
                    self.stride, self.padding, self.bias is None)
        return s


class SVDConv2dM(Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 hp_dict=None, name=str,
                 dense_w=None, dense_b=None):
        super(SVDConv2dM, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if padding_mode != 'zeros':
            raise ValueError("padding_mode must be zero in this mode")
        if groups != 1:
            raise ValueError("groups must be 1 in this mode")
        if kernel_size[0] * kernel_size[1] != 1:
            raise ValueError('kernel_size must be 1 in this mode')
        if stride[0] * stride[1] != 1:
            raise ValueError('stride must be 1')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ranks = hp_dict.ranks[name]
        self.rank = self.ranks if isinstance(self.ranks, int) else self.ranks[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        self.left_factor = Parameter(torch.Tensor(self.rank, self.in_channels, 1, 1))
        self.right_factor = Parameter(torch.Tensor(self.out_channels, self.rank, 1, 1))

        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            u, s, v = np.linalg.svd(dense_w.detach().squeeze().cpu().numpy(), full_matrices=False)
            u = u[:, :self.rank]
            s = s[:self.rank]
            v = v[:self.rank, :]
            self.right_factor.data = torch.from_numpy(u)
            self.left_factor.data = torch.from_numpy(np.diag(s) @ v)

        else:
            self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.left_factor)
        init.xavier_uniform_(self.right_factor)

    def forward(self, x):
        out = F.linear(x.permute(0, 2, 3, 1), self.left_factor)
        out = F.linear(out, self.right_factor, self.bias).permute(0, 3, 1, 2)
        return out
