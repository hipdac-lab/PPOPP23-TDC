
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

from ttd import ten2tt


class TTConv2dM(Module):
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
        super().__init__()

        self.tt_shapes = list(hp_dict.tt_shapes[name])
        self.tt_order = len(self.tt_shapes)
        channels = 1
        for i in range(len(self.tt_shapes)):
            channels *= self.tt_shapes[i]
            if channels == out_channels:
                self.out_tt_order = i + 1
                self.in_tt_order = self.tt_order - self.out_tt_order - 1
                break
        self.out_tt_shapes = self.tt_shapes[:self.out_tt_order]
        self.in_tt_shapes = self.tt_shapes[self.out_tt_order + 1:]

        assert in_channels == int(np.prod(self.in_tt_shapes))
        assert out_channels == int(np.prod(self.out_tt_shapes))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.tt_ranks = list(hp_dict.ranks[name])
        self.out_tt_ranks = self.tt_ranks[:self.out_tt_order + 1]
        self.in_tt_ranks = self.tt_ranks[self.out_tt_order + 1:]

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode

        self.in_tt_cores = ParameterList([Parameter(torch.Tensor(
            self.in_tt_ranks[i], self.in_tt_shapes[i], self.in_tt_ranks[i + 1]))
            for i in range(self.in_tt_order)])

        self.core_kernel = Parameter(Tensor(self.out_tt_ranks[-1], self.in_tt_ranks[0], *self.kernel_size))

        self.out_tt_cores = ParameterList([Parameter(torch.Tensor(
            self.out_tt_ranks[i], self.out_tt_shapes[i], self.out_tt_ranks[i + 1]))
            for i in range(self.out_tt_order)])

        if bias:
            self.bias = Parameter(torch.zeros(self.out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            w = dense_w.detach().cpu().numpy()
            w = np.reshape(w, [self.out_channels, self.in_channels, self.kernel_size[0]*self.kernel_size[1]])
            w = np.transpose(w, [0, 2, 1])
            tt_cores = ten2tt(w, self.tt_shapes, self.tt_ranks)

            for i in range(len(tt_cores)):
                if i < self.out_tt_order:
                    self.out_tt_cores[i].data = torch.from_numpy(tt_cores[i])
                elif i == self.out_tt_order:
                    self.core_kernel.data = torch.from_numpy(tt_cores[i]).permute(0, 2, 1).reshape(
                        self.out_tt_ranks[-1], self.in_tt_ranks[0], self.kernel_size[0], self.kernel_size[1])
                else:
                    self.in_tt_cores[i - self.out_tt_order - 1].data = torch.from_numpy(tt_cores[i])

        else:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.out_tt_order):
            init.xavier_uniform_(self.out_tt_cores[i])
        for i in range(self.in_tt_order):
            init.xavier_uniform_(self.in_tt_cores[i])
        init.xavier_uniform_(self.core_kernel)
        # init.kaiming_uniform_(self.cores[i], a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)

    def get_ranks(self):
        str_ranks = [str(r) for r in self.tt_ranks]
        return ', '.join(str_ranks)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out = x.permute(0, 2, 3, 1)
        for i in range(self.in_tt_order - 1, -1, -1):
            out = torch.mm(
                self.in_tt_cores[i].reshape(self.in_tt_ranks[i], self.in_tt_shapes[i] * self.in_tt_ranks[i + 1]),
                out.reshape(-1, self.in_tt_shapes[i] * self.in_tt_ranks[i + 1]).t()).t()
        out = out.reshape(batch_size, height, width, self.in_tt_ranks[0]).permute(0, 3, 1, 2)

        out = F.conv2d(out, self.core_kernel, None, self.stride, self.padding, self.dilation, self.groups)
        _, _, height_, width_ = out.shape

        out = out.permute(0, 2, 3, 1)
        for i in range(self.out_tt_order - 1, -1, -1):
            out = torch.mm(
                self.out_tt_cores[i].reshape(self.out_tt_ranks[i] * self.out_tt_shapes[i], self.out_tt_ranks[i + 1]),
                out.reshape(-1, self.out_tt_ranks[i + 1]).t())
            out = out.reshape(self.out_tt_ranks[i], -1).t()

        out = out.reshape(self.out_channels, batch_size, height_, width_).permute(1, 0, 2, 3)
        if self.bias is not None:
            out += self.bias

        return out

    def forward_flops(self, x):
        tt_params = 0
        tt_flops = 0
        batch_size, channels, height, width = x.shape
        out = x.permute(0, 2, 3, 1)
        for i in range(self.in_tt_order - 1, -1, -1):
            a = self.in_tt_cores[i].reshape(self.in_tt_ranks[i], self.in_tt_shapes[i] * self.in_tt_ranks[i + 1])
            b = out.reshape(-1, self.in_tt_shapes[i] * self.in_tt_ranks[i + 1]).t()
            out = torch.mm(a, b).t()
            tt_flops += a.shape[0] * a.shape[1] * b.shape[1] / 1000 / 1000
        out = out.reshape(batch_size, height, width, self.in_tt_ranks[0]).permute(0, 3, 1, 2)

        out = F.conv2d(out, self.core_kernel, None, self.stride, self.padding, self.dilation, self.groups)
        _, _, height_, width_ = out.shape
        tt_flops += height_ * width_ * self.core_kernel.numel() / 1000 / 1000

        out = out.permute(0, 2, 3, 1)
        for i in range(self.out_tt_order - 1, -1, -1):
            a = self.out_tt_cores[i].reshape(self.out_tt_ranks[i] * self.out_tt_shapes[i], self.out_tt_ranks[i + 1])
            b = out.reshape(-1, self.out_tt_ranks[i + 1]).t()
            out = torch.mm(a, b).reshape(self.out_tt_ranks[i], -1).t()
            tt_flops += a.shape[0] * a.shape[1] * b.shape[1] / 1000 / 1000

        out = out.reshape(self.out_channels, batch_size, height_, width_).permute(1, 0, 2, 3)
        if self.bias is not None:
            out += self.bias

        base_flops = height_ * width_ * self.kernel_size[0] * self.kernel_size[1] * self.in_channels * \
                     self.out_channels / 1000 / 1000

        for core in self.in_tt_cores:
            tt_params += core.numel()
        for core in self.out_tt_cores:
            tt_params += core.numel()
        tt_params += self.core_kernel.numel()
        base_params = self.kernel_size[0] * self.kernel_size[1] * self.in_channels * self.out_channels

        print('baseline # params: {:.2f}K, tt # params: {:.2f}K'.format(base_params / 1000, tt_params / 1000))
        print('baseline # flops: {:.2f}M, tt # flops: {:.2f}M'.format(base_flops, tt_flops))

        return out, base_flops, tt_flops


class TTConv2dR(Module):
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
                 hp_dict=None,
                 name=str,
                 dense_w: Tensor = None,
                 dense_b: Tensor = None,
                 ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(TTConv2dR, self).__init__()

        self.tt_shapes = list(hp_dict.tt_shapes[name])
        self.tt_order = len(self.tt_shapes)
        channels = 1
        for i in range(len(self.tt_shapes)):
            channels *= self.tt_shapes[i]
            if channels == out_channels:
                self.out_tt_order = i + 1
                self.in_tt_order = self.tt_order - self.out_tt_order - 1
                break
        self.out_tt_shapes = self.tt_shapes[:self.out_tt_order]
        self.in_tt_shapes = self.tt_shapes[self.out_tt_order + 1:]

        # output channels are in front of input channels in the original kernel
        self.tt_ranks = list(hp_dict.ranks[name])
        self.out_tt_ranks = self.tt_ranks[:self.out_tt_order + 1]
        self.in_tt_ranks = self.tt_ranks[self.out_tt_order + 1:]

        assert in_channels == int(np.prod(self.in_tt_shapes))
        assert out_channels == int(np.prod(self.out_tt_shapes))

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

        self.out_tt_cores = ParameterList([Parameter(torch.Tensor(
            self.out_tt_ranks[i], self.out_tt_shapes[i], self.out_tt_ranks[i + 1]))
            for i in range(self.out_tt_order)])

        self.conv_core = Parameter(torch.Tensor(self.out_tt_ranks[-1], self.filter_dim, self.in_tt_ranks[0]))

        self.in_tt_cores = ParameterList([Parameter(torch.Tensor(
            self.in_tt_ranks[i], self.in_tt_shapes[i], self.in_tt_ranks[i + 1]))
            for i in range(self.in_tt_order)])

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
            if dense_b is not None:
                self.bias.data = dense_b
        else:
            self.register_parameter('bias', None)

        if dense_w is not None:
            w = dense_w.detach().cpu().numpy()
            w = np.reshape(w, [self.out_channels, self.in_channels, -1])
            tt_shapes = self.out_tt_shapes + [w.shape[-1]] + self.in_tt_shapes
            tt_cores = ten2tt(w, tt_shapes, self.tt_ranks)

            for i in range(len(tt_cores)):
                if i < self.out_tt_order:
                    self.out_tt_cores[i].data = torch.from_numpy(tt_cores[i])
                elif i == self.out_tt_order:
                    self.conv_core.data = torch.from_numpy(tt_cores[i])
                else:
                    self.in_tt_cores[i - self.out_tt_order - 1].data = torch.from_numpy(tt_cores[i])

        else:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.out_tt_order):
            init.xavier_uniform_(self.out_tt_cores[i])
        init.xavier_uniform_(self.conv_core)
        for i in range(self.in_tt_order):
            init.xavier_uniform_(self.in_tt_cores[i])
        weight = self._recover_weight()
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _recover_weight(self):
        w = self.out_tt_cores[0]
        for i in range(1, self.out_tt_order):
            w = torch.mm(w.reshape(-1, self.out_tt_ranks[i]), self.out_tt_cores[i].reshape(self.out_tt_ranks[i], -1))
        w = torch.mm(w.reshape(-1, self.out_tt_ranks[-1]), self.conv_core.reshape(self.out_tt_ranks[-1], -1))
        for i in range(0, self.in_tt_order):
            w = torch.mm(w.reshape(-1, self.in_tt_ranks[i]), self.in_tt_cores[i].reshape(self.in_tt_ranks[i], -1))

        w = w.reshape(self.out_channels, self.filter_dim, self.in_channels).permute(0, 1, 2).reshape(self.kernel_shape)
        return w

    def _conv_forward(self, x, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(x, self._recover_weight())
