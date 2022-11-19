


import re
from collections import OrderedDict
from functools import partial

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.jit.annotations import List
import utils

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import BatchNormAct2d, create_norm_act, BlurPool2d, create_classifier
from timm.models.registry import register_model
from TKConv import TKConv2dC, TKConv2dM, TKConv2dR
from TTConv import TTConv2dM, TTConv2dR
from SVDConv import SVDConv2dC, SVDConv2dM, SVDConv2dR

__all__ = ['TenDenseNet']


def _cfg(url=''):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.conv0', 'classifier': 'classifier',
    }


class TenDenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, norm_layer=BatchNormAct2d,
                 drop_rate=0., memory_efficient=False,
                 conv=TKConv2dC, stage=None, id=None, hp_dict=None, dense_dict=None):
        super(TenDenseLayer, self).__init__()
        self.add_module('norm1', norm_layer(num_input_features)),
        w_name = 'features.denseblock' + str(stage) + '.denselayer' + str(id) + '.conv1.weight'
        if w_name in hp_dict.ranks:
            self.add_module('conv1', SVDConv2dC(
                num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False,
                hp_dict=hp_dict, name=w_name, dense_w=None if not dense_dict else dense_dict[w_name])),
        else:
            self.add_module('conv1', nn.Conv2d(
                num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', norm_layer(bn_size * growth_rate)),
        w_name = 'features.denseblock' + str(stage) + '.denselayer' + str(id) + '.conv2.weight'
        if w_name in hp_dict.ranks:
            self.add_module('conv2', conv(
                bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False,
                hp_dict=hp_dict, name=w_name, dense_w=None if not dense_dict else dense_dict[w_name])),
        else:
            self.add_module('conv2', nn.Conv2d(
                bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bottleneck_fn(self, xs):
        # type: (List[torch.Tensor]) -> torch.Tensor
        concated_features = torch.cat(xs, 1)
        bottleneck_output = self.conv1(self.norm1(concated_features))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, x):
        # type: (List[torch.Tensor]) -> bool
        for tensor in x:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, x):
        # type: (List[torch.Tensor]) -> torch.Tensor
        def closure(*xs):
            return self.bottleneck_fn(xs)

        return cp.checkpoint(closure, *x)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (List[torch.Tensor]) -> (torch.Tensor)
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, x):
        # type: (torch.Tensor) -> (torch.Tensor)
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, x):  # noqa: F811
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bottleneck_fn(prev_features)

        new_features = self.conv2(self.norm2(bottleneck_output))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class TenDenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, norm_layer=nn.ReLU,
                 drop_rate=0., memory_efficient=False,
                 conv=TKConv2dC, stage=None, hp_dict=None, dense_dict=None):
        super(TenDenseBlock, self).__init__()
        for i in range(num_layers):
            layer = TenDenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                conv=conv, stage=stage, id=i+1, hp_dict=hp_dict, dense_dict=dense_dict
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class TenDenseTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, norm_layer=nn.BatchNorm2d, aa_layer=None,
                 stage=None, hp_dict=None, dense_dict=None):
        super(TenDenseTransition, self).__init__()
        self.add_module('norm', norm_layer(num_input_features))
        w_name = 'features.transition' + str(stage) + '.conv.weight'
        if w_name in hp_dict.ranks:
            self.add_module('conv', SVDConv2dC(
                num_input_features, num_output_features, kernel_size=1, stride=1, bias=False,
                hp_dict=hp_dict, name=w_name, dense_w=None if dense_dict is None else dense_dict[w_name]))
        else:
            self.add_module('conv', nn.Conv2d(
                num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if aa_layer is not None:
            self.add_module('pool', aa_layer(num_output_features, stride=2))
        else:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class TenDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), bn_size=4, stem_type='',
                 num_classes=1000, in_chans=3, global_pool='avg',
                 norm_layer=BatchNormAct2d, aa_layer=None, drop_rate=0, memory_efficient=False,
                 aa_stem_only=True,
                 conv=TKConv2dC, hp_dict=None, dense_dict=None):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(TenDenseNet, self).__init__()

        # Stem
        deep_stem = 'deep' in stem_type  # 3x3 deep stem
        num_init_features = growth_rate * 2
        if aa_layer is None:
            stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            stem_pool = nn.Sequential(*[
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                aa_layer(channels=num_init_features, stride=2)])
        if deep_stem:
            stem_chs_1 = stem_chs_2 = growth_rate
            if 'tiered' in stem_type:
                stem_chs_1 = 3 * (growth_rate // 4)
                stem_chs_2 = num_init_features if 'narrow' in stem_type else 6 * (growth_rate // 4)
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False)),
                ('norm0', norm_layer(stem_chs_1)),
                ('conv1', nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False)),
                ('norm1', norm_layer(stem_chs_2)),
                ('conv2', nn.Conv2d(stem_chs_2, num_init_features, 3, stride=1, padding=1, bias=False)),
                ('norm2', norm_layer(num_init_features)),
                ('pool0', stem_pool),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(in_chans, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', norm_layer(num_init_features)),
                ('pool0', stem_pool),
            ]))
        self.feature_info = [
            dict(num_chs=num_init_features, reduction=2, module=f'features.norm{2 if deep_stem else 0}')]
        current_stride = 4

        # DenseBlocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = TenDenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                conv=conv, stage=i+1, hp_dict=hp_dict, dense_dict=dense_dict
            )
            module_name = f'denseblock{(i + 1)}'
            self.features.add_module(module_name, block)
            num_features = num_features + num_layers * growth_rate
            transition_aa_layer = None if aa_stem_only else aa_layer
            if i != len(block_config) - 1:
                self.feature_info += [
                    dict(num_chs=num_features, reduction=current_stride, module='features.' + module_name)]
                current_stride *= 2
                trans = TenDenseTransition(
                    num_input_features=num_features, num_output_features=num_features // 2,
                    norm_layer=norm_layer, aa_layer=transition_aa_layer,
                    stage=i+1, hp_dict=hp_dict, dense_dict=dense_dict)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', norm_layer(num_features))

        self.feature_info += [dict(num_chs=num_features, reduction=current_stride, module='features.norm5')]
        self.num_features = num_features

        # Linear layer
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.classifier = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x):
        return self.features(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        # both classifier and block drop?
        # if self.drop_rate > 0.:
        #     x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x


def _filter_torchvision_pretrained(state_dict):
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def _ten_densenet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000,
                  conv=TKConv2dC, hp_dict=None, dense_dict=None, **kwargs):
    if 'num_classes' in kwargs.keys():
        num_classes = kwargs.get('num_classes')
    model = TenDenseNet(growth_rate, block_config, num_classes=num_classes,
                        conv=conv, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)

    if dense_dict is not None:
        ten_dict = model.state_dict()
        for key in ten_dict.keys():
            if key in dense_dict.keys():
                ten_dict[key] = dense_dict[key]
        model.load_state_dict(ten_dict)
    return model


@register_model
def tkc_densenet121(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet121', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _ten_densenet(growth_rate=32, block_config=(6, 12, 24, 16),
                          conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkc_densenet264(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet264', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _ten_densenet(growth_rate=48, block_config=(6, 12, 64, 48),
                          conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkc_densenet201(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('resnet201', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _ten_densenet(growth_rate=32, block_config=(6, 12, 48, 32),
                          conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    baseline = 'densenet201'
    baseline_model = timm.create_model(baseline)
    base_n_params = 0
    for name, p in baseline_model.named_parameters():
        if p.requires_grad:
            if 'conv' in name:
                print('\'{}\': {},'.format(name, list(p.shape)))
            base_n_params += p.numel()
    model_name = 'tkc_' + baseline
    hp_dict = utils.get_hp_dict(model_name, ratio='2')
    model = timm.create_model(model_name, hp_dict=hp_dict, decompose=None)
    x = torch.randn([1, 3, 224, 224])
    y = model(x)
    n_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'conv' in name:
            print('\'{}\': {},'.format(name, list(p.shape)))
        n_params += p.numel()
    print('Total # parameters: {}'.format(n_params))
    print('Compression ratio: {:.2f}'.format(base_n_params/n_params))

