

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast
import timm
import utils
import re

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.fx_features import register_notrace_module
from timm.models.layers import ClassifierHead
from timm.models.registry import register_model

from TKConv import TKConv2dC, TKConv2dM, TKConv2dR
from TTConv import TTConv2dM, TTConv2dR
from SVDConv import SVDConv2dC, SVDConv2dM, SVDConv2dR


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'features.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = {
    'vgg11': _cfg(url='https://download.pytorch.org/models/vgg11-bbd30ac9.pth'),
    'vgg13': _cfg(url='https://download.pytorch.org/models/vgg13-c768596a.pth'),
    'vgg16': _cfg(url='https://download.pytorch.org/models/vgg16-397923af.pth'),
    'vgg19': _cfg(url='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'),
    'vgg11_bn': _cfg(url='https://download.pytorch.org/models/vgg11_bn-6002323d.pth'),
    'vgg13_bn': _cfg(url='https://download.pytorch.org/models/vgg13_bn-abd245e5.pth'),
    'vgg16_bn': _cfg(url='https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'),
    'vgg19_bn': _cfg(url='https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'),
}

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


@register_notrace_module  # reason: FX can't symbolically trace control flow in forward method
class TenConvMlp(nn.Module):

    def __init__(self, in_features=512, out_features=4096, kernel_size=7, mlp_ratio=1.0,
                 drop_rate: float = 0.2, act_layer: nn.Module = None, conv_layer: nn.Module = None,
                 ten_conv=TKConv2dC, hp_dict=None, dense_dict=None):
        super(TenConvMlp, self).__init__()
        self.input_kernel_size = kernel_size
        mid_features = int(out_features * mlp_ratio)
        w_name = 'pre_logits.fc1.weight'
        if w_name in hp_dict.ranks:
            self.fc1 = ten_conv(in_features, mid_features, kernel_size, bias=True, hp_dict=hp_dict,
                                name=w_name, dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.fc1 = conv_layer(in_features, mid_features, kernel_size, bias=True)
        self.act1 = act_layer(True)
        self.drop = nn.Dropout(drop_rate)
        w_name = 'pre_logits.fc2.weight'
        if w_name in hp_dict.ranks:
            self.fc2 = SVDConv2dC(mid_features, out_features, 1, bias=True, hp_dict=hp_dict,
                                  name=w_name, dense_w=None if dense_dict is None else dense_dict[w_name])
        else:
            self.fc2 = conv_layer(mid_features, out_features, 1, bias=True)
        self.act2 = act_layer(True)

    def forward(self, x):
        if x.shape[-2] < self.input_kernel_size or x.shape[-1] < self.input_kernel_size:
            # keep the input size >= 7x7
            output_size = (max(self.input_kernel_size, x.shape[-2]), max(self.input_kernel_size, x.shape[-1]))
            x = F.adaptive_avg_pool2d(x, output_size)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class TenVGG(nn.Module):

    def __init__(
            self,
            cfg: List[Any],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            mlp_ratio: float = 1.0,
            act_layer: nn.Module = nn.ReLU,
            conv_layer: nn.Module = nn.Conv2d,
            norm_layer: nn.Module = None,
            global_pool: str = 'avg',
            drop_rate: float = 0.,
            ten_conv=TKConv2dC,
            hp_dict=None,
            dense_dict=None
    ) -> None:
        super(TenVGG, self).__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.feature_info = []
        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool2d
        layers: List[nn.Module] = []
        id = 0
        for v in cfg:
            last_idx = len(layers) - 1
            if v == 'M':
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{last_idx}'))
                layers += [pool_layer(kernel_size=2, stride=2)]
                id += 1
                net_stride *= 2
            else:
                v = cast(int, v)
                w_name = 'features.' + str(id) + '.weight'
                if w_name in hp_dict.ranks:
                    conv2d = ten_conv(prev_chs, v, kernel_size=3, padding=1, hp_dict=hp_dict, name=w_name,
                                      dense_w=None if dense_dict is None else dense_dict[w_name])
                else:
                    conv2d = conv_layer(prev_chs, v, kernel_size=3, padding=1)
                if norm_layer is not None:
                    layers += [conv2d, norm_layer(v), act_layer(inplace=True)]
                    id += 3
                else:
                    layers += [conv2d, act_layer(inplace=True)]
                    id += 2
                prev_chs = v
        self.features = nn.Sequential(*layers)
        self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{len(layers) - 1}'))
        self.pre_logits = TenConvMlp(
            prev_chs, self.num_features, 7, mlp_ratio=mlp_ratio,
            drop_rate=drop_rate, act_layer=act_layer, conv_layer=conv_layer,
            ten_conv=ten_conv, hp_dict=hp_dict, dense_dict=dense_dict)
        self.head = ClassifierHead(
            self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        self._initialize_weights()

    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head = ClassifierHead(
            self.num_features, self.num_classes, pool_type=global_pool, drop_rate=self.drop_rate)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _filter_fn(state_dict):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        k_r = k
        k_r = k_r.replace('classifier.0', 'pre_logits.fc1')
        k_r = k_r.replace('classifier.3', 'pre_logits.fc2')
        k_r = k_r.replace('classifier.6', 'head.fc')
        if 'classifier.0.weight' in k:
            v = v.reshape(-1, 512, 7, 7)
        if 'classifier.3.weight' in k:
            v = v.reshape(-1, 4096, 1, 1)
        out_dict[k_r] = v
    return out_dict


def _ten_vgg(cfg: List[Any],
             num_classes=1000,
             ten_conv=TKConv2dC,
             hp_dict=None,
             dense_dict=None, **kwargs):
    if 'num_classes' in kwargs.keys():
        num_classes = kwargs.get('num_classes')
    model = TenVGG(cfg, num_classes, ten_conv=ten_conv, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if dense_dict is not None:
        ten_dict = model.state_dict()
        for key in ten_dict.keys():
            if key in dense_dict.keys():
                ten_dict[key] = dense_dict[key]
        model.load_state_dict(ten_dict)
    return model


@register_model
def tkc_vgg16(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('vgg16', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _ten_vgg(cfgs['vgg16'], ten_conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict, **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


@register_model
def tkc_vgg16_bn(hp_dict, decompose=False, pretrained=False, path=None, **kwargs):
    if decompose:
        if pretrained:
            dense_dict = timm.create_model('vgg16_bn', pretrained=True).state_dict()
        else:
            dense_dict = torch.load(path, map_location='cpu')
    else:
        dense_dict = None
    model = _ten_vgg(cfgs['vgg16'], norm_layer=nn.BatchNorm2d,
                     ten_conv=TKConv2dC, hp_dict=hp_dict, dense_dict=dense_dict,  **kwargs)
    if pretrained and not decompose:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    baseline = 'vgg16_bn'
    baseline_model = timm.create_model(baseline)
    base_n_params = 0
    for name, p in baseline_model.named_parameters():
        if p.requires_grad:
            base_n_params += p.numel()
    model_name = 'tkc_' + baseline
    hp_dict = utils.get_hp_dict(model_name, ratio='10')
    model = timm.create_model(model_name, hp_dict=hp_dict, decompose=True, pretrained=True)
    # model = timm.create_model(baseline)
    x = torch.randn([1, 3, 224, 224])
    y = model(x)
    n_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print('\'{}\': {},'.format(name, list(p.shape)))
        n_params += p.numel()
    print('Total # parameters: {}'.format(n_params))
    print('Compression ratio: {:.2f}'.format(base_n_params/n_params))
