

import io
import os
import time
from collections import defaultdict, deque
import datetime
import math
import re

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i+1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i+1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('*INFO: Distributed mode disabled.')
        args.distributed = False
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def normal_adjust_lr(optimizer, init_lr, epoch, epochs, base=0.2):
    if (epoch + 1) > 0.95 * epochs:
        factor = math.pow(base, 3) * 0.5
    elif (epoch + 1) > 0.8 * epochs:
        factor = math.pow(base, 3)
    elif (epoch + 1) > 0.6 * epochs:
        factor = math.pow(base, 2)
    elif (epoch + 1) > 0.3 * epochs:
        factor = base
    else:
        factor = 1

    lr = init_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_hp_dict(model_name, ratio, format='none', tt_type='general'):
    match = re.match(r'(tk|tt|svd)?[crm]?_?(.+)', model_name)
    if match.group(1):
        format, model_name = match.group(1), match.group(2)
    if 'tk' == format and 'deit_tiny_patch16_224' == model_name:
        if ratio == '2':
            from hp_dicts.tk_deit_tiny_patch16_224_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tt' == format and 'deit_tiny_patch16_224' == model_name:
        if ratio == '2':
            from hp_dicts.tt_deit_tiny_patch16_224_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tt' == format and 'deit_small_patch16_224' == model_name:
        if ratio == '2':
            from hp_dicts.tt_deit_small_patch16_224_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'resnet32' == model_name:
        if ratio == '1.5':
            from hp_dicts.tk_resnet32_hp import HyperParamsDictRatio1p5x as hp_dict
        elif ratio == '2':
            from hp_dicts.tk_resnet32_hp import HyperParamsDictRatio2x as hp_dict
        elif ratio == '3':
            from hp_dicts.tk_resnet32_hp import HyperParamsDictRatio3x as hp_dict
        elif ratio == '5':
            from hp_dicts.tk_resnet32_hp import HyperParamsDictRatio5x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tt' == format and 'resnet32' == model_name:
        if ratio == '3':
            from hp_dicts.tt_resnet32_hp import HyperParamsDictRatio3x as hp_dict
        elif ratio == '5':
            from hp_dicts.tt_resnet32_hp import HyperParamsDictRatio5x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'resnet56' == model_name:
        if ratio == '2':
            from hp_dicts.tk_resnet56_hp import HyperParamsDictRatio2x as hp_dict
        elif ratio == '3':
            from hp_dicts.tk_resnet56_hp import HyperParamsDictRatio3x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tt' == format and 'resnet56' == model_name:
        if ratio == '2':
            from hp_dicts.tt_resnet56_hp import HyperParamsDictRatio2x as hp_dict
        elif ratio == '3':
            from hp_dicts.tt_resnet56_hp import HyperParamsDictRatio3x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'resnet18' == model_name:
        if ratio == '2':
            from hp_dicts.tk_resnet18_hp import HyperParamsDictRatio2x as hp_dict
        elif ratio == 'sc':
            from hp_dicts.tk_resnet18_hp import HyperParamsDictSC as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tt' == format and 'resnet18' == model_name:
        if ratio == '2' and tt_type == 'general':
            from hp_dicts.tt_resnet18_hp import HyperParamsDictGeneralRatio2x as hp_dict
        elif ratio == '2' and tt_type == 'special':
            from hp_dicts.tt_resnet18_hp import HyperParamsDictSpecialRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tt' == format and 'resnet50' == model_name:
        if ratio == '3' and tt_type == 'general':
            from hp_dicts.tt_resnet50_hp import HyperParamsDictGeneralRatio3x as hp_dict
        elif ratio == '3' and tt_type == 'special':
            from hp_dicts.tt_resnet50_hp import HyperParamsDictSpecialRatio3x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'resnet50' == model_name:
        if ratio == '3':
            from hp_dicts.tk_resnet50_hp import HyperParamsDictRatio3x as hp_dict
        elif ratio == 'sc':
            from hp_dicts.tk_resnet50_hp import HyperParamsDictSC as hp_dict
        elif ratio == '10':
            from hp_dicts.tk_resnet50_hp import HyperParamsDictRatio10x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'mobilenetv2' == model_name:
        if ratio == '2':
            from hp_dicts.tk_mobilenetv2_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tt' == format and 'mobilenetv2' == model_name:
        if ratio == '2':
            from hp_dicts.tt_mobilenetv2_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'svd' == format and 'mobilenetv2' == model_name:
        if ratio == '2':
            from hp_dicts.svd_mobilenetv2_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'mobilenetv2_cifar' == model_name:
        if ratio == '2':
            from hp_dicts.tk_mobilenetv2_cifar_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'svd' == format and 'mobilenetv2_cifar' == model_name:
        if ratio == '2':
            from hp_dicts.svd_mobilenetv2_cifar_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'densenet40' == model_name:
        if ratio == '2':
            from hp_dicts.tk_densenet40_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'densenet121' == model_name:
        if ratio == '2':
            from hp_dicts.tk_densenet121_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'densenet201' == model_name:
        if ratio == '2':
            from hp_dicts.tk_densenet201_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif 'tk' == format and 'vgg16' == model_name:
        if ratio == '2':
            from hp_dicts.tk_vgg16_hp import HyperParamsDictRatio2x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    elif format == 'tk' and model_name == 'vgg16_bn':
        if ratio == '2':
            from hp_dicts.tk_vgg16_bn_hp import HyperParamsDictRatio2x as hp_dict
        elif ratio == '10':
            from hp_dicts.tk_vgg16_bn_hp import HyperParamsDictRatio10x as hp_dict
        else:
            raise Exception('ERROR: Unsupported compression ratio!')
    else:
        hp_dict = None

    return hp_dict
