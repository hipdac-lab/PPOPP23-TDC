import os
import json
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import utils


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10('~/datasets/cifar10', train=is_train, transform=transform)
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100('~/datasets/cifar100', train=is_train, transform=transform)
    elif args.dataset == 'imagenet':
        if os.path.exists('2080.work'):
            data_path = '~/datasets/imagenet/'
        elif os.path.exists('dgx.work'):
            data_path = '/raid/data/ilsvrc2012/'
        else:
            data_path = '/home/datasets/imagenet/'
        root = os.path.join(data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        raise Exception('ERROR: Unsupported dataset!')

    return dataset


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if args.dataset == 'cifar10':
        t.append(transforms.Normalize((0.4914, 0.4822, 0.4465),
                                      (0.2023, 0.1994, 0.2010)))
    elif args.dataset == 'cifar100':
        t.append(transforms.Normalize((0.5071, 0.4867, 0.4408),
                                      (0.2675, 0.2565, 0.2761)))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def get_data_loader(is_train, args):
    if args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('~/datasets/cifar10',
                                         train=True, download=True,
                                         transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       transforms.ToTensor(),
                                                                       transforms.Normalize(
                                                                           (0.4914, 0.4822, 0.4465),
                                                                           (0.2023, 0.1994, 0.2010))]))
        dataset_val = datasets.CIFAR10('~/datasets/cifar10', train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.2023, 0.1994, 0.2010))
                                       ]))
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('~/datasets/cifar100',
                                          train=True, download=True,
                                          transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                                        transforms.RandomHorizontalFlip(),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(
                                                                            (0.5071, 0.4867, 0.4408),
                                                                            (0.2675, 0.2565, 0.2761))]))
        dataset_val = datasets.CIFAR100('~/datsets/cifar100', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                 (0.2675, 0.2565, 0.2761))
                                        ]))
    elif args.dataset == 'imagenet':

        if args.image_folder:
            #             train_path = os.path.join(args.data_path, 'train')
            #             dataset_train = datasets.ImageFolder(train_path,
            #                                                  transform=transforms.Compose(
            #                                                      [transforms.RandomResizedCrop(args.input_size),
            #                                                       transforms.RandomHorizontalFlip(),
            #                                                       transforms.ToTensor(),
            #                                                       transforms.Normalize((0.485, 0.456, 0.406),
            #                                                                            (0.229, 0.224, 0.225))]))

            val_path = os.path.join(args.data_path, 'val')
            dataset_val = datasets.ImageFolder(val_path,
                                               transform=transforms.Compose([
                                                   transforms.Resize(int(256 / 224 * args.input_size)),
                                                   transforms.CenterCrop(args.input_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                                        (0.229, 0.224, 0.225))
                                               ]))
        else:
            dataset_train = datasets.ImageNet(args.data_path, split='train',
                                              transform=transforms.Compose(
                                                  [transforms.RandomResizedCrop(args.input_size),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.485, 0.456, 0.406),
                                                                        (0.229, 0.224, 0.225))]))

            dataset_val = datasets.ImageNet(args.data_path, split='val',
                                            transform=transforms.Compose([
                                                transforms.Resize(int(256 / 224 * args.input_size)),
                                                transforms.CenterCrop(args.input_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                     (0.229, 0.224, 0.225))
                                            ]))
    elif args.dataset == 'mnist':
        dataset_train = datasets.MNIST('~/datasets/mnist', train=True, download=True,
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))
        dataset_val = datasets.MNIST('~/datasets/mnist', train=False, download=True,
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Normalize((0.1307,), (0.3081,))]))
    else:
        raise Exception('ERROR: Unspecified dataset!')

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    else:
        sampler_train = None
        sampler_val = None
    if is_train:
        data_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=sampler_train is None,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory, sampler=sampler_train)
    else:
        data_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=args.pin_memory, sampler=sampler_val)

    return data_loader