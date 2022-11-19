from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy

from datasets import get_data_loader
import torch
import timm
import os

import utils
from parse_args import parse_args

import resnet_inet_tt
import densenet_inet_tt
import vgg_tt


@torch.no_grad()
def evaluate(data_loader, model, device, print_freq=100):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Evaluation Result: Acc@1 {top1.global_avg:.3f}%, Acc@5 {top5.global_avg:.3f}%, loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def eval(model, args):
    device = torch.device(args.device)
    model.to(device)

    val_loader = get_data_loader(False, args)
    evaluate(val_loader, model, device, args.print_freq)


if __name__ == '__main__':
    args = parse_args()
    args.num_classes = 1000
    args.image_folder = True

    if not os.path.exists('./saved_models'):
        os.mkdir('./saved_models')

    if args.model == 'tkc_resnet18':
        hp_dict = utils.get_hp_dict(args.model, 'sc')
        args.model_path = './saved_models/tkc_resnet18.pt'
        file_id = '1UgP1zFavHGc7Jjre0YyXTxXCBYyJ-aqq'
    elif args.model == 'tkc_resnet50':
        hp_dict = utils.get_hp_dict(args.model, 'sc')
        args.model_path = './saved_models/tkc_resnet50.pt'
        file_id = '1Jhhrjlvd9byLIb7cUL2aU6d6i8kWOHbz'
    elif args.model == 'tkc_vgg16':
        args.model = 'tkc_vgg16_bn'
        hp_dict = utils.get_hp_dict(args.model, '10')
        args.model_path = './saved_models/tkc_vgg16.pt'
        file_id = '1_zYHW4xE7DUkV54pmPgX0wDYRIwxU9gI'
    elif args.model == 'tkc_densenet121':
        hp_dict = utils.get_hp_dict(args.model, '2')
        args.model_path = './saved_models/tkc_densenet121.pt'
        file_id = '1u8MXt1Z2XfTdiyABfx-nGwvsCzMxTaMo'
    elif args.model == 'tkc_densenet201':
        hp_dict = utils.get_hp_dict(args.model, '2')
        args.model_path = './saved_models/tkc_densenet201.pt'
        file_id = ''
    else:
        raise Exception('ERROR: Unsupported model!')

    print('Evaluating...')
    model = timm.create_model(
        args.model,
        pretrained=True,
        num_classes=args.num_classes,
        decompose=False,
        path=args.model_path,
        hp_dict=hp_dict,
    )
    eval(model, args)