# Copyright (C) 2025 Denso IT Laboratory, Inc.
# All Rights Reserved

import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import numpy as np

from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, SequentialLR, LinearLR, ConstantLR

from convert_sas import convert_layers
from resnet18_model import resnet18, wide_resnet18

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.utils.shampoo_utils import GraftingType

import timm

from torchvision.models import resnet50
from torchvision.models import resnet34


# teacher_model_name = "timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k"
# teacher_model_name = "timm/tf_efficientnet_b3.ns_jft_in1k"
# teacher_model_name = "timm/regnety_016.tv2_in1k"
teacher_model_name = "timm/rexnet_150.nav_in1k"
teacher_model = timm.create_model(teacher_model_name, pretrained=True)


results_dir = "/path/to/your/results_dir/"



def create_dir_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def setup_scheduler(optimizer, args):
    # Main LR Scheduler Setup
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min)
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR, and ExponentialLR are supported.")

    # Warmup Scheduler Setup
    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = LinearLR(optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = ConstantLR(optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs)
        else:
            raise RuntimeError(f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported.")
        
        # Combine warmup and main scheduler
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs])
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler


def fast_collate(batch, memory_format):
    """Based on fast_collate from the APEX example"""
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        nump_array = np.copy(nump_array)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

def parse():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                       'to have subdirectories named "train" and "val"; alternatively,\n' +
                       'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=['resnet18', 'wide_resnet18'],
                        help='model architecture: resnet18 (default) or wide_resnet18')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size per process (default: 256)')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument('--fp16-mode', default=False, action='store_true',
                        help='Enable half precision mode.')
    parser.add_argument('--loss-scale', type=float, default=1)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')
    
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--lr-scheduler", default="cosineannealinglr", type=str, help="the lr scheduler (default: cosine)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")

    parser.add_argument("--experiment_name", default="DefaultExperiment", type=str,
                    help="Name of the experiment (used to create unique result folders)")

    # added
    parser.add_argument(
        '--use_sas',
        action='store_true',
        help='If set, use_sas is True. Otherwise, it is False.'
    )
    parser.add_argument(
        '--use_shampoo',
        action='store_true',
        help='If set, use Shampoo optimizer instead of SGD.'
    )

    parser.add_argument('--mixup', action='store_true',
                        help='Enable MixUp augmentation (requires torchvision.transforms.v2)')
    parser.add_argument('--mixup-alpha', type=float, default=1.0,
                        help='MixUp α parameter (default: 1.0)')


    args = parser.parse_args()
    return args



def distillation_loss(student_logits, teacher_outputs, temperature=1.0):      
    """
    reference
    (https://arxiv.org/abs/2211.16231)

    """

    # student_logits [B, C]
    student_logits = F.log_softmax(student_logits / temperature, dim=-1)
    
    # teacher_outputs [B, C]
    soft_labels = F.softmax(teacher_outputs / temperature, dim=-1)

    
    loss = F.kl_div(student_logits, soft_labels, reduction='batchmean') * (temperature ** 2)
    return loss





# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def main():
    global best_prec1, args
    best_prec1 = 0
    args = parse()

    # added
    use_sas = args.use_sas

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not args.experiment_name:
        args.experiment_name = "DefaultExperiment"

    parent_experiment_dir_name = f"{args.experiment_name}_{current_time}"  
    parent_experiment_dir = os.path.join(results_dir, parent_experiment_dir_name)  

    # Create the parent folder
    create_dir_if_not_exists(parent_experiment_dir) 

    log_dir = os.path.join(parent_experiment_dir, "log")
    tensorboard_dir = os.path.join(parent_experiment_dir, "tensorboard")
    ckpt_dir = os.path.join(parent_experiment_dir, "ckpt")

    create_dir_if_not_exists(log_dir)
    create_dir_if_not_exists(tensorboard_dir)
    create_dir_if_not_exists(ckpt_dir)

    result_suffix = f"epoch{args.epochs}_{current_time}"

    log_file_path = os.path.join(log_dir, f"log_{result_suffix}.txt")
    
    tensorboard_log_path = os.path.join(tensorboard_dir, f"tensorboard_{result_suffix}")

    checkpoint_filename = os.path.join(ckpt_dir, f"checkpoint_{result_suffix}.pth.tar")
    best_model_filename = os.path.join(ckpt_dir, f"model_best_{result_suffix}.pth.tar")


    if not len(args.data):
        raise Exception("error: No data set provided")

    if args.test:
        print("Test mode - only 10 iterations")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = 0

    print("fp16_mode = {}".format(args.fp16_mode))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_prec1 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.arch == 'wide_resnet18':
        model = wide_resnet18()
    else:
        model = resnet18()

    model, count = convert_layers(model, use_sas)
    print(f"Converted {count} layers in {model.__class__.__name__}")
    model = model.to(args.gpu)


   
    if args.local_rank == 0:
        print("=== Model Architecture ===")
        print(model) 
        print("=== End of Model Architecture ===")


    if hasattr(torch, 'channels_last') and  hasattr(torch, 'contiguous_format'):
        if args.channels_last:
            memory_format = torch.channels_last
        else:
            memory_format = torch.contiguous_format
        model = model.cuda().to(memory_format=memory_format)
    else:
        model = model.cuda()

    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256.

    if args.use_shampoo:
        optimizer = DistributedShampoo( 
                                        model.parameters(),
                                        lr=0.1,
                                        betas=(0., 0.999),
                                        epsilon=1e-12,
                                        momentum=0.9,
                                        weight_decay=1e-04,
                                        max_preconditioner_dim=8192,
                                        precondition_frequency=100,
                                        grafting_type=GraftingType.SGD,
                                    )
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    scheduler = setup_scheduler(optimizer, args)

    if args.distributed:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        torch.cuda.current_stream().wait_stream(s)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()


    writer = SummaryWriter(log_dir=tensorboard_log_path)


    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                global best_prec1
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])

                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()

    # Data loading code
    if len(args.data) == 1:
        traindir = os.path.join(args.data[0], 'train')
        valdir = os.path.join(args.data[0], 'val')
    else:
        traindir = args.data[0]
        valdir= args.data[1]

    if args.arch == "inception_v3":
        raise RuntimeError("Currently, inception_v3 is not supported by this example.")
        # crop_size = 299
        # val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256


    train_dataset = datasets.ImageFolder(traindir,
                                            transforms.Compose([transforms.RandomResizedCrop(crop_size),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)]))
    val_dataset = datasets.ImageFolder(valdir,
                                        transforms.Compose([transforms.Resize(val_size),
                                                            transforms.CenterCrop(crop_size)]))

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    collate_fn = lambda b: fast_collate(b, memory_format)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=(train_sampler is None),
                                                num_workers=args.workers,
                                                pin_memory=True,
                                                sampler=train_sampler,
                                                collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True,
                                                sampler=val_sampler,
                                                collate_fn=collate_fn)

    num_classes = len(train_loader.dataset.classes)

    # MixUp（v2 API）
    if args.mixup:
        from torchvision.transforms.v2 import MixUp
        mixup_fn = MixUp(alpha=args.mixup_alpha, num_classes=num_classes)
    else:
        mixup_fn = None


    if args.evaluate:
        validate(val_loader, model, 0, writer)
        writer.close()
        return

    scaler = torch.cuda.amp.GradScaler(init_scale=args.loss_scale,
                                       growth_factor=2,
                                       backoff_factor=0.5,
                                       growth_interval=100,
                                       enabled=args.fp16_mode)
    total_time = AverageMeter()


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        avg_train_time = train(train_loader, teacher_model, model, scaler, optimizer, epoch, writer, mixup_fn=mixup_fn)
        total_time.update(avg_train_time)

        # Update the learning rate
        scheduler.step()

        if args.test:
            break

        # evaluate on validation set
        [prec1, prec5] = validate(val_loader, model, epoch, writer)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if args.use_shampoo:
                optimizer_state = None
            else:
                optimizer_state = optimizer.state_dict()

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer_state
            }, is_best, filename=checkpoint_filename, best_model_filename=best_model_filename)
            if epoch == args.epochs - 1:
                print('##Top-1 {0}\n'
                      '##Top-5 {1}\n'
                      '##Perf  {2}'.format(
                      prec1,
                      prec5,
                      args.total_batch_size / total_time.avg))
                
    writer.close()

class data_prefetcher():
    """Based on prefetcher from the APEX example"""
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        if input is None:
            raise StopIteration
        return input, target

def train(train_loader, teacher_model, model, scaler, optimizer, epoch, writer, mixup_fn=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    total_time = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    data_iterator = data_prefetcher(train_loader)
    data_iterator = iter(data_iterator)

    # Transfer the teacher model to GPU and set it to evaluation mode
    teacher_model = teacher_model.to(args.gpu)
    teacher_model.eval()

    # with open(log_file_path, 'a') as log_file:
    for i, data in enumerate(data_iterator):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        input, target = data
        train_loader_len = len(train_loader)

        input, target = input.to(args.gpu), target.to(args.gpu)

        # original labels
        hard_target = target

        # MixUp（per batch）
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)

        with torch.no_grad():
            # [B, C]
            teacher_outputs = teacher_model(input)


        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        if args.test:
            if i > 10:
                break

        with torch.cuda.amp.autocast(enabled=args.fp16_mode):
            student_outputs = model(input) # [B, C]
            loss = distillation_loss(student_outputs, teacher_outputs)


        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")

        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        scaler.scale(loss).backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        scaler.step(optimizer)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        scaler.update()

        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.
            
            output = student_outputs

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, hard_target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            total_time.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0:

                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Train/Loss', losses.val, global_step)
                writer.add_scalar('Train/Prec@1', top1.val, global_step)
                writer.add_scalar('Train/Prec@5', top5.val, global_step)
                writer.add_scalar('Train/BatchTime', batch_time.val, global_step)
                writer.add_scalar('Train/TotalTime', total_time.sum, epoch)

                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Speed {3:.3f} ({4:.3f})\t'
                    'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    'total_training_time {total_time.sum:.3f}\t'
                    'current_time {current_time}'.format(
                    epoch, i, train_loader_len,
                    args.world_size*args.batch_size/batch_time.val,
                    args.world_size*args.batch_size/batch_time.avg,
                    batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5,
                    total_time=total_time,
                    current_time=current_time))
                """
                log_file.write(f'Epoch: [{epoch}][{i}/{train_loader_len}]\t'
                                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                f'Speed {args.world_size * args.batch_size / batch_time.val:.3f} '
                                f'({args.world_size * args.batch_size / batch_time.avg:.3f})\t'
                                f'Loss {losses.val:.10f} ({losses.avg:.4f})\t'
                                f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                                f'total_training_time {total_time.sum:.3f}\t'
                                f'current_time {current_time}\n')
                """
        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()

    return batch_time.avg

def validate(val_loader, model, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    data_iterator = data_prefetcher(val_loader)
    data_iterator = iter(data_iterator)



    # with open(log_file_path, 'a') as log_file:
    for i, data in enumerate(data_iterator):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        input, target = data
        val_loader_len = len(val_loader)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            global_step = epoch * len(val_loader) + i
            writer.add_scalar('Validation/Loss', losses.val, global_step)
            writer.add_scalar('Validation/Prec@1', top1.val, global_step)
            writer.add_scalar('Validation/Prec@5', top5.val, global_step)
            writer.add_scalar('Validation/BatchTime', batch_time.val, global_step)


            print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Speed {2:.3f} ({3:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                'current_time {current_time}'.format(
                i, val_loader_len,
                args.world_size * args.batch_size / batch_time.val,
                args.world_size * args.batch_size / batch_time.avg,
                batch_time=batch_time, loss=losses,
                top1=top1, top5=top5,
                current_time=current_time))
            """
            log_file.write(
                    f'Test: [{i}/{val_loader_len}]\t'
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Speed {args.world_size * args.batch_size / batch_time.val:.3f} '
                    f'({args.world_size * args.batch_size / batch_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                    f'current_time {current_time}\n'
                )
            """
    if writer and args.local_rank == 0:
        # Log final validation metrics
        writer.add_scalar('Validation/Prec@1_Avg', top1.avg, epoch)
        writer.add_scalar('Validation/Prec@5_Avg', top5.avg, epoch)
        """
        with open(log_file_path, 'a') as log_file:
            log_file.write(
                f'Final Validation Metrics at Epoch {epoch}\n'
                f'Prec@1 Average: {top1.avg:.3f}\n'
                f'Prec@5 Average: {top5.avg:.3f}\n'
            )
        """
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    return [top1.avg, top5.avg]


def save_checkpoint(state, is_best, filename, best_model_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_model_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

if __name__ == '__main__':
    main()






