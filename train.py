import os
import random
import re
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch import hub
from torch.backends import cudnn

from config import Config

from training import models
from training.datasets import get_dataloader
from training.tools.train_utils import parse_args, validate, train

CONFIG = Config()
hub.set_dir(CONFIG['TORCH_HOME'])

torch.backends.cudnn.benchmark = True


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("Initializing Networks")
    model = models.__dict__[args.arch](pretrained=True, num_classes=2)
    model_cfg = model.default_cfg

    print("Initializing Distribution")
    if not torch.cuda.is_available():
        print('Using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    print("Initializing Data Loader")
    train_sampler, train_loader, val_loader = get_dataloader(model_cfg, args)

    loss_functions = {
        "classifier_loss": nn.CrossEntropyLoss().cuda(args.gpu),
        "binary_map_loss": nn.CrossEntropyLoss().cuda(args.gpu),
        "map_loss": nn.MSELoss().cuda(args.gpu)
    }
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    start_epoch = 1
    best_acc1 = 0.
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            state_dict = checkpoint['state_dict']
            if args.distributed:
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            print("Loaded checkpoint '{}' (epoch {}, best_acc {})".format(args.resume, start_epoch - 1, best_acc1))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, args)
        return

    print("Start Training")
    is_main_node = not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)
    better_acc = False
    best_pw_acc = 0.
    for epoch in range(start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(train_loader, model, scaler, optimizer, loss_functions, epoch, args)

        if (epoch % 2 == 0 or epoch == args.epochs) and is_main_node:
            acc1, pw_acc = validate(val_loader, model, args)
            better_acc = best_acc1 < acc1 or best_pw_acc < pw_acc
            if better_acc:
                best_acc1, best_pw_acc = acc1, pw_acc

        # save_model = (better_acc or (epoch == args.epochs) or (epoch % 5 == 0)) and is_main_node
        save_model = better_acc and is_main_node
        if save_model:
            print('Save model')
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'best_acc1': best_acc1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "scaler": scaler.state_dict()
            }, os.path.join('weights', '{}_{}_{}.pt'.format(args.arch, args.prefix, epoch)))
        better_acc = False


def main():
    args = parse_args()
    print(args)
    os.makedirs('weights', exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. ' +
                      'This will turn on the CUDNN deterministic setting, ' +
                      'which can slow down your training considerably! ' +
                      'You may see unexpected behavior when restarting from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(fn=main_worker, args=(ngpus_per_node, args), nprocs=ngpus_per_node)
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
