import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

num_gpus = torch.cuda.device_count()


def setup(rank=0, world_size=num_gpus):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def setup_norank(world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl")


def setup_sharedfile(world_size):
    dist.init_process_group("nccl",
                            init_method='file:///mnt/nfs/sharedfile',
                            world_size=world_size,
                            group_name='hpmlGroup')


def cleanup():
    dist.destroy_process_group()


def distributed_wrapper(rank, func, world_size, *func_args):
    """
    This is an example function
    """
    assert len(func_args) == 9
    start_epoch, epoch_total, train_dl, test_dl, model, criterion, optimizer, scheduler, alpha = func_args
    setup(rank, world_size)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    func(start_epoch, epoch_total, train_dl, test_dl, ddp_model, criterion, optimizer, scheduler, alpha)
    cleanup()


def distributed_warpper_runner(distributed_wrapper, world_size, *func_args):
    mp.spawn(distributed_wrapper, args=(world_size, *func_args, ), nprocs=world_size, join=True)


def draw_graph(xs, ys, labels, title, metric):
    plt.figure(figsize=(12, 8))
    if isinstance(xs[0], list) or isinstance(xs[0], np.ndarray):
        for x_list, y_list, label in zip(xs, ys, labels):
            plt.plot(x_list, y_list, label=label, linewidth=2)
        plt.xticks(xs[0])
    else:
        plt.plot(xs, ys, label=labels, linewidth=2)
        plt.xticks(xs)

    plt.xlabel("Epoch/Iteration")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(title + ".png")


