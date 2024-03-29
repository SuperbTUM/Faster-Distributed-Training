import torch
import torch.nn as nn
from torch.optim import SGD
# from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable
from torch import autocast
from torch.cuda.amp import GradScaler
from fairscale.optim.grad_scaler import ShardedGradScaler

from transformers import AutoTokenizer
from transformer import Transformer

import os
import dill as pickle
import time
import argparse
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator

import warnings

warnings.filterwarnings("ignore")

from utils import *
from ngd_optimizer import NGD
from torch.utils.data.distributed import DistributedSampler
# from torchdata.datapipes.iter import IterableWrapper
from torch.utils.data.backward_compatibility import worker_init_fn

import madgrad

# from torch_ort import ORTModule
"""
from torch_ort import ORTModule
model = ORTModule(model)
"""

from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)

import re
from typing import Sequence
from gensim.parsing.preprocessing import remove_stopwords

training_accuracy = []
testing_accuracy = []

device = "cuda" if torch.cuda.is_available() else "cpu"
# used to create key-data pair for map-style dataset
i = -1


def transformation(data):
    global i
    i += 1
    return (i, data)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


def remove_url(data):
    return re.sub(r'\s*(?:https?://)?www\.\S*\.[A-Za-z]{2,5}\s*', ' ', data).strip()


class AG_NEWS_DATASET:
    def __init__(self, tokenizer=None, batch_size=32, max_sen_len=None):
        self.tokenizer = tokenizer
        self.specials = ['[UNK]', '[PAD]', '[CLS]']
        self.batch_size = batch_size
        self.max_sen_len = 100
        train_ds = AG_NEWS(root='./data', split='train')
        self.train_ds = train_ds.to_map_datapipe(key_value_fn=transformation)
        # self.train_ds = IterableWrapper([(i, data) for i, data in enumerate(train_ds)])
        self.test_ds = AG_NEWS(root='./data', split='test')

    def generate_batch(self, data_batch):
        batch_label, batch_sentence_raw = zip(*data_batch)
        batch_sentence = [remove_stopwords(remove_url(striphtml(x))) for x in batch_sentence_raw]

        tmp = self.tokenizer(batch_sentence, padding='longest', return_tensors='pt')

        batch_sentence = tmp['input_ids']
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        batch_token_ids = tmp["token_type_ids"]
        attn_mask = tmp['attention_mask']

        return batch_sentence, batch_label, batch_token_ids, attn_mask

    def load_data(self, distributed=False, num_workers=2):
        if distributed:
            train_sampler = DistributedSampler(dataset=self.train_ds)
            train_dl = DataLoaderX(dataset=self.train_ds,
                                   batch_size=self.batch_size,
                                   sampler=train_sampler,
                                   collate_fn=self.generate_batch,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   persistent_workers=True,
                                   # worker_init_fn=worker_init_fn,
                                   drop_last=True)
        else:
            train_dl = DataLoaderX(self.train_ds,
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   collate_fn=self.generate_batch,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   persistent_workers=True,
                                   worker_init_fn=worker_init_fn,
                                   drop_last=True)
            train_sampler = None
        test_dl = DataLoaderX(self.test_ds,
                              batch_size=self.batch_size,
                              shuffle=False,
                              collate_fn=self.generate_batch,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=True,
                              worker_init_fn=worker_init_fn)

        return train_dl, test_dl, train_sampler


def get_tokenizer_size():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer.vocab_size


def load_dataloader(batch_size, num_workers):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dl, test_dl, train_sampler = AG_NEWS_DATASET(tokenizer, batch_size=batch_size).load_data(args.distributed, num_workers)
    return train_dl, test_dl, train_sampler


def load_best_performance(args, num_class):
    best_acc = 1. / num_class
    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/transformer_ckpt.pth')
        best_acc = max(best_acc, checkpoint['acc'])
        start_epoch = checkpoint['epoch']
    return best_acc, start_epoch


def load_model(args, num_class, vocab):
    # loss func
    loss_fn = nn.CrossEntropyLoss()
    # get model
    model = Transformer(num_class, vocab, alpha=args.alpha)
    model = model.to(device)
    if not args.distributed:
        model = nn.DataParallel(model)
    # model = ORTModule(model)
    cudnn.enabled = True
    cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/transformer_ckpt.pth')
        model.load_state_dict(checkpoint['net'])
    return model, loss_fn


def mixup_data(x, y, alpha=.99):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        distri = torch.distributions.beta.Beta(alpha, alpha)
        lam = distri.sample().item()
    else:
        lam = alpha

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(model, criterion, ngd, rank=0):
    ##
    train_dl, test_dl, train_sampler = load_dataloader(args.batch_size, args.workers)

    model.train()
    peak_memory_allocated = 0
    ##
    if args.distributed:
        lr = args.lr * 4
    else:
        lr = args.lr
    if ngd:
        optimizer = NGD(model.parameters(), lr=lr, weight_decay=0., momentum=0.9)
    else:
        # optimizer = SGD(model.parameters(), lr=lr, weight_decay=0., momentum=0.9)
        optimizer = madgrad.MirrorMADGRAD(model.parameters(), lr=lr, weight_decay=0., momentum=0.9)
        # if args.distributed:
        #     optimizer = ZeroRedundancyOptimizer(params=model.parameters(), optimizer_class=madgrad.MirrorMADGRAD, lr=lr, weight_decay=0., momentum=0.9)
    # scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)
    scheduler = OneCycleLR(optimizer, max_lr=lr * 5, epochs=epochs_total - start_epoch,
                           steps_per_epoch=len(train_dl),
                           cycle_momentum=True)
    
    pos_index = torch.arange(start=0, end=512, device=device)
    # if args.distributed:
    #     pos_index = pos_index.to(rank)

    for epoch in range(start_epoch, epochs_total):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        correct = torch.zeros(1, device=device)#.to(rank)
        total = torch.zeros(1, device=device)#.to(rank)
        batch_idx = 0
        iterator = tqdm(train_dl)

        start = time.monotonic()
        for tokens, labels, token_types, masks in iterator:
            labels = labels.to(device, non_blocking=True) - 1
            tokens = tokens.to(device, non_blocking=True)
            token_types = token_types.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            # if args.distributed:
            #     labels = labels.to(rank)
            #     tokens = tokens.to(rank)
            #     token_types = token_types.to(rank)
            #     masks = masks.to(rank)

            # tokens, labels_a, labels_b, lam = mixup_data(tokens, labels,
            #                                              alpha)
            # tokens, labels_a, labels_b = map(Variable, (tokens, labels_a, labels_b))

            # compute gradient and do SGD step
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device, dtype=torch.float16):
                logits, index, lam = model(tokens, token_types, pos_index, masks.view(masks.shape[0], 1, 1, masks.shape[1]))
                labels_a = labels
                labels_b = labels[index]
                # loss = criterion(logits, labels)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            scaler.step(optimizer)
            scaler.update()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(labels_a.data).sum().float()
                            + (1 - lam) * predicted.eq(labels_b.data).sum().float())

            descriptor = "batch idx: {}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(batch_idx,
                                                                                    loss / (batch_idx + 1),
                                                                                    100. * (correct / total).cpu().item(), int(correct.cpu().item()),
                                                                                    int(total.cpu().item()))
            iterator.set_description(descriptor)
            batch_idx += 1
        end = time.monotonic()
        if args.distributed:
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        ################
        training_time_epoch[epoch - start_epoch] += end - start
        training_accuracy.append(100. * (correct / total).cpu().item())
        scheduler.step()
        test(epoch, test_dl, model, rank)

    if rank == 0:
        peak_memory_allocated += torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        print("Peak memory allocated: {:.2f} GB".format(peak_memory_allocated / 1024 ** 3))


def test(epoch, dataloader, model, rank=0):
    global best_acc
    model.eval()
    iterator = tqdm(dataloader)
    correct = 0
    total = 0
    batch_idx = 0
    index = torch.arange(start=0, end=512, device=device)
    # if args.distributed:
    #     index = index.to(rank)
    with torch.no_grad():
        for tokens, labels, token_types, masks in iterator:
            labels = labels.to(device) - 1
            tokens = tokens.to(device)
            token_types = token_types.to(device)
            masks = masks.to(device)
            # if args.distributed:
            #     labels = labels.to(rank)
            #     tokens = tokens.to(rank)
            #     token_types = token_types.to(rank)
            #     masks = masks.to(rank)
            logits = model(tokens, token_types, index, masks.view(masks.shape[0], 1, 1, masks.shape[1]))
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            descriptor = "batch idx: {}, Acc: {:.3f} ({}/{})".format(batch_idx, 100. * correct / total, correct,
                                                                     total)
            iterator.set_description(descriptor)
            batch_idx += 1

    acc = 100. * correct / total
    testing_accuracy.append(acc)
    # Save checkpoint.
    if rank == 0:
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/transformer_ckpt.pth')
            best_acc = acc
        # if args.distributed:
        #     dist.barrier()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", help="batch size", default=128, type=int)
    parser.add_argument("--epoch", help="epoch", default=50, type=int)
    parser.add_argument("--lr", help="learning rate", default=1e-4, type=float)
    parser.add_argument("--resume", help="continue training", action="store_true")
    parser.add_argument("--workers", help="number of workers", default=2, type=int)
    parser.add_argument("--alpha", help="alpha for beta distribution", default=0.99, type=float)
    parser.add_argument("--distributed", help="whether to turn on distributed training", action="store_true")
    parser.add_argument("--ngd", action="store_true")
    args = parser.parse_args()
    return args


args = parse()
num_class = 4
vocab = get_tokenizer_size()
best_acc, start_epoch = load_best_performance(args, num_class)
scaler = ShardedGradScaler()#GradScaler()

epochs_total = args.epoch
alpha = args.alpha
ngd = args.ngd

training_time_epoch = np.zeros((epochs_total - start_epoch,))


# def main_ddp(rank, world_size):
def main_ddp(world_size):
    global model
    # setup(rank, world_size)
    setup_norank(world_size)
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    model, criterion = load_model(args, num_class, vocab)

    # ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    ddp_model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=size_based_auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        cpu_offload=CPUOffload(offload_params=True),
    )
    print("***********************************************************rank: ", rank)
    train(ddp_model, criterion, ngd, rank)
    draw_graph([np.arange(start=start_epoch, stop=start_epoch + args.epoch) for _ in range(2)],
               [training_accuracy, testing_accuracy], ["training", "testing"], "Transformer accuracy curve", "accuracy")
    draw_graph(np.arange(start=start_epoch, stop=start_epoch + args.epoch), training_time_epoch, "training time",
               "Transformer time for training", "time(sec.)")
    cleanup()


if __name__ == "__main__":
    assert device == "cuda"
    # args = parse()
    # num_class = 4
    # train_dl, test_dl, vocab = load_dataloader(args.batch_size, args.workers)
    # model, criterion, optimizer, scheduler, best_acc, start_epoch = load_model(args, num_class, vocab)
    # scaler = GradScaler()
    torch.cuda.empty_cache()

    if not args.distributed:
        model, criterion = load_model(args, num_class, vocab)
        train(model, criterion, ngd)
        draw_graph([np.arange(start=start_epoch, stop=start_epoch + args.epoch) for _ in range(2)],
                   [training_accuracy, testing_accuracy], ["training", "testing"], "Transformer accuracy curve",
                   "accuracy")
        draw_graph(np.arange(start=start_epoch, stop=start_epoch + args.epoch), training_time_epoch, "training time",
                   "Transformer time for training", "time(sec.)")
    else:
        # -----------------distributed training---------------------- #

        world_size = torch.cuda.device_count()
        main_ddp(world_size)
        # distributed_warpper_runner(main_ddp, world_size)