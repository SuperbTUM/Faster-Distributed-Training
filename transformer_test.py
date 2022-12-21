import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable
from torch import autocast
from torch.cuda.amp import GradScaler

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
from torchdata.datapipes.iter import IterableWrapper

# from torch_ort import ORTModule
"""
from torch_ort import ORTModule
model = ORTModule(model)
"""

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


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    max_size = sequences[0].size()


class AG_NEWS_DATASET:
    def __init__(self, tokenizer=None, batch_size=32, max_sen_len=None, sample_rate=1.0):
        self.tokenizer = tokenizer
        self.specials = ['[UNK]', '[PAD]', '[CLS]']
        self.batch_size = batch_size
        self.max_sen_len = 100
        self.sample_rate = sample_rate
        train_ds = AG_NEWS(root='./data', split='train')
        self.train_ds = train_ds.to_map_datapipe(key_value_fn=transformation)
        test_ds = AG_NEWS(root='./data', split='test')
        test_ds = IterableWrapper([(i, data) for i, data in enumerate(test_ds)])
        self.test_ds = test_ds.to_map_datapipe()


    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (lab, sen) in data_batch:
            batch_sentence.append(sen)
            batch_label.append(lab)

        tmp = self.tokenizer(batch_sentence, padding='longest', return_tensors='pt')

        batch_sentence = tmp['input_ids']
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        attn_mask = tmp['attention_mask']

        return batch_sentence, batch_label, attn_mask

    def load_data(self, distributed=False, num_workers=2):

        # print(self.train_ds.__len__())

        # self.train_ds = torch.utils.data.Subset(self.train_ds, list(range(0, len(self.train_ds), 1000)))
        # self.test_ds = torch.utils.data.Subset(self.test_ds, list(range(0, len(self.test_ds), 10)))

        # print(self.train_ds.__len__())

        if distributed:
            train_sampler = DistributedSampler(dataset=self.train_ds)
            train_dl = DataLoaderX(dataset=self.train_ds,
                                   batch_size=self.batch_size,
                                   sampler=train_sampler,
                                   collate_fn=self.generate_batch,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   persistent_workers=True)
        else:
            train_dl = DataLoaderX(self.train_ds,
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   collate_fn=self.generate_batch,
                                   num_workers=num_workers,
                                   pin_memory=True,
                                   persistent_workers=True)
        test_dl = DataLoaderX(self.test_ds,
                              batch_size=self.batch_size,
                              shuffle=False,
                              collate_fn=self.generate_batch,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=True)

        return train_dl, test_dl


def get_tokenizer_size():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer.vocab_size


def load_dataloader(batch_size, num_workers):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dl, test_dl = AG_NEWS_DATASET(tokenizer, batch_size=batch_size).load_data(args.distributed, num_workers)
    return train_dl, test_dl


def load_best_performance(args, num_class):
    best_acc = 1. / num_class
    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        best_acc = max(best_acc, checkpoint['acc'])
        start_epoch = checkpoint['epoch']
    return best_acc, start_epoch


def load_model(args, num_class, vocab):
    # loss func
    loss_fn = nn.CrossEntropyLoss()
    # get model
    model = Transformer(num_class, vocab)
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
        checkpoint = torch.load('./checkpoint/ckpt.pth')
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


def train(model, criterion, alpha, ngd, rank=0):
    ##
    train_dl, test_dl = load_dataloader(args.batch_size, args.workers)

    model.train()
    correct = 0
    total = 0
    batch_idx = 0
    iterator = tqdm(train_dl)
    peak_memory_allocated = 0
    ##
    if ngd:
        optimizer = NGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    for epoch in range(start_epoch, epochs_total):
        start = time.monotonic()
        for tokens, labels, masks in iterator:
            labels = labels.to(device, non_blocking=True) - 1
            tokens = tokens.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if args.distributed:
                labels = labels.to(rank)
                tokens = tokens.to(rank)
                masks = masks.to(rank)

            tokens, labels_a, labels_b, lam = mixup_data(tokens, labels,
                                                         alpha)
            tokens, labels_a, labels_b = map(Variable, (tokens, labels_a, labels_b))

            # compute gradient and do SGD step
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device, dtype=torch.float16):
                logits = model(tokens, masks.view(masks.shape[0], 1, 1, masks.shape[1]))
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
            correct += predicted.eq(labels).sum().item()

            descriptor = "batch idx: {}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(batch_idx,
                                                                                    loss / (batch_idx + 1),
                                                                                    100. * correct / total, correct,
                                                                                    total)
            iterator.set_description(descriptor)
            batch_idx += 1
        end = time.monotonic()
        ################
        training_time_epoch[epoch - start_epoch] += end - start
        scheduler.step()
        test(epoch, test_dl, model, rank)

    if rank == 0:
        training_accuracy.append(100. * correct / total)
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
    with torch.no_grad():
        for tokens, labels, masks in iterator:
            labels = labels.to(device) - 1
            tokens = tokens.to(device)
            masks = masks.to(device)
            if args.distributed:
                labels = labels.to(rank)
                tokens = tokens.to(rank)
                masks = masks.to(rank)
            logits = model(tokens, masks.view(masks.shape[0], 1, 1, masks.shape[1]))
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
    parser.add_argument("--weight_decay", help="weight decay", default=0, type=float)
    parser.add_argument("--momentum", help="momentum", default=0, type=float)
    args = parser.parse_args()
    return args


args = parse()
num_class = 4
vocab = get_tokenizer_size()
best_acc, start_epoch = load_best_performance(args, num_class)
scaler = GradScaler()

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
    model, criterion = load_model(args, num_class, vocab)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    print("***********************************************************rank: ", rank)
    train(ddp_model, criterion, alpha, ngd, rank)
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
        train(model, criterion, alpha, ngd)
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
