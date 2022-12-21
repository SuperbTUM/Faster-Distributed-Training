'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import os
import pickle
import time
from PIL import Image
import numpy as np
import argparse
from typing import Any, Callable, List, Optional, Tuple

from resnet import resnet50
from prefetch_generator import BackgroundGenerator

try:
    from torch import autocast
    from torch.cuda.amp import GradScaler
    use_torch_extension = True
except:
    from apex import amp
    use_torch_extension = False

from utils import *
from ngd_optimizer import NGD
from torchvision_utils import download_and_extract_archive, check_integrity

testing_acc = []


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument("--epoch", default=50, type=int, help="epoch num for training")
    parser.add_argument("--alpha", default=0.99, type=float, help="alpha value for beta distribution")
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--meta_learning", action="store_true", help="adaptive lambda in mixup")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--ngd", action="store_true")
    args = parser.parse_args()
    return args


class StandardTransform:
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)

class VisionDataset(torch.utils.data.Dataset):
    """
    Base Class For making datasets which are compatible with torchvision.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.
    Args:
        root (string): Root directory of dataset.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    .. note::
        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """

    _repr_indent = 4

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        # _log_api_usage_once(self)
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def extra_repr(self) -> str:
        return ""


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        #
        # Transform to tensor
        img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


def get_classes():
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return classes


def get_dataset():
    # Data
    print('==> Preparing data..')
    transform_train = nn.Sequential(
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    )
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    transform_test = nn.Sequential(
        # transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    )
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return trainset, testset


def data_preparation(trainset, batch_size, workers):
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # trainset = torchvision.datasets.CIFAR10(
    #     root='./data', train=True, download=True, transform=transform_train)
    if distributed:
        train_sampler = DistributedSampler(dataset=trainset)
        trainloader = DataLoaderX(trainset, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=workers, pin_memory=True, persistent_workers=True)
    else:
        trainloader = DataLoaderX(
            trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True)
    return trainloader


def data_preparation_test(testset, batch_size, workers):
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoaderX(
        testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True)
    return testloader


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


def mixup_data_meta(x, y, rank):
    batch_size = x.size(0)
    lam = torch.sigmoid(nn.Parameter(torch.rand(batch_size, 1, 1, 1, device=device)).to(rank))
    index = torch.randperm(batch_size, device=device).to(rank)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# class lam_meta(torch.autograd.Function):
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def forward(ctx, inputs, targets, lam):
#         batch_size = inputs.size(0)
#         index = torch.randperm(batch_size, device=device)
#         mixed_x = lam * inputs + (1 - lam) * inputs[index, :]
#         y_a, y_b = targets, targets[index]
#         return mixed_x, y_a, y_b, lam
#
#     @staticmethod
#     @torch.cuda.amp.custom_bwd
#     def backward(ctx, grad_out):
#         return None, None, None, grad_out
#
#
# class lam_meta_module(nn.Module):
#     def __init__(self, batch_size):
#         super(lam_meta_module, self).__init__()
#         self.lam = torch.sigmoid(nn.Parameter(torch.rand(batch_size, 1, 1, 1, device=device)))
#
#     def forward(self, inputs, targets):
#         return lam_meta.apply(inputs, targets, self.lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_criterion_meta(criterion, pred, y_a, y_b, lam):
    criterion.reduction = "none"
    return torch.mean(lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b))


def get_model(args, classes):
    # Model
    print('==> Building model..')
    net = resnet50(len(classes)).to(device)
    if device == 'cuda':
        if use_torch_extension and not args.distributed:
            net = torch.nn.DataParallel(net)
        cudnn.enabled = True
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/resnet_ckpt.pth')
        net.load_state_dict(checkpoint['net'])

    criterion = nn.CrossEntropyLoss()
    return net, criterion


def get_optimizer(net):
    if args.distributed:
        lr = args.lr * 4
    else:
        lr = args.lr
    if args.ngd:
        optimizer = NGD(net.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.2)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return optimizer, scheduler


# If model is not specified as DDP
def average_gradients(model):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= 4


# Training
def train(trainset, testloader, net, criterion, alpha, meta_learning, rank=0):
    optimizer, scheduler = get_optimizer(net)
    trainloader = data_preparation(trainset, args.bs, args.workers)
    # testloader = data_preparation_test(testset, args.bs, args.workers)
    for epoch in range(start_epoch, start_epoch + args.epoch):
        net.train()
        train_loss = torch.zeros(1).to(rank)
        correct = torch.zeros(1).to(rank)
        total = torch.zeros(1).to(rank)
        batch_idx = 0
        peak_memory_allocated = 0
        iterator = tqdm(trainloader)
        ############################
        start = time.monotonic()
        if use_torch_extension:
            for inputs, targets in iterator:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                if distributed:
                    inputs, targets = inputs.to(rank, non_blocking=True), targets.to(rank, non_blocking=True)
                if meta_learning:
                    inputs, targets_a, targets_b, lam = mixup_data_meta(inputs, targets, rank)
                else:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                              targets_a, targets_b))

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=device, dtype=torch.float16):
                    outputs = net(inputs)
                    # loss = criterion(outputs, targets)
                    if meta_learning:
                        loss = mixup_criterion_meta(criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                scaler.scale(loss).backward()
                # if args.distributed:
                #     average_gradients(net)
                scaler.unscale_(optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (lam * predicted.eq(targets_a.data).sum().float()
                            + (1 - lam) * predicted.eq(targets_b.data).sum().float())

                descriptor = "batch idx: {}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(
                    batch_idx,
                    train_loss.cpu().item()/(batch_idx+1),
                    100.*(correct/total).cpu().item(),
                    int(correct.cpu().item()),
                    int(total.cpu().item()))
                iterator.set_description(descriptor)
                batch_idx += 1
        else:
            opt_level = "O1"
            net, optimizer = amp.initialize(net, optimizer, opt_level)
            net = nn.DataParallel(net)
            for inputs, targets in iterator:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                if distributed:
                    inputs, targets = inputs.to(rank), targets.to(rank)
                if meta_learning:
                    inputs, targets_a, targets_b, lam = mixup_data_meta(inputs, targets, rank)
                else:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                                   alpha)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                              targets_a, targets_b))
                optimizer.zero_grad(set_to_none=True)

                outputs = net(inputs)
                # loss = criterion(outputs, targets)
                if meta_learning:
                    loss = mixup_criterion_meta(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (lam * predicted.eq(targets_a.data).sum().float()
                            + (1 - lam) * predicted.eq(targets_b.data).sum().float())

                descriptor = "batch idx: {}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(
                    batch_idx,
                    train_loss.cpu().item()/(batch_idx+1),
                    100.*(correct/total).cpu().item(),
                    int(correct.cpu().item()),
                    int(total.cpu().item()))
                iterator.set_description(descriptor)
                batch_idx += 1
        ################
        end = time.monotonic()

        if args.distributed:
            dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)

        training_time_epoch[epoch-start_epoch] += (end - start)
        training_acc[epoch-start_epoch] += (100. * correct / total).cpu().item()
        peak_memory_allocated += torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        print("Peak memory allocated: {:.2f} GB".format(peak_memory_allocated/1024 ** 3))

        test(epoch, testloader, criterion, net, rank)
        scheduler.step()


def test(epoch, testloader, criterion, net, rank=0):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    criterion.reduce = "mean"
    with torch.no_grad():
        iterator = tqdm(testloader)
        for inputs, targets in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            if distributed:
                inputs, targets = inputs.to(rank), targets.to(rank)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            try:
                test_loss += loss.item()
            except ValueError:
                test_loss += loss.mean().item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            descriptor = "batch idx: {}, Acc: {:.3f} ({}/{})".format(batch_idx, 100. * correct / total, correct,
                                                                     total)
            iterator.set_description(descriptor)
            batch_idx += 1

    acc = 100.*correct/total
    testing_acc.append(acc)
    # Save checkpoint.
    if rank == 0:
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/resnet_ckpt.pth')
            best_acc = acc
        # if args.distributed:
        #     dist.barrier()


def load_best_performance(args, num_class):
    best_acc = 1. / num_class
    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/resnet_ckpt.pth')
        best_acc = max(best_acc, checkpoint['acc'])
        start_epoch = checkpoint['epoch']
    return best_acc, start_epoch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = parse()
distributed = args.distributed
classes = get_classes()
best_acc, start_epoch = load_best_performance(args, len(classes))
trainset, testset = get_dataset()
testloader = data_preparation_test(testset, args.bs, args.workers)


training_acc = np.zeros((args.epoch, ))
training_time_epoch = np.zeros((args.epoch, ))

if use_torch_extension:
    scaler = GradScaler()


# def main_ddp(rank, world_size):
def main_ddp(world_size):
    global testing_acc
    setup_norank(world_size)
    rank = dist.get_rank()
    model, criterion = get_model(args, classes)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    train(trainset, testloader, ddp_model, criterion, args.alpha, args.meta_learning, rank)
    testing_acc = np.asarray(testing_acc)
    draw_graph([np.arange(start=start_epoch, stop=start_epoch+args.epoch) for _ in range(2)],
        [training_acc, testing_acc], ["training", "testing"], "Resnet accuracy curve", "accuracy")
    draw_graph(np.arange(start=start_epoch, stop=start_epoch+args.epoch), training_time_epoch, "training time",
            "Resnet time for training", "time(sec.)")
    cleanup()


if __name__ == "__main__":
    assert device == "cuda"
    torch.manual_seed(123456)
    torch.cuda.empty_cache()
    if distributed:
        world_size = torch.cuda.device_count()
        main_ddp(world_size)
        # distributed_warpper_runner(main_ddp, world_size)
    else:
        model, criterion = get_model(args, classes)
        train(trainset, testloader, model, criterion, args.alpha, args.meta_learning)
        draw_graph([np.arange(start=start_epoch, stop=start_epoch+args.epoch) for _ in range(2)],
                [training_acc, testing_acc], ["training", "testing"], "Resnet accuracy curve", "accuracy")
        draw_graph(np.arange(start=start_epoch, stop=start_epoch+args.epoch), training_time_epoch, "training time",
                "Resnet time for training", "time(sec.)")
