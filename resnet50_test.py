'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import os
import argparse

from resnet import resnet50
from prefetch_generator import BackgroundGenerator

try:
    from torch import autocast
    from torch.cuda.amp import GradScaler
    use_torch_extension = True
except:
    from apex import amp
    use_torch_extension = False


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
    args = parser.parse_args()
    return args


def data_preparation(batch_size, workers):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoaderX(
        trainset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoaderX(
        testset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


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


def mixup_data_meta(x, y):
    lam = torch.sigmoid(nn.Parameter(torch.rand(1, device=device)))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
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
        if use_torch_extension:
            net = torch.nn.DataParallel(net)
        cudnn.enabled = True
        cudnn.benchmark = True

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return net, criterion, optimizer, scheduler, best_acc, start_epoch


# Training
def train(epoch, trainloader, net, optimizer, criterion, alpha, meta_learning):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    iterator = tqdm(trainloader)
    peak_memory_allocated = 0
    if use_torch_extension:
        for inputs, targets in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            if meta_learning:
                inputs, targets_a, targets_b, lam = mixup_data_meta(inputs, targets)
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
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            descriptor = "batch idx: {}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            iterator.set_description(descriptor)
            batch_idx += 1
    else:
        opt_level = "O1"
        net, optimizer = amp.initialize(net, optimizer, opt_level)
        net = nn.DataParallel(net)
        for inputs, targets in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            if meta_learning:
                inputs, targets_a, targets_b, lam = mixup_data_meta(inputs, targets)
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
            correct += predicted.eq(targets).sum().item()

            descriptor = "batch idx: {}, Loss: {:.3f} | Acc: {:.3f} ({}/{})".format(batch_idx,
                                                                                    train_loss / (batch_idx + 1),
                                                                                    100. * correct / total, correct,
                                                                                    total)
            iterator.set_description(descriptor)
            batch_idx += 1
    peak_memory_allocated += torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    print("Peak memory allocated: {:.2f} GB".format(peak_memory_allocated/1024 ** 3))


def test(epoch, testloader, net):
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
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            descriptor = "batch idx: {}, Acc: {:.3f} ({}/{})".format(batch_idx, 100. * correct / total, correct,
                                                                     total)
            iterator.set_description(descriptor)
            batch_idx += 1

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == "cuda"
    args = parse()
    torch.manual_seed(123456)
    trainloader, testloader, classes = data_preparation(args.bs, args.workers)
    model, criterion, optimizer, scheduler, best_acc, start_epoch = get_model(args, classes)
    if use_torch_extension:
        scaler = GradScaler()
    for epoch in range(start_epoch, start_epoch+args.epoch):
        train(epoch, trainloader, model, optimizer, criterion, args.alpha, args.meta_learning)
        test(epoch, testloader, model)
        scheduler.step()
