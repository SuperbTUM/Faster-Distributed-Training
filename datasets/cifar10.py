import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


class Cifar10:
    def __init__(self, root="./data") -> None:
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.root = root 
        self.classes = ['plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.__trainset = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=self.transform_train)  
        self.__testset = torchvision.datasets.CIFAR10(root=self.root, train=False, download=True, transform=self.transform_test)


    def create_trainloader(self, batch_size=128, shuffle=True, num_workers=2):
        train_loader = torch.utils.data.DataLoader(self.__trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_loader

    def create_testloader(self, batch_size=100, shuffle=True, num_workers=2):
        test_loader = torch.utils.data.DataLoader(self.__testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return test_loader