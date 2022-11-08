import torch 
import torch.nn 
import torch.nn.functional as F

import time 

from datasets import Dataset
from optimizers import Optimizer
from losses import Criterion

from utils import ProgressBar

from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.fused import FusedResNet18, FusedResNet34, FusedResNet50, FusedResNet101, FusedResNet152


RESNETS = {
    "ResNet18": ResNet18, 
    "ResNet34": ResNet34, 
    "ResNet50": ResNet50, 
    "ResNet101": ResNet101, 
    "ResNet152": ResNet152,
    "FusedResNet18": FusedResNet18, 
    "FusedResNet34": FusedResNet34,
    "FusedResNet50": FusedResNet50, 
    "FusedResNet101": FusedResNet101, 
    "FusedResNet152": FusedResNet152
}


MODELS = RESNETS


class Model:
    def __init__(self, conf) -> None:
        self.conf = conf
        self._load()
        self.progress_bar = ProgressBar(bar_length=conf.progress_bar_length)

    def _load(self) -> None:
        conf = self.conf
        self.model = MODELS[conf.model]()
        if conf.resume:
            path = "./saved_models/" + conf.model_name + ".pth"
            self.model.load_state_dict(path)

    def train(self, trainloader) -> list:
        conf = self.conf

        self.model.train()
        if conf.device == 'default': 
            conf.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif conf.device in ['cuda', 'cpu']:
            pass 
        else:
            raise ValueError("Device is not appliable.")
        device = torch.device(conf.device)

        self.model.to(device)

        optimizer = Optimizer(self.model, conf)
        criterion = Criterion(conf)

        train_loss = 0.0
        total = 0.0
        correct = 0.0
        for e in range(conf.epochs):
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                self.progress_bar.update(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            # after all batches

            self.save_model()
        
    def save_model(self) -> None:
        conf = self.conf
        path = "./saved_models/" + conf.model_name + ".pth"
        torch.save(self.model.state_dict(), path)

    def evaluate(self, testloader) -> None:
        conf = self.conf
        self.model.eval()
        path = "./saved_models/" + conf.model_name + ".pth"

        if conf.device == 'default': 
            conf.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif conf.device in ['cuda', 'cpu']:
            pass 
        else:
            raise ValueError("Device is not appliable.")
        device = torch.device(conf.device)

        self.model.to(device)
        
        criterion = Criterion(conf)

        test_loss = 0.0
        total = 0.0
        correct = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                self.progress_bar.update(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        return 