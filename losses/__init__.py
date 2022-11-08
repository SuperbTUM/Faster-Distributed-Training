import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss
}

def Criterion(conf):
    return LOSSES[conf.criterion]()
