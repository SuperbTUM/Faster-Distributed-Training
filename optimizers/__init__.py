import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


OPTIMIZERS = {
    "SGD": optim.SGD,
    "SGD-nesterov": optim.SGD, 
    "Adam": optim.Adam,
    "Adagrad": optim.Adagrad,
    "Adadelta": optim.Adadelta,
}

def Optimizer(model, conf):
    if conf.optimizer == "SGD":
        return OPTIMIZERS[conf.optimizer](
            model.parameters(), 
            lr=conf.learning_rate,
            momentum=conf.momentum,
            weight_decay=conf.weight_decay
        )
    elif conf.optimizer == "SGD-nesterov":
        return OPTIMIZERS[conf.optimizer](
            model.parameters(), 
            lr=conf.learning_rate,
            momentum=conf.momentum,
            weight_decay=conf.weight_decay,
            nesterov=True,
        )
    elif conf.optimizer == "Adam":
        return OPTIMIZERS[conf.optimizer](
            model.parameters(), 
            betas=eval(conf.betas),
            eps=conf.eps,
            lr=conf.learning_rate,
            weight_decay=conf.weight_decay
        )
    elif conf.optimizer in ["Adagrad", "Adadelta"]:
        return OPTIMIZERS[conf.optimizer](
            model.parameters(), 
            lr=conf.learning_rate,
            weight_decay=conf.weight_decay
        )
    else:
        raise NotImplementedError
