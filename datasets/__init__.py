from datasets.cifar10 import Cifar10

# __all__ = [Cifar10]
DATASETS = {
    "cifar_10": Cifar10
}


def Dataset(conf):
    if conf.dataset in DATASETS:
        return DATASETS[conf.dataset]()
    else:
        return None 