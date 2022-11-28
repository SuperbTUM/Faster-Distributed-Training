import argparse

from config import Conf
from models import Model
from datasets import Dataset

def main(conf):
    model = Model(conf=conf)

    if conf.train:
        trainloader = Dataset(conf=conf).create_trainloader(batch_size=conf.batch_size, shuffle=conf.shuffle, num_workers=conf.num_workers)
        model.train(trainloader=trainloader)
    if conf.eval:
        trainloader = Dataset(conf=conf).create_trainloader(batch_size=conf.batch_size, shuffle=conf.shuffle, num_workers=conf.num_workers)
        model.evaluate(trainloader=trainloader)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument("-c", "--conf", default="./conf/train.json", type=str, help="conf path")
    args = parser.parse_args()

    conf = Conf(args.conf)
    print("Hyperparameters & Settings:", conf)

    main(conf)
