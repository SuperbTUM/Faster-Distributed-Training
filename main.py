import argparse

from models import Model
from datasets import Dataset

def main(args):
    conf = args
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
    parser.add_argument('--epochs', '-e', default=5, type=int, help='epochs')
    parser.add_argument('--device', default='default', type=str,
                        help='device, if not specified, it will be cuda if any, otherwise cpu')
    parser.add_argument('--dataset', '-d', default='cifar_10',
                        type=str, help='dataset to use')
    parser.add_argument('--shuffle', '-s',
                        action="store_false", help='if shuffle dataset')
    parser.add_argument('--num_workers', default=2,
                        type=int, help='dataset to use')
    parser.add_argument('--optimizer', '-o', default='SGD',
                        type=str, help='optimizer')
    parser.add_argument('--model', '-m', default='ResNet18',
                        type=str, help='model')
    parser.add_argument('--model_name', '-n',
                        default='my_model', type=str, help='model name')
    parser.add_argument('--learning_rate', '-l', default=0.1,
                        type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9,
                        type=float, help='momentum for optimizer')
    parser.add_argument('--betas', default="(0.9, 0.999)",
                        type=str, help="coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))")
    parser.add_argument('--eps', default=1e-08,
                        type=float, help='term added to the denominator to improve numerical stability (default: 1e-8)')
    parser.add_argument('--batch_size', '-b', default=128,
                        type=int, help='batch size')
    parser.add_argument('--weight_decay', '-w', default=5e-4,
                        type=float, help='weight_decay')
    parser.add_argument('--criterion', '-c',
                        default='cross_entropy', type=str, help='loss')
    # or default = 65
    parser.add_argument('--progress_bar_length', default=65,
                        type=int, help="training progress bar length")

    parser.add_argument('--train', action="store_true", help='train the model')
    parser.add_argument('--eval', action="store_true", help='eval the model')

    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')

    args = parser.parse_args()

    print("Hyperparameters & Settings:", args)

    main(args)
