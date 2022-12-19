## Introduction

This project aims to propose a faster distributed training approach with natural gradient descent, mixup, Apex/ORT training, along with a series of training acceleration tricks including non-blocking data loading, module fusion and distributed training with omp and torchrun. We evaluate our proposal with two major deep learning architectures, CNN and transformer, with the tasks being classification tasks.



## Quick Start

Let's start by checking at least four GPUs existing in your server (if distributed training is enabled) with CUDA installed. You can check with `nvidia-smi` command.  Once completed, let's move on with dependency installation.

```bash
pip install -r requirements.txt
```

There are two main entrances you can start with: `resnet50_test.py` and `transformer_test.py` . While `resnet50_test.py` is for resnet50+Cifar10 training, `transformer_test.py` is for transformer+AGNews training.

If you have multi-GPUs and wish to train the model with distributed training, you can execute the bash file with the following commands:

```bash
bash run_distributed.sh
```

If you wish to train the model without distributed training, there are a variety of commands to activate training, one example will be

```bash
python resnet50_test.py --bs 64 --ngd --meta_learning
```

If you wish to fine tune the training hyper-parameters prior to official training, there is also a script `transformer_tuning.py` to generate a subset of the dataset and quickly train with a minor dataset available for learning rate, momentum and weight decay tuning. You can write a shell script to apply grid search.



## Datasets

[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html): image-classification dataset with 10 classes

[AGNews](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html): text-classification dataset with 4 classes



## Meet the Authors

Mingzhe Hu, Columbia Engineering

Lisen Dai, Columbia Engineering