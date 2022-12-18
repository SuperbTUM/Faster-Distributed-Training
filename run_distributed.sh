#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=12 python -m torch.distributed.run --master_port=12355 --nnodes=1 --nproc_per_node=4 ./resnet50_test.py --workers 4 --bs 256 --distributed --meta_learning --ngd
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=12 python -m torch.distributed.run --master_port=12355 --nnodes=1 --nproc_per_node=4 ./transformer_test.py --workers 4 --batch_size 64 --distributed --ngd