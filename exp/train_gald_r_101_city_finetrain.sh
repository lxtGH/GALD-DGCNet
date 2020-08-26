#!/usr/bin/env bash

# train the net (suppose 8 gpus)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 train_distribute.py --data_set cityscapes \
--data_dir "/data/lxt/CityScapes/" \
--data_list "./data/cityscapes/train.lst" \
--arch GALD_res101 \
--restore_from "/data/lxt/pretrained/resnet50-deep.pth" \
--input_size 769 \
--batch_size_per_gpu 1 \
--learning_rate 0.01 \
--num_steps 50000 \
--save_dir "./save_gald_r101" \
--ohem 1 --ohem_thres 0.7 --ohem_keep 100000 \
--log_file "./save_gald_r101/gald_res101.log"