#!/usr/bin/env bash

# train the net (suppose 8 gpus)
python eval.py --data_set cityscapes \
--data_dir "./dataset/cityscapes" \
--data_list "./data/cityscapes/val.txt" \
--arch DualSeg_res50 \
--rgb 1 \
--restore_from "./save_dualseg_r50/DualSeg_res50_final.pth" \
--output_dir "./dual_seg_r50"