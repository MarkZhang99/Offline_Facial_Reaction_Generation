#!/bin/sh

python train.py \
--test \
--model-pth "results/mhp-epoch95-seed1.pth" \
--neighbor-pattern 'nearest' \
--seed 1