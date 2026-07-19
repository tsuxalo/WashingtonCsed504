#!/bin/bash
P="C:/Users/truer/.conda/envs/uw-csed504/python.exe"
# The big ViT is the long pole (~5h). Give it a card to itself.
"$P" -W ignore train.py --model vit_base --gpu 0 --epochs 40 --batch 512 > logs/vit_base.log 2>&1
