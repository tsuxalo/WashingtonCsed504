#!/bin/bash
P="C:/Users/truer/.conda/envs/uw-csed504/python.exe"
# 1. the parameter-matched ViT, now properly regularized
"$P" -W ignore train.py --model vit      --gpu 1 --epochs 60 --batch 512 > logs/vit.log 2>&1
# 2. ResNet-50, with zero_init_residual + grad clipping to cure the NaN
"$P" -W ignore train.py --model resnet50 --gpu 1 --epochs 40 --batch 512 > logs/resnet50.log 2>&1
# 3. CONTROL: does mixup/CutMix help the CNN too?  If it does, the ViT's gain is not architectural.
"$P" -W ignore train.py --model resnet18 --gpu 1 --epochs 40 --batch 512 --strong-aug \
     --tag resnet18_aug > logs/resnet18_aug.log 2>&1
