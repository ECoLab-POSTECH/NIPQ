#!/bin/bash
batch='256'
lr='0.04'
decay='1e-5'
epoch='40'

model='mobilenetv2'
dataset='imagenet'

a_scale='1'
w_scale='1'
bops_scale='3'

mode='bops'
target='5.3'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py \
        --model $model --mode $mode --target $target --bops_scale ${bops_scale} --ts \
        --warmuplen 3 --ft_epoch 3 \
        --dataset $dataset --lr $lr --decay $decay  \
        --epoch $epoch --batch $batch --a_scale $a_scale --w_scale $w_scale\
        --ckpt ./checkpoint/${date}/${model}_${dataset}_${mode}_${epoch}_${a_scale}_${w_scale}_target_${target} \
        >& ./log/${model}_${dataset}_${mode}_${epoch}_${a_scale}_${w_scale}_target_${target}.log &