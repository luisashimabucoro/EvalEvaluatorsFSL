#!/bin/bash

echo $'########################################################'
echo $'###---           Emb-FSL CLIP Experiments         ---###'
echo $'########################################################\n\n'

IFS="/"
for way in ${1}; do 
    for shot in ${2}; do
        for boot_seed in ${5}; do
            echo "Testing $way way $shot shot model"
            CUDA_VISIBLE_DEVICES=$6 python3 run_tester.py --test_shot $shot --test_way $way --eval_types $3 --test_folds $4 --boot_seed $boot_seed
        done
    done
done