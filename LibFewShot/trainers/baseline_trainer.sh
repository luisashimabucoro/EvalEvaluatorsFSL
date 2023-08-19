#!/bin/bash

# Baseline
echo $'########################################################'
echo $'###---      Baseline Cross-Domain Experiments     ---###'
echo $'########################################################\n\n'

IFS="/"
for way in ${1}; do 
    for shot in ${2}; do 
        echo "Training $way way $shot shot model"
        if [ "$5" = "meta" ] ; then
            python3 run_trainer_baselinemd.py --shot_num $shot --test_shot $shot --test_way $way --n_gpu $3 --device_ids $4
        elif [ "$5" = "cifar" ] ; then
            python3 run_trainer_baseline_cifar.py --shot_num $shot --test_shot $shot --test_way $way --n_gpu $3 --device_ids $4
        else
            python3 run_trainer_baseline.py --shot_num $shot --test_shot $shot --test_way $way --n_gpu $3 --device_ids $4
        fi
    done
done