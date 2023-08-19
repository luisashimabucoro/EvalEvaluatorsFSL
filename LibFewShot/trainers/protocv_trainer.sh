#!/bin/bash

# Baseline
echo $'########################################################'
echo $'###---      ProtoNet Cross-Domain Experiments     ---###'
echo $'########################################################\n\n'

IFS="/"
for way in ${1}; do 
    for shot in ${2}; do 
        echo "Training $way way $shot shot model"
        if [ "$5" = "meta" ] ; then
            echo "Meta Album model"
            python3 run_trainer_protocvmd.py --shot_num $shot --test_shot $shot --way_num 19 --test_way $way --n_gpu $3 --device_ids $4
        elif [ "$5" = "cifar" ] ; then
            echo "CIFAR-FS model"
            python3 run_trainer_protocv_cifar.py --shot_num $shot --test_shot $shot --way_num 20 --test_way $way --n_gpu $3 --device_ids $4
        else
            echo "miniImageNet model"
            python3 run_trainer_protocv.py --shot_num $shot --test_shot $shot --way_num 20 --test_way $way --n_gpu $3 --device_ids $4
        fi
    done
done