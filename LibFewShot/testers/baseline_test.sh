#!/bin/bash

echo $'########################################################'
echo $'###---      Baseline Cross-Domain Experiments     ---###'
echo $'########################################################\n\n'

IFS="/"
for way in ${1}; do 
    for shot in ${2}; do
        for boot_seed in ${13}; do
            echo "Testing $way way $shot shot model"
            if [ "${10}" = "0" ] ; then
            python3 ./testers/create_tester_dict.py --shot_num $shot --test_shot $shot --way_num $way --test_way $way --test_folds $3 --n_gpu $4 --device_ids $5 --model $6 --dataset $7 --eval_types $8 --experiment_dir $9 --fixed_classes ${10} --stability_test ${11} --test_batch ${12} --boot_seed $boot_seed
            else
            python3 ./testers/create_tester_dict.py --shot_num $shot --test_shot $shot --way_num $shot --test_way $way --test_folds $3 --n_gpu $4 --device_ids $5 --model $6 --dataset $7 --eval_types $8 --experiment_dir $9 --fixed_classes ${10} --stability_test ${11} --test_batch ${12} --boot_seed $boot_seed
            fi

            if [ "$7" = "MetaAlbumMini" ] ; then
                python3 run_test_baselinemd.py 
            elif [ "$7" = "cifar100" ] ; then
                python3 run_test_baseline_cifar.py 
            else
                python3 run_test_baseline.py 
            fi
        done
    done
done