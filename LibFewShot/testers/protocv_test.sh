#!/bin/bash

echo $'########################################################'
echo $'###---            ProtoNetCV Experiments          ---###'
echo $'########################################################\n\n'

IFS="/"
for way in ${1}; do 
    for shot in ${2}; do 
        echo "Testing $way way $shot shot model"
        if [ "${10}" = "0" ] ; then
        python3 ./testers/create_tester_dict.py --shot_num $shot --test_shot $shot --way_num $way --test_way $way --test_folds $3 --n_gpu $4 --device_ids $5 --model $6 --dataset $7 --eval_types $8 --experiment_dir $9 --fixed_classes ${10} --stability_test ${11} --test_batch ${12}
        else
        python3 ./testers/create_tester_dict.py --shot_num $shot --test_shot $shot --way_num $shot --test_way $way --test_folds $3 --n_gpu $4 --device_ids $5 --model $6 --dataset $7 --eval_types $8 --experiment_dir $9 --fixed_classes ${10} --stability_test ${11} --test_batch ${12}
        fi

        if [ "$7" = "MetaAlbumMini" ] ; then
            python3 run_test_protomd.py 
        elif [ "$7" = "cifar100" ] ; then
            python3 run_test_proto_cifar.py 
        else
            python3 run_test_protocv.py 
        fi
        done
    done
done