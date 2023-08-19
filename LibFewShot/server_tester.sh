#!/bin/bash

echo $'########################################################'
echo $'###---           Model Testing Routine            ---###'
echo $'########################################################\n\n'

function create_gpu_list () {
    gpu_list=""

    for i in $(seq $1 $2); do
    token="$i"
    gpu_list="${gpu_list}${gpu_list:+,}$token"
    done
}


# create session so screen doesnt expire

gpus_used=0
total_gpus=$(nvidia-smi --list-gpus | wc -l)
host_name=$(hostname)

read -p "How many models would you like to train in this server? " n_models
read -p "Session name: " session_name
read -p "Experiment directory name: " exp_dir
read -p "List of ways to be tested: " way_list
read -p "List of shots to be tested: " shot_list
read -p "List of folds to be tested: " fold_list
read -p "Bootstrapping seeds: " boot_seed_list
read -p "Fixed classes (0 if not required): " fixed_classes
read -p "Test batch (-1 if not required): " test_batch
read -p "Stability experiment? (true or false) " stability_bool

screen -d -m -l -S $session_name


for ((j=0;j<$n_models;j++)); do
    # create now window using screen command
    if [ $j -ne 0 ] 
    then
        screen -S $session_name -X screen $j
    fi
    # creating screen for model
    read -p "Model name (script): " model_name
    read -p "Model name (path): " model_path
    read -p "Dataset name: " dataset_name
    read -p "Types of evaluations: " eval_list
    echo "$(( $total_gpus - $gpus_used )) GPUs available."
    read -p "Number of GPUs for this model: " n_gpu
    read -p "GPU list: " gpu_list
    # create_gpu_list $gpus_used $(( $gpus_used + $n_gpu - 1))
    gpus_used=$(( $gpus_used + $n_gpu ))
    echo "GPU(s) allocated: ${gpu_list}"

    screen -S $session_name -p $j -X title $model_name

    echo $host_name
    if [ "$host_name" != "daisy1.inf.ed.ac.uk" ] ; then
        screen -S $session_name -p $j -X stuff $"cd myproject; source ./bin/activate; cd ..;\n"
    fi
    screen -S $session_name -p $j -X stuff $"chmod a+x ./testers/${model_name}_test.sh;\n"
    if [ "$host_name" != "daisy1.inf.ed.ac.uk" ] ; then
        screen -S $session_name -p $j -X stuff $"longjob -28day -c \"./testers/${model_name}_test.sh $way_list $shot_list $fold_list $n_gpu $gpu_list $model_path $dataset_name $eval_list $exp_dir $fixed_classes $stability_bool $test_batch $boot_seed_list\";\n"
        screen -S $session_name -p $j -X stuff $"$1\n"
    else
        screen -S $session_name -p $j -X stuff $"conda activate research\n"
        screen -S $session_name -p $j -X stuff $"./testers/${model_name}_test.sh $way_list $shot_list $fold_list $n_gpu $gpu_list $model_path $dataset_name $eval_list $exp_dir $fixed_classes $stability_bool $test_batch $boot_seed_list;\n"
    fi
done

