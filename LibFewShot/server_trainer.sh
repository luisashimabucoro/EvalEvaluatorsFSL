#!/bin/bash

echo $'########################################################'
echo $'###---           Model Training Routine           ---###'
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
read -p "Dataset: " dataset_name
read -p "Session name: " session_name
read -p "List of ways to be tested: " way_list
read -p "List of shots to be tested: " shot_list

screen -d -m -l -S $session_name


for ((j=0;j<$n_models;j++)); do
    # create now window using screen command
    if [ $j -ne 0 ] 
    then
        screen -S $session_name -X screen $j
    fi
    # creating screen for model
    echo $'\n\n########################################'
    read -p "Model name: " model_name
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
    screen -S $session_name -p $j -X stuff $"chmod a+x ./trainers/${model_name}_trainer.sh;\n"
    if [ "$host_name" != "daisy1.inf.ed.ac.uk" ] ; then
        screen -S $session_name -p $j -X stuff $"longjob -28day -c \"./trainers/${model_name}_trainer.sh $way_list $shot_list $n_gpu $gpu_list $dataset_name \";\n"
        screen -S $session_name -p $j -X stuff $"$1\n"
    else
        screen -S $session_name -p $j -X stuff $"conda activate research\n"
        screen -S $session_name -p $j -X stuff $"./trainers/${model_name}_trainer.sh $way_list $shot_list $n_gpu $gpu_list $dataset_name;\n"
    fi
    # screen -S $session_name -p $j -X stuff $"sleep 2\n"
done