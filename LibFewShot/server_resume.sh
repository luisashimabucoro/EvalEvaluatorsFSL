#!/bin/bash

echo $'########################################################'
echo $'###---            Model Resume Routine            ---###'
echo $'########################################################\n\n'

function create_gpu_list () {
    gpu_list=""

    for i in $(seq $1 $2); do
    token="$i"
    gpu_list="${gpu_list}${gpu_list:+,}$token"
    done
}


gpus_used=0
total_gpus=$(nvidia-smi --list-gpus | wc -l)
host_name=$(hostname)

read -p "Session name: " session_name
read -p "Model name: " model_name
read -p "Dataset: " dataset_name
read -p "Path: " model_path
read -p "Way number: " way_num
read -p "Shot number: " shot_num

screen -d -m -l -S $session_name

read -p "Number of GPUs for this model: " n_gpu
read -p "GPU list: " gpu_list

screen -S $session_name -p 0 -X title $model_name

if [ "$host_name" != "daisy1.inf.ed.ac.uk" ] ; then
    screen -S $session_name -p 0 -X stuff $"cd myproject; source ./bin/activate; cd ..;\n"
    screen -S $session_name -p 0 -X stuff $"python3 ./trainers/create_resume_dict.py --way $way_num --shot $shot_num --path $model_path --model $model_name --dataset $dataset_name;\n"
    screen -S $session_name -p 0 -X stuff $"longjob -28day -c \"python3 run_trainer_resume.py --n_gpu $n_gpu --device_ids $gpu_list\";\n"
    screen -S $session_name -p 0 -X stuff $"$1;\n"
else
    screen -S $session_name -p 0 -X stuff $"conda activate research;\n"
    screen -S $session_name -p 0 -X stuff $"python3 ./trainers/create_resume_dict.py --way $way_num --shot $shot_num --path $model_path --model $model_name --dataset $dataset_name;\n"
    screen -S $session_name -p 0 -X stuff $"python3 run_trainer_resume.py --n_gpu $n_gpu --device_ids $gpu_list;\n"
fi