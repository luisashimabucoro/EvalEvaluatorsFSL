#!/bin/bash

read -p "Session name: " session_name
read -p "List of ways to be tested: " way_list
read -p "List of shots to be tested: " shot_list
read -p "List of folds to be tested: " fold_list
read -p "Bootstrapping seeds: " boot_seed_list
read -p "Types of evaluations: " eval_list
read -p "GPU id: " gpu

screen -d -m -l -S $session_name
screen -S $session_name -p 0 -X stuff $"conda activate code\n"
screen -S $session_name -p 0 -X stuff $"chmod u+x test.sh\n"
screen -S $session_name -p 0 -X stuff $"./test.sh $way_list $shot_list $eval_list $fold_list $boot_seed_list $gpu;\n"