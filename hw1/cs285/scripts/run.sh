#!/bin/bash

IS_DAGGER=$1
HYPERPARAMETER=$2
SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
SOURCE="cs285/scripts/run_hw1.py"
FLAGS_BC="--n_iter 1 --ep_len 1000 --batch_size 1000 \
--eval_batch_size 5000 --train_batch_size 100 --video_log_freq -1"

FLAGS_DAGGER="--ep_len 1000 --batch_size 1000 \
--eval_batch_size 5000 --train_batch_size 100 --video_log_freq -1 --do_dagger"
#declare -a bmarks=("Ant" "HalfCheetah" "Hopper" "Humanoid" "Walker2d")
declare -a bmarks=("Ant" "Hopper")

cd $SCRIPT_DIR/../../
if [[ $IS_DAGGER -gt 0 ]]; then
    for bmark in "${bmarks[@]}" 
    do 
        echo "RUNNING $bmark with DAgger"
        for i in $(seq 2 10); do
            python $SOURCE $FLAGS_DAGGER --expert_policy_file cs285/policies/experts/$bmark.pkl --env_name $bmark-v2 \
            --exp_name bc_$bmark --expert_data cs285/expert_data/expert_data_$bmark-v2.pkl --n_iter $i
        done
    done
else
    for bmark in "${bmarks[@]}" 
    do 
        if [[ $HYPERPARAMETER -gt 0 ]]; then
        for i in 10 25 50 75 100 200 500; do
            echo "RUNNING $bmark with Train Batch Size $i"
            python $SOURCE $FLAGS_BC --expert_policy_file cs285/policies/experts/$bmark.pkl --env_name $bmark-v2 \
            --exp_name bc_$bmark --expert_data cs285/expert_data/expert_data_$bmark-v2.pkl --train_batch_size $i
        done
        else
            echo "RUNNING $bmark with Behavioral Cloning"
            python $SOURCE $FLAGS_BC --expert_policy_file cs285/policies/experts/$bmark.pkl --env_name $bmark-v2 \
            --exp_name bc_$bmark --expert_data cs285/expert_data/expert_data_$bmark-v2.pkl
        fi 
    done
fi


