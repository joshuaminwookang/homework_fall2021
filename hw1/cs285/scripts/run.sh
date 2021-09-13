#!/bin/bash

IS_DAGGER=$1
SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
SOURCE="cs285/scripts/run_hw1.py"
FLAGS_BC="--n_iter 1 --ep_len 1000 --batch_size 1000 \
--eval_batch_size 5000 --train_batch_size 100 --video_log_freq -1"

FLAGS_DAGGER="--n_iter 10 --ep_len 1000 --batch_size 1000 \
--eval_batch_size 5000 --train_batch_size 100 --video_log_freq -1 --do_dagger"
declare -a bmarks=("Ant" "HalfCheetah" "Hopper" "Humanoid" "Walker2d")
#declare -a bmarks=("Ant")

cd $SCRIPT_DIR/../../
if [[ $IS_DAGGER -gt 0 ]]; then
    for bmark in "${bmarks[@]}" 
    do 
        echo "RUNNING $bmark with DAgger"
        python $SOURCE $FLAGS_DAGGER --expert_policy_file cs285/policies/experts/$bmark.pkl --env_name $bmark-v2 \
        --exp_name bc_$bmark --expert_data cs285/expert_data/expert_data_$bmark-v2.pkl
    done
else
    for bmark in "${bmarks[@]}" 
    do 
        echo "RUNNING $bmark with Behavioral Cloning"
        python $SOURCE $FLAGS_BC --expert_policy_file cs285/policies/experts/$bmark.pkl --env_name $bmark-v2 \
        --exp_name bc_$bmark --expert_data cs285/expert_data/expert_data_$bmark-v2.pkl
    done
fi
