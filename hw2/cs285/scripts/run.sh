#!/bin/bash
#set -e

EXP_NUM=$1
SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
SOURCE="cs285/scripts/run_hw2.py"

declare -a pendulum_b_sizes=("50" "100" "500" "1000" )
declare -a pendulum_r_sizes=( "0.005" "0.01" "0.02" "0.05")  

cd $SCRIPT_DIR/../../
if [[ $EXP_NUM == 1 ]]; then
    echo "Experiment 1: CartPole"
    python $SOURCE --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa
    python $SOURCE --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa
    python $SOURCE --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na
    python $SOURCE --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa
    python $SOURCE --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa
    python $SOURCE --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na
elif [[ $EXP_NUM == 2 ]]; then
    echo "Experiment 2: Inverted Pendulum"
    for b in "${pendulum_b_sizes[@]}"
    do 
        for r in "${pendulum_r_sizes[@]}" 
        do 
            python $SOURCE --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg --exp_name q2_b$b_r$r 
        done
    done
elif [[ $EXP_NUM == 3 ]]; then
    echo "Experiment 3: Luncar Landing; target 180 eval_return"
     python $SOURCE --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005
elif [[ $EXP_NUM == 4 ]]; then
    echo "Experiment 4: HalfCheetah"
    if [[ $SEARCH == true ]]; then
        declare -a b_sizes=("10000" "30000" "50000" )
        declare -a r_sizes=( "0.005" "0.01" "0.02")  
        for b in "${b_sizes[@]}"
        do 
            for r in "${r_sizes[@]}" 
            do 
                python $SOURCE --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline --exp_name q4_search_b$b_lr$r_rtg_nnbaseline 
            done
        done
    else
        python $SOURCE --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r --exp_name q4_$b_lr$r
        python $SOURCE --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --exp_name q4_b$b_lr$r_rtg
        python $SOURCE --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r --nn_baseline --exp_name q4_b$b_lr$r_nnbaseline 
        python $SOURCE --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline --exp_name q4_b$b_lr$r_rtg_nnbaseline 
    fi
elif [[ $EXP_NUM == 5 ]]; then
    echo "Experiment 5: GAE with Hopperv2"
    declare -a labmdas=("0" "0.95" "0.99" "1" )
    for lambda in "${lambdas[@]}"
    do 
        python $SOURCE --env_name Hopper-v2 --ep_len 1000 --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 -rtg --nn_baseline --action_noise_std 0.5 -gae_lambda $lambda --exp_name q5_b2000_r0.001_lambda$lambda
    done
else
    echo "Nothing to be done"
fi


