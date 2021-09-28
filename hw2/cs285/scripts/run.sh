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
    echo "Experiment 2: Inverted Pendulum"
    declare -a b_sizes=("10000" "30000" "50000" )
    declare -a r_sizes=( "0.005" "0.01" "0.02")  
    for b in "${b_sizes[@]}"
    do 
        for r in "${r_sizes[@]}" 
        do 
            python $SOURCE --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg --exp_name q2_b$b_r$r 
        done
    done
else
    echo "Nothing to be done"
fi


