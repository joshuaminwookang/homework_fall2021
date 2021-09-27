#!/bin/bash
#set -e

EXP_NUM=$1
SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
SOURCE="cs285/scripts/run_hw2.py"

#declare -a pendulum_b_sizes=("100" "1000")
#declare -a pendulum_r_sizes=("5e-3" "1e-2")
declare -a pendulum_b_sizes=("50" "100" "500" "1000" )
declare -a pendulum_r_sizes=( "5e-3" "5e-2" "5e-1")

cd $SCRIPT_DIR/../../
if [[ $EXP_NUM == 1 ]]; then
    echo "Experiment 1: CartPole"
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na
elif [[ $EXP_NUM == 2 ]]; then
    echo "Experiment 2: Inverted Pendulum"
    for b in "${pendulum_b_sizes[@]}"
    do 
        for r in "${pendulum_r_sizes[@]}" 
        do 
            python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg --exp_name q2_b$b_r$r 
        done
    done
else
    echo "Nothing to be done"
fi


