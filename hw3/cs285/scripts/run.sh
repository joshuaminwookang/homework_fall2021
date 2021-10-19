#!/bin/bash
#set -e

EXP_NUM=$1
SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
SOURCE_DQN="cs285/scripts/run_hw3_dqn.py"
SOURCE_AC="cs285/scripts/run_hw3_actor_critic.py"

cd $SCRIPT_DIR/../../
if [[ $EXP_NUM == 1 ]]; then
    echo "Experiment 1: DQN LunarLander (Q3) Different Learning Rates"
    # declare -a pendulum_b_sizes=("50" "100" "200" "500" "1000" )
    # declare -a pendulum_r_sizes=( "0.005" "0.01" "0.02" "0.05")  
    # for b in "${pendulum_b_sizes[@]}"
    # do 
    #     for r in "${pendulum_r_sizes[@]}" 
    #     do 
    #         echo $b $r
    #         python $SOURCE --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg --exp_name q2_b"$b"_r"$r" 
    #     done
    # done
    python $SOURCE_DQN --env_name LunarLander-v3 --learning_freq 1 --exp_name q3_hparam1
    python $SOURCE_DQN --env_name LunarLander-v3 --learning_freq 2 --exp_name q3_hparam2
    python $SOURCE_DQN --env_name LunarLander-v3 --learning_freq 4 --exp_name q3_hparam3
    python $SOURCE_DQN --env_name LunarLander-v3 --learning_freq 8 --exp_name q3_hparam4

elif [[ $EXP_NUM == 2 ]]; then
    echo "Experiment 2: DDQN LunarLander"
    python $SOURCE_DQN --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1
    python $SOURCE_DQN --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2
    python $SOURCE_DQN --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3
    python $SOURCE_DQN --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --seed 1 --double_q
    python $SOURCE_DQN --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --seed 2 --double_q
    python $SOURCE_DQN --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --seed 3 --double_q
else
    echo "Nothing to be done"
fi


