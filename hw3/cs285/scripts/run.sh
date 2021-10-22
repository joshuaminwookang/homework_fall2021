#!/bin/bash
#set -e

EXP_NUM=$1
SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
SOURCE_DQN="cs285/scripts/run_hw3_dqn.py"
SOURCE_AC="cs285/scripts/run_hw3_actor_critic.py"

cd $SCRIPT_DIR/../../
if [[ $EXP_NUM == 3 ]]; then
    echo "Experiment 3: DQN LunarLander (Q3) Different Learning Rates"
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
elif [[ $EXP_NUM == 4 ]]; then
    echo "Experiment 4: Actor Critic CartPole"
    python $SOURCE_AC --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_1 -ntu 1 -ngsptu 1
    python $SOURCE_AC --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_1_100 -ntu 1 -ngsptu 100
    python $SOURCE_AC --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_100_1 -ntu 100 -ngsptu 1
    python $SOURCE_AC --env_name CartPole-v0 -n 100 -b 1000 --exp_name q4_ac_10_10 -ntu 10 -ngsptu 10
elif [[ $EXP_NUM == 5 ]]; then
    echo "Experiment 5: Actor Critic HalfCheetah"
    # python $SOURCE_AC --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 \
    #     --exp_name q5_10_10 -ntu 10 -ngsptu 10
    python $SOURCE_AC --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 \
        -b 30000 -eb 1500 -lr 0.02 --exp_name q5_10_10 -ntu 10 -ngsptu 10
else
    echo "Nothing to be done"
fi


