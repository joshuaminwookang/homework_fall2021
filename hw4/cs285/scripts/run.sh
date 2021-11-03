#!/bin/bash
#set -e

EXP_NUM=$1
SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
SOURCE_DQN="cs285/scripts/run_hw3_dqn.py"
SOURCE_AC="cs285/scripts/run_hw3_actor_critic.py"

cd $SCRIPT_DIR/../../
if [[ $EXP_NUM == 1 ]]; then
    echo "Experiment 1: MBRL with Random Policy "
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
    # python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch1x32 --env_name cheetah-cs285-v0 \
    #     --add_sl_noise --n_iter 1 \
    #     --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 \
    #     --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1 \
    #     --mpc_action_sampling_strategy 'random'
    python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n5_arch2x250 --env_name cheetah-cs285-v0 \
        --add_sl_noise --n_iter 1 \
        --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 \
        --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'
    python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch2x250 --env_name cheetah-cs285-v0 \
        --add_sl_noise --n_iter 1 \
        --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 \
        --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1 \
        --mpc_action_sampling_strategy 'random'
elif [[ $EXP_NUM == 2 ]]; then
    echo "Experiment 2: MBRL with Trained MPC Policy"
    python cs285/scripts/run_hw4_mb.py --exp_name q2_obstacles_singleiteration --env_name obstacles-cs285-v0 \
    --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 \
    --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 \
    --mpc_action_sampling_strategy 'random'
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


