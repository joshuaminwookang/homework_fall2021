#!/bin/bash
#set -e

EXP_NUM=$1
SCRIPT_DIR="$( dirname "$( readlink -f "${BASH_SOURCE[0]}" )" )"
SOURCE_DQN="cs285/scripts/run_hw3_dqn.py"
SOURCE_AC="cs285/scripts/run_hw3_actor_critic.py"

cd $SCRIPT_DIR/../../
if [[ $EXP_NUM == 0 ]]; then
    echo "Experiment 1-1: RND vs random exploration  "
    python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd \
    --unsupervised_exploration --exp_name q1_easy_rnd --num_timesteps 20000
    python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 \
    --unsupervised_exploration --exp_name q1_easy_random --num_timesteps 20000
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --unsupervised_exploration --exp_name q1_medium_rnd --num_timesteps 20000
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
    --unsupervised_exploration --exp_name q1_medium_random --num_timesteps 20000
elif [[ $EXP_NUM == 1 ]]; then
    echo "Experiment 1-2: custom (count) exploration "
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
    --unsupervised_exploration --use_count_model --exp_name q1_alg_med
    # python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 \
    # --unsupervised_exploration --use_count_model  --exp_name q1_alg_med
elif [[ $EXP_NUM == 2 ]]; then
    echo "Experiment 2-1: Offline Learning with CQL vs DQN"
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn \
        --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql \
        --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_expl_re_transform \
        --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 \
        --exploit_rew_shift=1.0 --exploit_rew_scale=100.0
elif [[ $EXP_NUM == 3 ]]; then
    echo "Experiment 2-2,3: Ablation study; exploration data size and cql_alpha"
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0.1 \
    --unsupervised_exploration --exp_name q2_cql_numsteps_5000
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0.1 \
    --unsupervised_exploration --exp_name q2_cql_numsteps_15000
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --num_exploration_steps=5000 --offline_exploitation --cql_alpha=0 \
    --unsupervised_exploration --exp_name q2_dqn_numsteps_5000
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --num_exploration_steps=15000 --offline_exploitation --cql_alpha=0 \
    --unsupervised_exploration --exp_name q2_dqn_numsteps_15000
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --unsupervised_exploration --offline_exploitation --cql_alpha=0.02 \
    --exp_name q2_alpha_0.02
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --unsupervised_exploration --offline_exploitation --cql_alpha=0.5 \
    --exp_name q2_alpha_0.5
elif [[ $EXP_NUM == 4 ]]; then
    echo "Experiment 3: Supevised Exploration with Mixed Reward Bonus"
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_medium_dqn
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_medium_cql
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd \
    --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_medium_cql_transform_re \
    --exploit_rew_shift=1.0 --exploit_rew_scale=100.0
    python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
    --num_exploration_steps=20000 --cql_alpha=0.0 --exp_name q3_hard_dqn
    python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
    --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_hard_cql
    python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd \
    --num_exploration_steps=20000 --cql_alpha=1.0 --exp_name q3_hard_cql_transform_re \
    --exploit_rew_shift=1.0 --exploit_rew_scale=100.0
elif [[ $EXP_NUM == 5 ]]; then
    echo "Experiment 5: AWAC"
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
    --exp_name q5_awac_medium_unsupervised_lam_1 --use_rnd \
    --unsupervised_exploration --awac_lambda=1 --num_exploration_steps=20000
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
    --exp_name q5_awac_medium_unsupervised_lam_10 --use_rnd \
    --unsupervised_exploration --awac_lambda=1 --num_exploration_steps=20000
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
    --exp_name q5_awac_medium_unsupervised_lam_50 --use_rnd \
    --unsupervised_exploration --awac_lambda=1 --num_exploration_steps=20000
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
    --exp_name q5_awac_medium_supervised_lam_1 --use_rnd \
    --awac_lambda=1 --num_exploration_steps=20000
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
    --exp_name q5_awac_medium_supervised_lam_10 --use_rnd \
    --awac_lambda=1 --num_exploration_steps=20000
    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
    --exp_name q5_awac_easy_unsupervised_lam_1 --use_rnd \
    --unsupervised_exploration --awac_lambda=1 --num_exploration_steps=20000
    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
    --exp_name q5_awac_easy_unsupervised_lam_10 --use_rnd \
    --unsupervised_exploration --awac_lambda=1 --num_exploration_steps=20000
    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
    --exp_name q5_awac_easy_supervised_lam_1 --use_rnd \
    --awac_lambda=1 --num_exploration_steps=20000
    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 \
    --exp_name q5_awac_easy_supervised_lam_10 --use_rnd \
    --awac_lambda=1 --num_exploration_steps=20000
else
    echo "Nothing to be done"
fi


