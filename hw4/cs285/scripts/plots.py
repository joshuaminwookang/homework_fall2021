from pandas.io.sql import DatabaseError
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob, os.path, os
import tensorflow as tf


#Problem-specific setup variables
TAGNAME = "Experiment"

#global params
fig_dims = (12,8)
axis_label = 16
legend_label = 14
axis_scale = 3.0

def get_section_results_dqn(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    # for e in tf.train.summary_iterator(file):
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == "Train_AverageReturn":
                Y.append(v.simple_value)
            elif v.tag == "Train_BestReturn":
                Z.append(v.simple_value)
    return X, Y, Z

def get_section_results_ac(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    # for e in tf.train.summary_iterator(file):
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            # elif v.tag == 'Eval_AverageReturn':
            elif v.tag == "Eval_AverageReturn":
                Y.append(v.simple_value)
    return X, Y

# Load the data as Pandas DataFrame
def load_data(pathname):
    search_path = os.path.normpath(os.path.join(os.getcwd(), pathname))
    csv = glob.glob(search_path)
    filename=os.path.basename(csv[0])
    tag=filename[filename.find('pg_') + 1:filename.find('.csv')]
    data = pd.read_csv(csv[0], delimiter=",")
    data[TAGNAME] = tag
    data["Iteration"] = range(0, len(data[TAGNAME]))
    return data

def load_data_from_dir(search,  dqn, best_results, log_freq=1):
    dfs = []
    search_path = os.path.normpath(os.path.join(os.getcwd(), search))
    csvs = glob.glob(search_path)
    for f in csvs:
        X = []
        Y = []
        Y2 = []
        columns = [] 
        filename=os.path.dirname(f)
        tag=filename[filename.find('/q') + 1:filename.find('-v')]
        if dqn:
            X, Y, Y2 = get_section_results_dqn(f)
            columns = ["Train_EnvstepsSoFar", "Train_AverageReturn"]

        else :
            X, Y = get_section_results_ac(f)
            columns = ["Train_EnvstepsSoFar", "Eval_AverageReturn" ]

        data = pd.DataFrame([ [int(x),y] for x,y in zip(X, Y)], columns = columns)
        data[TAGNAME] = tag
        data["Iteration"] = range(0, len(data[TAGNAME]))
        data["Iteration"] *= log_freq
        dfs.append(data)
        if best_results:
            data2 = pd.DataFrame([ [int(x),y] for x,y in zip(X, Y2)], columns = ["Train_EnvstepsSoFar", "Train_AverageReturn"])
            data2[TAGNAME] = tag + "_Best_Return"
            dfs.append(data2)
    return dfs

def load_data_from_dir_average(search, tag):
    search_path = os.path.normpath(os.path.join(os.getcwd(), search))
    csvs = glob.glob(search_path)
    X_all = []
    Y_all = []
    for f in csvs:
        X, Y, _ = get_section_results_dqn(f)
        columns = ["Train_EnvstepsSoFar", "Train_AverageReturn"]
        X_all.append(X)
        Y_all.append(Y)
    # Y_avg = [np.mean([a,b,c]) for a, b,c in zip(Y_all[0], Y_all[1],Y_all[2])]
    Y_avg = np.mean(Y_all, axis=0)
    data = pd.DataFrame([ [int(x),y] for x,y in zip(X_all[0], Y_avg)], columns = columns)
    data[TAGNAME] = tag
    # data["Iteration"] = range(0, len(data[TAGNAME]))
    return data


"""
    From a list of DataFrames, plot all data in a single plot (with legend)
    Goal: compare learning curves with some score metric (y_vars[1]) over some predictor (y_vars[0])
"""
def plot_stacked_learning_curves(dfs, vars, title, plot_type="scatter", subtitle=""):
    total_df = pd.DataFrame()
    # min_size = np.amin([len(df.index) for df in dfs])
    for df in dfs:
        total_df = total_df.append(df)
    total_df = total_df.pivot(index=vars[0], columns=TAGNAME, values=vars[1])
    fig = plt.gcf()
    fig.set_size_inches(fig_dims)
    sns.set_style("darkgrid")

    with sns.plotting_context(font_scale=axis_scale):
        if (plot_type == "scatter"):
            ax = sns.scatterplot(data=total_df.iloc[100:200])
        else :
            ax = sns.lineplot(data=total_df)
        ax.set_xlabel(vars[0], fontsize=axis_label, weight='bold')
        ax.set_ylabel(vars[1], fontsize=axis_label, weight='bold')
        if vars[0] == "Train_EnvstepsSoFar":
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.legend(fontsize=legend_label,loc="best", prop={'weight': 'bold'})
        plt.title(title+"\n"+subtitle, fontsize=axis_label, weight='bold')

    plt.savefig(title+'.png',  format='png', dpi=300)
    #plt.savefig(title+vars[0]+'_'+vars[1]+'.png',  format='png', dpi=300)

    plt.close() 

"""
main function
"""
def main():
    dfs_q1 = load_data_from_dir("data/q1*/event*", True, True)
    plot_stacked_learning_curves(dfs_q1, ['Train_EnvstepsSoFar', 'Train_AverageReturn'], "Q1_MsPacman-v0_DQN", plot_type="line")

    dfs_q2_dqn = load_data_from_dir_average("data/q2_dqn*/event*", "Q2_LunarLander_DQN")
    dfs_q2_ddqn = load_data_from_dir_average("data/q2_doubledqn*/event*", "Q2_LunarLander_DDQN")
    plot_stacked_learning_curves([dfs_q2_dqn, dfs_q2_ddqn], ['Train_EnvstepsSoFar', 'Train_AverageReturn'], "Q2_LunarLander_DQN_vs_DDQN", plot_type="line")
    
    dfs_q3 = load_data_from_dir("data/q3*/event*", True, False)
    plot_stacked_learning_curves(dfs_q3, ['Train_EnvstepsSoFar', 'Train_AverageReturn'], "Q3_LunarLander_DQN_vary_LR", plot_type="line", subtitle="learning_freq = 1, 2, 4, 8")
    
    dfs_q4 = load_data_from_dir("data/q4*/event*", False, False, log_freq=10)
    plot_stacked_learning_curves(dfs_q4, ['Iteration', 'Eval_AverageReturn'], "Q4_AC_CartPole", plot_type="line", subtitle="Num_Target_Updates, Num_Grad_Steps_Per_Update")
    
    dfs_q5_pendulum = load_data_from_dir("data/q5_10_10_Inverted*/event*", False, False, log_freq=10)
    plot_stacked_learning_curves(dfs_q5_pendulum, ['Iteration', 'Eval_AverageReturn'], "Q5_AC_InvertedPendulum", plot_type="line", subtitle="-ntu 10 and -ngstu 10")
    
    dfs_q5_cheetah = load_data_from_dir("data/q5_10_10_HalfCheetah*/event*", False, False)
    plot_stacked_learning_curves(dfs_q5_cheetah, ['Iteration', 'Eval_AverageReturn'], "Q5_AC_HalfCheetah", plot_type="line", subtitle="-ntu 10 and -ngstu 10")
    # plot_stacked_learning_curves(dfs_q1_sb, ['Iteration', 'Eval_AverageReturn'], "Q1 CartPole SB -n 100 -b 1000", plot_type="line")
    # plot_stacked_learning_curves(dfs_q1_lb, ['Iteration', 'Eval_AverageReturn'], "Q1 CartPole LB -n 100 -b 5000", plot_type="line")

    # dfs_q2 = [load_data("data/q2_pg_q2_b50_r0.02_*/*.csv"), load_data("data/q2_pg_q2_b200_r0.04_*/*.csv")]
    # plot_stacked_learning_curves(dfs_q2, ['Iteration', 'Eval_AverageReturn'], "Q2 Inverted Pendulum", plot_type="line")

    # dfs_q3 = load_data_from_dir("data/q2_pg_q3*/*.csv")
    # plot_stacked_learning_curves(dfs_q3, ['Iteration', 'Eval_AverageReturn'], "Q3 LunarLander -n 100 RTG_NNBaseline", plot_type="line")

    # dfs_q4_search = load_data_from_dir("data/q2_pg_q4_search_*/*.csv")
    # plot_stacked_learning_curves(dfs_q4_search, ['Iteration', 'Eval_AverageReturn'], "Q4 HalfCheetah Search", plot_type="line")
    # dfs_q4_final = load_data_from_dir("data/q2_pg_q4_b*/*.csv")
    # plot_stacked_learning_curves(dfs_q4_final, ['Iteration', 'Eval_AverageReturn'], "Q4 HalfCheetah b 30000 lr 0.02", plot_type="line")

    # dfs_q5 = load_data_from_dir("data/q2_pg_q5*/*.csv")
    # plot_stacked_learning_curves(dfs_q5, ['Iteration', 'Eval_AverageReturn'], "Q5 HopperV2 GAE", plot_type="line")
    # plot_stacked(dfs, ['Index', 'Slice_LUTs'], plot_type="scatter")
    # for df in dfs:
    #     if df.iloc[0]['Benchmark'] == "or1200":
    #         plot_single(df, "Vivado_vs_ABC", ['ABC_Delay', 'Path_Delay'], plot_type="scatter")
    #         plot_single(df, "Vivado_vs_ABC", ['ABC_Area', 'Slice_LUTs'], plot_type="scatter")



if __name__ == "__main__":
    main()
