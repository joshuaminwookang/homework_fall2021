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

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y

# Load the data as Pandas DataFrame
def load_data(pathname):
    search_path = os.path.normpath(os.path.join(os.getcwd(), pathname))
    csv = glob.glob(search_path)
    filename=os.path.basename(csv[0])
    tag=filename[filename.find('pg_') + 1:filename.find('.csv')]
    data = pd.read_csv(csv[0], delimiter=",")
    data["Experiment"] = tag
    data["Iteration"] = range(0, len(data[TAGNAME]))
    return data

def load_data_from_dir(search):
    dfs = []
    search_path = os.path.normpath(os.path.join(os.getcwd(), search))
    csvs = glob.glob(search_path)
    for f in csvs:
        filename=os.path.basename(f)
        tag=filename[filename.find('pg_') + 1:filename.find('.csv')]
        data = pd.read_csv(f, delimiter=",")
        data["Experiment"] = tag
        data["Iteration"] = range(0, len(data[TAGNAME]))
        dfs.append(data)
    return dfs

"""
    From a list of DataFrames, plot all data in a single plot (with legend)
    Goal: compare learning curves with some score metric (y_vars[1]) over some predictor (y_vars[0])
"""
def plot_stacked_learning_curves(dfs, y_vars, title, plot_type="scatter"):
    total_df = pd.DataFrame()
    # min_size = np.amin([len(df.index) for df in dfs])
    for df in dfs:
        # relative_df = df.copy()
        # relative_df[y_vars[1]] = df[y_vars[1]] / df[y_vars[1]].median()
        total_df = total_df.append(df)
    total_df = total_df.pivot(index=y_vars[0], columns=TAGNAME, values=y_vars[1])

    fig = plt.gcf()
    fig.set_size_inches(fig_dims)
    sns.set_style("darkgrid")

    with sns.plotting_context(font_scale=axis_scale):
        if (plot_type == "scatter"):
            ax = sns.scatterplot(data=total_df.iloc[100:200])
        else :
            ax = sns.lineplot(data=total_df)
        ax.set_xlabel(y_vars[0], fontsize=axis_label, weight='bold')
        ax.set_ylabel(y_vars[1], fontsize=axis_label, weight='bold')
        plt.legend(fontsize=legend_label,loc="best", prop={'weight': 'bold'})
        plt.title(title, fontsize=axis_label, weight='bold')

    plt.savefig(title+y_vars[0]+'_'+y_vars[1]+'.png',  format='png', dpi=300)
    plt.close() 

"""
main function
"""
def main():
    dfs_q1_sb = load_data_from_dir("data/q2_pg_q1_sb*/*.csv")
    dfs_q1_lb = load_data_from_dir("data/q2_pg_q1_lb*/*.csv")

    plot_stacked_learning_curves(dfs_q1_sb, ['Iteration', 'Eval_AverageReturn'], "Q1 CartPole SB -n 100 -b 1000", plot_type="line")
    plot_stacked_learning_curves(dfs_q1_lb, ['Iteration', 'Eval_AverageReturn'], "Q1 CartPole LB -n 100 -b 5000", plot_type="line")

    dfs_q2 = [load_data("data/q2_pg_q2_b50_r0.02_*/*.csv"), load_data("data/q2_pg_q2_b200_r0.04_*/*.csv")]
    plot_stacked_learning_curves(dfs_q2, ['Iteration', 'Eval_AverageReturn'], "Q2 Inverted Pendulum", plot_type="line")

    dfs_q3 = load_data_from_dir("data/q2_pg_q3*/*.csv")
    plot_stacked_learning_curves(dfs_q3, ['Iteration', 'Eval_AverageReturn'], "Q3 LunarLander -n 100 RTG_NNBaseline", plot_type="line")

    dfs_q4_search = load_data_from_dir("data/q2_pg_q4_search_*/*.csv")
    plot_stacked_learning_curves(dfs_q4_search, ['Iteration', 'Eval_AverageReturn'], "Q4 HalfCheetah Search", plot_type="line")
    dfs_q4_final = load_data_from_dir("data/q2_pg_q4_b*/*.csv")
    plot_stacked_learning_curves(dfs_q4_final, ['Iteration', 'Eval_AverageReturn'], "Q4 HalfCheetah b 30000 lr 0.02", plot_type="line")

    dfs_q5 = load_data_from_dir("data/q2_pg_q5*/*.csv")
    plot_stacked_learning_curves(dfs_q5, ['Iteration', 'Eval_AverageReturn'], "Q5 HopperV2 GAE", plot_type="line")
    # plot_stacked(dfs, ['Index', 'Slice_LUTs'], plot_type="scatter")
    # for df in dfs:
    #     if df.iloc[0]['Benchmark'] == "or1200":
    #         plot_single(df, "Vivado_vs_ABC", ['ABC_Delay', 'Path_Delay'], plot_type="scatter")
    #         plot_single(df, "Vivado_vs_ABC", ['ABC_Area', 'Slice_LUTs'], plot_type="scatter")



if __name__ == "__main__":
    main()
