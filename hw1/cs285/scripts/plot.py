import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob, os.path, os

# Load the data .
q1_hopper_dfs = []
q1_ant_dfs = []
q2_hopper_dfs = []
q2_ant_dfs = []


q1_csv_paths=os.path.normpath(os.path.join(os.getcwd(), "data/q1_*.csv"))
q1_csvs = glob.glob(q1_csv_paths)
q2_csv_paths=os.path.normpath(os.path.join(os.getcwd(), "data/q2_*.csv"))
q2_csvs = glob.glob(q2_csv_paths)

for f in q1_csvs:
    filename=os.path.basename(f)
    data = pd.read_csv(f)
    if "Hopper" in filename:
        q1_hopper_dfs.append(data)
    else:
        q1_ant_dfs.append(data)

for f in q2_csvs:
    filename=os.path.basename(f)
    num_iter=filename[filename.rfind('_')+1: filename.find('.')]
    data = pd.read_csv(f)
    data['Dagger_Itreations'] = int(num_iter)

    if "Hopper" in filename:
        q2_hopper_dfs.append(data)
    else: 
        q2_ant_dfs.append(data)




def plot_q1(q1_hopper_dfs):
    q1_hopper_df = pd.DataFrame()
    for df in q1_hopper_dfs:
        q1_hopper_df = q1_hopper_df.append(df)  
    q1_hopper_df = q1_hopper_df.sort_values(by=['Train_Batch_Size'])

    sns.set_style("darkgrid")
    ax = sns.pointplot(x='Train_Batch_Size', y ='Eval_AverageReturn',  data=q1_hopper_df, dodge=True, join=False);

    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)

    # Calculate the type of error to plot as the error bars
    # Make sure the order is the same as the points were looped over
    ax.errorbar(x_coords, y_coords, yerr=q1_hopper_df['Eval_StdReturn'], fmt=' ', zorder=-1)
    plt.title("Hopper")
    plt.savefig('q1-2.png',  format='png')
    plt.savefig('q1-2.eps',  format='eps')
    plt.close() 

def plot_q2(q2_dfs, bmark):
    q2_df = pd.DataFrame()
    for df in q2_dfs:
        q2_df = q2_df.append(df)

    if bmark == "Hopper":
        bc_baseline = q1_hopper_dfs[1].loc[0, 'Eval_AverageReturn']
    else:
        bc_baseline = q1_ant_dfs[1].loc[0,'Eval_AverageReturn']
        
    expert_baseline = q2_df.iloc[0].loc['Initial_DataCollection_AverageReturn']

    q2_df = q2_df.sort_values(by=['Dagger_Itreations'])
    # PART 2
    sns.set_style("darkgrid")
    color='steelblue' if bmark == "Hopper" else 'coral'

    
    ax = sns.pointplot(x='Dagger_Itreations', y ='Eval_AverageReturn', color=color, data=q2_df, dodge=True, join=False);
    bottom, top = plt.ylim() 
    # plt.ylim(bottom, expert_baseline + 1000)
    ax.axhline(bc_baseline, ls='--', color='red')
    ax.text(2,bc_baseline + (top-bottom)//20, "BC Agent")

    ax.axhline(expert_baseline, ls='-.', color='black')
    ax.text(4,expert_baseline- (top-bottom)//10, "Expert Policy")
    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)

    # Calculate the type of error to plot as the error bars
    # Make sure the order is the same as the points were looped over
    ax.errorbar(x_coords, y_coords, yerr=q2_df['Eval_StdReturn'], ecolor=color, fmt=' ', zorder=-1)
    plt.title(bmark)
    plt.savefig('q2_{}.png'.format(bmark),  format='png')
    plt.savefig('q2_{}.eps'.format(bmark),  format='eps')
    plt.close() 


def main():
    plot_q1(q1_hopper_dfs)
    plot_q2(q2_hopper_dfs, "Hopper")
    plot_q2(q2_ant_dfs, "Ant")
if __name__ == "__main__":
    main()