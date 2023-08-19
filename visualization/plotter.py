# -*- coding: utf-8 -*-
from abc import abstractmethod

import torch
from torch import nn
import math
import yaml
import argparse
import os
import seaborn as sns


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats.mstats import spearmanr
import plotly.graph_objs as go

"""
Input:
    - One or more .csv files
    - config file defining plot specifications

Output:
    - different types of plots 
    - pernalize details within plot


Plots:
    - One table for each dataset with oracle and cross-val losses and accuracies (plus diff). Rows are models (DONE)
    - Line-chart (x-axis: cv shots, y-axis: accuracy) with 9 lines (model x measurement methods)
    - Line-chart (x-axis: cv shots, y-axis: MAE between oracle and cv) with 3 lines (model)
    - checkpoints_path/{model}/{dataset}
    - {eval_type}
GOAL: Create plots and then functions


- Compute confidence interval in accuracy table and include std (DONE)
- Create accuracy table for algorithms (x-axis shot and y-axis accuracy) (DONE)
- Turn MAE into plots (1 plot per algorithm, one line per evaluation method) 

"""

def get_conf_interval(mean_acc, n_samples=600):
    # mean_acc = mean_acc/100
    return 1.96 * mean_acc.std() / np.sqrt(n_samples)
    # return (1.96 * np.sqrt((mean_acc * (1 - mean_acc)) / n_samples))

def get_MAE(a, b):
    return (a - b).abs().mean()

def get_table_data(dataframe_list, tb_type='acc_table'):
    output = []

    if tb_type == 'acc_table':
        for model in dataframe_list:
            row = []
            for shot_df in model:
                row.append(f"{shot_df['test_accuracy'].mean():.2f}±{get_conf_interval(shot_df['test_accuracy'].mean()):.3f}")
            output.append(row)
    elif tb_type == 'general_table':
        for (model, (oracle, ho, cv5, cv1000, boot)) in dataframe_list:
            row = []
            if model == 'BaselinePlusCV':
                row.append(f"{oracle['test_accuracy'].mean():.2f}±{get_conf_interval(oracle['test_accuracy']):.3f}")
                row.append(f"{ho['test_accuracy'].mean():.2f}±{get_conf_interval(ho['test_accuracy']):.3f}")
                row.append(f"{cv5['test_accuracy'].mean():.2f}±{get_conf_interval(cv5['test_accuracy']):.3f}")
                row.append(f"{cv1000['test_accuracy'].mean():.2f}±{get_conf_interval(cv1000['test_accuracy']):.3f}")
                row.append(f"-")
                row.append(f"{get_MAE(oracle['test_accuracy'], ho['test_accuracy']):.2f}")
                row.append(f"{get_MAE(oracle['test_accuracy'], cv5['test_accuracy']):.2f}")
                row.append(f"{get_MAE(oracle['test_accuracy'], cv1000['test_accuracy']):.2f}")
                row.append(f"-")
            else:
                row.append(f"{oracle['test_accuracy'].mean():.2f}±{get_conf_interval(oracle['test_accuracy']):.3f}")
                row.append(f"{ho['test_accuracy'].mean():.2f}±{get_conf_interval(ho['test_accuracy']):.3f}")
                row.append(f"{cv5['test_accuracy'].mean():.2f}±{get_conf_interval(cv5['test_accuracy']):.3f}")
                row.append(f"{cv1000['test_accuracy'].mean():.2f}±{get_conf_interval(cv1000['test_accuracy']):.3f}")
                # row.append(f"-")
                # row.append(f"{get_MAE(oracle['test_accuracy'], ho['test_accuracy']):.2f}")
                # row.append(f"{get_MAE(oracle['test_accuracy'], cv5['test_accuracy']):.2f}")
                # row.append(f"{get_MAE(oracle['test_accuracy'], cv1000['test_accuracy']):.2f}")
                # row.append(f"-")
                row.append(f"{boot['test_accuracy'].mean():.2f}±{get_conf_interval(boot['test_accuracy']):.3f}")
                row.append(f"{get_MAE(oracle['test_accuracy'], ho['test_accuracy']):.2f}")
                row.append(f"{get_MAE(oracle['test_accuracy'], cv5['test_accuracy']):.2f}")
                row.append(f"{get_MAE(oracle['test_accuracy'], cv1000['test_accuracy']):.2f}")
                row.append(f"{get_MAE(oracle['test_accuracy'], boot['test_accuracy']):.2f}")
            # row.append(f"{get_MAE(oracle['loss'], cv['loss']):.2f}")
            output.append(row)

    return output

def make_bold(headers):
    bold_headers = []
    for header in headers:
      bold_headers.append(f"$\\bf{{{header}}}$")
    
    return bold_headers

def create_table(data, column_headers, row_headers, filename='comparison_table2.png', figsize=(8,4), title=None):
    # Get colormaps
    rcolors = plt.cm.Greys(np.full(len(row_headers), 0.2))
    ccolors = plt.cm.Greys(np.full(len(column_headers), 0.2))

    plt.figure(linewidth=10,
            tight_layout={'pad':1},
            figsize=figsize)
    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=data,
                        rowLabels=make_bold(row_headers),
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=make_bold(column_headers),
                        loc='center')
    
    the_table.set_fontsize(14)
    the_table.scale(1, 2)

    # Hiding axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)

    plt.suptitle(title)
    plt.draw()

    fig = plt.gcf()
    plt.savefig(filename,
                edgecolor=fig.get_edgecolor(),
                facecolor=fig.get_facecolor(),
                dpi=1000)

    print(f"Table successfully created!")

def get_data(data_path, config):
    shot = []
    acc_values = []
    std = []

    # oracle_data = pd.read_csv(f"{data_path}/{config['models']}-5fold-oracle.csv")
    for idx in range(config['shot_num']):
        # path = f"{data_path}/{config['models']}-{idx}fold-cross_val.csv"

        try: 
            df = pd.read_csv(data_path[idx])
        except:
            continue

        # MAE_value = get_MAE(oracle_data['loss'], df['loss'])
        shot.append(idx+1)
        acc_values.append(df['accuracy'].mean())
        std.append(df['accuracy'].std())
    
    df_dict = {'shot' : shot, 'acc' : acc_values, 'std' : std}
    sample_df = pd.DataFrame(df_dict)

    return sample_df

def create_linechart(data_path, config_dict, save_path):
    """
    data : list of DataFrames, each containing three columns (k_shot, accuracy and loss)
    config_dict : dictionary containing plot specifications
    save_path : path where plot should be saved    
    
    - Line-chart (x-axis: cv shots, y-axis: accuracy) with 9 lines (model x measurement methods)
    - Line-chart (x-axis: cv shots, y-axis: MAE between oracle and cv) with 3 lines (model)
    - checkpoints_path/{model}/{dataset}
    - {model_name}-{k_fold}fold-{eval_type}.csv
    """

    # include function to alter data here
    data = []
    for model in config_dict['models']:
        data.append(get_data(data_path[model], config_dict))
        
    print(data)

    fig = go.Figure()

    for idx, df in enumerate(data):
        fig.add_traces(go.Scatter(
        name=config_dict['models'][idx],
        x=df['shot'],
        y=df['acc'],
        mode='lines',
        line=dict(color=config_dict['color_list'][idx], dash=config_dict['dash'])))

        if config_dict['std']:
            fig.add_traces(go.Scatter(
            name='Upper Bound',
            x=df['shot'],
            y=df['acc']+df['std'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False))

            fig.add_traces(go.Scatter(
            name='Lower Bound',
            x=df['shot'],
            y=df['acc']-df['std'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor=config_dict['color_fill_list'][idx],
            fill='tonexty',
            showlegend=False))

        fig.update_layout(
        yaxis_title='Accuracy',
        xaxis_title='Shots',
        title='Accuracy comparison plot',
        hovermode="x",
        plot_bgcolor='rgba(192,192,192, 0.25)',
        autosize=False,
        width=500,
        height=300)

    print(f"Saving image...")
    fig.write_image(save_path, width=800, height=400)
    # print(fig.to_image(format="png"))
    print(f"Chart created successfully!")

def get_rank_data(model_dict):
    oracle_acc = []
    alternative_acc = []
    spearman_corr = []
    for ((model_name), (oracle_df, alternative_df)) in model_dict.items():
        results = stats.spearmanr(oracle_df['test_accuracy'], alternative_df['test_accuracy'])
        spearman_corr.append(f"{results.correlation:.4f}")
        # oracle_acc.append(f"{oracle_df['test_accuracy'].mean():.2f}")
        # alternative_acc.append(f"{alternative_df['test_accuracy'].mean():.2f}")
    
    return spearman_corr

def create_rank_table(alternative_eval_list, model_list, model_dir_list, dataset, way_num, shot_num):

    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2']
    # datasets = ['miniImageNet']
    # ways = ['5']
    # shots = ['5']
    # folds = ['5', '25']

    eval_dict = {}
    for alter_eval in alternative_eval_list:
        df_dict = {}
        # get rank coeff for each alter_eval
        for model, model_dir in zip(model_list, model_dir_list):
            data_path = os.path.join(model_path, model_dir, dataset, f'{way_num}way-{shot_num}shot')                 
            oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
            alternative_df = pd.read_csv(f"{data_path}/{model}-{alter_eval}.csv")
            df_dict[model] = [oracle_df, alternative_df]
        eval_dict[alter_eval] = list(get_rank_correlation(df_dict))
        print(eval_dict)
    
    print(eval_dict)
        
    alternative_eval_list = ['Hold Out', 'CV 5Fold', 'CV LOO', 'Bootstrapping']
    create_table(list(eval_dict.values()), model_dir_list, alternative_eval_list, filename='model_corr_10.png', figsize=(8,4), title=None)


def get_rank_correlation(model_dict):
    """
    model_dict : {'Baseline' : [oracle_df, alter_df], 'R2D2' : ...}
    """
    correlations = get_rank_data(model_dict)
    # results = stats.spearmanr(oracle_acc, alter_acc)
    print(correlations)
    return correlations


def ranking_calculation():
    alternative_eval_list = ['hold_out', 'cross_val-5fold', 'cross_val-1000fold', 'bootstrapping']
    models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    dataset = 'miniImageNet'
    create_rank_table(alternative_eval_list, models, models_dir, dataset, 5, 10)

def within_domain_table(datasets, shots):
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    # model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    model_path = '/raid/s2589574/Models/CLIP'
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2', 'MAML']
    models = ['CLIP']
    models_dir = ['ViT-B32']
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2']
    # datasets = ['miniImageNet']
    ways = [5]
    # shots = [5]
    # shots = [5, 10, 20, 50]
    # folds = [5, 1000]

    df_list = []

    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    if model == 'BaselinePlusCV':
                        oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        ho_df = pd.read_csv(f"{data_path}/{model}-hold_out.csv")
                        cv5_df = pd.read_csv(f"{data_path}/{model}-cross_val-5fold.csv")
                        cv1000_df = pd.read_csv(f"{data_path}/{model}-cross_val-1000fold.csv")
                        boot_df = []                     
                    else:  
                        oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        ho_df = pd.read_csv(f"{data_path}/{model}-hold_out.csv")
                        cv5_df = pd.read_csv(f"{data_path}/{model}-cross_val-5fold.csv")
                        if (model == 'MAML') and (shot == 50):
                            cv1000_df = get_dataframe(data_path, f"{model}-cross_val-1000fold")
                        else:
                            cv1000_df = pd.read_csv(f"{data_path}/{model}-cross_val-1000fold.csv")
                        boot_df = pd.read_csv(f"{data_path}/{model}-bootstrapping.csv")
                    df_list.append((oracle_df, ho_df, cv5_df, cv1000_df, boot_df))
    
    column_headers = ['Oracle Acc', 'HO Acc', 'CV 5F Acc', 'CV LOO Acc', 'Boot Acc', 'MAE HO', 'MAE CV5', 'MAE CVLOO', 'MAE Boot']
    model_data_tuple = zip(models, df_list)
    data = get_table_data(model_data_tuple, tb_type='general_table')
    print(data)
    create_table(data, column_headers, models, filename=f'WD_{datasets[0]}_{shots[0]}shot.png')
    print(f"Accuracy and MAE for {datasets[0]} {shots[0]}-shot created!")



# Create accuracy table for algorithms (x-axis shot and y-axis accuracy)
def create_algo_acc_table(models, models_dir, shots, filename='Algo_accuracy_table.png'):
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2']
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2']
    datasets = ['miniImageNet']
    ways = [5]
    # shots = [5, 10, 20, 50]
    # shots = [10]
    # folds = [5, 1000]

    df_list = []
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                algo_list = []
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                    algo_list.append(oracle_df)
            df_list.append(algo_list)
    
    column_headers = [str(i) for i in shots]
    # model_data_tuple = zip(models, df_list)
    data = get_table_data(df_list, tb_type='acc_table')
    print(data)
    create_table(data, column_headers, models_dir, filename='Algo_accuracy_table.png')   

# Turn MAE into plots (1 plot per algorithm, one line per evaluation method) 
def create_MAE_plots(datasets, ways, shots, models, models_dir, filename="MAE_plots.pdf"):
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2']
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2']
    alt_eval = ['hold_out', 'cross_val-5fold', 'cross_val-1000fold', 'bootstrapping']
    alt_eval_dict = {'hold_out' : 'Hold Out', 'cross_val-5fold' : 'CV 5-Fold', 'cross_val-1000fold' : 'CV LOO', 'bootstrapping' : 'Bootstrapping'}
    # shots = [5, 10, 20, 50]
    
    df_dict = {}
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                algo_list = []
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    if ((model == 'MAML') and (shot == 50)):
                        oracle_df = get_dataframe(data_path, f"{model}-oracle")
                    else:
                        oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                    algo_list.append(oracle_df)
            df_dict[model_dir] = algo_list

    eval_dict = {model : {} for model in models_dir}
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                for eval in alt_eval:
                    algo_list = []
                    for shot in shots:
                        data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                        try:
                            if ((model == 'MAML') and (shot == 50)):
                                df = get_dataframe(data_path, f"{model}-{eval}")
                            else:
                                df = pd.read_csv(f"{data_path}/{model}-{eval}.csv")
                        except:
                            df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        # df = pd.read_csv(f"{data_path}/{model}-{eval}.csv")
                            # df = pd.read_csv(f"{data_path}/{model}-{eval}.csv")
                        algo_list.append(df)
                        # print(algo_list)
                    eval_dict[model_dir][eval] = algo_list
                    

    # plt.rcParams.update(plt.rcParamsDefault)
    # plt.style.use('ggplot')
    # fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    # plt.rcParams["font.family"] = "Times New Roman"

    # # iterating through algos
    # for i in range(2):
    #     for j in range(3):
    #         model_idx = (i*3) + j
    #         for eval_method, eval_method_title in alt_eval_dict.items():
    #             # baseline++cv doesnt have bootstrapping
    #             if (eval_method == 'bootstrapping') and (models[model_idx] == 'BaselinePlusCV'):
    #                 continue

    #             data = [get_MAE(oracle_df['test_accuracy'], alt_df['test_accuracy']) for oracle_df, alt_df in zip(df_dict[models_dir[model_idx]], eval_dict[models_dir[model_idx]][eval_method])]
    #             data = [3 if mae_value == 0 else mae_value for mae_value in data]
    #             print(models_dir[model_idx])
    #             print(eval_method)
    #             print(data)
    #             # dfs = df_dict[models_dir[model_idx]]
    #             if ((i == 0) and (j == 2)):
    #                 axs[i, j].plot(shots, data, marker='o', label=eval_method_title)
    #                 axs[i, j].legend()
    #             else:
    #                 axs[i, j].plot(shots, data, marker='o')
    #             axs[i, j].set_title(models_dir[model_idx], fontweight="bold", fontname = "Times New Roman")
    #             axs[i, j].set_xlim([3, 52])
    #             axs[i, j].set_ylim([3, 62])
    #             if i == 1:
    #                 axs[i, j].set_xlabel("Shot Number", fontname = "Times New Roman")
    #             if j == 0:    
    #                 axs[i, j].set_ylabel("MAE", fontname = "Times New Roman")
    #             for tick in axs[i, j].get_xticklabels():
    #                 tick.set_fontname("Times New Roman")
    #             for tick in axs[i, j].get_yticklabels():
    #                 tick.set_fontname("Times New Roman")

    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, len(models), figsize=(15, 3))
    plt.rcParams["font.family"] = "Times New Roman"

    # iterating through algos
    for i in range(len(models)):
        # for j in range(3):
            # model_idx = (i*3) + j
            for eval_method, eval_method_title in alt_eval_dict.items():
                # baseline++cv doesnt have bootstrapping
                if (eval_method == 'bootstrapping') and (models[i] == 'BaselinePlusCV'):
                    continue

                data = [get_MAE(oracle_df['test_accuracy'], alt_df['test_accuracy']) for oracle_df, alt_df in zip(df_dict[models_dir[i]], eval_dict[models_dir[i]][eval_method])]
                data = [3 if mae_value == 0 else mae_value for mae_value in data]
                print(models_dir[i])
                print(eval_method)
                print(data)
                # dfs = df_dict[models_dir[i]]
                if ((i == 3)):
                    axs[i].plot(shots, data, marker='o', label=eval_method_title)
                    axs[i].legend()
                else:
                    axs[i].plot(shots, data, marker='o')
                axs[i].set_title(models_dir[i], fontweight="bold", fontname = "Times New Roman")
                axs[i].set_xlim([3, 52])
                axs[i].set_ylim([3, 62])
                # if i == 1:
                axs[i].set_xlabel("Shot Number", fontname = "Times New Roman")
                # if j == 0:    
                axs[i].set_ylabel("MAE", fontname = "Times New Roman")
                for tick in axs[i].get_xticklabels():
                    tick.set_fontname("Times New Roman")
                for tick in axs[i].get_yticklabels():
                    tick.set_fontname("Times New Roman")
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.set(xlabel='x-label', ylabel='y-label')

    plt.savefig(filename, format="pdf", bbox_inches="tight")

# Turn MAE into plots (1 plot per algorithm, one line per evaluation method) 
def create_ACC_plots(datasets, ways, shots, models, models_dir, filename="Acc_plots.pdf"):
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2']
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2']
    alt_eval = ['hold_out', 'cross_val-5fold', 'cross_val-1000fold', 'bootstrapping']
    alt_eval_dict = {'hold_out' : 'Hold Out', 'cross_val-5fold' : 'CV 5-Fold', 'cross_val-1000fold' : 'CV LOO', 'bootstrapping' : 'Bootstrapping'}
    # shots = [5, 10, 20, 50]
    
    df_dict = {}
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                algo_list = []
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    oracle_df = get_dataframe(data_path, f"{model}-oracle")
                    algo_list.append(oracle_df)
            df_dict[model_dir] = algo_list

    eval_dict = {model : {} for model in models_dir}
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                for eval in alt_eval:
                    algo_list = []
                    for shot in shots:
                        data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                        # try:
                        df = get_dataframe(data_path, f"{model}-{eval}")
                        # except:
                            # df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        # df = pd.read_csv(f"{data_path}/{model}-{eval}.csv")
                            # df = pd.read_csv(f"{data_path}/{model}-{eval}.csv")
                        algo_list.append(df)
                        # print(algo_list)
                    eval_dict[model_dir][eval] = algo_list
                    

    # print(eval_dict)
    # print(eval_dict)
    # plot code
    # fig, axs = plt.subplots(2, 3)
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    plt.rcParams["font.family"] = "Times New Roman"

    # iterating through algos
    for i in range(2):
        for j in range(3):
            model_idx = (i*3) + j
            if model_idx == 5:
                break
            for eval_method, eval_method_title in alt_eval_dict.items():
                # baseline++cv doesnt have bootstrapping
                if (eval_method == 'bootstrapping') and (models[model_idx] == 'BaselinePlusCV'):
                    continue

                data = [alt_df['test_accuracy'].mean() for oracle_df, alt_df in zip(df_dict[models_dir[model_idx]], eval_dict[models_dir[model_idx]][eval_method])]
                # data = [3 if mae_value == 0 else mae_value for mae_value in data]
                print(models_dir[model_idx])
                print(eval_method)
                print(data)
                # dfs = df_dict[models_dir[model_idx]]
                if ((i == 0) and (j == 2)):
                    axs[i, j].plot(shots, data, marker='o', label=eval_method_title)
                    axs[i, j].legend()
                else:
                    axs[i, j].plot(shots, data, marker='o')
                axs[i, j].set_title(models_dir[model_idx], fontweight="bold", fontname = "Times New Roman")
                axs[i, j].set_xlim([3, 52])
                axs[i, j].set_ylim([0, 80])
                if i == 1:
                    axs[i, j].set_xlabel("Shot Number", fontname = "Times New Roman")
                if j == 0:    
                    axs[i, j].set_ylabel("Accuracy", fontname = "Times New Roman")
                for tick in axs[i, j].get_xticklabels():
                    tick.set_fontname("Times New Roman")
                for tick in axs[i, j].get_yticklabels():
                    tick.set_fontname("Times New Roman")
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for ax in axs.flat:
    #     ax.set(xlabel='x-label', ylabel='y-label')

    plt.savefig(filename, format="pdf", bbox_inches="tight")


def create_num_folds_experiment_plot(datasets, ways, shots, models, models_dir, exp_type='MAE', filename="folds_experiment.pdf"):
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    fold_dict = {2 : [], 3 : [], 5 : [], 10 : [], 1000: []}
    fold_for_plot = [2, 3, 5, 10, 25]

    # we have to get the oracle accuracy for each algo to define the threshold line 
    #   and get the different accuracies for each fold value

    # getting oracle accuracy
    df_dict = {}
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                algo_list = []
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                    algo_list.append(oracle_df)
            df_dict[model_dir] = algo_list

    # getting fold information
    data = []
    data_dict = {}
    for dataset in datasets:
        for way in ways:
            for model, model_dir in zip(models, models_dir):
                data = []
                for fold_value in fold_dict.keys():
                    for shot in shots:
                        data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                        df = pd.read_csv(f"{data_path}/{model}-cross_val-{fold_value}fold.csv")
                        if exp_type == 'acc':
                            data.append(df['test_accuracy'].mean())
                        elif exp_type == 'MAE':
                            data.append(get_MAE(df_dict[model_dir][0]['test_accuracy'], df['test_accuracy']))
                    data_dict[model_dir] = data
                        # print(algo_list)
    
    print(fold_dict)
    print(data)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('ggplot')
    fig, axs = plt.subplots(figsize=(6, 6))
    plt.rcParams["font.family"] = "Times New Roman"

    # iterating through algos
    # for i in range(2):
        # for j in range(3):
    
    for idx in range(len(models_dir)):
            
        # model_idx = (i*3) + j
        data_plot = data_dict[models_dir[idx]]

        axs.set_xlim([0, 27])
        if exp_type == 'acc':
            axs.set_ylim([0, 80])
        else:
            axs.set_ylim([5, 50])
        axs.plot(fold_for_plot, data_plot, marker='o', label=models_dir[idx])
        axs.legend()
        # axs.set_title(models_dir[idx], fontweight="bold", fontname = "Times New Roman")
        # if i == 1:
        axs.set_xlabel("Fold Number", fontname = "Times New Roman")
        # if j == 0:    
        axs.set_ylabel("MAE", fontname = "Times New Roman")


    for tick in axs.get_xticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight('bold')
        tick.set_fontsize(14)
    for tick in axs.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight('bold')
        tick.set_fontsize(14)


    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)


    # plot data in grouped manner of bar type
    plt.savefig(f"folds_experiment-{exp_type}.pdf", format="pdf", bbox_inches="tight")
    print(f"Plot folds_experiment-{exp_type}.pdf saved!")

def create_cvloo_experiment_plot(datasets, models, models_dir):
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    ways = [2, 3, 5, 6, 10, 15]
    # getting oracle accuracy
    df_dict = {}
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            algo_list = {}
            for way in ways:
                data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{int(30/way)}shot', 'cv_loo_experiment')

                # data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{int(30/way)}shot', 'cv_loo_experiment')
                oracle_df = pd.read_csv(f"{data_path}/{model}-{way}way_{int(30/way)}shot-oracle.csv")
                # try:
                cv_df = pd.read_csv(f"{data_path}/{model}-{way}way_{int(30/way)}shot-cross_val-1000fold.csv")
                # except:
                    # cv_df = pd.read_csv(f"{data_path}/{model}-{way}way_{int(30/way)}shot-oracle.csv")
                algo_list[way] = get_MAE(oracle_df['test_accuracy'], cv_df['test_accuracy'])
            df_dict[model_dir] = algo_list
    
    for key, value in df_dict.items():
        print(f"{key} -> {value}")

    # # fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    # fig, axs = plt.subplots(figsize=(6, 6))
    # plt.rcParams["font.family"] = "Times New Roman"

    # # iterating through algos
    # for i in range(2):
    #     for j in range(2):
    #         model_idx = (i*2) + j
    #         data_plot = df_dict[models_dir[model_idx]].values()

    #         data_plot = [3 if data_point == 0 else data_point for data_point in data_plot]
    #         axs[i, j].set_xlim([0, 17])
    #         axs[i, j].set_ylim([3, 40])
    #         axs[i, j].plot(ways, data_plot, marker='o')
    #         axs[i, j].set_title(models_dir[model_idx], fontweight="bold", fontname = "Times New Roman")
    #         if i == 1:
    #             axs[i, j].set_xlabel("Way Number", fontname = "Times New Roman")
    #         if j == 0:    
    #             axs[i, j].set_ylabel("MAE", fontname = "Times New Roman")
    #         for tick in axs[i, j].get_xticklabels():
    #             tick.set_fontname("Times New Roman")
    #         for tick in axs[i, j].get_yticklabels():
    #             tick.set_fontname("Times New Roman")


    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1,
    #                 right=0.9,
    #                 top=0.9,
    #                 wspace=0.3,
    #                 hspace=0.3)


    # fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('ggplot')
    fig, axs = plt.subplots(figsize=(6, 6))
    plt.rcParams["font.family"] = "Times New Roman"

    # iterating through algos
    for i in range(len(models_dir)):
            # model_idx = (i*2) + j
            data_plot = df_dict[models_dir[i]].values()

            # data_plot = [3 if data_point == 0 else data_point for data_point in data_plot]
            axs.set_xlim([1, 16])
            axs.set_ylim([0, 35])
            axs.plot(ways, data_plot, marker='o', label=models_dir[i])
            axs.legend()
            # axs.set_title(models_dir[model_idx], fontweight="bold", fontname = "Times New Roman")
            # if i == 1:
            axs.set_xlabel("Way Number", fontname = "Times New Roman", fontsize=12)
            # if j == 0:    
            axs.set_ylabel("MAE", fontname = "Times New Roman", fontsize=12)
            for tick in axs.get_xticklabels():
                tick.set_fontname("Times New Roman")
                tick.set_fontweight('bold')
                tick.set_fontsize(14)
            for tick in axs.get_yticklabels():
                tick.set_fontname("Times New Roman")
                tick.set_fontweight('bold')
                tick.set_fontsize(14)


    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1,
    #                 right=0.9,
    #                 top=0.9,
    #                 wspace=0.3,
    #                 hspace=0.3)

    # plot data in grouped manner of bar type
    plt.savefig(f"fixed_sample.pdf", format="pdf", bbox_inches="tight")
    print(f"Plot fixed_sample.pdf saved!")

def create_acc_histograms():
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2']
    datasets = ['miniImageNet']
    ways = [5]
    shots = [5]
    # shots = [5, 10, 20, 50]
    # folds = [5, 1000]
    alt_eval = ['hold_out', 'cross_val-5fold', 'cross_val-1000fold', 'bootstrapping']
    alt_eval_dict = {'hold_out' : 'Hold Out', 'cross_val-5fold' : 'CV 5-Fold', 'cross_val-1000fold' : 'CV LOO', 'bootstrapping' : 'Bootstrapping'}

    df_dict = {}

    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    if model == 'BaselinePlusCV':
                        # oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        ho_df = pd.read_csv(f"{data_path}/{model}-hold_out.csv")
                        cv5_df = pd.read_csv(f"{data_path}/{model}-cross_val-5fold.csv")
                        cv1000_df = pd.read_csv(f"{data_path}/{model}-cross_val-1000fold.csv")
                        boot_df = []                     
                    else:  
                        # oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        ho_df = pd.read_csv(f"{data_path}/{model}-hold_out.csv")
                        cv5_df = pd.read_csv(f"{data_path}/{model}-cross_val-5fold.csv")
                        cv1000_df = pd.read_csv(f"{data_path}/{model}-cross_val-1000fold.csv")
                        boot_df = pd.read_csv(f"{data_path}/{model}-bootstrapping.csv")
                    df_dict[model_dir] = [ho_df, cv5_df, cv1000_df, boot_df]
    
    fig, axs = plt.subplots(5, 4, figsize=(15, 20))
    plt.rcParams["font.family"] = "Times New Roman"

    fig, big_axes = plt.subplots( figsize=(15, 20) , nrows=5, ncols=1, sharey=True) 

    for row, big_ax in enumerate(big_axes, start=1):
        # print(row)
        big_ax.set_title(f"{models_dir[row-1]} \n", fontsize=22, fontweight="bold", fontname="Times New Roman")

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    for i in range(5):
        for j in range(4):
            plot_idx = (i * 4) + j + 1 
            ax = fig.add_subplot(5,4,plot_idx)
            ax.set_title(alt_eval_dict[alt_eval[j]])
            data_plot = df_dict[models_dir[i]][j]['test_accuracy'].astype(int)

            # data_plot = [3 if data_point == 0 else data_point for data_point in data_plot]
            ax.set_xlim([-2, 102])
            # ax.set_ylim([0, 210])
            ax.hist(data_plot, bins=range(0, 100 + 5, 5), edgecolor = "black")
            ax.set_title(alt_eval_dict[alt_eval[j]], fontweight="bold", fontname = "Times New Roman")
            if i == 4:
                ax.set_xlabel("Accuracy", fontname = "Times New Roman")
            if j == 0:    
                ax.set_ylabel("Frequency", fontname = "Times New Roman")
            for tick in ax.get_xticklabels():
                tick.set_fontname("Times New Roman")
            for tick in ax.get_yticklabels():
                tick.set_fontname("Times New Roman")


    # for model_idx, model in enumerate(models_dir):
    #     for eval_idx, (alt_eval, alt_eval_name) in enumerate(alt_eval_dict.items()):
    #         # model_idx = (i*2) + j
    #         data_plot = df_dict[model][eval_idx]['test_accuracy'].astype(int)

    #         # data_plot = [3 if data_point == 0 else data_point for data_point in data_plot]
    #         axs[model_idx, eval_idx].set_xlim([-2, 102])
    #         # axs[model_idx, eval_idx].set_ylim([0, 210])
    #         axs[model_idx, eval_idx].hist(data_plot, bins=range(0, 100 + 5, 5), edgecolor = "black")
    #         axs[model_idx, eval_idx].set_title(alt_eval_name, fontweight="bold", fontname = "Times New Roman")
    #         if model_idx == 4:
    #             axs[model_idx, eval_idx].set_xlabel("Accuracy", fontname = "Times New Roman")
    #         if eval_idx == 0:    
    #             axs[model_idx, eval_idx].set_ylabel("Frequency", fontname = "Times New Roman")
    #         for tick in axs[model_idx, eval_idx].get_xticklabels():
    #             tick.set_fontname("Times New Roman")
    #         for tick in axs[model_idx, eval_idx].get_yticklabels():
    #             tick.set_fontname("Times New Roman")


    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.4)


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def subplots_centered(nrows, ncols, figsize, nfigs):
    """
    Modification of matplotlib plt.subplots(),
    useful when some subplots are empty.
    
    It returns a grid where the plots
    in the **last** row are centered.
    
    Inputs
    ------
        nrows, ncols, figsize: same as plt.subplots()
        nfigs: real number of figures
    """
    assert nfigs < nrows * ncols, "No empty subplots, use normal plt.subplots() instead"
    
    fig = plt.figure(figsize=figsize)
    axs = []
    
    m = nfigs % ncols
    m = range(1, ncols+1)[-m]  # subdivision of columns
    gs = gridspec.GridSpec(nrows, m*ncols)

    for i in range(0, nfigs):
        row = i // ncols
        col = i % ncols

        if row == nrows-1: # center only last row
            off = int(m * (ncols - nfigs % ncols) / 2)
        else:
            off = 0

        ax = plt.subplot(gs[row, m*col + off : m*(col+1) + off])
        axs.append(ax)
        
    return fig, axs

def create_acc_histograms_new():
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    # model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    datasets = ['miniImageNet']
    ways = [5]
    shots = [5]
    # shots = [5, 10, 20, 50]
    # folds = [5, 1000]
    alt_eval = ['hold_out', 'cross_val-5fold', 'cross_val-1000fold', 'bootstrapping']
    alt_eval_dict = {'hold_out' : 'Hold Out', 'cross_val-5fold' : 'CV 5-Fold', 'cross_val-1000fold' : 'CV LOO', 'bootstrapping' : 'Bootstrapping'}

    df_dict = {}
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    eval_dfs = []
                    for eval_type, eval_name in alt_eval_dict.items():
                        if (model == 'BaselinePlusCV') and (eval_type == 'bootstrapping'):
                            continue
                        oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        alt_df = pd.read_csv(f"{data_path}/{model}-{eval_type}.csv").assign(Evaluation=eval_name)
                        alt_df['test_accuracy'] -= oracle_df['test_accuracy']
                        alt_df['test_accuracy'] = alt_df['test_accuracy'].abs()
                        eval_dfs.append(alt_df)
            df_dict[model_dir] = pd.concat(eval_dfs)
    
    print(df_dict['Baseline'].shape)
    
    # fig, axs = plt.subplots(2, 3, figsize=(15, 6))
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('ggplot')
    fig, axs = subplots_centered(nrows=2, ncols=3, figsize=(15,6), nfigs=5)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.weight"] = "bold"

    for i in range(2):
        for j in range(3):
            plot_idx = (i * 3) + j
            if plot_idx == 5:
                break
            # if plot_idx == 5:
                # axs[plot_idx].set_axis_off()
                # break
            axs[plot_idx].set_ylim([-1, 100])
            sns.boxplot(ax=axs[plot_idx], x=df_dict[models_dir[plot_idx]]["Evaluation"], y=df_dict[models_dir[plot_idx]]["test_accuracy"], palette="tab10")
            if plot_idx in [0, 3]:
                axs[plot_idx].set_ylabel("MAE Frequency", fontname="Times New Roman")
            else:
                axs[plot_idx].set_ylabel(" ", fontname="Times New Roman")

            axs[plot_idx].set_title(models_dir[plot_idx], fontweight="bold", fontname = "Times New Roman", fontsize=12)
            axs[plot_idx].set(xlabel=None)
            for tick in axs[plot_idx].get_xticklabels():
                tick.set_fontname("Times New Roman")
                tick.set_fontweight('bold')
            for tick in axs[plot_idx].get_yticklabels():
                tick.set_fontname("Times New Roman")
                tick.set_fontweight('bold')


    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.4)


    # plot data in grouped manner of bar type
    plt.savefig(f"acc_boxplots.pdf", format="pdf", bbox_inches="tight")
    print(f"Plot acc_boxplots.pdf saved!")

def create_teaser_figure():
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML', 'BaselinePlusCV',]
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML', 'Baseline++CV']
    models = ['Baseline']
    models_dir = ['Baseline']
    datasets = ['miniImageNet']
    ways = [5]
    shots = [5]
    # shots = [5, 10, 20, 50]
    # folds = [5, 1000]
    alt_eval = ['hold_out', 'cross_val-5fold', 'cross_val-1000fold', 'bootstrapping']
    alt_eval_dict = {'hold_out' : 'Hold Out', 'cross_val-5fold' : 'CV 5-Fold', 'cross_val-1000fold' : 'CV LOO', 'bootstrapping' : 'Bootstrapping'}

    df_dict = {}
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    eval_dfs = []
                    for eval_type, eval_name in alt_eval_dict.items():
                        if (model == 'BaselinePlusCV') and (eval_type == 'bootstrapping'):
                            continue
                        oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        alt_df = pd.read_csv(f"{data_path}/{model}-{eval_type}.csv")
                        # alt_df['test_accuracy'] -= oracle_df['test_accuracy']
                        # alt_df['test_accuracy'] = alt_df['test_accuracy']
                        eval_dfs.append(alt_df)
            df_dict[model_dir] = [oracle_df, eval_dfs]
    
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, 4, figsize=(15, 3))
    # fig, axs = subplots_centered(nrows=3, ncols=2, figsize=(12,9), nfigs=6)
    plt.rcParams["font.family"] = "Times New Roman"

    print(df_dict)
    print(df_dict[models_dir[0]][0])
    print(df_dict[models_dir[0]][1][2])

    sns.set()
    # for i in range(1):
    for j in range(4):
        plot_idx = j
        # if plot_idx == 5:
            # axs[plot_idx].set_axis_off()
            # break
        # sns.boxplot(ax=axs[i, j], x=df_dict[models_dir[plot_idx]]["Evaluation"], y=df_dict[models_dir[plot_idx]]["test_accuracy"], palette="tab10")
        plt.xlim(0,100)
        plt.ylim(0,100)
        # axs[j].set_xlim(xlim)
        # sns.set()
        # sns.regplot(ax=axs[j], y=df_dict[models_dir[0]][0].sample(n=100, random_state=42)["test_accuracy"], x=df_dict[models_dir[0]][1][plot_idx].sample(n=100, random_state=42)["test_accuracy"], color=(1, 0.38, 0.27, 0.8), truncate=False)
        axs[j].scatter(y=df_dict[models_dir[0]][0].sample(n=100, random_state=2)["test_accuracy"], x=df_dict[models_dir[0]][1][plot_idx].sample(n=100, random_state=2)["test_accuracy"], color=(1, 0.38, 0.27, 0.8))
        # b, a = np.polyfit(df_dict[models_dir[0]][0]["test_accuracy"], df_dict[models_dir[0]][1][plot_idx]["test_accuracy"], deg=1)
        # Create sequence of 100 numbers from 0 to 100 
        # xseq = np.linspace(0, 100, num=1000)
        # axs[j].plot(xseq, a + b * xseq, color=(1, 0.38, 0.27, 0.8), lw=2.5)
        # sns.despine()
        # x0, x1 = axs[j].get_xlim()
        # y0, y1 = axs[j].get_ylim()
        # x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        # y = np.array([2, 5, 6, 7, 9, 12, 16, 19])
        x_low = 0.9*min(df_dict[models_dir[0]][1][plot_idx]["test_accuracy"])
        x_high = 1.1*max(df_dict[models_dir[0]][1][plot_idx]["test_accuracy"])
        x_extended = np.linspace(x_low, x_high, 100)

        p1 = np.polyfit(df_dict[models_dir[0]][1][plot_idx]["test_accuracy"], df_dict[models_dir[0]][0]["test_accuracy"], 1) 

        #find line of best fit
        # a, b = np.polyfit(df_dict[models_dir[0]][1][plot_idx]["test_accuracy"], df_dict[models_dir[0]][0]["test_accuracy"], 1)
        axs[j].plot(x_extended, np.polyval(p1,x_extended), color=(1, 0.38, 0.27, 0.8), lw=2.5)     
        #add points to plot
        # plt.scatter(x, y)

        #add line of best fit to plot
        lims = [-1, 101]
        axs[j].plot(lims, lims, color='green')
        axs[j].set_title(alt_eval_dict[alt_eval[plot_idx]], fontweight="bold", fontname = "Times New Roman", fontsize=12)
        if j == 0:
            axs[j].set_ylabel("Oracle accuracy", fontname="Times New Roman")
        axs[j].set_xlabel("Estimated accuracy", fontname="Times New Roman")
        # axs[j].set(xlabel=None)
        for tick in axs[j].get_xticklabels():
            tick.set_fontname("Times New Roman")
        for tick in axs[j].get_yticklabels():
            tick.set_fontname("Times New Roman")
        
        # axs[j].set_title("Seaborn colorblind style sheet")

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.4)


    # plot data in grouped manner of bar type
    plt.savefig(f"teaser_fig.pdf", format="pdf", bbox_inches="tight")
    print(f"Plot teaser_fig.pdf saved!")


def create_stability_test_plot():
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2', 'MAML']
    models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2', 'MAML']
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2']
    datasets = ['miniImageNet']
    ways = [5]
    shots = [5]
    # shots = [5, 10, 20, 50]
    # folds = [5, 1000]
    alt_eval = ['hold_out', 'cross_val-5fold', 'cross_val-1000fold', 'bootstrapping']
    alt_eval_dict = {'hold_out' : 'Hold Out', 'cross_val-5fold' : 'CV 5-Fold', 'cross_val-1000fold' : 'CV LOO', 'bootstrapping' : 'Bootstrapping'}

    df_dict = {}

    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                    stability_df = pd.read_csv(f"{data_path}/stability_test/{model}-oracle.csv")
                    df_dict[model_dir] = [oracle_df, stability_df]
    
    fig = plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    ax = fig.add_axes([0,0,1,1])
    ax.set_ylabel("Per episode MAE", fontname = "Times New Roman", fontsize=16)
    my_cmap = plt.get_cmap("tab10")
    # langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    mae_values = [get_MAE(oracle_df['test_accuracy'], stability_df['test_accuracy']) for oracle_df, stability_df in df_dict.values()]
    ax.bar(models_dir, mae_values, color=my_cmap.colors)

    plt.savefig(f"stability_exp.pdf", format="pdf", bbox_inches="tight")
    print("Stability Plot created!")

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def get_dataframe(path, filename):
    df_list = []
    all_files = os.listdir(path)

    for file in all_files:
        if filename in file:
            df_list.append(str(file))
    
    if len(df_list) == 1:
        print(pd.read_csv(os.path.join(path, f'{filename}.csv')).shape)
        return pd.read_csv(os.path.join(path, f'{filename}.csv'))
    
    df_list.sort(key=natural_keys)
    partial_df_list = []
    for i in range(len(df_list)):
        print(df_list[i])
        partial_df_list.append(pd.read_csv(os.path.join(path, f'{df_list[i]}')))
    
    final_df = pd.concat(partial_df_list)
    print(final_df.shape)
    # print(final_df.shape)

    return final_df

# Get average spearman correlation coeff per evaluation method
def get_avg_corr(eval_accs, acc_col='test_accuracy', test_samples=600):
    div_samples = test_samples
    sum_corr_coeff = 0
    for i in range(test_samples):
        # print(i)
        oracle_accs = []
        alt_accs = []
        for oracle_df, alt_df in eval_accs:
            oracle_accs.append(oracle_df.iloc[i]['test_accuracy'])
            alt_accs.append(alt_df.iloc[i][acc_col])
        # print(f"Oracle accs: {oracle_accs}")
        # print(f"Alt accs: {alt_accs}")
        print(oracle_accs)
        print(alt_accs)
        results = spearmanr(oracle_accs, alt_accs)
        print(results)
        # print(results.correlation)
        sum_corr_coeff += results.correlation
    return sum_corr_coeff / div_samples



def create_spearman_plot():
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    # datasets = ['miniImageNet', 'cifar100', 'MetaAlbumMini']
    datasets = ['miniImageNet', 'cifar100', 'MetaAlbumMini']
    datasets_dict = {'miniImageNet' : 'miniImageNet', 'cifar100' : 'CIFAR-FS', 'MetaAlbumMini' : 'Meta Album'}
    # datasets = ['miniImageNet', 'cifar100']
    # datasets_dict = {'miniImageNet' : 'miniImageNet', 'cifar100' : 'CIFAR-FS'}
    ways = [5]
    shots = [5]
    # shots = [5, 10, 20, 50]
    # folds = [5, 1000]
    alt_eval = ['hold_out', 'cross_val-5fold', 'cross_val-1000fold', 'bootstrapping']
    alt_eval_dict = {'hold_out' : 'Hold Out', 'cross_val-5fold' : 'CV 5-Fold', 'cross_val-1000fold' : 'CV LOO', 'bootstrapping' : 'Bootstrapping'}

    df_dict = {'miniImageNet' : {}, 'cifar100' : {}, 'MetaAlbumMini' : {}}
    for dataset in datasets:
        print(dataset)
        for eval_type, eval_name in alt_eval_dict.items():
            print(eval_type)
            eval_dfs = []
            for way in ways:
                for shot in shots:
                    for model, model_dir in zip(models, models_dir):
                        data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                        # if (model == 'BaselinePlusCV') and (eval_type == 'bootstrapping'):
                        #     continue
                        print(model)
                        oracle_df = get_dataframe(data_path, f"{model}-oracle")
                        alt_df = get_dataframe(data_path, f"{model}-{eval_type}")
                        eval_dfs.append((oracle_df, alt_df))
            df_dict[dataset][eval_type] = eval_dfs
    
    print(df_dict)

    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1, len(datasets), figsize=(15, 3))
    # fig, axs = subplots_centered(nrows=3, ncols=2, figsize=(12,9), nfigs=6)
    plt.rcParams["font.family"] = "Times New Roman"
    for dataset_idx in range(len(datasets)):
        corr_list = []
        for eval_type, eval_name in alt_eval_dict.items():
            corr_list.append(get_avg_corr(df_dict[datasets[dataset_idx]][eval_type], 1800 if dataset_idx == 2 else 600))

        my_cmap = plt.get_cmap("tab10")
        axs[dataset_idx].set_ylim([0.0, 0.45])
        # print(corr_list)
        # print(alt_eval_dict.values())
        axs[dataset_idx].bar(alt_eval_dict.values(), corr_list, color=my_cmap.colors)
        axs[dataset_idx].set_title(datasets_dict[datasets[dataset_idx]], fontweight="bold", fontname = "Times New Roman", fontsize=16)
        if dataset_idx == 0:
            axs[dataset_idx].set_ylabel("Average Spearman's Coefficient", fontname="Times New Roman")
        axs[dataset_idx].set(xlabel=None)
        for tick in axs[dataset_idx].get_xticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontweight('bold')
        for tick in axs[dataset_idx].get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontweight('bold')

    plt.savefig(f"rank_bars.pdf", format="pdf", bbox_inches="tight")
    print("rank_bars.pdf created!")

def multi_domain_table(datasets, shots):
    # model_path = '/afs/inf.ed.ac.uk/user/s25/s2589574/BEPE/Models/'
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2', 'MAML']
    models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2', 'MAML']
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2']
    # datasets = ['miniImageNet']
    ways = [5]
    # shots = [5]
    # shots = [5, 10, 20, 50]
    # folds = [5, 1000]

    df_list = []

    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    if model == 'BaselinePlusCV':
                        # oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        # ho_df = pd.read_csv(f"{data_path}/{model}-hold_out.csv")
                        # cv5_df = pd.read_csv(f"{data_path}/{model}-cross_val-5fold.csv")
                        # cv1000_df = pd.read_csv(f"{data_path}/{model}-cross_val-1000fold.csv")
                        oracle_df = get_dataframe(data_path, f"{model}-oracle")
                        ho_df = get_dataframe(data_path, f"{model}-hold_out")
                        cv5_df = get_dataframe(data_path, f"{model}-cross_val-5fold")
                        cv1000_df = get_dataframe(data_path, f"{model}-cross_val-1000fold")
                        boot_df = []                     
                    else:  
                        # oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                        # ho_df = pd.read_csv(f"{data_path}/{model}-hold_out.csv")
                        # cv5_df = pd.read_csv(f"{data_path}/{model}-cross_val-5fold.csv")
                        # cv1000_df = pd.read_csv(f"{data_path}/{model}-cross_val-1000fold.csv")
                        # boot_df = []        
                        oracle_df = get_dataframe(data_path, f"{model}-oracle")
                        ho_df = get_dataframe(data_path, f"{model}-hold_out")
                        cv5_df = get_dataframe(data_path, f"{model}-cross_val-5fold")
                        cv1000_df = get_dataframe(data_path, f"{model}-cross_val-1000fold")
                        boot_df = get_dataframe(data_path, f"{model}-bootstrapping")
                        # boot_df = pd.read_csv(f"{data_path}/{model}-bootstrapping.csv")
                    df_list.append((oracle_df, ho_df, cv5_df, cv1000_df, boot_df))
    
    column_headers = ['Oracle Acc', 'HO Acc', 'CV 5F Acc', 'CV LOO Acc', 'Boot Acc', 'MAE HO', 'MAE CV5', 'MAE CVLOO', 'MAE Boot']
    model_data_tuple = zip(models, df_list)
    data = get_table_data(model_data_tuple, tb_type='general_table')
    print(data)
    create_table(data, column_headers, models, filename=f'MD_{datasets[0]}_{shots[0]}shot.png')
    print(f"Accuracy and MAE for {datasets[0]} {shots[0]}-shot created!")

def get_aggr_reg(oracle_df_list, eval_df_list, acc_column='test_accuracy', n_models=5, episodes=600):
    aggreg_acc = 0
    acc_array = []
    # models_list = [0, 0, 0, 0, 0]
    for episode_idx in range(episodes):
        best_acc = 0
        best_acc_idx = 0
        for algo_idx in range (n_models):
            if eval_df_list[algo_idx].iloc[episode_idx][acc_column] > best_acc:
                best_acc = eval_df_list[algo_idx].iloc[episode_idx][acc_column]
                best_acc_idx = algo_idx
        # models_list[best_acc_idx] += 1
    # print(models_list)
    # estimated_acc = oracle_df_list[models_list.index(max(models_list))]
    # estimated_acc = oracle_df_list[models_list.index(max(models_list))]
    # oracle_acc = oracle_df_list[models_list.index(max(models_list))]
        aggreg_acc += oracle_df_list[best_acc_idx].iloc[episode_idx]['test_accuracy']
        acc_array.append(oracle_df_list[best_acc_idx].iloc[episode_idx]['test_accuracy'])
    # regret = 0


    return ((aggreg_acc / episodes), (np.array(acc_array)))

import json

def calculate_aggreg_acc_and_regret():
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2']
    datasets = ['miniImageNet', 'cifar100', 'MetaAlbumMini']
    datasets_dict = {'miniImageNet' : 'miniImageNet', 'cifar100' : 'CIFAR-FS', 'MetaAlbumMini' : 'Meta Album'}
    alt_eval = ['hold_out', 'cross_val-5fold', 'cross_val-1000fold', 'bootstrapping', 'oracle']
    alt_eval_dict = {'hold_out' : 'Hold Out', 'cross_val-5fold' : 'CV 5-Fold', 'cross_val-1000fold' : 'CV LOO', 'bootstrapping' : 'Bootstrapping', 'oracle' : 'Oracle'}
    ways = [5]
    shots = [5]

    oracle_dict = {'miniImageNet' : {}, 'cifar100' : {}, 'MetaAlbumMini' : {}}
    for dataset in datasets:
        for way in ways:
            for shot in shots:
                df_list = []
                for model, model_dir in zip(models, models_dir):
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                    # if (model == 'BaselinePlusCV') and (eval_type == 'bootstrapping'):
                    #     continue
                    # print(model)
                    if (dataset == 'MetaAlbumMini'):
                        df = get_dataframe(data_path, f"{model}-oracle")
                    else:
                        df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                    # oracle_df = get_dataframe(data_path, f"{model}-oracle")
                    # alt_df = get_dataframe(data_path, f"{model}-{eval_type}")
                    # eval_dfs.append(oracle_df)
                    df_list.append(df)
                oracle_dict[dataset] = df_list
    


    df_dict = {'miniImageNet' : {}, 'cifar100' : {}, 'MetaAlbumMini' : {}}
    for dataset in datasets:
        for eval_type, eval_name in alt_eval_dict.items():
            eval_dfs = []
            for way in ways:
                for shot in shots:
                    for model, model_dir in zip(models, models_dir):
                        data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot')
                        if (((dataset == 'miniImageNet') and (model == 'MAML') and (shot == 50)) or (dataset == 'MetaAlbumMini')):
                            df = get_dataframe(data_path, f"{model}-{eval_type}")
                        else:
                            df = pd.read_csv(f"{data_path}/{model}-{eval_type}.csv")
                        # alt_df = get_dataframe(data_path, f"{model}-{eval_type}")
                        eval_dfs.append(df)
                        
                        # eval_dfs.append((oracle_df, alt_df))
            df_dict[dataset][eval_type] = eval_dfs
    
    print(oracle_dict)
    # print(df_dict)

    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for eval_type, eval_name in alt_eval_dict.items():
            acc_value, acc_array = get_aggr_reg(oracle_dict[dataset], df_dict[dataset][eval_type], n_models=5, episodes=1800 if dataset == 'MetaAlbumMini' else 600)
            print(acc_array.shape)
            results[dataset][eval_type] = f"{acc_value} +- {get_conf_interval(acc_array, 1800 if dataset == 'MetaAlbumMini' else 600)}"


    print(json.dumps(results, indent=4))

# https://www.geeksforgeeks.org/create-a-grouped-bar-plot-in-matplotlib/
def create_boot_analysis_plot_mae(datasets, ways, shots, models, models_dir, filename="boot-analysis_experiment.pdf"):
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'

    # we have to get the oracle accuracy for each algo to define the threshold line 
    #   and get the different accuracies for each fold value

    # getting oracle accuracy
    oracle_df_list = []
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                algo_list = []
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot', 'bootstrapping_analysis')
                    oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                    oracle_df_list.append(oracle_df)
                    # algo_list.append(oracle_df)

    # getting bootstrapping information
    data_list = []
    for dataset in datasets:
        for way in ways:
            for model, model_dir in zip(models, models_dir):
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot', 'bootstrapping_analysis')
                    df = pd.read_csv(f"{data_path}/{model}-bootstrapping.csv")
                data_list.append(df)
                # print(algo_list)
    
    print(oracle_df_list)
    print(data_list)
    # return

    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('ggplot')
    fig, axs = plt.subplots(figsize=(6, 6))
    plt.rcParams["font.family"] = "Times New Roman"

    # iterating through algos
    # for i in range(2):
        # for j in range(3):
    
    # for idx in range(len(models_dir)):
            
    # model_idx = (i*3) + j
    for model_idx, model_name in enumerate(models):
        x_axis = list(range(1,51))
        axs.set_xlim([1, 50])
        axs.set_ylim([15, 30])
        y_axis_values = []
        for boot_value in range(1,51):
            value = get_MAE(oracle_df_list[model_idx]['test_accuracy'], data_list[model_idx][f'{boot_value}-round'])
            y_axis_values.append(value)
        
        print(y_axis_values)
        axs.plot(x_axis, y_axis_values, label=model_name)
        axs.legend()
    # axs.set_title(models_dir[idx], fontweight="bold", fontname = "Times New Roman")
    # if i == 1:
    axs.set_xlabel("Bootstrapping iterations", fontname = "Times New Roman")
    # if j == 0:    
    axs.set_ylabel("MAE", fontname = "Times New Roman")


    for tick in axs.get_xticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight('bold')
        tick.set_fontsize(14)
    for tick in axs.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight('bold')
        tick.set_fontsize(14)


    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)


    # plot data in grouped manner of bar type
    plt.savefig(f"boot_analysis_mae.pdf", format="pdf", bbox_inches="tight")
    print(f"Plot boot_analysis_mae.pdf saved!")


def create_boot_analysis_plot_rank(datasets, ways, shots, models, models_dir, filename="boot-analysis_experiment.pdf"):
    model_path = '/mnt/invinciblefs/scratch/lushima/Models/'

    # we have to get the oracle accuracy for each algo to define the threshold line 
    #   and get the different accuracies for each fold value

    # getting oracle accuracy
    oracle_df_list = []
    for dataset in datasets:
        for model, model_dir in zip(models, models_dir):
            for way in ways:
                algo_list = []
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot', 'bootstrapping_analysis')
                    oracle_df = pd.read_csv(f"{data_path}/{model}-oracle.csv")
                    oracle_df_list.append(oracle_df)
                    # algo_list.append(oracle_df)

    # getting bootstrapping information
    data_list = []
    for dataset in datasets:
        for way in ways:
            for model, model_dir in zip(models, models_dir):
                for shot in shots:
                    data_path = os.path.join(model_path, model_dir, dataset, f'{way}way-{shot}shot', 'bootstrapping_analysis')
                    df = pd.read_csv(f"{data_path}/{model}-bootstrapping.csv")
                data_list.append(df)
                # print(algo_list)
    
    print(oracle_df_list)
    print(data_list)
    # return

    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('ggplot')
    fig, axs = plt.subplots(figsize=(6, 6))
    plt.rcParams["font.family"] = "Times New Roman"

    # iterating through algos
    # for i in range(2):
        # for j in range(3):
    
    # for idx in range(len(models_dir)):
            
    # model_idx = (i*3) + j
    x_axis = list(range(1,51))
    axs.set_xlim([1, 50])
    axs.set_ylim([0.1, 0.2])
    y_axis_values = []
    for boot_value in range(1,51):
        corr_zip = list(zip(oracle_df_list, data_list))
        value = get_avg_corr(corr_zip, acc_col=f'{boot_value}-round')
        y_axis_values.append(value)
    
    print(y_axis_values)
    axs.plot(x_axis, y_axis_values)
    # axs.legend()
    # axs.set_title(models_dir[idx], fontweight="bold", fontname = "Times New Roman")
    # if i == 1:
    axs.set_xlabel("Bootstrapping iterations", fontname = "Times New Roman")
    # if j == 0:    
    axs.set_ylabel("Spearman's correlation coefficient", fontname = "Times New Roman")


    for tick in axs.get_xticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight('bold')
        tick.set_fontsize(14)
    for tick in axs.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontweight('bold')
        tick.set_fontsize(14)


    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3)


    # plot data in grouped manner of bar type
    plt.savefig(f"boot_analysis_rank.pdf", format="pdf", bbox_inches="tight")
    print(f"Plot boot_analysis_rank.pdf saved!")

def main():
    # get_dataframe('/mnt/invinciblefs/scratch/lushima/Models/ProtoNet/MetaAlbumMini/5way-5shot/', 'ProtoNet-hold_out')

    # ! TABLES
    # within_domain_table(['cifar100'], [5])
    # within_domain_table(['miniImageNet'], [5])
    within_domain_table(['food-101'], [5])
    within_domain_table(['food-101'], [10])
    within_domain_table(['food-101'], [20])
    within_domain_table(['food-101'], [50])
    # within_domain_table(['miniImageNet'], [10])
    # within_domain_table(['miniImageNet'], [20])
    # within_domain_table(['miniImageNet'], [50])
    # multi_domain_table(['MetaAlbumMini'], [5])
    # models = ['Baseline', 'ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'ProtoNet', 'R2D2']
    # shots = [5, 10, 20, 50]
    # create_algo_acc_table(models, models_dir, shots, filename='Algo_accuracy_table.png')
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2', 'MAML']
    # shots = [5, 10]
    # create_algo_acc_table(models, models_dir, shots, filename='Algo_accuracy_table2.png')

    # ! MAE PLOTS
    # we dont hav 
    # models = ['Baseline', 'BaselinePlus', 'R2D2', 'BaselinePlusCV', 'ProtoNet', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'R2D2', 'Baseline++CV', 'ProtoNet', 'MAML']
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'MAML']
    # create_MAE_plots(['miniImageNet'], [5], [5, 10, 20, 50], models, models_dir)
    # create_MAE_plots(['miniImageNet'], [5], [5, 10, 20, 50], models, models_dir)
    # models = ['BaselinePlus','ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    # create_MAE_plots(['miniImageNet'], [5], [5, 10], models, models_dir, "MAE_plots_MAML.pdf")

    # ranking_calculation()


    # ! FOLD EXPERIMENT
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'BaselinePlusCV', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'Baseline++CV', 'MAML']
    # create_num_folds_experiment_plot(['miniImageNet'], [5], [5], models, models_dir, exp_type='MAE')

    # ! FIXED SAMPLE EXPERIMENT
    # models = ['Baseline', 'BaselinePlus', 'BaselinePlusCV', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'Baseline++CV', 'ProtoNet', 'R2D2', 'MAML']
    # models = ['Baseline', 'BaselinePlus','ProtoNet', 'R2D2']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2']
    # create_cvloo_experiment_plot(['miniImageNet'], models, models_dir)

    # ! HISTOGRAMS
    create_acc_histograms_new()

    # ! STABILITY TEST
    # create_stability_test_plot()

    # ! TEASER FIGURE
    # create_teaser_figure()

    # ! RANKING
    # create_spearman_plot()

    # ! AGGREGATED ACCURACY
    # calculate_aggreg_acc_and_regret()

    # ! BOOTSTRAPPING ANALYSIS
    # models = ['Baseline', 'BaselinePlus', 'ProtoNet', 'R2D2', 'MAML']
    # models_dir = ['Baseline', 'Baseline++', 'ProtoNet', 'R2D2', 'MAML']
    # create_boot_analysis_plot_mae(['miniImageNet'], [5], [5], models, models_dir)
    # create_boot_analysis_plot_rank(['miniImageNet'], [5], [5], models, models_dir)

if __name__ == '__main__':
    main()