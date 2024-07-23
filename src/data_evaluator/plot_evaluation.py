import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ipywidgets import Image
import io


def plot_correlation_ratio_diff_(corr_real, corr_synth, norm_diff):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.5), sharey=True)

    axs[0].set_title('Real')

    sns.heatmap(corr_real['corr'], annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[0], cbar=False, vmin=0, vmax=1, cmap='Blues')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=0)

    axs[1].set_title('Synth')
    g2 = sns.heatmap(corr_synth['corr'], annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[1], cbar=False, vmin=0, vmax=1, cmap='Blues')
    g2.set_yticklabels([])
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")

    diff = pd.Series.abs(corr_real['corr'] - corr_synth['corr'])
    axs[2].set_title('Diff - Norm Diff : {}'.format(norm_diff))
    g2 = sns.heatmap(diff, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[2], cbar=True, vmin=0, vmax=1, cmap='Blues')
    g2.set_yticklabels([])
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.close(fig)  # Close the figure to avoid displaying it directly in the notebook
    return fig


def plot_correlation_diff_(corr_real, corr_synth, norm_diff):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.5), sharey=True)

    cors = corr_real.iloc[1:, 0:-1]
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))
    axs[0].set_title('Données Réelles')
    sns.heatmap(cors, annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[0], mask=cors_mask, cbar=False, vmin=0, vmax=1, cmap='Blues')
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")

    cors = corr_synth.iloc[1:, 0:-1]
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))
    axs[1].set_title('Données Synthétiques')
    g2 = sns.heatmap(cors, annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[1], mask=cors_mask, cbar=False, vmin=0, vmax=1, cmap='Blues')
    g2.set(yticklabels=[])
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")

    diff = pd.Series.abs(corr_real - corr_synth)
    cors = diff.iloc[1:, 0:-1]
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))
    axs[2].set_title('Différence des Matrices - Norme : {:.2f}'.format(norm_diff))
    g2 = sns.heatmap(cors, annot=True, annot_kws={"fontsize": 8}, linewidths=.3, ax=axs[2], mask=cors_mask, cbar=True, vmin=0, vmax=1, cmap='Blues')
    g2.set(yticklabels=[])
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.close(fig)  # Close the figure to avoid displaying it directly in the notebook
    return fig


def box_plot_results(results, name, x_labels) :
    
    fig = plt.figure()
    fig.suptitle(name)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(x_labels)
    plt.show()

def histo_plot_utility_compare(utilities):
    for utility_key in utilities:
        utilities[utility_key]['Model'] = utilities[utility_key].index
        utilities[utility_key]['data'] = utility_key
    
    plt.figure(figsize=(8, 6)) 
    sns.barplot(y='f1', x='Model', hue='data', data=pd.concat([df for df in utilities.values()]))
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return Image(value=buf.read(), format='png', width=600, height=400)


def accuracy_compare(utilities) :
    acc_dict = {}

    for utility_key in utilities :
        key = utility_key + "_acc"
        acc_dict[key] = utilities[utility_key]['accuracy'].to_list()

    return pd.DataFrame.from_dict(acc_dict)