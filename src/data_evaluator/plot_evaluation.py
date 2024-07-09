import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



def plot_correlation_ratio_diff(corr_real, corr_synth, norm_diff) :
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.5), sharey=True)

    axs[0].set_title('Real')
    sns.heatmap(corr_real['corr'],  annot=True, annot_kws={"fontsize":8}, linewidths=.3, ax=axs[0], cbar=False, vmin=0, vmax=1, cmap='Blues')

    axs[1].set_title('Synth')
    g2 = sns.heatmap(corr_synth['corr'],  annot=True, annot_kws={"fontsize":8}, linewidths=.3, ax=axs[1], cbar=False, vmin=0, vmax=1, cmap='Blues')
    # g2.set(yticklabels=[])  

    diff = pd.Series.abs(corr_real['corr'] - corr_synth['corr'])

    axs[2].set_title('Diff - Norm Diff : {}'.format(norm_diff))
    g2 = sns.heatmap(diff, annot_kws={"fontsize":8}, linewidths=.3, ax=axs[2], cbar=False, vmin=0, vmax=1, cmap='Blues')
    # g2.set(yticklabels=[])

    plt.show()


def plot_correlation_diff(corr_real, corr_synth, norm_diff) :
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.5), sharey=True)


    cors = corr_real.iloc[1:, 0:-1]
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))
    axs[0].set_title('Données Réelles')
    sns.heatmap(cors,  annot=True, annot_kws={"fontsize":8}, linewidths=.3, ax=axs[0], mask=cors_mask, cbar=False, vmin=0, vmax=1, cmap='Blues')

    cors = corr_synth.iloc[1:, 0:-1]
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))
    axs[1].set_title('Données Synthétiques')
    g2 = sns.heatmap(cors,  annot=True, annot_kws={"fontsize":8}, linewidths=.3, ax=axs[1], mask=cors_mask, cbar=False, vmin=0, vmax=1, cmap='Blues')
    g2.set(yticklabels=[])  

    diff = pd.Series.abs(corr_real - corr_synth)
    cors = diff.iloc[1:, 0:-1]
    cors_mask = np.triu(np.ones_like(cors, dtype=bool)) - np.identity(len(cors))
    axs[2].set_title('Différence des Matrices - Norme : {:.2f}'.format(norm_diff))
    g2 = sns.heatmap(cors,  annot=True, annot_kws={"fontsize":8}, linewidths=.3, ax=axs[2], mask=cors_mask, cbar=True, vmin=0, vmax=1, cmap='Blues')
    g2.set(yticklabels=[])

    plt.show()

def box_plot_results(results, name, x_labels) :
    
    fig = plt.figure()
    fig.suptitle(name)
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(x_labels)
    plt.show()

def histo_plot_utility_compare(utilities) :
    # fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

    for utility_key in utilities :
        utilities[utility_key]['Model'] = utilities[utility_key].index
        utilities[utility_key]['data'] = utility_key
    
    plt.figure(figsize=(8, 6)) 
    sns.barplot(y='f1',x='Model',hue='data',data=pd.concat([df for df in utilities.values()]))
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.show()


    # axs[0][0].set_title('F1 score')    
    # sns.barplot(y='F1',x='Model',hue='data',data=pd.concat([df for df in utilities.values()]), ax=axs[0][0])
    # axs[0][0].legend(loc='lower right')
    # axs[0][0].set_xticks([])
    # # axs[0][0].set_xticklabels(axs[0][0].get_xticklabels(), rotation=45)

    # axs[0][1].set_title('Precision')
    # sns.barplot(y='precision',x='Model',hue='data',data=pd.concat([df for df in utilities.values()]), ax=axs[0][1])
    # axs[0][1].legend(loc='lower right')
    # axs[0][1].set_xticks([])
    # # axs[0][1].set_xticklabels(axs[0][1].get_xticklabels(), rotation=45)

    # axs[0][2].set_title('Recall')
    # sns.barplot(y='recall',x='Model',hue='data',data=pd.concat([df for df in utilities.values()]), ax=axs[0][2])
    # axs[0][2].legend(loc='lower right')
    # axs[0][2].set_xticks([])
    # # axs[0][2].set_xticklabels(axs[0][2].get_xticklabels(), rotation=45)

    # axs[0][3].set_title('F1')
    # sns.barplot(y='f1',x='Model',hue='data',data=pd.concat([df for df in utilities.values()]), ax=axs[0][3])
    # axs[0][3].set_xticks([])
    # # axs[0][3].set_xticklabels(axs[0][3].get_xticklabels(), rotation=45)
    # axs[0][3].legend(loc='lower right')

    # axs[1][0].set_title('Balanced Accuracy')    
    # sns.barplot(y='balanced_accuracy',x='Model',hue='data',data=pd.concat([df for df in utilities.values()]), ax=axs[1][0])
    # axs[1][0].legend(loc='lower right')
    # axs[1][0].set_xticks(axs[1][0].get_xticks())
    # axs[1][0].set_xticklabels(axs[1][0].get_xticklabels(), rotation=45)

    # axs[1][1].set_title('Precision weighted')
    # sns.barplot(y='precision_weighted',x='Model',hue='data',data=pd.concat([df for df in utilities.values()]), ax=axs[1][1])
    # axs[1][1].legend(loc='lower right')
    # axs[1][1].set_xticks(axs[1][1].get_xticks())
    # axs[1][1].set_xticklabels(axs[1][1].get_xticklabels(), rotation=45)

    # axs[1][2].set_title('Recall weighted')
    # sns.barplot(y='recall_weighted',x='Model',hue='data',data=pd.concat([df for df in utilities.values()]), ax=axs[1][2])
    # axs[1][2].legend(loc='lower right')
    # axs[1][2].set_xticks(axs[1][2].get_xticks())
    # axs[1][2].set_xticklabels(axs[1][2].get_xticklabels(), rotation=45)

    # axs[1][3].set_title('F1 weighted')
    # sns.barplot(y='f1_weighted',x='Model',hue='data',data=pd.concat([df for df in utilities.values()]), ax=axs[1][3])
    # axs[1][3].set_xticks(axs[1][3].get_xticks())
    # axs[1][3].set_xticklabels(axs[1][3].get_xticklabels(), rotation=45)
    # axs[1][3].legend(loc='lower right')
    plt.show()

def accuracy_compare(utilities) :
    acc_dict = {}

    for utility_key in utilities :
        key = utility_key + "_acc"
        acc_dict[key] = utilities[utility_key]['accuracy'].to_list()

    return pd.DataFrame.from_dict(acc_dict)


 
