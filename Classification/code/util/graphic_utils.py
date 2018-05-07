import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set(font_scale=1.5, style="whitegrid")

def print_confusion_matrix(name, confusion_matrix, class_names):
    """Draws a nice confusion matrix"""
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names)
    df_cm.to_csv('graphics/data/conf-{}.csv'.format(name))
    fig = plt.figure(figsize=(8, 8))
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", linewidths=1,
                              cmap="Greens", cbar=False)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=90, ha='center', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    heatmap.yaxis.set_label_coords(-0.1, 0.5)
    heatmap.xaxis.set_label_coords(0.5, -0.1)
    fig.savefig('graphics/conf-{}.pdf'.format(name))


def plot(d):
    """creates plots for f1 score, precision and recall"""
    for p in ['f1','precision','recall']:
        pal = sns.color_palette("muted")[:2]+[sns.color_palette("muted")[3]] if (len(d['class'].unique()) == 3) else  sns.color_palette("muted")
        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        sns.barplot(x="feature", y=p, ci="sd",hue="class",palette=pal, data=d,dodge=True)
        plt.ylim(ymax = 1,ymin=0)
        plt.legend(ncol=4,loc='upper center', bbox_to_anchor=(0.5, 1.08))
        #plt.title("recall")
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_yticks(np.arange(0.0, 1.05, 0.05),minor=True)
        ax.yaxis.grid(which='minor', linestyle='--')
        ax.yaxis.grid(which='major', linestyle='-')
        plt.xlabel('')
        plt.ylabel('')
        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=15)
        fig.savefig("graphics/{}-{}.pdf".format(p,len(d['class'].unique()) == 3))
