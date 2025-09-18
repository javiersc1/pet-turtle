import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_two_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    C = 101
    labels = np.arange(0, C)
    cm_turtle = np.load("/home/javier/Desktop/turtle/figures/turtle_food101lt_confusion.npy")
    cm_pet = np.load("/home/javier/Desktop/turtle/figures/turtlepet_food101lt_confusion.npy")
    vmax = max(cm_turtle.max(), cm_pet.max())
    vmin = min(cm_turtle.min(), cm_pet.min())

    color = "Blues"
    # First heatmap
    sns.heatmap(
        pd.DataFrame(cm_turtle, index=labels, columns=labels),
        annot=False,
        fmt='d',
        cmap=color,
        cbar=False,
        vmax=vmax,
        vmin=vmin,
        linewidths=0,
        linecolor='gray',
        ax=axes[0]
    )
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_ylabel('True label',fontsize=14)
    axes[0].set_xlabel('Predicted label',fontsize=14)
    
    axes[0].set_title("TURTLE", fontsize=20)
    
    # Second heatmap
    sns.heatmap(
        pd.DataFrame(cm_pet, index=labels, columns=labels),
        annot=False,
        fmt='d',
        cmap=color,
        cbar=False,
        linewidths=0,
        linecolor='gray',
        ax=axes[1],
        vmax=vmax,
        vmin=vmin,
    )
    axes[1].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_ylabel('True label', fontsize=14)
    axes[1].yaxis.set_label_position('right')
    axes[1].set_xlabel('Predicted label', fontsize=14)
    axes[1].set_title("PET-TURTLE", fontsize=20)

    for spine in axes[0].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    for spine in axes[1].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig("figures/confusion.png", bbox_inches='tight', dpi=600)
    plt.close()


plot_two_heatmaps()
