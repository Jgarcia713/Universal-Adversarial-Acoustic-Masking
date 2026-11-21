"""Create plots from the CSVs written by paired evaluation.

Plots accuracy across carriers and masking conditions, and visualizes
the most frequent misclassifications for a selected carrier. Figures
can be regenerated directly from saved results.
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy(summary_csv,output=None):
    summary=pd.read_csv(summary_csv)
    ax=summary.pivot(index="carrier",columns="condition",values="mean").plot(kind="bar",figsize=(11,5))
    ax.set_ylabel("Classifier accuracy (lower = better)")
    ax.set_ylim(0,1)
    plt.tight_layout()
    if output:plt.savefig(output,dpi=180,bbox_inches="tight")
    else:plt.show()

def plot_top_errors(predictions_csv,carrier="fan",top_n=12,output=None):
    df=pd.read_csv(predictions_csv)
    df=df[(df.carrier==carrier)&(~df.correct)]
    table=pd.crosstab(df.condition,df.predicted_label)
    top=df.predicted_label.value_counts().head(top_n).index
    frac=table.reindex(columns=top,fill_value=0).div(df.groupby("condition").size(),axis=0)*100
    ax=plt.gca();im=ax.imshow(frac.to_numpy(),aspect="auto",cmap="Blues")
    ax.set_yticks(range(len(frac)),frac.index)
    ax.set_xticks(range(len(top)),[chr(97+int(x)) for x in top])
    ax.set_title(f"{carrier}: percentage of incorrect predictions")
    plt.colorbar(im,ax=ax,label="% of errors")
    plt.tight_layout()
    if output:plt.savefig(output,dpi=180,bbox_inches="tight")
    else:plt.show()
