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

# Categorical slots from the validated reference palette (light mode), in fixed order.
SERIES_COLORS = ["#257de0","#f47230","#00b894","#f2a600","#ee7fac","#008a17","#6a4ce2","#ec4647"]
TRANSFER_CONDITIONS = {"carrier_only":"Carrier\nonly","random_noise":"Random\nnoise",
                       "original_uap":"Original\nUAP","multi_scratch":"Multi-carrier\n(scratch)",
                       "bank_mean":"Bank mean\n(32 masks)"}

def plot_transfer(comparison_csv,output=None,model_labels=None,conditions=TRANSFER_CONDITIONS):
    """Grouped bars of mean accuracy per condition: one panel per carrier, one color per model."""
    import numpy as np
    df=pd.read_csv(comparison_csv)
    models=[c[len("accuracy_"):] for c in df.columns if c.startswith("accuracy_")]
    if len(models)>len(SERIES_COLORS): raise ValueError("Too many models for one chart")
    model_labels=model_labels or {}
    carriers=list(dict.fromkeys(df.carrier))
    fig,axes=plt.subplots(1,len(carriers),figsize=(4.3*len(carriers),4.2),sharey=True,squeeze=False)
    x=np.arange(len(conditions));width=0.6/len(models)
    for ax,carrier in zip(axes[0],carriers):
        sub=df[df.carrier==carrier].set_index("condition").reindex(list(conditions))
        for i,model in enumerate(models):
            ax.bar(x+(i-(len(models)-1)/2)*width,sub[f"accuracy_{model}"],width,color=SERIES_COLORS[i],
                   edgecolor="white",linewidth=2,label=model_labels.get(model,model))
        ax.set_title(carrier.capitalize(),fontsize=11,color="#191919",loc="left")
        ax.set_xticks(x,list(conditions.values()),fontsize=8,color="#666666")
        ax.set_ylim(0,1);ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"{v:.0%}"))
        ax.tick_params(axis="y",labelsize=8,colors="#666666",length=0);ax.tick_params(axis="x",length=0)
        ax.yaxis.grid(True,color="#e6e6e6",linewidth=0.8);ax.set_axisbelow(True)
        for side in ("top","right","left"):ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#bbbbbb")
    axes[0][0].set_ylabel("Letter accuracy (lower = masking works better)",fontsize=9,color="#666666")
    fig.tight_layout(rect=(0,0,1,0.86))
    handles,labels=axes[0][0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper left",bbox_to_anchor=(0.005,0.94),ncol=len(models),
               frameon=False,fontsize=9,handlelength=1.2,columnspacing=1.6)
    fig.suptitle("Do masks optimised against the original CNN transfer to a new model?",
                 x=0.01,y=0.995,ha="left",fontsize=12,color="#191919")
    if output:fig.savefig(output,dpi=200,bbox_inches="tight",facecolor="white")
    else:plt.show()
    plt.close(fig)
