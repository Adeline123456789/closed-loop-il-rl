"""Figures derived from raw-CSV summaries only; called only with measured rows."""
from pathlib import Path


def plot_summary(summary,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[r for r in summary if r['status']=='COMPLETED']
    if not rows:raise ValueError('No measured rows: do not generate a performance plot')
    fig,axes=plt.subplots(1,2,figsize=(max(8,len(rows)*1.2),4),layout='constrained')
    labels=[f"{r['arm']} / seed {r['seed']}\n{r['split']} / {r['policy_sha'][:7]}" for r in rows]
    for ax,metric,title in zip(axes,['stable_success','overload'],['Stable geometric success','Overload (all episodes)']):
        values=[r[metric+'_rate'] for r in rows]
        ax.bar(range(len(rows)),values,color='#416b96')
        for i,r in enumerate(rows):
            low,high=r[metric+'_ci_low'],r[metric+'_ci_high']
            if low!='NA':ax.vlines(i,low,high,color='black',linewidth=1.5)
        ax.set_xticks(range(len(rows)),labels,rotation=35,ha='right',fontsize=8)
        ax.set_ylim(0,1.05);ax.set_title(title);ax.set_ylabel('Fraction')
    fig.suptitle('Descriptive outcomes; intervals resample parent episodes')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160);plt.close(fig)
