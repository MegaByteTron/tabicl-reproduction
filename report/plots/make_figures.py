from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,default=Path('upload'));p.add_argument('--output-dir',type=Path,default=Path('report_figures'));args=p.parse_args();out=args.output_dir;out.mkdir(exist_ok=True,parents=True)
m=pd.read_json(args.input_dir/'metrics.jsonl',lines=True);d=pd.read_json(args.input_dir/'eval_datasets.jsonl',lines=True)
assert not d.duplicated(['step','dataset_id']).any()
assert d.groupby('step').dataset_id.nunique().eq(57).all()
keys=['accuracy','balanced_accuracy','roc_auc','log_loss','ece'];agg=d.groupby('step')[keys].mean();e=m.dropna(subset=['eval/accuracy']).sort_values('step');t=m.dropna(subset=['train/loss']).sort_values('step');first=d[d.step==d.step.min()].set_index('dataset_id');last=d[d.step==d.step.max()].set_index('dataset_id');step=int(d.step.max())
for k in keys:np.testing.assert_allclose(agg[k],e.set_index('step')['eval/'+k],atol=1e-12)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#344054','text.color':'#182230','axes.edgecolor':'#98a2b3','grid.color':'#e4e7ec','grid.linewidth':.65,'savefig.dpi':240,'pdf.fonttype':42})
blue='#2166ac';orange='#d97706';green='#16817a'
def finish(fig,name,title,note):
 fig.suptitle(title,fontsize=16,fontweight='bold',x=.07,ha='left',y=.99)
 fig.text(.07,.015,note,fontsize=8,color='#667085')
 fig.tight_layout(rect=(.02,.055,.99,.94))
 for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),bbox_inches='tight',facecolor='white')
 plt.close(fig)
def axis(a):a.grid(axis='y');a.set_axisbelow(True)
fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for a,key,epoch,label in zip(axs,['train/loss','train/accuracy'],['epoch/loss','epoch/accuracy'],['Training loss','Training accuracy']):
 a.plot(t.step,t[key],color=blue,alpha=.18,lw=.7,label='Logged 10-step metric');a.plot(e.step,e[epoch],color=blue,lw=1.8,label='Logged epoch average');a.set(xlabel='Training step',ylabel=label);axis(a)
axs[1].yaxis.set_major_formatter(PercentFormatter(1));axs[0].legend(frameon=False,fontsize=8)
finish(fig,'01_training','Pretraining on synthetic tables','Partial run • 959 training log points; 95 epoch averages • No artificial smoothing; last training step 9,590')
fig,axs=plt.subplots(2,2,figsize=(10,7))
for a,k,label in zip(axs.flat,keys[:4],['Accuracy','Balanced accuracy','ROC AUC','Log loss']):
 a.plot(agg.index,agg[k],color=blue,lw=2);a.scatter([step],[agg[k].iloc[-1]],color=orange,zorder=3);a.set(xlabel='Training step',ylabel=label);axis(a)
 a.set_title(f'{label}: {agg[k].iloc[-1]:.4f}',loc='left',fontsize=11)
 if k in ['accuracy','balanced_accuracy']:a.yaxis.set_major_formatter(PercentFormatter(1))
finish(fig,'02_validation','Validation performance during pretraining','Equal-weight mean over 57 datasets • 95 evaluations, steps 100–9,500 • AUC over training is not a ROC curve')
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for a,k,label in zip(axs,['accuracy','balanced_accuracy'],['Accuracy','Balanced accuracy']):
 g=d.groupby('step')[k];lo=g.quantile(.25);hi=g.quantile(.75)
 a.fill_between(lo.index,lo,hi,color=blue,alpha=.15,label='25th–75th dataset percentiles');a.plot(agg.index,agg[k],color=blue,lw=2,label='Dataset mean');a.set(xlabel='Training step',ylabel=label,ylim=(0,1));a.yaxis.set_major_formatter(PercentFormatter(1));axis(a)
axs[0].legend(frameon=False,fontsize=8,loc='lower right')
finish(fig,'03_dataset_spread','Performance varies across datasets','Shading describes variability across datasets, not confidence intervals or repeated-run uncertainty • Partial run')
fig,axs=plt.subplots(1,2,figsize=(10,4.8))
for a,k,label in zip(axs,['accuracy','balanced_accuracy'],['Accuracy','Balanced accuracy']):
 ids=last.index;a.scatter(first.loc[ids,k],last.loc[ids,k],s=28,color=blue,alpha=.8,edgecolor='white',linewidth=.4);a.plot([0,1],[0,1],ls='--',color='#98a2b3',lw=1);a.set(xlim=(0,1.02),ylim=(0,1.02),xlabel='First evaluation (step 100)',ylabel='Latest evaluation (step 9,500)',title=label);a.set_aspect('equal');a.xaxis.set_major_formatter(PercentFormatter(1));a.yaxis.set_major_formatter(PercentFormatter(1));axis(a)
 improved=int((last[k]>first.loc[ids,k]).sum());a.text(.05,.93,f'{improved}/57 datasets improved',transform=a.transAxes,fontsize=9)
finish(fig,'04_first_vs_latest','Learning gains by dataset','One point per dataset • Above the dashed diagonal = improvement • First evaluation is not an untrained baseline')
order=last.sort_values('balanced_accuracy').index
fig,axs=plt.subplots(1,2,figsize=(12,11))
for a,ids in zip(axs,[order[:29],order[29:]]):
 z=last.loc[ids];y=np.arange(len(z));a.hlines(y,z.balanced_accuracy,z.accuracy,color='#cbd5e1',lw=2);a.scatter(z.accuracy,y,color=blue,s=21,label='Accuracy');a.scatter(z.balanced_accuracy,y,color=orange,s=21,label='Balanced accuracy');a.set_yticks(y,labels=[str(v) for v in ids],fontsize=8);a.set(xlim=(0,1.03),xlabel='Score',ylabel='OpenML dataset ID');a.xaxis.set_major_formatter(PercentFormatter(1));axis(a)
axs[0].legend(frameon=False,fontsize=9,loc='lower right')
finish(fig,'05_per_dataset','Latest evaluation: all 57 datasets','Step 9,500 • Ordered by balanced accuracy, first column then second • Gaps warrant checking class imbalance and per-class errors')
fig,axs=plt.subplots(1,2,figsize=(10,4.6))
a=axs[0]
for multiclass,color,label in [(False,blue,'Binary'),(True,green,'Multiclass')]:
 z=d[(d.n_classes>2)==multiclass];g=z.groupby('step').balanced_accuracy.mean();a.plot(g.index,g,color=color,lw=2,label=f'{label} (n={z.dataset_id.nunique()})')
a.set(xlabel='Training step',ylabel='Mean balanced accuracy',ylim=(0,1));a.yaxis.set_major_formatter(PercentFormatter(1));a.legend(frameon=False);axis(a)
a=axs[1];a.plot(agg.index,agg.ece,color=green,lw=2);a.set(xlabel='Training step',ylabel='Mean logged ECE',title='Calibration error (lower is better)');axis(a)
finish(fig,'06_task_types_calibration','Task types and calibration','Equal-weight dataset averages • Groups differ in difficulty; this is descriptive, not a controlled comparison • ECE binning follows logger')
agg.to_csv(out/'validation_means.csv');last.to_csv(out/'latest_per_dataset.csv')
print(agg.iloc[[0,-1]].to_string());print('created',len(list(out.glob('*.png'))),'figures')
