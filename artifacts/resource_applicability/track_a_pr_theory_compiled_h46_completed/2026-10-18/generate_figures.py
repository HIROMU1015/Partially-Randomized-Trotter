"""Render frozen comparison rows only; no molecule, sampling or compiler calls."""
from pathlib import Path
from fractions import Fraction as F
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
colors={'A':'#3366aa','B':'#d98820','NATIVE_MIXED':'#24815d'}
labels={'B1_p12_q1_2nd':'PF p12 q1 S2','B1_p12_q1_4th':'PF p12 q1 S4','B2_p3_q1_2nd_r4_K2':'PR p3 q1 r4 K2','B2_p3_q8_2nd_r2_K4':'PR p3 q8 r2 K4','B1_p19_q8_4th':'PF p19 q8 S4','B2_p10_q8_2nd_r1_K2':'PR p10 q8 r1 K2','B2_p15_q8_2nd_r2_K4':'PR p15 q8 r2 K4'}
for filename,caption,fieldlo,fieldhi in [('expectation_vs_saved','Theory expectation vs saved finite sample (not a CI)','rho_lower','rho_upper'),('same_event_comparisons','Same saved event: compiled / theoretical count','rho_lower','rho_upper')]:
 data=json.loads((ROOT/(filename+'.json')).read_text());fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
 for ri,system in enumerate(('H4','H6')):
  ids=list(dict.fromkeys(r['candidate_id'] for r in data if r['system']==system))
  for ci,metric in enumerate(('RZ','CX')):
   ax=axes[ri,ci];ax.axhline(1,color='#555555',lw=1,ls='--')
   for model_i,model in enumerate(colors):
    for axis_i,axis in enumerate(('cosine','sine')):
     for j,cid in enumerate(ids):
      r=next(r for r in data if (r['system'],r['candidate_id'],r['model'],r['axis'],r['metric'])==(system,cid,model,axis,metric))
      x=j+(model_i-1)*.23+(axis_i-.5)*.055;lo=float(F(r[fieldlo]));hi=float(F(r[fieldhi]))
      ax.vlines(x,lo,hi,color=colors[model],lw=2)
      ax.plot([x],[lo],marker='o' if axis=='cosine' else 's',ms=5,color=colors[model],fillstyle='full' if axis=='cosine' else 'none',label=model+' / '+axis if j==0 else None)
      if hi!=lo:ax.plot([x],[hi],marker='_',ms=7,color=colors[model])
   ax.set_title(system+' / '+metric);ax.set_xticks(range(len(ids)),[labels[c] for c in ids],rotation=18,ha='right');ax.set_ylabel('compiled / theory');ax.grid(axis='y',alpha=.2)
 handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=3,fontsize=9)
 fig.suptitle(caption+'\nH4 DF12 / H6 DF19, T=0.8, symmetric-directional, P=0; A/B are decomposition models',fontsize=11)
 fig.tight_layout(rect=(0,.1,1,.91));fig.savefig(ROOT/(filename+'.svg'),metadata={'Creator':'Matplotlib; frozen comparison rows only'});svg=ROOT/(filename+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n');plt.close(fig)
