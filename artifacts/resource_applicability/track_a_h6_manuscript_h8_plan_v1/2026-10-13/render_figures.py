"""Plot saved scalar tables only. No scientific/circuit calculation."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','svg.hashsalt':'h6-manuscript-v1'})

def rows(name):
    with (HERE/name).open() as f: return list(csv.DictReader(f))

def save(fig,name):
    fig.tight_layout(rect=(0,.07,1,1))
    fig.savefig(HERE/name,metadata={'Date':None})
    p=HERE/name
    p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
    plt.close(fig)

r=rows('h6_precision_results.csv');p=rows('h6_fixed_panel.csv');x=list(range(4))
fig,ax=plt.subplots(figsize=(7.2,4.2))
ax.plot(x,[float(a['exploration_PR_PF_ratio']) for a in r],'o--',color='#888888',label='Exploration: 8 samples, selected K4')
ax.errorbar([t-.035 for t in x],[float(a['fixed_K4_confirmation_ratio']) for a in r],
    yerr=[2*float(a['fixed_K4_confirmation_SE'])/float(a['PF_G']) for a in r],fmt='o-',
    color='#2563eb',capsize=4,label='Fixed K4: fresh 32 samples')
ax.errorbar([t+.035 for t in x],[float(a['postconfirmation_ratio']) for a in r],
    yerr=[2*float(a['postconfirmation_SE'])/float(a['PF_G']) for a in r],fmt='s:',
    color='#e68122',capsize=3,label='Post-confirmation point minimum')
ax.set_xticks(x,[a['epsilon'] for a in r]);ax.set_xlabel('Required complex-signal error epsilon')
ax.set_ylabel('Shot-weighted RZ cost / full PF cost (P = 0)')
ax.set_ylim(.33,.49);ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.text(.5,.012,'Bars: descriptive +/- 2 SE; no confidence bound or population winner.',ha='center',fontsize=8)
fig.subplots_adjust(bottom=.2);save(fig,'FigH6_1_precision.svg')

fig,ax=plt.subplots(figsize=(7.2,4.2))
shot=[float(a['postconfirmation_shots'])/float(a['PF_shots']) for a in r]
circuit=[float(a['postconfirmation_G'])/float(a['postconfirmation_shots'])/(float(a['PF_G'])/float(a['PF_shots'])) for a in r]
ax.plot(x,shot,'o-',label='Total shot ratio',color='#6b4c9a')
ax.plot(x,circuit,'s-',label='Shot-weighted mean RZ per shot ratio',color='#20806c')
ax.plot(x,[float(a['postconfirmation_ratio']) for a in r],'^-',label='Product: total RZ ratio',color='#2563eb')
ax.axhline(1,color='#888888',ls='--',lw=1);ax.set_xticks(x,[a['epsilon'] for a in r])
ax.set_xlabel('Required complex-signal error epsilon');ax.set_ylabel('PR / full PF')
ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.text(.5,.012,'Post-confirmation point selections; state preparation excluded; quantum shots not executed.',ha='center',fontsize=8)
fig.subplots_adjust(bottom=.2);save(fig,'FigH6_2_cost_factors.svg')

s=rows('h6_pilot_strata.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.9))
axes[0].errorbar([0,1],[float(a['RZ_mean']) for a in s],
 yerr=[2*float(a['RZ_SE']) for a in s],fmt='o',capsize=5,color='#2563eb')
axes[0].set_xticks([0,1],['s0: n = 32','s2: n = 8']);axes[0].set_ylabel('Mean full-wrapper RZ / shot')
axes[0].set_title('Observed conditional cost means')
axes[1].bar([0,1],[float(a['probability']) for a in s],color=['#20806c','#e68122'])
axes[1].set_yscale('log');axes[1].set_xticks([0,1],['s0','s2']);axes[1].set_ylabel('Known trajectory probability')
axes[1].set_title('q8 / r2 / K2, prefix 5')
for ax in axes:ax.grid(alpha=.2,axis='y')
fig.text(.5,.012,'Point mean update +0.00734%; multiple rare orders and rare components remain unbounded.',ha='center',fontsize=8)
fig.subplots_adjust(bottom=.2);save(fig,'FigH6_3_pilot.svg')
