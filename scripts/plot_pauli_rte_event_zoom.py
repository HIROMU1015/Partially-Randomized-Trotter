#!/usr/bin/env python3
"""Plot completed saved G rows near the H6 minimum; no new accounting."""
import csv
from fractions import Fraction as Q
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pauli_rte_event_extension/2026-10-26'
def main():
    assert (OUT/'STUDY_STOP.json').exists()
    with (OUT/'event_cost_trends.csv').open() as f:rows=list(csv.DictReader(f))
    with (ROOT/'artifacts/pauli_rte_event_extension_preparation/2026-10-26/saved_baseline_quotes.csv').open() as f:base=list(csv.DictReader(f))
    plt.rcParams.update({'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,5,figsize=(17,4),layout='constrained')
    for ax,e in zip(axs,['0.05','0.01','0.005','0.001','0.0005']):
        for cohort,marker in [('exploration','x'),('confirmation','o')]:
            rr=sorted([r for r in rows if r['molecule']=='H6' and r['epsilon_sig']==e and r['q']=='8' and r['K']=='2' and int(r['s'])>=64 and r['cohort']==cohort and r['G_exact']],key=lambda r:int(r['s']))
            ax.errorbar([int(r['s']) for r in rr],[float(Q(r['G_exact'])) for r in rr],yerr=[2*float(r['G_SE']) for r in rr],marker=marker,linestyle='--' if cohort=='exploration' else '-',color='tab:blue',label=cohort+'; 2SE')
        h=next(r for r in base if r['molecule']=='H6' and r['epsilon_sig']==e and r['group']=='Hybrid')
        ax.axhline(float(Q(h['G_exact'])),color='black',linestyle=':',label='Saved Hybrid (n=48)')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title('epsilon='+e);ax.set_xlabel('s=q*r, q=8');ax.set_xticks([64,128,256,512,1024],labels=['64','128','256','512','1024']);ax.tick_params(axis='x',labelsize=8);ax.grid(alpha=.2)
    axs[0].set_ylabel('Conditional G (compiled RZ x shot)');axs[-1].legend(fontsize=7)
    fig.suptitle('H6 K2, s>=64: hypothetical rho=0; exploration n=8; confirmation n=48; lines are visual guides')
    for ext in ['png','svg','pdf']:fig.savefig(OUT/'figures'/('H6_G_vs_s_extension_zoom.'+ext),dpi=160)
    plt.close(fig)
if __name__=='__main__':main()
