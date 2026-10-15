"""Plot published saved scalars only; no trajectory, reference or compile calls."""
import argparse
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--preview');a=p.parse_args()
    bundle=Path(a.bundle)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    read=lambda n:json.loads((bundle/n).read_bytes())
    comparisons=read('comparisons_v1.json');signals=read('signal_summary_v1.json');rows=read('wrapper_projection_v1.json')
    times=(.4,.8,1.6);eps=(.05,.01,.005,.001,.0005)
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    ratios=[[next(r['ratio'] for r in comparisons if
        (r['T'],r['epsilon'],r['factor'],r['metric'])==(T,e,1.,'RZ')) for T in times] for e in eps]
    heat=[[math.log2(v) if v is not None else float('nan') for v in row] for row in ratios]
    im=axes[0,0].imshow(heat,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1.2,vcenter=0,vmax=3),aspect='auto')
    axes[0,0].set(xticks=range(3),xticklabels=times,yticks=range(5),yticklabels=eps,
        xlabel='Evolution time T (fixed delta = 0.2)',ylabel='Required complex-signal precision',
        title='PR / best eligible deterministic cost (P = 0)')
    for i,row in enumerate(ratios):
        for j,v in enumerate(row):axes[0,0].text(j,i,'ineligible' if v is None else f'{v:.3f}',ha='center',va='center',fontsize=11,
            color='white' if v is not None and (v<.6 or v>4) else 'black')
    cb=fig.colorbar(im,ax=axes[0,0],shrink=.8,ticks=(-1,0,1,2,3));cb.ax.set_yticklabels(('0.5x','1x','2x','4x','8x'))
    methods=(('B0','B0 p15 S2'),('B0I','B0+I p15 S2'),('B1S2','Full PF S2'),('B1S4','Full PF S4'),('B2','PR p5 r4 K2 S2'))
    for method,label in methods:
        def select(T):
            return next(s for s in signals if s['T']==T and
                (s['cell']['method']==method or (method.startswith('B1') and s['cell']['method']=='B1' and
                    (s['cell']['formula']=='2nd')==(method=='B1S2'))))
        selected=[select(T) for T in times]
        axes[0,1].plot(times,[s['decomposition']['total'] for s in selected],marker='o',label=label)
        averages=[]
        for T,s in zip(times,selected):
            subset=[r for r in rows if r['T']==T and r['cell_id']==s['cell']['id']]
            averages.append(sum(r['metrics']['RZ'] for r in subset)/len(subset))
        axes[1,1].plot(times,averages,marker='o',label=label)
    axes[0,1].set(yscale='log',xlabel='T',ylabel='Absolute complex-signal bias',title='Fixed-delta approximation error')
    axes[0,1].legend(fontsize=8)
    pr=[next(s for s in signals if s['T']==T and s['cell']['method']=='B2') for T in times]
    axes[1,0].plot(times,[s['B'] for s in pr],'-o',label='B')
    axes[1,0].plot(times,[s['B']**2 for s in pr],'-s',label='B squared')
    axes[1,0].set(xlabel='T',ylabel='Normalization multiplier',title='PR: same local law, more occurrences')
    axes[1,0].legend()
    axes[1,1].set(yscale='log',xlabel='T',ylabel='Mean RZ count per axis',title='Full-wrapper cost before shots')
    axes[1,1].legend(fontsize=8)
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        ax.set_xticks(times);ax.grid(alpha=.2)
    fig.suptitle('H6 / DF rank 19: fixed five-configuration panel\nPR: 32 independent trajectories per T; point costs, no population-mean guarantee',fontsize=12)
    fig.savefig(bundle/'T_precision_cost_and_causes_v1.svg',metadata={'Date':None})
    svg=bundle/'T_precision_cost_and_causes_v1.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    if a.preview:fig.savefig(a.preview,dpi=140)
    plt.close(fig)


if __name__=='__main__':main()
