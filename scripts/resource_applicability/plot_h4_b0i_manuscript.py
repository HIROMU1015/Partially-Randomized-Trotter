"""Publication figures from saved scalar tables only, one process/thread."""
import argparse
import csv
import math
import os
from pathlib import Path
import resource
import signal

REPO=Path(__file__).resolve().parents[2]
BUNDLE=REPO/'artifacts/resource_applicability/track_a_h4_b0i_manuscript_integration/2026-10-14'
GEOMETRIES=('0.70','0.80','0.90','1.10','1.40','1.60')
COLORS={'B0_PLUS_I':'#d67a28','B2':'#257bb1'}


def read(name):
    with (BUNDLE/name).open(newline='') as f:return list(csv.DictReader(f))


def log_edges(xs):
    inner=[math.sqrt(a*b) for a,b in zip(xs,xs[1:])]
    return [xs[0]**2/inner[0]]+inner+[xs[-1]**2/inner[-1]]


def main():
    resource.setrlimit(resource.RLIMIT_AS,(1024*2**20,1024*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(120,120));signal.alarm(180)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BUNDLE/'figures');args=parser.parse_args()
    for key in ('MPLCONFIGDIR','TMPDIR','XDG_CACHE_HOME'):
        value=Path(os.environ[key]).resolve()
        if not value.is_relative_to(Path('/home/AbeHiromu')):raise ValueError('private home cache/temp required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'h4-b0i-measured-manuscript-v04'})
    args.output.mkdir(parents=True,exist_ok=False)
    factors=read('cost_factors_1812.csv');uncertainty=read('uncertainty_comparison_1812.csv')
    groups={g:[r for r in factors if r['geometry']==g] for g in GEOMETRIES}
    xs=[float(r['epsilon']) for r in groups[GEOMETRIES[0]]]

    def save(fig,name):
        for ext in ('png','svg','pdf'):
            metadata={'Creator':'H4 B0+I saved-value manuscript v0.4'}
            if ext=='pdf':metadata.update(CreationDate=None,ModDate=None)
            if ext=='svg':metadata['Date']=None
            fig.savefig(args.output/(name+'.'+ext),dpi=200,bbox_inches='tight',metadata=metadata)
            if ext=='svg':
                path=args.output/(name+'.'+ext)
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)

    def precision_axis(ax):
        ax.set_xscale('log');ax.set_xlim(.005,.1)
        ax.set_xticks([.005,.01,.02,.05,.1]);ax.set_xticklabels(['.005','.01','.02','.05','.1'])
        ax.axvline(.05,color='gray',ls=':',lw=.8);ax.grid(alpha=.16)

    fig,ax=plt.subplots(figsize=(10,3.5))
    values=[[int(r['point_min_method']=='B0_PLUS_I') for r in groups[g]] for g in GEOMETRIES]
    ax.pcolormesh(log_edges(xs),range(7),values,cmap=ListedColormap([COLORS['B2'],COLORS['B0_PLUS_I']]),vmin=0,vmax=1,shading='flat',rasterized=True)
    precision_axis(ax);ax.set(yticks=[i+.5 for i in range(6)],yticklabels=GEOMETRIES,
        ylabel='H4 distance (angstrom)',xlabel='Complex-signal precision epsilon',
        title='Point minimum after adding measured B0+I: 397 B0+I / 1415 B2 grid points')
    ax.legend(handles=[Patch(color=COLORS['B2'],label='B2'),Patch(color=COLORS['B0_PLUS_I'],label='B0+I')],loc='upper left',bbox_to_anchor=(1.01,1))
    fig.tight_layout();save(fig,'figure_1_measured_method_map')

    fig,axes=plt.subplots(2,3,figsize=(11.4,6.1),sharex=True,sharey=True)
    for ax,g in zip(axes.flat,GEOMETRIES):
        rows=groups[g];ys=[float(r['ratio_work']) for r in rows]
        ax.plot(xs,ys,color=COLORS['B0_PLUS_I'],lw=1.7)
        ax.fill_between(xs,ys,1,where=[v<1 for v in ys],color=COLORS['B0_PLUS_I'],alpha=.18)
        ax.axhline(1,color='black',lw=.8)
        for r in rows:
            if float(r['epsilon']) in (.005,.05,.1):ax.plot(float(r['epsilon']),float(r['ratio_work']),'o',color='black',ms=4)
        precision_axis(ax);ax.set(title=g+' angstrom',xlabel='Signal precision epsilon',ylabel='G(B0+I) / G(B2)')
    fig.suptitle('Measured family minima; each saved precision reselects prefix/q and B2 configuration')
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'figure_2_measured_cost_ratio')

    fig,axes=plt.subplots(1,3,figsize=(12.2,4.1),sharey=True)
    for ax,e in zip(axes,(.005,.05,.1)):
        rows=[r for r in factors if float(r['epsilon'])==e];x=range(6)
        for offset,key,color,label in [(-.25,'ratio_shots','#56a08a','Total shots'),(0,'ratio_effective_RZ','#cbac47','Effective RZ/shot'),(.25,'ratio_work',COLORS['B0_PLUS_I'],'RZ work = product')]:
            ax.bar([i+offset for i in x],[float(r[key])-1 for r in rows],.24,bottom=1,color=color,label=label)
        ax.axhline(1,color='black',lw=.7);ax.set(yscale='log',xticks=list(x),xticklabels=GEOMETRIES,
            xlabel='H4 distance (angstrom)',title=f'epsilon = {e}',ylabel='B0+I / B2 ratio')
        ax.grid(axis='y',alpha=.16)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,.91))
    fig.suptitle('Reselected minima: shot ratio times effective circuit-cost ratio')
    fig.tight_layout(rect=(0,0,1,.83));save(fig,'figure_3_shot_circuit_factors')

    fig,axes=plt.subplots(2,3,figsize=(11.4,6.1),sharex=True,sharey=True)
    for ax,g in zip(axes.flat,GEOMETRIES):
        rows=[r for r in uncertainty if r['geometry']==g]
        gap=[100*float(r['relative_gap']) for r in rows];width=[100*float(r['two_SE_over_B2']) for r in rows]
        ax.plot(xs,gap,color=COLORS['B2'],lw=1.3)
        ax.fill_between(xs,[a-b for a,b in zip(gap,width)],[a+b for a,b in zip(gap,width)],color=COLORS['B2'],alpha=.25,label='Descriptive +/-2 paired SE')
        ax.axhline(0,color='black',lw=.8);precision_axis(ax);ax.set_yscale('symlog',linthresh=2)
        ax.set(title=g+' angstrom',xlabel='Signal precision epsilon',ylabel='100 [G(B2)-G(B0+I)] / G(B2)')
    axes[0,0].legend(fontsize=8);fig.suptitle('374 points below B2 point estimate minus 2SE; 23 additional B0+I point minima\nSame 32 trajectories select and estimate B2; band is not a confidence interval')
    fig.tight_layout(rect=(0,0,1,.90));save(fig,'figure_4_paired_sample_comparison')

    matched=read('matched_B0_B0I_120.csv');chosen=[r for r in factors if float(r['epsilon'])==.05]
    pairs=[]
    for r in chosen:
        pairs.append(next(m for m in matched if m['geometry']==r['geometry'] and m['B0I_template']==r['B0I_template']))
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharex=True)
    for j,axis in enumerate(('cosine','sine')):
        for method,color,offset in [('B0','#777777',-.18),('B0I',COLORS['B0_PLUS_I'],.18)]:
            axes[0,j].bar([i+offset for i in range(6)],[float(r[method+'_bias_'+axis]) for r in pairs],.35,color=color,label=method.replace('B0I','B0+I'))
            counts=[float(r[method+'_N_'+('cos' if axis=='cosine' else 'sin')]) if r[method+'_N_'+('cos' if axis=='cosine' else 'sin')] else math.nan for r in pairs]
            axes[1,j].plot([i+offset for i in range(6)],counts,linestyle='',marker='o',ms=6,color=color,label=method.replace('B0I','B0+I'))
            for i,v in enumerate(counts):
                if math.isnan(v):axes[1,j].text(i+offset,.98,'ineligible',rotation=90,fontsize=7,ha='center',va='top',transform=axes[1,j].get_xaxis_transform())
        axes[0,j].axhline(.05/math.sqrt(2),color='black',ls='--',lw=.8,label='epsilon / sqrt(2)')
        axes[0,j].set(title=axis,ylabel='Saved absolute axis bias')
        axes[1,j].set(yscale='log',ylabel='Required shots',xlabel='Distance and same prefix/q in both methods',
            xticks=list(range(6)),xticklabels=[r['geometry']+'\np'+r['prefix']+'/q'+r['q'] for r in pairs])
    axes[0,0].legend(fontsize=8);fig.suptitle('Matched B0 versus B0+I at epsilon=.05: phase changes bias/headroom/shots\nDisplayed configurations are selected by B0+I; per-axis compiled RZ is unchanged')
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'figure_S1_matched_phase_bias_shots')
    assets=list(args.output.iterdir());total=sum(p.stat().st_size for p in assets)
    if len(assets)!=15 or total>12*2**20:raise ValueError('figure output count/byte cap')
    print(f'15 PNG/SVG/PDF assets, {total} bytes; new science/compile/sampling=0')


if __name__=='__main__':main()
