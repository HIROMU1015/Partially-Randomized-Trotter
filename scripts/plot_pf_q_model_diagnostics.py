"""Render the saved 450-axis diagnostic; no model evaluation or fitting."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from run_pf_q_model_diagnostics import OUT, write, sha

COLORS={'B0':'#0072B2','B0I':'#D55E00'}
SYSTEMS=('H6_DF19','H8_DF28')
LABELS={'H6_DF19':'H6 / DF19','H8_DF28':'H8 / DF28'}
FORMULAS=('2nd','4th')
PF_LABELS={'2nd':'S2','4th':'S4 (composition hypothesis)'}


def main():
    if (OUT/'rendering_completion.json').exists():raise ValueError('RENDERING_ALREADY_COMPLETED')
    completion=json.loads((OUT/'numeric_completion.json').read_text())
    for name,digest in completion['result_hashes'].items():
        if sha(OUT/name)!=digest:raise ValueError('SAVED_RESULT_CHANGED:'+name)
    data=json.loads((OUT/'axis_errors.json').read_text());rows=data['rows']
    grouped=json.loads((OUT/'grouped_errors.json').read_text())['rows']
    if len(rows)!=450:raise ValueError('FIXED_450_SAVED_AXES')
    with tempfile.TemporaryDirectory(prefix='pf-q-model-render-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
            'svg.hashsalt':'pf-q-model-diagnostic-450-v1','pdf.fonttype':42})
        paths=[];panels={}
        footer='Conditional saved-value diagnostic; H6 profile differs; runtime equivalence unproven.\nFull Hadamard-wrapper RZ, no state preparation or shots; cosine/sine may overlap.'
        def save(fig,name):
            fig.text(.5,.015,footer,ha='center',va='bottom',fontsize=8)
            fig.tight_layout(rect=(0,.12,1,.94))
            for ext in ('png','svg','pdf'):
                path=OUT/(name+'.'+ext)
                metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Software':'Matplotlib saved PF diagnostic'}
                fig.savefig(path,dpi=180,metadata=metadata)
                if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
                paths.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size))
            plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,5.7))
        for ax,system in zip(axes,SYSTEMS):
            subset=[r for r in rows if r['system']==system]
            x=[r['measured_RZ'] for r in subset];y=[r['predicted_RZ'] for r in subset]
            lo=min(x+y)*.9;hi=max(x+y)*1.1
            ax.plot([lo,hi],[lo,hi],color='black',lw=1,label='Equal RZ')
            ax.fill_between([lo,hi],[lo*.98,hi*.98],[lo*1.02,hi*1.02],color='0.8',alpha=.6,label='±2% of measured')
            for pf in FORMULAS:
                for method in COLORS:
                    for axis in ('cosine','sine'):
                        picked=[r for r in subset if (r['PF'],r['method'],r['axis'])==(pf,method,axis)]
                        ax.scatter([r['measured_RZ'] for r in picked],[r['predicted_RZ'] for r in picked],s=28,
                            marker='o' if pf=='2nd' else '^',edgecolors=COLORS[method],
                            facecolors=COLORS[method] if axis=='cosine' else 'none',alpha=.7,
                            label=f'{PF_LABELS[pf].split(" ")[0]} {method}' if axis=='cosine' else None)
            if system=='H6_DF19':
                picked=[r for r in subset if r['candidate_id']=='B0_p11_q4_2nd']
                ax.scatter([r['measured_RZ'] for r in picked],[r['predicted_RZ'] for r in picked],
                    marker='*',s=130,color='#CC79A7',zorder=5,label='B0 p11 q4 S2')
            ax.set(xscale='log',yscale='log',xlim=(lo,hi),ylim=(lo,hi),
                xlabel='Directly compiled RZ',ylabel='Saved model prediction (RZ)',
                title=f'{LABELS[system]}: {len(subset)} axes')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
            panels['predicted_vs_measured/'+system]=[r['task_id'] for r in subset]
        fig.suptitle('Saved q-dependent model vs direct compilation',fontsize=14)
        save(fig,'predicted_vs_measured')

        fig,axes=plt.subplots(2,2,figsize=(13,8))
        histograms=[]
        for i,system in enumerate(SYSTEMS):
            values=[r['signed_relative_error_percent'] for r in rows if r['system']==system]
            low=min(min(values),-2.2);high=max(max(values),2.2)
            bins=np.linspace(low,high,21)
            for j,pf in enumerate(FORMULAS):
                ax=axes[i,j];subset=[r for r in rows if (r['system'],r['PF'])==(system,pf)]
                for method in COLORS:
                    selected=[r for r in subset if r['method']==method]
                    counts,edges,_=ax.hist([r['signed_relative_error_percent'] for r in selected],bins=bins,
                        histtype='step',lw=1.7,color=COLORS[method],label=method)
                    histograms.append(dict(system=system,PF=pf,method=method,
                        bin_edges_percent=edges.tolist(),counts=counts.astype(int).tolist(),n_axes=len(selected)))
                ax.axvline(0,color='0.5',lw=.7)
                for value in (-2,2):ax.axvline(value,color='black',ls='--',lw=.8)
                ax.set(title=f'{LABELS[system]} / {PF_LABELS[pf]} (n={len(subset)})',
                    xlabel='Signed error: 100 × (prediction − measured) / measured (%)',ylabel='Axis count')
                ax.legend(fontsize=8);ax.grid(alpha=.15)
                panels[f'relative_error_distribution/{system}/{pf}']=[r['task_id'] for r in subset]
        fig.suptitle('Relative-error distribution; dashed lines are ±2%',fontsize=14)
        save(fig,'relative_error_distribution')

        for dimension in ('q','prefix'):
            fig,axes=plt.subplots(2,2,figsize=(13,8))
            for i,system in enumerate(SYSTEMS):
                for j,pf in enumerate(FORMULAS):
                    ax=axes[i,j];subset=[r for r in rows if (r['system'],r['PF'])==(system,pf)]
                    for method in COLORS:
                        for axis in ('cosine','sine'):
                            selected=[r for r in subset if (r['method'],r['axis'])==(method,axis)]
                            ax.scatter([int(r[dimension]) for r in selected],[r['signed_relative_error_percent'] for r in selected],
                                s=17,color=COLORS[method],marker='o' if axis=='cosine' else 'x',alpha=.5,
                                label=method+' '+axis)
                        means=[r for r in grouped if r['grouping']=='system+PF+method+'+dimension
                               and (r['group']['system'],r['group']['PF'],r['group']['method'])==(system,pf,method)]
                        means.sort(key=lambda r:int(r['group'][dimension]))
                        ax.plot([int(r['group'][dimension]) for r in means],
                            [r['mean_signed_relative_error_percent'] for r in means],
                            lw=1.3,color=COLORS[method],label=method+' mean')
                    for value in (-2,2):ax.axhline(value,color='black',ls='--',lw=.8)
                    ax.axhline(0,color='0.5',lw=.7)
                    ticks=sorted({int(r[dimension]) for r in subset})
                    if dimension=='q':ax.set_xscale('log',base=2)
                    ax.set(xticks=ticks,xticklabels=[str(v) for v in ticks],xlabel=dimension,
                        ylabel='Signed relative error (%)',title=f'{LABELS[system]} / {PF_LABELS[pf]}')
                    ax.grid(alpha=.15);ax.legend(fontsize=6,ncol=2)
                    panels[f'error_vs_{dimension}/{system}/{pf}']=[r['task_id'] for r in subset]
            fig.suptitle(f'Error vs {dimension}; lines are descriptive saved-axis means, no fit',fontsize=13)
            save(fig,'error_vs_'+dimension)
        write('figure_data.json',dict(SOURCE=data['SOURCE'],axis_errors_sha256=sha(OUT/'axis_errors.json'),
            point_columns=['task_id','system','PF','method','axis','q','prefix','predicted_RZ','measured_RZ','signed_relative_error_percent','model_point_class'],
            points=[{k:r[k] for k in ['task_id','system','PF','method','axis','q','prefix','predicted_RZ','measured_RZ','signed_relative_error_percent','model_point_class']} for r in rows],
            panel_task_ids=panels,histograms=histograms,mean_lines_source='grouped_errors.json',
            captions=dict(
                predicted_vs_measured='450 saved conditional comparisons. Color: B0/B0I; marker: S2/S4; filled/open: cosine/sine. Band is ±2% of measured RZ. Axes can overlap. No model was evaluated.',
                relative_error_distribution='Signed saved-value error; 20 shared bins within each system. All points retained. Dashed ±2% boundaries use measured RZ. Counts are axes, not independent samples.',
                error_vs_q='All saved axes by q, S2/S4 and system. Lines are unweighted method means pooled over saved prefixes/cosine/sine; no fit or causal q interpretation.',
                error_vs_prefix='All saved axes by prefix, S2/S4 and system. Lines are unweighted method means pooled over saved q/cosine/sine; no fit or causal prefix interpretation.')))
        write('rendering_completion.json',dict(status='PASS_FOUR_FIGURES_TWELVE_EXPORTS',SOURCE=data['SOURCE'],
            renderer_source_sha256=sha(Path(__file__)),matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
            figures=paths,figure_data_sha256=sha(OUT/'figure_data.json'),new_model_predictions=0,new_fit=0))
        print(json.dumps(dict(status='PASS_RENDERING',figures=4,exports=len(paths))))


if __name__=='__main__':main()
