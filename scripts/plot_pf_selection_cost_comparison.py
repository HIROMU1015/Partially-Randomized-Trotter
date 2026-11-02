"""Render saved 70-condition PF/PR results only; no ranking or accounting."""
import json
import os
import tempfile

from run_pf_selection_cost_comparison import OUT,read,sha,write


def main():
    if (OUT/'rendering_completion.json').exists():raise ValueError('ALREADY_RENDERED')
    completed=read(OUT/'numeric_completion.json')
    for name,value in completed['result_sha256'].items():
        if sha(OUT/name)!=value:raise ValueError('SAVED_RESULT_CHANGED:'+name)
    rows=read(OUT/'selection_comparison.json')['rows'];summary=read(OUT/'summary.json')
    if len(rows)!=70:raise ValueError('EXACT70')
    with tempfile.TemporaryDirectory(prefix='pf-selection-render-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
            'svg.hashsalt':'ALL_SAVED458-70-v1','pdf.fonttype':42,'xtick.labelsize':8,'ytick.labelsize':8})
        figures=[];panels={};points=[]
        for row in rows:
            from fractions import Fraction
            points.append(dict(condition_id=row['condition_id'],system=row['system'],bias_mode=row['bias_mode'],
                original_scope=row['original_scope'],epsilon=row['epsilon'],factor=row['allowance_factor'],
                baseline_G=float(Fraction(row['baseline']['G_exact'])),replacement_G=float(Fraction(row['replacement']['G_exact'])),
                total_change_percent=float(Fraction(row['winner_total_change_percent_exact'])),
                ratio_baseline=row['PR_over_PF_baseline'],ratio_replacement=row['PR_over_PF_replacement'],
                best_candidate_changed=row['best_candidate_changed'],q_changed=row['q_changed']))
        footer='Conditional historical saved-cohort comparison: H6 profile differs; runtime equivalence unproven; S4 is a composition hypothesis.\nInitial-run global final gate remains missing. Saved PR n32 costs/SE/cohorts and shots unchanged.'
        def save(fig,name):
            fig.text(.5,.012,footer,ha='center',va='bottom',fontsize=7)
            fig.tight_layout(rect=(0,.075,1,.95))
            for ext in ('png','svg','pdf'):
                path=OUT/(name+'.'+ext)
                metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Software':'Matplotlib saved PF selection comparison'}
                fig.savefig(path,dpi=180,metadata=metadata)
                if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
                figures.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size))
            plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5.5))
        colors={'H6_DF19':'#0072B2','H8_DF28':'#D55E00'}
        for system,color in colors.items():
            selected=[r for r in points if r['system']==system]
            axes[0].scatter([r['baseline_G'] for r in selected],[r['replacement_G'] for r in selected],
                color=color,s=26,alpha=.65,label=f'{system} ({len(selected)} scopes)')
            axes[1].scatter([r['baseline_G'] for r in selected],[r['total_change_percent'] for r in selected],
                color=color,s=26,alpha=.65,label=system)
        low=min(min(p['baseline_G'],p['replacement_G']) for p in points)*.8
        high=max(max(p['baseline_G'],p['replacement_G']) for p in points)*1.2
        axes[0].plot([low,high],[low,high],color='black',lw=.8,label='Equal total RZ')
        axes[0].set(xscale='log',yscale='log',xlabel='Baseline best PF total RZ',ylabel='Replacement best PF total RZ',title='Best PF total cost')
        axes[1].axhline(0,color='black',lw=.8)
        axes[1].set(xscale='log',xlabel='Baseline best PF total RZ',ylabel='100 × (replacement / baseline − 1) (%)',title='Change in best PF cost')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(f'ALL_SAVED458: 70 scopes / 30 endpoints; best candidate changes = {summary["all_scopes"]["best_candidate_changed"]}',fontsize=12)
        panels['PF_cost_change']=[p['condition_id'] for p in points]
        save(fig,'PF_cost_change')

        groups=sorted({(p['system'],p['bias_mode'],p['original_scope']) for p in points})
        fig,axes=plt.subplots(4,2,figsize=(12,13))
        scope_names={'old_grid_S2':'old32','fine_grid_S2':'fine S2 (96)','fine_grid_S2_plus_truncated_S4':'S2 + S4 (136)',
                     'old32':'old32','strengthened184':'strengthened184'}
        for ax,(system,bias,scope) in zip(axes.flat,groups):
            selected=[p for p in points if (p['system'],p['bias_mode'],p['original_scope'])==(system,bias,scope)]
            for factor,color,marker in [('1.0','#0072B2','o'),('10.0','#D55E00','s')]:
                subset=sorted([p for p in selected if p['factor']==factor],key=lambda p:float(p['epsilon']))
                ax.plot([float(p['epsilon']) for p in subset],[p['ratio_baseline'] for p in subset],
                    color=color,ls='--',marker=marker,markersize=4,alpha=.65,label='Baseline ×'+factor)
                ax.plot([float(p['epsilon']) for p in subset],[p['ratio_replacement'] for p in subset],
                    color=color,ls='-',marker=marker,markersize=3,label='Direct ×'+factor)
            ax.axhline(1,color='black',lw=.9)
            ax.set(xscale='log',yscale='log',xlabel='epsilon',ylabel='Saved PR / best PF total RZ',
                title=f'{system} / {"noncancellation" if "NONCANCELLATION" in bias else "signed"} / {scope_names[scope]}')
            ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
            panels['PR_PF_ratios/'+'/'.join((system,bias,scope))]=[p['condition_id'] for p in selected]
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.05,.8,'Below 1: saved PR costs less\nAbove 1: selected PF costs less\n\nScopes and factor rows overlap.\nAll 70 conditions retained; no new PR cohort.\nThis is a point comparison, not a population guarantee.',va='top',fontsize=10)
        fig.suptitle('Baseline vs direct-price PF selection: PR/PF ratios',fontsize=14)
        save(fig,'PR_PF_ratios')
        write('figure_data.json',dict(SOURCE=completed['SOURCE'],points=points,panel_condition_ids=panels,
            source_selection_comparison_sha256=sha(OUT/'selection_comparison.json'),
            captions=dict(PF_cost_change='All 70 approved scope conditions, with overlapping scopes. The right panel compares the two selected minima, which may be different candidates. No conditions removed.',
                PR_PF_ratios='All seven system/bias/scope groups, five epsilons and two allowance factors. Dashed: saved mixed model/measured baseline. Solid: available direct-price replacement. Horizontal line: equal PR/PF total. PR is fixed n32.')))
        write('rendering_completion.json',dict(status='PASS_TWO_FIGURES_SIX_EXPORTS',SOURCE=completed['SOURCE'],
            figures=figures,matplotlib_version=matplotlib.__version__,figure_data_sha256=sha(OUT/'figure_data.json'),
            numeric_analysis_reexecuted=False))
        print(json.dumps(dict(status='PASS_RENDERING',figures=2,exports=6)))


if __name__=='__main__':main()
