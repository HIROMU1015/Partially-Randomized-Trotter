"""Render completed stage B30 and verbatim saved70; no new cost/rank analysis."""
import os
import tempfile
from fractions import Fraction
from run_h6_noncancellation_stage_b_rerun import OUT,read,sha,write


def main():
    if (OUT/'rendering_completion.json').exists():raise ValueError('ALREADY_RENDERED_NO_RETRY')
    completed=read(OUT/'numeric_completion.json')
    for name,value in completed['result_sha256'].items():
        if sha(OUT/name)!=value:raise ValueError('SAVED_RESULT_CHANGED:'+name)
    rows=read(OUT/'combined_selection100.json')['rows'];summary=read(OUT/'summary.json')
    if len(rows)!=100:raise ValueError('EXACT100')
    with tempfile.TemporaryDirectory(prefix='h6-stageB-render-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none',
            'svg.hashsalt':'H6-stageB30-with-saved70','pdf.fonttype':42})
        points=[];figures=[];panels={}
        for r in rows:
            points.append(dict(condition_id=r['condition_id'],system=r['system'],bias_mode=r['bias_mode'],
                scope=r['original_scope'],epsilon=r['epsilon'],factor=r['allowance_factor'],origin=r['numeric_origin'],
                baseline_G=float(Fraction(r['baseline']['G_exact'])),replacement_G=float(Fraction(r['replacement']['G_exact'])),
                total_change_percent=float(Fraction(r['winner_total_change_percent_exact'])),
                baseline_ratio=r['PR_over_PF_baseline'],replacement_ratio=r['PR_over_PF_replacement']))
        footer='Conditional saved-cohort diagnostic: profile/runtime differences; S4 hypothesis; initial-run final gate missing.\n70 prior rows copied; 30 H6 noncancellation rows use stage A shots and a newly completed baseline. Fixed PR n32; scopes overlap.'
        def save(fig,name):
            fig.text(.5,.012,footer,ha='center',va='bottom',fontsize=7)
            fig.tight_layout(rect=(0,.065,1,.96))
            for ext in ('png','svg','pdf'):
                path=OUT/(name+'.'+ext)
                metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Software':'Matplotlib saved-stageB comparison'}
                fig.savefig(path,dpi=170,metadata=metadata)
                if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
                figures.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size))
            plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5.8))
        for label,color,marker,subset in [
            ('Previous H6 signed (30)','#0072B2','o',[p for p in points if p['system']=='H6_DF19' and p['origin']=='VERBATIM_PREVIOUS70']),
            ('Previous H8 both bias modes (40)','#D55E00','s',[p for p in points if p['system']=='H8_DF28']),
            ('New H6 noncancellation (30)','#009E73','^',[p for p in points if p['origin']=='NEW_STAGE_B_30'])]:
            axes[0].scatter([p['baseline_G'] for p in subset],[p['replacement_G'] for p in subset],s=30,color=color,marker=marker,alpha=.6,label=label)
            axes[1].scatter([p['baseline_G'] for p in subset],[p['total_change_percent'] for p in subset],s=30,color=color,marker=marker,alpha=.6,label=label)
        low=min(min(p['baseline_G'],p['replacement_G']) for p in points)*.8
        high=max(max(p['baseline_G'],p['replacement_G']) for p in points)*1.2
        axes[0].plot([low,high],[low,high],color='black',lw=.8)
        axes[0].set(xscale='log',yscale='log',xlabel='Best PF total RZ: mixed-price baseline',ylabel='Best PF total RZ: direct-price replacement',title='Selected PF total costs')
        axes[1].axhline(0,color='black',lw=.8)
        axes[1].set(xscale='log',xlabel='Best PF total RZ: mixed-price baseline',ylabel='Change of selected minimum (%)',title='Direct-price effect on minimum cost')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(f'100 overlapping scopes / 40 endpoints: PF candidate changes {summary["all100"]["best_candidate_changed"]}',fontsize=12)
        panels['PF_cost_change100']=[p['condition_id'] for p in points];save(fig,'PF_cost_change100')
        groups=sorted({(p['system'],p['bias_mode'],p['scope']) for p in points})
        if len(groups)!=10:raise ValueError('TEN_GROUPS_REQUIRED')
        fig,axes=plt.subplots(5,2,figsize=(12,16))
        names={'old_grid_S2':'old32','fine_grid_S2':'fine S2 (96)','fine_grid_S2_plus_truncated_S4':'S2 + S4 (136)',
               'old32':'old32','strengthened184':'strengthened184'}
        for ax,(system,bias,scope) in zip(axes.flat,groups):
            group=[p for p in points if (p['system'],p['bias_mode'],p['scope'])==(system,bias,scope)]
            for factor,color,marker in [('1.0','#0072B2','o'),('10.0','#D55E00','s')]:
                subset=sorted([p for p in group if p['factor']==factor],key=lambda p:float(p['epsilon']))
                x=[float(p['epsilon']) for p in subset]
                ax.plot(x,[p['baseline_ratio'] for p in subset],color=color,ls='--',marker=marker,markersize=4,alpha=.65,label='Mixed baseline x'+factor)
                ax.plot(x,[p['replacement_ratio'] for p in subset],color=color,ls='-',marker=marker,markersize=3,label='Direct price x'+factor)
            ax.axhline(1,color='black',lw=.9)
            kind='noncancellation' if 'cancellation' in bias.lower() else 'signed'
            source='new stage B' if group[0]['origin']=='NEW_STAGE_B_30' else 'saved previous'
            ax.set(xscale='log',yscale='log',xlabel='epsilon',ylabel='Saved PR / best PF total RZ',title=f'{system} {kind} / {names[scope]} ({source})')
            ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
            panels['PR_PF_ratios100/'+'/'.join((system,bias,scope))]=[p['condition_id'] for p in group]
        fig.suptitle('Saved PR/PF point-cost ratios: below 1 PR costs less; above 1 PF costs less',fontsize=13)
        save(fig,'PR_PF_ratios100')
        write('figure_data.json',dict(SOURCE=completed['SOURCE'],points=points,panel_condition_ids=panels,
            combined_result_sha256=sha(OUT/'combined_selection100.json'),
            captions=dict(PF_cost_change100='100 overlapping scopes,40 endpoints; previous70 copied and new30 stageA-completed baseline distinguished. No old70 numeric replay.',
                PR_PF_ratios100='10 system/bias/domain panels; five epsilons and two allowance factors. PR saved union8/union6 n32, not independent P. Conditional point ratios, no population guarantee.')))
        write('rendering_completion.json',dict(status='PASS_TWO_FIGURES_SIX_EXPORTS',SOURCE=completed['SOURCE'],
            figures=figures,figure_data_sha256=sha(OUT/'figure_data.json'),matplotlib_version=matplotlib.__version__,
            old70_numeric_reanalysis=False))
        print(dict(status='PASS_RENDERING',figures=2,exports=6))


if __name__=='__main__':main()
