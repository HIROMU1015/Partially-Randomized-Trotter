"""PR-only presentation of frozen H4 MC tables; no statistical recomputation.

Only select/copy stored fields, format units and draw them. No raw trajectories,
resample arrays, sampler, compiler, bootstrap or scientific accounting is loaded.
"""
import ast
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'defa5f6dbc72910c0a46ae2fef304a584ae408e0'
OLD = ROOT/'artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19'
P_OLD = ROOT/'artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19'
OUT = ROOT/'artifacts/resource_applicability/track_a_h4_pr_cost_variability/2026-10-19'
HELPER = ROOT/'scripts/resource_applicability/build_track_a_manuscript_figures.py'
SELF = Path(__file__).resolve()
FIELDS = {
    'bootstrap_summary.csv': ['epsilon','cell_id','n','B','cumulative_mean','cumulative_SE',
        'bootstrap_mean','bootstrap_SE','full_sample_plugin_SE','conditional_empirical_SE',
        'q025','q05','q25','q50','q75','q95','q975','min','max',
        'no_observed_S2_fraction','cumulative_rare_trajectories'],
    'observed_strata.csv': ['epsilon','cell_id','stratum','count','observed_mean_G_exact',
        'observed_min_G','observed_max_G','empirical_cost_share','population_mean_certified'],
    'bootstrap_S2_groups.csv': ['epsilon','cell_id','n','S2_trajectory_draws','replicates',
        'mean_G','min_G','max_G'],
}
N_GRID = {'0.05':[4,8,12,16,32,64,128,256,512,1024,1108],
          '0.0005':[4,8,12,16,32,64,78]}


def require(ok, reason):
    if not ok: raise ValueError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def relative(path):
    return str(path.relative_to(ROOT))


def read_csv(path):
    with path.open(newline='') as stream: return list(csv.DictReader(stream))


def preserved_paths():
    paths={p for directory in (OLD,P_OLD) for p in directory.iterdir() if p.is_file()}
    for directory in (OLD,P_OLD):
        binding=json.loads((directory/('contract.json' if directory==OLD else 'execution_binding.json')).read_text())
        paths.update(ROOT/name for name in binding['source_hashes'])
    paths.update([ROOT/'docs/research/track_a_h4_mc_sample_size_20261019.md',
                  ROOT/'docs/research/track_a_h4_tol_p_completed_20261019.md'])
    return sorted(paths)


def figure_helper():
    spec=importlib.util.spec_from_file_location('h4_saved_figure_style',HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def draw(tables,endpoints,style):
    os.environ['MPLCONFIGDIR']=str(ROOT/'.h4_pr_cost_display_private/mpl_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    # Style, colours, band/line pattern and deterministic three-format export
    # follow the existing manuscript/H4 MC renderer. Every series is stored data.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'svg.hashsalt':'h4-pr-only-cost-variability-20261019'})
    blue=style.COLORS['B2']; green=style.COLORS['B3']; orange=style.COLORS['B1']
    footer=('H4 1 A | STO-3G | tol-only DF10 | T=0.8 | P=0 | saved shots x full-wrapper compiled RZ\n'
            'Deterministic prefix + stochastic RTE tail + full Hadamard wrapper. Stored values only; no new analysis.')
    def export(fig,name,detail):
        fig.text(.015,.018,footer+'\n'+detail,fontsize=7,color='#444444')
        fig.tight_layout(rect=(0,.115,1,.91))
        for ext in ('png','svg','pdf'):
            metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {}
            path=OUT/(name+'.'+ext)
            fig.savefig(path,dpi=180,bbox_inches='tight',metadata=metadata)
            if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    def n_axis(ax,ns):
        ticks=[n for n in ns if n not in (12,1024)] if 1108 in ns else ns
        ax.set_xscale('log',base=2);ax.set_xticks(ticks)
        ax.set_xticklabels([str(n) for n in ticks],rotation=45,ha='right',fontsize=8)
        ax.set_xlabel('MC sample size n (paired trajectories)');ax.grid(alpha=.18)
    def cost_axis(ax,epsilon,trajectory=False):
        # Unit conversion is a tick formatter; plotted values remain raw G_RZ.
        scale=1e6 if epsilon=='0.05' else 1e12
        power='6' if epsilon=='0.05' else '12'
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y/scale:.3f}'))
        label='PR cost per trajectory' if trajectory else 'Mean PR cost'
        ax.set_ylabel(label+r' $G_{\rm RZ}$  ($10^{'+power+'}$)')
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,ep in zip(axes,endpoints):
        eps=str(ep['epsilon']);rows=[r for r in tables['bootstrap_summary.csv'] if r['epsilon']==eps]
        ns=[int(r['n']) for r in rows]
        ax.fill_between(ns,[float(r['q025']) for r in rows],[float(r['q975']) for r in rows],
            color=blue,alpha=.22,label='Stored bootstrap 2.5-97.5 percentiles')
        ax.plot(ns,[float(r['bootstrap_mean']) for r in rows],'o-',color=blue,ms=4,label='Stored mean of bootstrap means')
        ax.plot(ns,[float(r['cumulative_mean']) for r in rows],'s-',color=green,ms=4,label='Stored cumulative MC mean')
        ax.axhline(float(Fraction(ep['full_sample']['G_exact'])),color='#666666',ls=':',label=f"Full saved PR mean, N={ep['N']}")
        ax.set_title(f"epsilon={ep['epsilon']} | {ep['cell_id']}",fontsize=10)
        n_axis(ax,ns);cost_axis(ax,eps);ax.legend(fontsize=7,loc='upper right')
    fig.suptitle('PR mean-cost estimation: smaller stored bootstrap spread at larger n',fontsize=14)
    export(fig,'pr_mean_cost_vs_n','Percentile band is conditional on saved pairs; it is not a population confidence interval. This is a mean estimate, not the total cost of n runs.')

    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,ep in zip(axes,endpoints):
        eps=str(ep['epsilon']);rows=[r for r in tables['bootstrap_summary.csv'] if r['epsilon']==eps]
        ns=[int(r['n']) for r in rows]
        ax.plot(ns,[float(r['bootstrap_SE']) for r in rows],'o-',color=blue,label='Stored SD of bootstrap means')
        ax.plot(ns,[float(r['conditional_empirical_SE']) for r in rows],'--',color='#555555',label='Stored empirical SD / sqrt(n)')
        ax.plot(ns,[float(r['cumulative_SE']) for r in rows],'s:',color=green,ms=4,label='Stored cumulative MC paired SE')
        ax.set_yscale('log');n_axis(ax,ns)
        ax.set_ylabel(r'SE of mean PR cost ($G_{\rm RZ}$ units)')
        ax.set_title(f"epsilon={ep['epsilon']} | N={ep['N']}")
        ax.legend(fontsize=8)
    fig.suptitle('PR standard error versus MC sample size n (log-log axes)',fontsize=14)
    export(fig,'pr_standard_error_vs_n','The sqrt(n) reference is an existing saved column. No new fit or SE calculation; the cumulative MC curve uses one nested sequence.')

    fig,axes=plt.subplots(1,2,figsize=(11,5.3))
    for ax,ep in zip(axes,endpoints):
        eps=str(ep['epsilon']);rows=[r for r in tables['observed_strata.csv'] if r['epsilon']==eps and r['observed_mean_G_exact']]
        for j,r in enumerate(rows):
            ax.vlines(j,float(r['observed_min_G']),float(r['observed_max_G']),color=blue,lw=3,alpha=.5)
            ax.plot(j,float(Fraction(r['observed_mean_G_exact'])),'D',color=blue,ms=8)
            ax.annotate('N='+r['count'],(j,float(r['observed_max_G'])),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xticks(range(len(rows)),[r['stratum'] for r in rows]);ax.set_xlim(-.5,len(rows)-.5)
        ax.margins(y=.15);ax.grid(axis='y',alpha=.18);cost_axis(ax,eps,trajectory=True)
        ax.set_title(f"epsilon={ep['epsilon']}: observed trajectory strata")
    fig.suptitle('Full PR cost within observed S0/S2 strata',fontsize=14)
    export(fig,'pr_observed_strata_cost','Diamonds: stored stratum mean. Lines: stored trajectory min-max, not SE/CI. Unobserved strata have no cost point; K2 excludes S4, K4 has no observed S4.')

    fig,axes=plt.subplots(2,3,figsize=(13,7.5))
    for i,ep in enumerate(endpoints):
        eps=str(ep['epsilon'])
        for ax,n in zip(axes[i],(4,32,ep['N'])):
            rows=[r for r in tables['bootstrap_S2_groups.csv'] if r['epsilon']==eps and int(r['n'])==n]
            xs=[int(r['S2_trajectory_draws']) for r in rows]
            ax.fill_between(xs,[float(r['min_G']) for r in rows],[float(r['max_G']) for r in rows],color=orange,alpha=.18,label='Stored group min-max')
            ax.plot(xs,[float(r['mean_G']) for r in rows],'o-',color=orange,ms=4,label='Stored group mean of mean costs')
            ax.set_title(f"epsilon={ep['epsilon']}, n={n}",fontsize=10)
            ax.set_xlabel('S2_SINGLE trajectory draws in one resample',fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=.18);cost_axis(ax,eps);ax.legend(fontsize=7)
    fig.suptitle('Observed S2 composition and PR mean-cost variation in saved bootstrap groups',fontsize=13)
    export(fig,'pr_saved_S2_groups','Group sizes are copied to the companion CSV. Sparse groups can fluctuate; grouping/correlation/quantiles are not recomputed, and this is not a causal effect estimate.')


def main():
    require(not OUT.exists() or ((OUT/'display_provenance.json').is_file()
        and json.loads((OUT/'display_provenance.json').read_text())['base_MC_commit']==BASE),
        'ONLY_OWNED_PRESENTATION_DESTINATION')
    # Metadata-only protection of the original P, MC SOURCE, tables, figures and
    # authorization/STOP records. Never observes unrelated worktree status.
    preservation={relative(p):sha(p) for p in preserved_paths()}
    inputs={name:sha(OLD/name) for name in [*FIELDS,'result.json']}
    for name,digest in inputs.items():
        blob=subprocess.check_output(['git','show',BASE+':'+relative(OLD/name)],cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest()==digest,'EXACT_BASE_INPUT:'+name)
    tree=ast.parse(SELF.read_text())
    blocked={'mean','std','stdev','variance','quantile','hist','histogram','corrcoef',
             'default_rng','bootstrap','evaluate_pairs','polyfit','curve_fit','sample','choice','integers'}
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
           for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
    require(not calls&blocked,'NO_STATISTICAL_OR_SAMPLING_CALLS')
    style=figure_helper()
    tables={name:read_csv(OLD/name) for name in FIELDS}
    original=json.loads((OLD/'result.json').read_text())
    endpoints=original['endpoints']
    require([ep['epsilon'] for ep in endpoints]==[.05,.0005],'ONLY_H4_TWO_PRECISIONS')
    for eps,ns in N_GRID.items():
        require([int(r['n']) for r in tables['bootstrap_summary.csv'] if r['epsilon']==eps]==ns,'SAVED_N_GRID')
    OUT.mkdir(parents=True,exist_ok=True)
    save_json('display_provenance.json',dict(base_MC_commit=BASE,base_MC_SOURCE=original['SOURCE'],
        renderer=relative(SELF),renderer_sha256=sha(SELF),reused_helper=relative(HELPER),helper_sha256=sha(HELPER),
        inputs_sha256=inputs,selected_fields=FIELDS,cost='saved shots x full compiled Hadamard wrapper RZ: deterministic prefix + stochastic RTE tail + wrapper',
        estimand='variation of the PR mean-cost estimator; not aggregate cost of n executions',
        operations=['copy stored fields without aggregation','parse numbers for rendering','format axes and tables','draw existing summaries'],
        new_sampling=0,new_compile=0,new_bootstrap=0,new_statistical_analysis=0,
        raw_trajectory_files_opened=False,bootstrap_replicates_opened=False,scientific_grant_issued=False,
        old_record_hashes=preservation,next_stage_authorized=False))
    for name,fields in FIELDS.items():
        # CSV values remain literal strings; no new scientific scalar is formed.
        with (OUT/('pr_'+name)).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            writer.writerows({k:r[k] for k in fields} for r in tables[name])
    refs=[]
    for ep in endpoints:
        refs.append(dict(epsilon=ep['epsilon'],cell_id=ep['cell_id'],N=ep['N'],
            G_exact=ep['full_sample']['G_exact'],paired_SE=ep['full_sample']['paired_SE'],
            sample_SD=ep['full_sample']['sample_SD'],cosine_shots=ep['saved_shots']['cosine'],
            sine_shots=ep['saved_shots']['sine']))
    with (OUT/'pr_full_sample_reference.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(refs[0]),lineterminator='\n');writer.writeheader();writer.writerows(refs)
    draw(tables,endpoints,style)
    # Structural checks: exact field-copy identity, file/picture integrity and
    # fixed input hashes. Never re-evaluates numerical statistics.
    row_counts={}
    for name,fields in FIELDS.items():
        copied=read_csv(OUT/('pr_'+name));expected=[{k:r[k] for k in fields} for r in tables[name]]
        require(copied==expected,'LITERAL_FIELD_COPY:'+name);row_counts[name]=len(copied)
    require(read_csv(OUT/'pr_full_sample_reference.csv')==[{k:str(v) for k,v in r.items()} for r in refs], 'FULL_REFERENCE_COPY')
    require(all(sha(ROOT/p)==digest for p,digest in preservation.items()),'ORIGINAL_RECORDS_UNCHANGED')
    from PIL import Image
    for path in OUT.glob('*.png'):
        with Image.open(path) as im:im.verify()
    require(len(list(OUT.glob('*.png')))==4 and len(list(OUT.glob('*.svg')))==4
        and len(list(OUT.glob('*.pdf')))==4,'FOUR_FIGURES_THREE_FORMATS')
    save_json('structural_audit.json',dict(status='PASS',original_files_SHA_unchanged=len(preservation),
        CSV_field_copy_identity=True,original_row_counts=row_counts,full_sample_reference_rows=2,
        source_AST_has_no_statistical_or_sampling_calls=True,figures_structurally_valid=12,
        new_sampling=0,new_compile=0,new_bootstrap=0,new_statistical_analysis=0,
        separate_AI_reviewer_used=False,mandatory_STOP=True))
    save_json('STOP.json',dict(status='COMPLETE_MANDATORY_STOP',next_stage_authorized=False,
        new_scientific_execution=False,original_STOP_unchanged=True))
    save_json('artifact_manifest.json',dict(base_MC_commit=BASE,
        files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'},
        status='COMPLETE_MANDATORY_STOP'))
    print('PR_ONLY_REDISPLAY_COMPLETE',row_counts,'new_statistics=0')


if __name__=='__main__':main()
