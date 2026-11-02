"""One authorized H4 saved-pair sample-size analysis; no molecular imports.

seal freezes SOURCE, inputs, runtime and scope. run consumes the one-shot grant.
audit reads saved outputs only; it never resamples. All caches stay in this tree.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import importlib.util
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import h4_dp_contract as c

ROOT = Path(__file__).resolve().parents[1]
BASE = '821cc0f6ced57168293ea166644024867b064ffb'
SCOPE = 'H4_TOL10_MC_SAMPLE_SIZE_20261019'
OLD = Path('artifacts/resource_applicability/track_a_h4_tol_p_completed/2026-10-19')
OUT = ROOT/'artifacts/resource_applicability/track_a_h4_mc_sample_size/2026-10-19'
PRIVATE = ROOT/'.h4_mc_sample_size_private'
SEED, B = 20261019, 2000
TARGETS = [
    dict(epsilon=.05, cell_id='B2_p3_q1_2nd_r4_K2', N=1108,
         PF_id='B0I_p5_q1_2nd', ns=[4,8,12,16,32,64,128,256,512,1024,1108]),
    dict(epsilon=.0005, cell_id='B2_p2_q8_2nd_r2_K4', N=78,
         PF_id='B0_p7_q8_2nd', ns=[4,8,12,16,32,64,78]),
]
STRATA = ('S0_ONLY', 'S2_SINGLE', 'S2_MULTI', 'S4_PRESENT')
REUSE = ['scripts/h4_dp_contract.py', 'src/trotterlib/h8_rare_tail_selection.py',
         'scripts/resource_applicability/build_track_a_manuscript_figures.py']
SOURCES = ['scripts/analyze_h4_mc_sample_size.py', 'tests/test_h4_mc_sample_size.py'] + REUSE
FORBIDDEN = dict(RTE_sampling=0, circuit_generation=0, compile=0, signal=0,
                 shot_recalculation=0, PF_reselection=0, new_scientific_seeds=0)


def pure_module(name, relative):
    # Avoid trotterlib.__init__, which imports molecular and compiler providers.
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


paired = pure_module('h4_mc_paired_helper', REUSE[1])
figures = pure_module('h4_mc_saved_figure_helper', REUSE[2])


def numerical_runtime():
    os.environ['MPLCONFIGDIR'] = str(PRIVATE/'mpl_cache')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return np, plt


def now():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def runtime_identity():
    from importlib import metadata
    packages = ['numpy','matplotlib','pillow','contourpy','cycler','fonttools',
                'kiwisolver','packaging','pyparsing','python-dateutil','six']
    paths = {str(Path(sys.executable).resolve())}
    versions = {}
    for name in packages:
        dist = metadata.distribution(name)
        versions[name] = dist.version
        for item in dist.files or []:
            path = Path(dist.locate_file(item)).resolve()
            if path.is_file() and path.suffix != '.pyc': paths.add(str(path))
    for mod in tuple(sys.modules.values()):
        name = getattr(mod, '__file__', None)
        if name and Path(name).is_file() and Path(name).suffix != '.pyc':
            paths.add(str(Path(name).resolve()))
    return dict(python=str(Path(sys.executable).resolve()), version=sys.version,
                packages=versions, files={p:c.sha(p) for p in sorted(paths)})


def seal():
    c.require(not OUT.exists(), 'UNUSED_ANALYSIS_DESTINATION')
    numerical_runtime()
    head = git('rev-parse','HEAD')
    c.require(git('merge-base',BASE,head) == BASE, 'FIXED_P_ANCESTOR')
    for name in SOURCES:
        raw = subprocess.check_output(['git','show',head+':'+name],cwd=ROOT)
        c.require(raw == (ROOT/name).read_bytes(), 'COMMITTED_SOURCE:'+name)
    inputs = {str(p.relative_to(ROOT)):c.sha(p) for p in sorted((ROOT/OLD).iterdir()) if p.is_file()}
    # Every inherited P SOURCE file is protected, without importing its providers.
    old_source = c.read(ROOT/OLD/'execution_binding.json')['source_hashes']
    protected = dict(inputs)
    protected.update({name:c.sha(ROOT/name) for name in old_source})
    c.require(all(protected[name] == value for name,value in old_source.items()), 'ORIGINAL_P_SOURCE')
    for name in inputs:
        blob = subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        c.require(blob == (ROOT/name).read_bytes(), 'ORIGINAL_P_INPUT:'+name)
    calls = dict(analysis_dispatch=1, full_sample_reconstructions=2,
                 cumulative_prefixes=1186, bootstrap_groups=18,
                 bootstrap_replicates=36000, **FORBIDDEN)
    contract = dict(scope=SCOPE, SOURCE=head, base_P_commit=BASE,
        source_hashes={name:c.sha(ROOT/name) for name in SOURCES}, inputs=inputs,
        protected=protected, runtime=runtime_identity(), targets=TARGETS,
        bootstrap=dict(seed=SEED, repetitions_per_n=B, replacement=True,
            sampling_unit='one saved trajectory containing BOTH cosine and sine',
            generator='numpy.PCG64(SeedSequence([20261019, endpoint_index, n]))',
            stream_rule='distinct deterministic streams for every endpoint/n',
            quantiles=[.025,.05,.25,.5,.75,.95,.975], quantile_method='linear',
            comparison='strict PR < fixed PF; ties separately counted',
            conditional_on_saved_empirical_distribution=True),
        cumulative='all prefixes in ascending saved replica order; no permutation',
        SE_definitions=dict(cumulative='sample SD / sqrt(n), n>=2',
            plugin='full saved sample SD / sqrt(n)',
            conditional='sqrt(((N-1)/N)*full sample variance/n)',
            bootstrap='SD of 2000 resampled means, ddof=1',
            within_resample='sample SD / sqrt(n) within each resampled trajectory set'),
        event_analysis='saved trajectory strata only; no raw event or law reconstruction',
        calls=calls, new_RTE_sampler_seeds=False, one_shot=True, automatic_retry=False,
        next_stage_authorized=False, public_push_authorized=False,
        resource_policy='vectorized per-n arrays; existing affinity/quota; no agent CPU/RAM/worker cap',
        unrelated_worktree_HEAD_or_status_is_a_gate=False, mandatory_STOP=True)
    c.save(OUT/'contract.json',contract)
    receipt = dict(scope=SCOPE, recorded_at_UTC=now(), authorization='EXPLICIT_USER',
        user_execution_instruction='ではこのチャットで作業を開始して',
        prior_explicit_scope='この範囲の保存データ数値再解析とbootstrapを、独立したone-shot解析1回として明示承認する。',
        scope_reference='Codex指示：H4のMC標本数依存性検証',
        SOURCE=head, contract_sha256=c.sha(OUT/'contract.json'), calls=calls,
        authorization_scope='preparation, one execution, audit, figures, local commits; no scope expansion',
        public_push_authorized=False, no_additional_user_checkpoint_required=True)
    c.save(OUT/'approval_receipt.json',receipt)
    c.save(OUT/'grant.json',dict(scope=SCOPE,valid=True,maximum_dispatches=1,
        contract_sha256=c.sha(OUT/'contract.json'),approval_receipt_sha256=c.sha(OUT/'approval_receipt.json')))
    print('SEALED',head, c.sha(OUT/'contract.json'))


def check_binding(contract):
    c.require(contract['scope']==SCOPE and contract['targets']==TARGETS, 'EXACT_ANALYSIS_SCOPE')
    c.require(contract['bootstrap']['seed']==SEED and contract['bootstrap']['repetitions_per_n']==B,
              'EXACT_BOOTSTRAP_SCOPE')
    for names in ('source_hashes','protected'):
        for name,sha in contract[names].items():
            c.require(c.sha(ROOT/name)==sha, 'FROZEN_BYTES:'+name)
    c.require(str(Path(sys.executable).resolve())==contract['runtime']['python']
              and sys.version==contract['runtime']['version'], 'PYTHON_IDENTITY')
    for name,sha in contract['runtime']['files'].items():
        c.require(c.sha(name)==sha, 'RUNTIME_BYTES:'+name)
    grant=c.read(OUT/'grant.json'); approval=c.read(OUT/'approval_receipt.json')
    c.require(grant['valid'] and grant['scope']==SCOPE and grant['maximum_dispatches']==1
        and grant['contract_sha256']==c.sha(OUT/'contract.json')==approval['contract_sha256']
        and grant['approval_receipt_sha256']==c.sha(OUT/'approval_receipt.json'), 'BOUND_ONE_SHOT_GRANT')


def restore_pairs(rows, ep, compiler):
    rows=sorted(rows,key=lambda r:r['replica'])
    c.require([r['replica'] for r in rows]==list(range(ep['n'])), 'EXACT_REPLICA_COVERAGE')
    values=[]
    for row in rows:
        c.require(row['cell_id']==ep['cell']['id'] and row['cohort']=='INDEPENDENT_H4_CONFIRMATION'
            and row['stratum'] in STRATA and set(row['axes'])==set(c.AXES), 'PAIRED_SCHEMA')
        for a in c.AXES:
            v=row['axes'][a]
            c.require(v['axis']==a and v['compiler_hash']==compiler and type(v['RZ']) is int
                      and v['RZ']>0 and len(v['fingerprint'])==64, 'SAVED_COMPILED_PAIR')
        cost=paired.paired_cost({a:row['axes'][a]['RZ'] for a in c.AXES},ep['saved_shots'])
        c.require(type(cost) is int and cost>0, 'INTEGER_PAIRED_COST')
        values.append(cost)
    stats=c.evaluate_pairs(rows,ep['saved_shots'],ep['PF_G_exact'],ep['expected_direction'])
    for key in ('G_exact','sample_variance_exact','point_direction','status'):
        c.require(stats[key]==ep['result'][key], 'REPRODUCE_SAVED_'+key)
    for key in ('sample_SD','paired_SE'):
        c.require(stats[key]==ep['result'][key], 'REPRODUCE_SAVED_'+key)
    c.require(dict(Counter(r['stratum'] for r in rows))=={s:n for s,n in ep['strata'].items() if n},
              'SAVED_STRATA_COUNTS')
    return rows,values,stats


def prefixes(rows,values,ep):
    out=[]; total=squares=rare=0
    for n,(row,value) in enumerate(zip(rows,values),1):
        total+=value; squares+=value*value; rare+=row['stratum']!='S0_ONLY'
        variance=Fraction(n*squares-total*total,n*(n-1)) if n>1 else None
        out.append(dict(epsilon=ep['epsilon'],cell_id=row['cell_id'],n=n,replica=row['replica'],
            stratum=row['stratum'],cosine_RZ=row['axes']['cosine']['RZ'],sine_RZ=row['axes']['sine']['RZ'],
            trajectory_G_exact=value,cumulative_sum_exact=total,cumulative_mean_exact=str(Fraction(total,n)),
            cumulative_mean=total/n,cumulative_SE=math.sqrt(variance/n) if n>1 else None,
            cumulative_rare_trajectories=rare,PR_below_PF=total<Fraction(ep['PF_G_exact'])*n))
    return out


def bootstrap(values,strata,n,endpoint_index,pf,sample_variance,np,replicates=B):
    c.require(n>=2 and max(values)*n < 2**63, 'INT64_EXACT_SUM_RANGE')
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED,endpoint_index,n])))
    indices=rng.integers(0,len(values),size=(replicates,n),dtype=np.int64)
    arr=np.asarray(values,dtype=np.int64)
    draws=arr[indices]
    totals=draws.sum(axis=1,dtype=np.int64)
    means=totals.astype(np.float64)/n
    delta=(draws-int(min(values))).astype(np.float64)
    variance=np.maximum(0.,(np.square(delta).sum(axis=1)-np.square(delta.sum(axis=1))/n)/(n-1))
    within=np.sqrt(variance/n)
    s2=np.asarray([s=='S2_SINGLE' for s in strata],dtype=np.int64)[indices].sum(axis=1)
    s4=np.asarray([s=='S4_PRESENT' for s in strata],dtype=np.int64)[indices].sum(axis=1)
    # PF is an integer in these two frozen endpoints; comparisons use exact sums.
    c.require(Fraction(pf).denominator==1, 'FIXED_INTEGER_PF')
    boundary=int(pf)*n
    wins=int(np.count_nonzero(totals<boundary)); ties=int(np.count_nonzero(totals==boundary))
    qs=np.quantile(means,[.025,.05,.25,.5,.75,.95,.975],method='linear')
    row=dict(n=n,B=replicates,bootstrap_mean=float(means.mean()),bootstrap_SE=float(means.std(ddof=1)),
        full_sample_plugin_SE=math.sqrt(sample_variance/n),
        conditional_empirical_SE=math.sqrt(sample_variance*(len(values)-1)/len(values)/n),
        q025=float(qs[0]),q05=float(qs[1]),q25=float(qs[2]),q50=float(qs[3]),q75=float(qs[4]),
        q95=float(qs[5]),q975=float(qs[6]),min=float(means.min()),max=float(means.max()),
        PR_below_PF_count=wins,PR_equal_PF_count=ties,PR_above_PF_count=replicates-wins-ties,
        PR_below_PF_fraction=wins/replicates,win_fraction_MC_SE=math.sqrt(wins/replicates*(1-wins/replicates)/replicates),
        within_resample_SE_mean=float(within.mean()),within_resample_SE_median=float(np.median(within)),
        no_observed_S2_fraction=float(np.mean(s2==0)),
        conditional_no_S2_probability=(1-strata.count('S2_SINGLE')/len(values))**n,
        corr_mean_cost_S2_fraction=float(np.corrcoef(means,s2)[0,1]) if np.std(s2)>0 and np.std(means)>0 else None,
        indices_sha256=c.hashlib.sha256(indices.tobytes()).hexdigest())
    groups=[]
    for k in np.unique(s2):
        mask=s2==k
        groups.append(dict(n=n,S2_trajectory_draws=int(k),replicates=int(mask.sum()),
            mean_G=float(means[mask].mean()),min_G=float(means[mask].min()),max_G=float(means[mask].max()),
            PR_below_PF_fraction=float(np.mean(totals[mask]<boundary))))
    samples=[dict(n=n,b=b,sum_G_exact=int(totals[b]),mean_G=float(means[b]),
        within_sample_SE=float(within[b]),S2_draws=int(s2[b]),S4_draws=int(s4[b])) for b in range(replicates)]
    return row,samples,groups


def save_gzip_csv(path,rows):
    f=io.StringIO(newline=''); writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    path.write_bytes(gzip.compress(f.getvalue().encode(),mtime=0))


def render(results,checkpoints,all_prefixes,samples,output):
    np,plt=numerical_runtime()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'svg.hashsalt':'h4-mc-sample-size-20261019'})
    color=figures.COLORS['B2']
    def save(fig,name):
        fig.text(.01,.01,'H4 1 A | STO-3G | tol-only DF10 | T=0.8 | P=0 | fixed saved shots\n'
            'Bootstrap: conditional on saved pairs, 2000/n; percentile band is NOT a population confidence interval.',fontsize=7)
        fig.tight_layout(rect=(0,.075,1,.94))
        for ext in ('png','svg','pdf'):
            md={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {}
            fig.savefig(output/(name+'.'+ext),dpi=180,bbox_inches='tight',metadata=md)
        plt.close(fig)
    for i,ep in enumerate(results):
        rows=[x for x in checkpoints if x['cell_id']==ep['cell_id']]
        pre=[x for x in all_prefixes if x['cell_id']==ep['cell_id'] and x['n']>=4]
        ns=np.array([x['n'] for x in rows]); mean=float(Fraction(ep['full_sample']['G_exact']))
        lo=np.array([x['q025'] for x in rows]); hi=np.array([x['q975'] for x in rows])
        med=np.array([x['q50'] for x in rows]); pf=float(Fraction(ep['PF_G_exact']))
        fig,axs=plt.subplots(2,2,figsize=(11,7))
        a=axs[0,0]; a.fill_between(ns,lo,hi,color=color,alpha=.2,label='Bootstrap 2.5-97.5 percentiles')
        a.plot(ns,med,'o-',color=color,label='Bootstrap median')
        a.plot([r['n'] for r in pre],[r['cumulative_mean'] for r in pre],color='#009e73',lw=1,label='Cumulative, replica order')
        a.axhline(mean,color='#777777',ls=':',label='Full saved PR mean')
        a.axhline(pf,color='#d55e00',ls='--',label='Fixed second-order PF')
        a.set(ylabel='Shot-weighted mean cost G_RZ',title='a  Absolute cost and fixed PF');a.legend(fontsize=6)
        a=axs[0,1];a.fill_between(ns,100*(lo/mean-1),100*(hi/mean-1),color=color,alpha=.2)
        a.plot(ns,100*(med/mean-1),'o-',color=color)
        a.plot([r['n'] for r in pre],[100*(r['cumulative_mean']/mean-1) for r in pre],color='#009e73',lw=1)
        a.axhline(0,color='#777777',ls=':')
        a.set(ylabel='100 (PR mean / full saved mean - 1)',title='b  Zoom: saved-PR fluctuations (%)')
        a=axs[1,0];a.loglog(ns,[r['bootstrap_SE'] for r in rows],'o-',color=color,label='SD of bootstrap means')
        a.loglog(ns,[r['conditional_empirical_SE'] for r in rows],'--',color='#777777',label='Empirical SD / sqrt(n)')
        a.set(ylabel='SE of mean cost',title='c  Conditional spread');a.legend(fontsize=7)
        a=axs[1,1];a.plot(ns,[r['PR_below_PF_fraction'] for r in rows],'o-',color=color,label='PR < fixed PF')
        a.plot(ns,[r['no_observed_S2_fraction'] for r in rows],'s--',color='#009e73',label='No observed S2 drawn')
        a.set(ylim=(-.04,1.04),ylabel='Fraction of bootstrap replicates',title='d  Cost ranking and S2 absence');a.legend(fontsize=7)
        for a in axs.flat:a.set_xscale('log',base=2);a.set_xlabel('Number of paired trajectories n');a.grid(alpha=.18)
        fig.suptitle(f"H4 sample-size dependence: epsilon={ep['epsilon']} | N={ep['N']}")
        save(fig,'h4_mc_convergence_'+('eps005' if i==0 else 'eps00005'))
    fig,axs=plt.subplots(2,4,figsize=(13,6))
    for i,ep in enumerate(results):
        full=float(Fraction(ep['full_sample']['G_exact'])); pf=float(Fraction(ep['PF_G_exact']))
        for a,n in zip(axs[i], [4,16,64,ep['N']]):
            ys=[r['mean_G']/full for r in samples if r['cell_id']==ep['cell_id'] and r['n']==n]
            a.hist(ys,bins=35,color=color,alpha=.75)
            # Show PF only within the distribution's vicinity, otherwise annotate it.
            if min(ys)-.03 <=pf/full<=max(ys)+.03:a.axvline(pf/full,color='#d55e00',ls='--',label='PF');a.legend(fontsize=6)
            else:a.text(.98,.9,f'PF/full PR={pf/full:.3f}',ha='right',transform=a.transAxes,fontsize=7)
            a.axvline(1,color='#777777',ls=':');a.set(title=f"epsilon={ep['epsilon']}, n={n}",xlabel='Bootstrap mean / full saved PR',ylabel='Replicates')
    fig.suptitle('Empirical distributions of paired mean costs; no population coverage claim')
    save(fig,'h4_mc_bootstrap_distributions')
    fig,axs=plt.subplots(1,2,figsize=(10,4.7))
    for a,ep in zip(axs,results):
        full=float(Fraction(ep['full_sample']['G_exact']))
        for n in (4,32,ep['N']):
            ss=[r for r in samples if r['cell_id']==ep['cell_id'] and r['n']==n]
            a.scatter([r['S2_draws']/n for r in ss],[r['mean_G']/full for r in ss],s=8,alpha=.22,label=f'n={n}')
        a.set(title=f"epsilon={ep['epsilon']}",xlabel='Fraction of drawn trajectories labelled S2_SINGLE',ylabel='Bootstrap mean / full saved PR')
        a.legend(fontsize=7);a.grid(alpha=.18)
    fig.suptitle('Observed S2 contribution in saved-pair resamples; unseen strata remain unknown')
    save(fig,'h4_mc_observed_S2_cost')


def consume_once(contract_sha256,private=PRIVATE):
    private.mkdir(parents=True,exist_ok=True)
    # Atomic, nonrefundable one-shot consumption. No retry path and no removal.
    with (private/'ONE_SHOT.json').open('x') as f:
        f.write(c.encoded(dict(scope=SCOPE,contract_sha256=contract_sha256,started_at_UTC=now())).decode())


def run():
    numerical_runtime()
    contract=c.read(OUT/'contract.json');check_binding(contract)
    consume_once(c.sha(OUT/'contract.json'))
    c.save(OUT/'calls_ledger.json',dict(status='RESERVED_NONREFUNDABLE',calls=contract['calls'],contract_sha256=c.sha(OUT/'contract.json')))
    started=time.monotonic()
    try:
        np,_=numerical_runtime()
        old=c.read(ROOT/OLD/'result.json')
        projection=c.read(ROOT/OLD/'comparison_projection.json')['rows']
        rows=c.json.loads(gzip.decompress((ROOT/OLD/'paired_costs_projection.json.gz').read_bytes()))['rows']
        c.require(len(rows)==1186 and len(old['comparisons'])==2 and old['status']=='COMPLETE_MANDATORY_STOP','FIXED_COMPLETED_P')
        c.require({r['cell_id'] for r in rows}=={t['cell_id'] for t in TARGETS},'ONLY_FIXED_COHORTS')
        endpoints=[]; checkpoints=[]; all_prefixes=[]; samples=[]; event_rows=[]; groups=[]
        for i,target in enumerate(TARGETS):
            ep=next(x for x in old['comparisons'] if x['cell']['id']==target['cell_id'])
            c.require(ep['n']==target['N'] and ep['epsilon']==target['epsilon'] and ep['PF_id']==target['PF_id'],'EXACT_FROZEN_ENDPOINT')
            selected,values,stats=restore_pairs([r for r in rows if r['cell_id']==target['cell_id']],ep,old['binding']['compiler'])
            prior=next(x for x in projection if x['PR_cell_id']==target['cell_id'])
            c.require(prior['PR_G_exact']==stats['G_exact'] and prior['PR_paired_SE']==stats['paired_SE']
                      and prior['PF_G_exact']==ep['PF_G_exact'],'SAVED_SECOND_PROJECTION')
            pre=prefixes(selected,values,ep);all_prefixes+=pre
            strata=[r['stratum'] for r in selected]
            endpoint=dict(target,PF_G_exact=ep['PF_G_exact'],saved_shots=ep['saved_shots'],
                full_sample=stats,full_sample_reproduction='EXACT_MEAN_VARIANCE_SE_MATCH',
                strata=ep['strata'],observed_min_G_exact=min(values),observed_max_G_exact=max(values),
                all_saved_pairs_below_PF=max(values)<Fraction(ep['PF_G_exact']),
                unseen_S2_multi_mean=None,unseen_S4_mean=None,S4_support='OUTSIDE_K2_SUPPORT' if i==0 else 'UNOBSERVED_SUPPORTED_STRATUM')
            for s in STRATA:
                subset=[v for v,r in zip(values,selected) if r['stratum']==s]
                event_rows.append(dict(epsilon=ep['epsilon'],cell_id=target['cell_id'],stratum=s,count=len(subset),
                    observed_mean_G_exact=str(Fraction(sum(subset),len(subset))) if subset else None,
                    observed_min_G=min(subset) if subset else None,observed_max_G=max(subset) if subset else None,
                    empirical_cost_share=sum(subset)/sum(values) if subset else None,
                    population_mean_certified=False))
            for n in target['ns']:
                summary,ss,gg=bootstrap(values,strata,n,i,ep['PF_G_exact'],Fraction(stats['sample_variance_exact']),np)
                common=dict(epsilon=ep['epsilon'],cell_id=target['cell_id'])
                summary.update(common,PF_G_exact=ep['PF_G_exact'],cumulative_mean=pre[n-1]['cumulative_mean'],
                    cumulative_mean_exact=pre[n-1]['cumulative_mean_exact'],cumulative_SE=pre[n-1]['cumulative_SE'],
                    cumulative_rare_trajectories=pre[n-1]['cumulative_rare_trajectories'])
                checkpoints.append(summary);samples += [dict(common,**r) for r in ss];groups += [dict(common,**r) for r in gg]
            endpoints.append(endpoint)
        result=dict(scope=SCOPE,SOURCE=contract['SOURCE'],base_P_commit=BASE,binding=old['binding'],
            endpoints=endpoints,calls=contract['calls'],bootstrap_seed=SEED,repetitions_per_n=B,
            condition='fixed empirical distribution of saved independent P trajectories; fixed PF and shots',
            population_mean_certified=False,rigorous_confidence_intervals=False,
            unseen_rare_support_recovered=False,new_PF_selection=False,next_stage_authorized=False)
        c.save(OUT/'result.json',result)
        for name,table in [('bootstrap_summary.csv',checkpoints),('trajectory_and_cumulative.csv',all_prefixes),
                           ('observed_strata.csv',event_rows),('bootstrap_S2_groups.csv',groups)]:
            figures.export_csv(OUT/name,table)
        save_gzip_csv(OUT/'bootstrap_samples.csv.gz',samples)
        # Numeric results persist before rendering. Plotting never resamples.
        render(endpoints,checkpoints,all_prefixes,samples,OUT)
        check_binding(contract)
        c.save(OUT/'execution_receipt.json',dict(scope=SCOPE,status='NUMERICS_AND_FIGURES_COMPLETE',
            SOURCE=contract['SOURCE'],contract_sha256=c.sha(OUT/'contract.json'),
            dispatches=1,automatic_retry=0,wall_seconds=time.monotonic()-started,
            vectorized=True,per_group_arrays='2000 by n, maximum n=1108; algorithm size, not resource cap',
            affinity=sorted(os.sched_getaffinity(0)),calls=contract['calls'],finished_at_UTC=now(),
            source_and_input_and_runtime_hashes_unchanged=True,separate_AI_reviewer_used=False))
        manifest={p.name:c.sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
        c.save(OUT/'output_manifest.json',dict(scope=SCOPE,files=manifest))
        audit()
        c.save(OUT/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',native_exit_code=0,
            mandatory_STOP=True,next_stage_authorized=False))
    except BaseException as exc:
        c.save(OUT/'terminal.json',dict(status='STOP_FAILED_NO_AUTOMATIC_RETRY',error=repr(exc),
                                      mandatory_STOP=True,next_stage_authorized=False))
        raise
    print('COMPLETE_MANDATORY_STOP',f'{time.monotonic()-started:.3f}s')


def audit():
    contract=c.read(OUT/'contract.json');check_binding(contract)
    manifest=c.read(OUT/'output_manifest.json')
    for name,sha in manifest['files'].items():c.require(c.sha(OUT/name)==sha,'SAVED_OUTPUT_BYTES:'+name)
    result=c.read(OUT/'result.json');receipt=c.read(OUT/'execution_receipt.json')
    c.require(result['calls']==contract['calls']==receipt['calls'] and receipt['dispatches']==1,'ONE_DISPATCH_CALLS')
    def table(name):
        with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))
    summaries=table('bootstrap_summary.csv');prefix=table('trajectory_and_cumulative.csv')
    with gzip.open(OUT/'bootstrap_samples.csv.gz','rt',newline='') as f:samples=list(csv.DictReader(f))
    c.require(len(summaries)==18 and len(samples)==36000 and len(prefix)==1186,'EXACT_OUTPUT_DIMENSIONS')
    for ep in result['endpoints']:
        ps=[r for r in prefix if r['cell_id']==ep['cell_id']]
        c.require([int(r['n']) for r in ps]==list(range(1,ep['N']+1)),'CUMULATIVE_COVERAGE')
        running=0
        for r in ps:
            cost=sum(ep['saved_shots'][a]*int(r[a+'_RZ']) for a in c.AXES)
            running+=cost
            c.require(cost==int(r['trajectory_G_exact']) and running==int(r['cumulative_sum_exact'])
                and Fraction(r['cumulative_mean_exact'])==Fraction(running,int(r['n'])),'SAVED_PAIRED_PREFIX_IDENTITY')
        c.require(Fraction(ps[-1]['cumulative_mean_exact'])==Fraction(ep['full_sample']['G_exact'])
            and float(ps[-1]['cumulative_SE'])==ep['full_sample']['paired_SE'],'FINAL_FULL_SAMPLE_IDENTITY')
        ss=[r for r in summaries if r['cell_id']==ep['cell_id']]
        c.require([int(r['n']) for r in ss]==ep['ns'],'FIXED_N_GRID')
        for r in ss:
            n=int(r['n']);bs=[x for x in samples if x['cell_id']==ep['cell_id'] and int(x['n'])==n]
            c.require([int(x['b']) for x in bs]==list(range(B)),'FIXED_BOOTSTRAP_REPLICATES')
            wins=ties=no_s2=0;means=[]
            for x in bs:
                total=int(x['sum_G_exact']);mean=float(x['mean_G']);means.append(mean)
                c.require(mean==total/n and n*ep['observed_min_G_exact']<=total<=n*ep['observed_max_G_exact'],
                          'BOOTSTRAP_EXACT_SUM_AND_EMPIRICAL_SUPPORT')
                c.require(0<=int(x['S2_draws'])<=n and int(x['S4_draws'])==0 and float(x['within_sample_SE'])>=0,'EVENT_SE_SCHEMA')
                wins+=total<int(ep['PF_G_exact'])*n;ties+=total==int(ep['PF_G_exact'])*n;no_s2+=int(x['S2_draws'])==0
            c.require(wins==int(r['PR_below_PF_count']) and ties==int(r['PR_equal_PF_count'])
                and wins/B==float(r['PR_below_PF_fraction']) and no_s2/B==float(r['no_observed_S2_fraction']),
                'EXACT_RANKING_EVENT_COUNTS')
            # Independent scalar audit of saved draws; no seed consumption/resampling.
            import statistics
            c.require(math.isclose(statistics.stdev(means),float(r['bootstrap_SE']),rel_tol=2e-13),'BOOTSTRAP_SD_AUDIT')
            ordered=sorted(means)
            for q,key in [(0.025,'q025'),(.05,'q05'),(.25,'q25'),(.5,'q50'),(.75,'q75'),(.95,'q95'),(.975,'q975')]:
                j=(B-1)*q;low=math.floor(j);high=math.ceil(j)
                value=ordered[low]+(ordered[high]-ordered[low])*(j-low)
                c.require(math.isclose(value,float(r[key]),rel_tol=2e-15),'QUANTILE_AUDIT')
    c.require(len(list(OUT.glob('*.png')))==4 and len(list(OUT.glob('*.svg')))==4
              and len(list(OUT.glob('*.pdf')))==4,'FOUR_FIGURES_THREE_FORMATS')
    from PIL import Image
    for path in OUT.glob('*.png'):
        with Image.open(path) as im:im.verify()
    c.save(OUT/'numerical_audit.json',dict(status='PASS',scope=SCOPE,full_sample_exact_matches=2,
        paired_prefix_identities=1186,bootstrap_groups=18,saved_bootstrap_replicates_audited=36000,
        exact_ranking_and_event_counts=True,scalar_SD_and_quantiles_checked=True,
        original_P_SOURCE_inputs_and_runtime_preserved=True,figures_validated=12,
        resampling_during_audit=0,separate_AI_reviewer_used=False,mandatory_STOP=True))
    print('NUMERICAL_AUDIT_PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['seal','run','audit'])
    action=parser.parse_args().action
    {'seal':seal,'run':run,'audit':audit}[action]()
