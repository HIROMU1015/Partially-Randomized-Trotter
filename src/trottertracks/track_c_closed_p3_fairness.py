"""Offer the RA adapter's identical fused fallbacks to canonical trajectories.

No new samples, D/PF choices, labels, or physics. Existing raw paired choices
are replayed; original files/policy/results remain intact. Alpha is frozen from
training before the stronger canonical holdout is evaluated.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip
import json
import math
import os
import resource
import time
import numpy as np
from . import track_c_closed_p3_resources as r
from .track_c_closed_p3 import *
from .track_c_compiler_a import compile_reference
from .track_c_d_resources import metrics
from .track_c_bottleneck import allocate
from trotterlib.rte import _make_event


def enhance(path):
    target=OUT/'fairness'/path.parent.name/path.name
    if target.exists():return str(target)
    with gzip.open(path,'rt') as f:x=json.load(f)
    prep=common()['preps'][3];d=r.canonical_tables(x['t'],x['n'])[0]
    events=[_make_event(c['indices'],prep.rte_preparation.symbolic_tail.components,d,1 if c['order']==2 else 0) for c in x['canonical_choices']]
    start=time.monotonic();vectors={'legacy':x['canonical']}
    for basis in ('full','singletons','joint'):
        q,_=build_events(events,'network',basis);vectors[basis+'_network']=metrics(compile_reference(q))
    best=min(vectors,key=lambda k:tuple(vectors[k]));assert vectors[best][0]<=x['canonical'][0]
    result=dict(source_commit=provenance()['source_commit'],input_raw=str(path.relative_to(OUT)),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        canonical_strong=vectors[best],canonical_original=x['canonical'],variants=vectors,choice=best,wall_seconds=time.monotonic()-start,
        max_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    target.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(target,'wt') as f:json.dump(result,f,separators=(',',':'))
    return str(target)


def data(phase):
    rs=r.load_prices(phase)
    for strata in rs.values():
        for rows in strata.values():
            for row in rows:
                p=OUT/'fairness'/phase/(cellkey(row['t'],row['n'])+f'_{int(row["rare"])}_{row["j"]}.json.gz')
                with gzip.open(p,'rt') as f:new=json.load(f)
                row['canonical']=new['canonical_strong']
    return rs


def run_prices(phase):
    r.setup()
    from .track_c_bottleneck_basis import configuration
    configuration()
    paths=sorted((OUT/phase).glob('*.json.gz'),key=lambda p:-int(p.name.split('_')[1]))
    workers=min(len(paths),len(os.sched_getaffinity(0)));start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed(pool.submit(enhance,p) for p in paths):f.result()
    save_json(OUT/('FAIR_'+phase.upper()+'_RUN.json'),dict(**provenance(),samples=len(paths),workers=workers,wall_seconds=time.monotonic()-start,
        scope='No new draw; give unchanged canonical the same full/single/joint fused fallbacks as Closed P3.'))
    print('FAIR_'+phase.upper()+'_COMPLETE',len(paths),flush=True)


def freeze():
    if (OUT/'FAIR_POLICY.json').exists():raise RuntimeError('Preserve frozen fair policy')
    rs=data('training');arms={}
    for arm in ('canonical','closed'):
        pp=r.points(rs,arm);alpha,ns=allocate(pp,0)
        arms[arm]=dict(alpha=alpha.tolist(),trials=ns.tolist(),training=r.account(pp,ns),qualifications=[p['qualification'] for p in pp])
    save_json(OUT/'FAIR_POLICY.json',dict(**provenance(),arms=arms,rule='Identical fused basis fallbacks for both arms; same fixed D/h/resolution; alpha from training only, before fair holdout evaluation.'))
    print('FAIR_POLICY_FROZEN',flush=True)


def finish():
    frozen=json.loads((OUT/'FAIR_POLICY.json').read_text());rs=data('holdout');out=json.loads((OUT/'FINAL.json').read_text());results={};weights={}
    for arm in ('canonical','closed'):
        pp=r.points(rs,arm);p=frozen['arms'][arm];ns=np.array(p['trials'],dtype=np.int64);a=r.account(pp,ns);w=r.price_weights(ns);weights[arm]=w;cov=r.covariance(rs,[(arm,w)])
        a.update(alpha=p['alpha'],alpha_sum_exact=str(sum(F.from_float(x) for row in p['alpha'] for x in row)),qualifications=p['qualifications'],MC_SE=np.sqrt(np.diag(cov)).tolist(),covariance=cov.tolist());results[arm]=a
    delta=np.array(results['canonical']['totals'])-results['closed']['totals'];cov=r.covariance(rs,[('canonical',weights['canonical']),('closed',{k:-v for k,v in weights['closed'].items()})])
    out.update(**provenance(),arms=results,savings=delta.tolist(),saving_fraction=(delta/np.array(results['canonical']['totals'])).tolist(),paired_saving_MC_SE=np.sqrt(np.diag(cov)).tolist(),
        fair_comparison='Both receive full/single/joint efficient fused fallbacks; canonical additionally retains all original legacy variants. Training/holdout kept separate; original raw and preliminary result preserved.')
    # Recompute attribution under the fair policy; never inherit the preliminary
    # fixed-trial ablation with an obsolete canonical arm.
    same_ns=np.array(frozen['arms']['canonical']['trials'],dtype=np.int64);same=r.account(r.points(rs,'closed'),same_ns)
    fixed=r.covariance(rs,[('canonical',weights['canonical']),('closed',{k:-v for k,v in weights['canonical'].items()})])
    out['fixed_canonical_trials_savings']=(np.array(results['canonical']['totals'])-same['totals']).tolist();out['fixed_canonical_trials_savings_MC_SE']=np.sqrt(np.diag(fixed)).tolist()
    save_json(OUT/'FAIR_FINAL.json',out)
    import csv
    fields=['arm','m','trials_X','trials_Y','rho_lower','beta_stat_lower','RZ','CX','serial_depth_upper','size']
    with (OUT/'FAIR_ROUND_RESOURCES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for arm,a in results.items():
            for row in a['rounds']:w.writerow(dict(arm=arm,m=row['m'],trials_X=row['trials'][0],trials_Y=row['trials'][1],rho_lower=row['rho_lower'],beta_stat_lower=row['beta_stat_lower'],**dict(zip(fields[-4:],row['total']))))
    print('FAIR_FINAL',out['saving_fraction'],out['paired_saving_MC_SE'],flush=True)


def run():
    run_prices('training');freeze();run_prices('holdout');finish()

if __name__=='__main__':run()
