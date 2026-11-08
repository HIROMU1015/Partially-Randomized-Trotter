"""Finite-menu h/resolution/alpha selection after the fixed Closed P3 study.

Old runners are never started. Matching saved signal/native qualifications and
exact microstep certificates are reused; new occurrence prices have an isolated
namespace. A policy is frozen before fresh holdout draws.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from functools import lru_cache
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import tarfile
import time
import numpy as np
import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
BASE = '2e38af3b12200c48c4f1b793638d223baff24e76'
OUT = Path(os.environ.get('TRACK_C_CLOSED_P3_JOINT_OPT_OUTPUT',str(ROOT/'artifacts/track_c_closed_p3_joint_opt/2026-11-08')))
os.environ['TRACK_C_CLOSED_P3_OUTPUT'] = str(OUT)
from . import track_c_closed_p3 as cp
from . import track_c_closed_p3_resources as cr
from . import track_c_closed_p3_validation as cv
from . import track_c_diagonal_fusion as df
from .track_c_diagonal_resources import qualify as fusion_qualify
from .track_c_bottleneck_certificate import supplement
from .track_c_bottleneck import identity, allocate, rigorous_shots
from .track_c_h4 import common, save_json
from .track_c_d_resources import split, metrics, overhead, assemble
from .track_c_compiler_a import compile_reference
from trotterlib.rte import _make_event

OLD = ROOT/'artifacts/track_c_closed_p3/2026-11-08'
CACHE = ROOT/'.cache/closed_p3_joint_opt'


def provenance():
    return dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
                study_base=BASE, common_sha256=hashlib.sha256((cp.INPUT/'common.pkl').read_bytes()).hexdigest())


@lru_cache(None)
def archived(study, name):
    with tarfile.open(ROOT/'artifacts'/study/'2026-11-08/RAW_DATA.tar.gz') as t:
        return json.load(t.extractfile(name))


def prepare():
    """Replay published inputs, or preserve an already copied private cache."""
    from .track_c_closed_p3_inputs import load_model, model_identity
    cp.INPUT.mkdir(parents=True, exist_ok=True)
    if not (cp.INPUT/'common.pkl').exists():
        with (cp.INPUT/'common.pkl').open('wb') as f: pickle.dump(load_model(OLD),f)
    expected=json.loads((OLD/'INPUT_REPLAY_FINAL.json').read_text())['scientific_model_sha256']
    assert model_identity(common()) == expected
    files={'policy.json':ROOT/'artifacts/track_c_diagonal_fusion/2026-11-08/POLICY.json',
           'catalogue.json':ROOT/'artifacts/track_c_joint_isometry/2026-11-08/CATALOGUE_FINAL.json',
           'joint_certificate.json':ROOT/'artifacts/track_c_joint_isometry/2026-11-08/CERTIFICATE.json'}
    for name,path in files.items():
        if not (cp.INPUT/name).exists(): (cp.INPUT/name).write_bytes(path.read_bytes())
        assert (cp.INPUT/name).read_bytes() == path.read_bytes()
    OUT.mkdir(parents=True,exist_ok=True)
    return expected


def walk_signals(v):
    if isinstance(v,dict):
        if all(k in v for k in ('case','m','h','rate','recipe','steps','raw')): yield v
        else:
            for x in v.values(): yield from walk_signals(x)
    elif isinstance(v,list):
        for x in v: yield from walk_signals(x)


def inventory():
    target=OUT/'MENU.json'
    if target.exists(): return json.loads(target.read_text())
    model=prepare(); raw={}; sources=[]
    for study,names in [('track_c_d_internal',['SIGNALS.json','G_RESULTS.json','NONRESONANT.json','STRONG_FULL.json','REGISTERED_COMPARISON.json']),
                        ('track_c_bottleneck',['JOINT_CANDIDATES.json','OPTIMIZATION.json'])]:
        for name in names:
            data=archived(study,name)
            for x in walk_signals(data): raw[identity(x)]=x
            sources.append(dict(study=study,file=name))
    for x in cp.policy()['selected']: raw[identity(x['signal'])]=x['signal']
    qs={}
    for name in ('QUALIFICATION.json','QUALIFICATION_EXTENSION.json','REGISTERED_QUALIFICATION.json'):
        for q in archived('track_c_d_internal',name)['rows']: qs[identity(q)]=q
    for q in archived('track_c_bottleneck','JOINT_NATIVE_SOURCE.json')['rows']: qs[identity(q)]=q
    certificate=json.loads((cp.INPUT/'joint_certificate.json').read_text())
    fixed={identity(s['signal']):s['qualification'] for s in cp.policy()['selected']}
    rows=[]; rejected=[]
    for key,q in sorted(qs.items()):
        assert key in raw, key
        x=raw[key]; p=q['dominant_overlap']
        reason=None
        if not q['closed']: reason='saved native qualification does not close'
        elif p<=.5 or q['dominant_accumulated_bias_abs']+math.asin(min(1.,(1-p)/p))>=.4: reason='nonresonant dominant-branch diagnostic fails'
        if reason: rejected.append(dict(key=list(key),reason=reason)); continue
        full=x['case']['kind'].startswith('full_')
        if not full:
            assert all(r['time']>0 and r['r']>=1 for r in x['recipe'])
            assert x['RTE_events']==x['steps']*sum(r['r'] for r in x['recipe'])
        cq=q if full else supplement(q,x,certificate)
        cq=fusion_qualify(cq,x)
        if key in fixed: cq=fixed[key]  # exact previous reference qualification
        assert cq['closed']
        rows.append(dict(key=list(key),signal=x,qualification=cq,family='Full' if full else 'PR',
                         qualification_origin='matching saved source/native row; uniform joint and diagonal supplements',
                         saved_native_qualification=q))
    # An event depends on exact lambda*t/n, not outer h or number of D steps.
    oldstage=json.loads((OLD/'STAGE1.json').read_text()); records=[]; reused=0
    byx={r['certificate']['x_exact']:r for r in oldstage['cells']}
    cells=sorted({(r['time'],r['r']) for s in rows for r in s['signal']['recipe']})
    for t,n in cells:
        x=str(F.from_float(common()['preps'][3].exact_rte_lambda_r)*F.from_float(t)/n)
        if x in byx:
            a=byx[x]; records.append({**a,'t':t,'n':n,'reused_from_cell':[a['t'],a['n']],
                                     'reuse_condition':'same exact lambda*t/n, p, Taylor order, bits, eta/rho and angle source'}); reused+=1
        else:
            records.append(cv.cell_check((t,n)))
            byx[x]=records[-1]
    save_json(OUT/'STAGE1.json',dict(**provenance(),cells=records,passed=True,scientific_model_sha256=model,
        inherited_artificial_and_operator_binding=oldstage['h4_generator_matrix_binding_error'],reused_exact_microstep_certificates=reused))
    for s in rows:
        if s['family']=='PR': s['closed_qualification']=cr.qualify(s)
    result=dict(**provenance(),rows=rows,rejected=rejected,original_saved_signal_rows=11988,
                saved_native_conditions=len(qs),source_assets=sources,cells=cells,
                boundary='Initial menu is every saved qualified nonresonant condition. No unqualified signal is treated as a certificate. Same PR menu in both arms; all saved strong Full conditions included.')
    save_json(target,result); print('MENU',len(rows),'CELLS',len(cells),flush=True); return result


@lru_cache(None)
def menu(): return json.loads((OUT/'MENU.json').read_text())


def det_price(task):
    case,h=task; path=OUT/'det_prices'/f'{case["name"]}_{h}.json'
    if path.exists(): return json.loads(path.read_text())
    # Reuse byte-identical deterministic source prices if already published.
    with tarfile.open(ROOT/'artifacts/track_c_diagonal_fusion/2026-11-08/RAW_DATA.tar.gz') as t:
        name='det_prices/'+path.name
        if name in t.getnames():
            row=json.load(t.extractfile(name)); assert row['case']==case and row['h']==h
            save_json(path,row); return row
    df.OUT=OUT
    return df.det((case,h))


def price_d():
    tasks={(json.dumps(s['signal']['case'],sort_keys=True),s['signal']['h']) for s in menu()['rows']}
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed(pool.submit(det_price,(json.loads(c),h)) for c,h in tasks): f.result()
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'det_prices').glob('*.json'))]
    save_json(OUT/'DETERMINISTIC.json',dict(**provenance(),rows=rows,groups=len(rows),
        count_rule='One RZ-first lexicographic circuit per group. Same native reference/term/merge/network choices for PR and Full. D/D junction counted once.'))


_canonical_native=cr.canonical_native
def canonical_native(t,n,choices):
    q,vs,_=_canonical_native(t,n,choices); circuits={'legacy':q}
    prep=common()['preps'][3]; d=cr.canonical_tables(t,n)[0]
    ev=[_make_event(c['indices'],prep.rte_preparation.symbolic_tail.components,d,1 if c['order']==2 else 0) for c in choices]
    for basis in ('full','singletons','joint'):
        circuits[basis+'_network']=compile_reference(cp.build_events(ev,'network',basis)[0])
    values={k:metrics(q) for k,q in circuits.items()}; k=min(values,key=lambda k:tuple(values[k]))
    return circuits[k],dict(legacy_variants=vs,efficient=values),k


def price_one(task):
    phase,t,n,rare,j=task; cr.setup()
    target=OUT/phase/(cp.cellkey(t,n)+f'_{int(rare)}_{j}.json.gz')
    if target.exists(): return str(target)
    seed=int.from_bytes(hashlib.sha256(f'P3_joint_opt:{phase}:{t}:{n}:{rare}:{j}'.encode()).digest()[:8],'big')
    start=time.monotonic(); choices,events,compact,p=cr.paired_paths(t,n,rare,seed)
    a,av,ac=canonical_native(t,n,choices); b,bv,bc=cr.closed_native(t,n,events)
    row=dict(**provenance(),t=t,n=n,rare=rare,j=j,seed=seed,canonical=metrics(a),closed=metrics(b),
             stratum_probability=dict(canonical=p[0],closed=p[1]),canonical_variants=av,closed_variants=bv,
             canonical_choice=ac,closed_choice=bc,canonical_choices=choices,closed_choices=compact,wall_seconds=time.monotonic()-start)
    target.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(target,'wt') as f: json.dump(row,f,separators=(',',':'),allow_nan=False)
    return str(target)


def reuse_training(cells):
    """Previous training + holdout become training here, never new holdout."""
    wanted=set(map(tuple,cells)); count=0
    with tarfile.open(OLD/'RAW_DATA.tar.gz') as t:
        members=set(t.getnames())
        for phase,offset in [('training',0),('holdout',16)]:
            for m in t.getmembers():
                if not m.name.startswith(phase+'/') or not m.name.endswith('.json.gz'): continue
                with gzip.open(io.BytesIO(t.extractfile(m).read()),'rt') as f: row=json.load(f)
                if (row['t'],row['n']) not in wanted: continue
                fair='fairness/'+m.name
                assert fair in members
                with gzip.open(io.BytesIO(t.extractfile(fair).read()),'rt') as f: row['canonical']=json.load(f)['canonical_strong']
                row['reused_source']=dict(commit=BASE,member=m.name,old_role=phase,new_role='training')
                row['j']+=offset
                path=OUT/'training'/(cp.cellkey(row['t'],row['n'])+f'_{int(row["rare"])}_{row["j"]}.json.gz')
                if not path.exists():
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with gzip.open(path,'wt') as f: json.dump(row,f,separators=(',',':'))
                count+=1
    return count


def prices(phase,cells,samples=32,record_name=None):
    cr.setup(); from .track_c_bottleneck_basis import configuration
    configuration(); reused=reuse_training(cells) if phase=='training' else 0
    tasks=[(phase,t,n,rare,j) for t,n in sorted(cells,key=lambda x:x[1],reverse=True) for rare in (False,True) for j in range(samples[cp.cellkey(t,n)] if isinstance(samples,dict) else samples)
           if not (OUT/phase/(cp.cellkey(t,n)+f'_{int(rare)}_{j}.json.gz')).exists()]
    start=time.monotonic(); workers=min(len(tasks),len(os.sched_getaffinity(0)))
    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for f in as_completed(pool.submit(price_one,t) for t in tasks): f.result()
    save_json(OUT/((record_name or phase.upper()+'_RUN')+'.json'),dict(**provenance(),cells=cells,samples_per_stratum=samples,new_samples=len(tasks),reused_samples=reused,
        workers=workers,wall_seconds=time.monotonic()-start,pairing='same seed/underlying uniform stream per arm within cell and rare stratum; each arm retains its own stratum probability'))
    print(phase.upper(),'COMPLETE',len(tasks),'REUSED',reused,flush=True)


@lru_cache(None)
def dets(): return {(x['case']['name'],x['h']):x for x in json.loads((OUT/'DETERMINISTIC.json').read_text())['rows']}


def shrink(q,extra):
    if extra==0: return q
    mp.iv.prec=192; r=cp.iv(q['rho_lower']); e=cp.iv(extra); ratio=e/r
    if not q['closed'] or extra>=q['rho_lower']: return {**q,'closed':False}
    phase=cp.sg.up(mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2)))
    rho=cp.sg.lower(r-e); beta=cp.sg.lower(cp.iv(q['beta_stat_lower'])-phase)
    return {**q,'rho_lower':rho,'beta_stat_lower':beta,'D_aggregation_disk':extra,'closed':rho>0 and beta>0}


@lru_cache(None)
def wrapper_price(ld,t): return np.array([metrics(overhead(ld,t,a)) for a in ('X','Y')],float)


def point(s,rs,arm):
    x=s['signal']; d=dets()[(x['name'],x['h'])]; g={k:np.array(v,float) for k,v in d['groups'].items()}; n=x['steps']; full=s['family']=='Full'
    a,b,j=('prefix','suffix','middle') if full else ('first','last','junction')
    bulk=n*(g[a]+g[b]+(0 if full else g['inside'])); boundary=(n-1)*(g[j]-g[a]-g[b])
    R=np.zeros(4)
    for recipe in x['recipe']:
        for rows in rs[cp.cellkey(recipe['time'],recipe['r'])].values():
            R+=n*rows[0]['stratum_probability'][arm]*np.mean([r[arm] for r in rows],axis=0)
    q=s['closed_qualification'] if arm=='closed' else s['qualification']
    # d_fused reports an exact rational aggregation error for the chosen group.
    error=n*(d['errors'][a]+d['errors'][b])+(n-1)*d['errors'][j]
    if not full: error+=n*d['errors']['inside']
    extra=cp.sg.up(2*mp.iv.sqrt(cp.iv(2))*cp.iv(error)) if error else 0.
    q=shrink(q,extra); over=wrapper_price(12 if full else 3,x['t'])
    return dict(key=s['key'],signal=x,qualification=q,D_bulk=bulk,D_boundary=boundary,D=bulk+boundary,R=R,overhead=over,mean=bulk+boundary+R+over,arm=arm)


def score(p,alpha):
    q=p['qualification']
    return sum(rigorous_shots(q['rho_lower'],q['beta_stat_lower'],a)*p['mean'][b,0] for b,a in enumerate(alpha))


def frozen_record(points,alpha,ns,history=None):
    assert sum(F.from_float(float(a)) for a in np.asarray(alpha).flat)<=F(1,20)
    return dict(selected=[dict(key=p['key'],signal=p['signal'],qualification=p['qualification']) for p in points],
                alpha=np.asarray(alpha).tolist(),trials=np.asarray(ns).tolist(),training=cr.account(points,np.asarray(ns)),history=history or [])


def optimize(points,prior_alpha=None):
    """Reuse cost-sensitive alpha proposals, alternate with integer-shot choices.

    This is a finite-menu coordinate search, not a global optimum certificate.
    Both methods use identical starts and deterministic tie rules.
    """
    byround=[[p for p in points if p['signal']['m']==m and p['qualification']['closed']] for m in range(18)]
    assert all(byround)
    best=None; history=[]
    for startalpha in (1/720,.01):
        chosen=[min(ps,key=lambda p:(score(p,[startalpha]*2),p['key'])) for ps in byround]; seen=set()
        while True:
            keys=tuple(tuple(p['key']) for p in chosen)
            if keys in seen: break
            from .track_c_closed_p3_joint_opt_alpha import discrete_allocate
            seen.add(keys); alpha,ns=discrete_allocate(chosen,prior_alpha); value=cr.account(chosen,ns)['totals'][0]
            history.append(dict(start_alpha=startalpha,cost=value,keys=[list(k) for k in keys]))
            if best is None or value<best[0]: best=(value,chosen[:],alpha.copy(),ns.copy())
            chosen=[min(ps,key=lambda p:(score(p,alpha[m]),p['key'])) for m,ps in enumerate(byround)]
    _,chosen,alpha,ns=best
    return frozen_record(chosen,alpha,ns,history)


def keys_for_fixed(): return [list(identity(x['signal'])) for x in cp.policy()['selected']]


def by_keys(points,keys):
    lookup={tuple(p['key']):p for p in points}
    return [lookup[tuple(k)] for k in keys]


def with_alpha(points,alpha):
    ns=np.array([[rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],a) for a in row] for p,row in zip(points,alpha)])
    return frozen_record(points,alpha,ns)


def freeze(filename='POLICY.json'):
    from .track_c_closed_p3_joint_opt_alpha import discrete_allocate
    target=OUT/filename
    if target.exists(): raise RuntimeError('Preserve frozen policy')
    rs=cr.load_prices('training'); rows=menu()['rows']; previous=json.loads((OLD/'FAIR_POLICY.json').read_text()); methods={}; pts={}
    for name,arm,family in [('A_canonical','canonical','PR'),('B_closed','closed','PR'),('C_Full','canonical','Full')]:
        pts[name]=[point(s,rs,arm) for s in rows if s['family']==family]
        prior=previous['arms'][arm]['alpha'] if family=='PR' else cp.policy('Full')['alpha']
        methods[name]=dict(arm=arm,family=family,**optimize(pts[name],prior))
    fixed=keys_for_fixed(); ablations={}
    for short,name in [('canonical','A_canonical'),('closed','B_closed')]:
        pp=by_keys(pts[name],fixed); oldalpha=previous['arms'][short]['alpha']
        ablations['fixed_'+short+'_old_alpha']=dict(arm=short,**with_alpha(pp,oldalpha))
        aa,nn=discrete_allocate(pp,oldalpha); ablations['fixed_'+short+'_new_alpha']=dict(arm=short,**frozen_record(pp,aa,nn))
        selected=by_keys(pts[name],[p['key'] for p in methods[name]['selected']])
        ablations['optimized_'+short+'_old_alpha']=dict(arm=short,**with_alpha(selected,oldalpha))
    aa_menu=by_keys(pts['B_closed'],[p['key'] for p in methods['A_canonical']['selected']]); aa,nn=discrete_allocate(aa_menu,previous['arms']['closed']['alpha'])
    ablations['closed_at_canonical_selection']=dict(arm='closed',**frozen_record(aa_menu,aa,nn))
    save_json(target,dict(**provenance(),methods=methods,ablations=ablations,objective='total expected native RZ; CX/depth/size of the SAME RZ-selected circuits',
        optimizer='identical two-start finite-menu coordinate search; existing cost-sensitive alpha proposal and prior-alpha start, then directed inverse-shot local integer exchanges (<=64 funding trial additions per proposal); not a global optimality claim',
        selection='training only; prior holdout is explicitly training here; fresh namespace holdout is not read until this policy exists'))
    print('POLICY_FROZEN', {k:v['training']['totals'][0] for k,v in methods.items()},flush=True)


def weight_map(p):
    w={}
    for s,ns in zip(p['selected'],p['trials']):
        for r in s['signal']['recipe']:
            k=cp.cellkey(r['time'],r['r']); w[k]=w.get(k,0)+int(sum(ns))*s['signal']['steps']
    return w


def evaluate(p,rs):
    lookup={tuple(s['key']):s for s in menu()['rows']}
    pp=[point(lookup[tuple(s['key'])],rs,p['arm']) for s in p['selected']]; ns=np.array(p['trials'],dtype=np.int64)
    result=cr.account(pp,ns); w=weight_map(p); cov=cr.covariance(rs,[(p['arm'],w)])
    result.update(MC_SE=np.sqrt(np.maximum(0,np.diag(cov))).tolist(),covariance=cov.tolist(),alpha=p['alpha'],
                  alpha_sum_exact=str(sum(F.from_float(a) for row in p['alpha'] for a in row)),
                  selected=p['selected'],all_source_native_closed=all(x['qualification']['closed'] for x in pp))
    from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase
    phases=[math.atan2(s['signal']['raw']['imag'],s['signal']['raw']['real']) for s in p['selected']]
    dec=reconstruct_rpe_phase(phases,[1<<m for m in range(18)]); err=abs(-dec['final_phase_estimate']/.02-common()['energy'])
    branch=all(abs(r['base_phase_estimate']+.02*common()['energy'])<=.4/(1<<m)+1e-10 for m,r in enumerate(dec['rounds']))
    assert branch and err<=1.5936e-4 and result['all_source_native_closed']
    result.update(decoder=dec,mean_branch_pass=branch,mean_energy_error=err,energy_statistical_bound=.4/(.02*(1<<17)))
    return result


def compare(a,b,results,policies,rs):
    x=np.array(results[a]['totals']); y=np.array(results[b]['totals'])
    cov=cr.covariance(rs,[(policies[a]['arm'],weight_map(policies[a])),(policies[b]['arm'],{k:-v for k,v in weight_map(policies[b]).items()})])
    return dict(before=a,after=b,saving=(x-y).tolist(),saving_fraction=((x-y)/x).tolist(),paired_MC_SE=np.sqrt(np.maximum(0,np.diag(cov))).tolist())


def finish():
    frozen=json.loads(policy_path().read_text()); policies={**frozen['methods'],**frozen['ablations']}; rs=cr.load_prices('holdout')
    results={k:evaluate(p,rs) for k,p in policies.items()}
    pairs=[('A_canonical','B_closed'),('fixed_closed_old_alpha','B_closed'),('fixed_canonical_old_alpha','A_canonical'),
           ('fixed_closed_old_alpha','fixed_closed_new_alpha'),('fixed_closed_old_alpha','optimized_closed_old_alpha'),
           ('optimized_closed_old_alpha','B_closed'),('A_canonical','closed_at_canonical_selection'),('closed_at_canonical_selection','B_closed')]
    comparisons=[compare(a,b,results,policies,rs) for a,b in pairs]
    full=np.array(results['C_Full']['totals'])
    save_json(OUT/'FINAL.json',dict(**provenance(),methods=results,comparisons=comparisons,
        Full_ratios={k:(np.array(results[k]['totals'])/full).tolist() for k in ('A_canonical','B_closed','fixed_closed_old_alpha')},
        metrics=['RZ','CX','modular_serial_depth_upper','size'],quantum_backend_shots=0,
        guarantee='saved Hermitian H4/exact original-H ground/ideal native/fresh independent bits; 36-axis exact alpha sum <=.05; all selected source/native/weighted budgets close',
        limits='Price expectations and SE are independent-holdout estimates. No continuum Hamiltonian inclusion, hardware, state preparation, global menu optimum or T-cost claim.'))
    with (OUT/'ROUNDS.csv').open('w',newline='') as f:
        fields=['method','m','case','h','resolution','microsteps_per_D_step','D_steps','trials_X','trials_Y','alpha_X','alpha_Y','rho','beta','PF_phase_upper','log_range','RZ','CX','serial_depth_upper','size']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for name,v in results.items():
            for s,r,a in zip(v['selected'],v['rounds'],v['alpha']):
                x=s['signal']; q=s['qualification']
                writer.writerow(dict(method=name,m=x['m'],case=x['name'],h=x['h'],resolution=x['rate'],microsteps_per_D_step=sum(z['r'] for z in x['recipe']),D_steps=x['steps'],
                    trials_X=r['trials'][0],trials_Y=r['trials'][1],alpha_X=a[0],alpha_Y=a[1],rho=q['rho_lower'],beta=q['beta_stat_lower'],PF_phase_upper=q['PF_phase_upper'],
                    log_range=q.get('RA_log_Wmax_upper',x['log_normalizer']),**dict(zip(fields[-4:],r['total']))))
    print('FINAL',comparisons[:2],flush=True)


def policy_path(): return OUT/('POLICY_REFINED.json' if (OUT/'POLICY_REFINED.json').exists() else 'POLICY.json')


def run(stage):
    if stage=='inventory': inventory();price_d()
    elif stage=='pilot':
        from .track_c_closed_p3_joint_opt_checks import pilot
        pilot()
    elif stage=='correctness':
        from .track_c_closed_p3_joint_opt_checks import correctness
        correctness()
    elif stage=='selected-correctness':
        from .track_c_closed_p3_joint_opt_checks import selected_correctness
        selected_correctness()
    elif stage=='training': prices('training',menu()['cells']);freeze()
    elif stage=='refine':
        assert not (OUT/'holdout').exists(),'Refinement must precede any holdout'
        # Top m17 alternatives from the initial training uncertainty diagnostic.
        # No new h/rate, physics, sampler, compiler or logical condition is added.
        cells=[(.32,512),(.32,1024),(.16,512)]
        prices('training',cells,256,'TRAINING_REFINEMENT_RUN');freeze('POLICY_REFINED.json')
    elif stage in ('holdout','holdout-more'):
        f=json.loads(policy_path().read_text()); allp=list(f['methods'].values())+list(f['ablations'].values())
        cells=sorted({(r['time'],r['r']) for p in allp for s in p['selected'] for r in s['signal']['recipe']})
        assert json.loads((OUT/'SELECTED_CORRECTNESS.json').read_text())['passed']
        plan=OUT/'HOLDOUT_PLAN.json';digest=hashlib.sha256(policy_path().read_bytes()).hexdigest()
        if stage=='holdout':
            assert not plan.exists(),'Preserve registered holdout plan'
            save_json(plan,dict(**provenance(),policy_sha256=digest,critical_cells=[[.16,512],[.32,1024]],
                initial_samples_per_stratum=128,other_samples_per_stratum=32,
                optional_precision_extension=256,extension_rule='Only if needed to resolve paired RZ precision; never retune the frozen policy. STOP after this bounded extension.',
                compiler='original direct compiler A; validated memoization helper is NOT used because a speed benefit was not established'))
        assert json.loads(plan.read_text())['policy_sha256']==digest
        count=128 if stage=='holdout' else 256
        samples={cp.cellkey(t,n):(count if (t,n) in ((.16,512),(.32,1024)) else 32) for t,n in cells}
        prices('holdout',cells,samples,'HOLDOUT_RUN' if stage=='holdout' else 'HOLDOUT_EXTENSION_RUN');finish()
    elif stage=='finish': finish()
    else: raise ValueError(stage)
