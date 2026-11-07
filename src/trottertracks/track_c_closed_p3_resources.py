"""Paired occurrence prices, fixed-policy RPE accounting and independent holdout.

Old SOURCE/output is read-only. Physical quantum shots are not acquired.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import time
import numpy as np
import mpmath as mp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from .track_c_closed_p3 import *
from .track_c_closed_p3_validation import h4_operators
from .track_c_compiler_a import compile_reference, PROFILE
from .track_c_d_resources import metrics, overhead, assemble
from .track_c_h4_price import conditional_flags, wrapper
from .track_c_joint_resources import initial_state
from .track_c_bottleneck import allocate, rigorous_shots
from .track_c_bottleneck_runwise import build_singletons
from .track_c_joint_isometry import build_joint
from .track_c_diagonal_fusion import build as old_fused
from . import track_c_joint_isometry as legacy_joint
from trotterlib.rte import _make_event
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder


@lru_cache(None)
def setup():
    # Configure only the legacy catalogue reader; no legacy run/prepare function.
    legacy_joint.OUT=INPUT
    (INPUT/'CATALOGUE_FINAL.json').write_bytes((INPUT/'catalogue.json').read_bytes()) if not (INPUT/'CATALOGUE_FINAL.json').exists() else None
    return json.loads((OUT/'STAGE1.json').read_text())


@lru_cache(None)
def certificate(t,n):return next(r['certificate'] for r in setup()['cells'] if r['t']==t and r['n']==n)


@lru_cache(None)
def canonical_tables(t,n):
    from .track_c_h4_finite_bits import integer_cdf
    prep=common()['preps'][3];d=finite_rte_distribution(prep.exact_rte_lambda_r*t/n,2)
    return d,integer_cdf([s.probability for s in prep.rte_preparation.symbolic_tail.components],BITS)


def paired_paths(t,n,rare,seed):
    """Exact dyadic label/group draws; legacy float conditional first-rare map for price MC only.

    Rare-stratum weights use each arm's own law. This stratified price sampler is
    distinct from the unconditioned DigitalSampler used for the shot model.
    """
    g=generator(t,n);d,labelcdf=canonical_tables(t,n);q0=float(d.order_probabilities[1]);q1=float(1-g.group_law[0])
    flags=[conditional_flags([p]*n,rare,np.random.default_rng(seed)) for p in (q0,q1)]
    bits0=BitStream(seed);bits1=BitStream(seed);sampler=DigitalSampler(g)
    choices=[];ra=[];compact=[]
    group_counts=[int(p*(1<<BITS)) for p in g.group_law[1:]];total=sum(group_counts)
    prefix=np.cumsum(np.array(group_counts,dtype=object)).tolist()
    for f0,f1 in zip(*flags):
        labels=[bisect_right(labelcdf,bits0.bits(BITS)) for _ in range(3 if f0 else 1)]
        choices.append(dict(indices=labels,order=2 if f0 else 0))
        if f1:
            gi=1+bisect_right(prefix,bits1.rng.randrange(total));first=g.groups[gi].first
            word=(first,sampler.conditional_draw(first,bits1));child=sampler.conditional_draw(first,bits1)
        else:gi=0;word=();child=sampler.draw(sampler.root,bits1)
        e=g.event(gi,word,child);ra.append(e);compact.append([gi,*word,child])
    weights=[]
    for p in (q0,q1):
        p_rare=-math.expm1(n*math.log1p(-p));weights.append(p_rare if rare else 1-p_rare)
    return choices,ra,compact,weights


def canonical_native(t,n,choices):
    setup();prep=common()['preps'][3];d=canonical_tables(t,n)[0]
    ev=[_make_event(c['indices'],prep.rte_preparation.symbolic_tail.components,d,1 if c['order']==2 else 0) for c in choices]
    req=DFRTEEventSequenceCircuitRequest(events=tuple(ev),component_specs=prep.rte_preparation.component_specs,controlled=True,ancilla_qubit=8,
        tail_id=prep.rte_preparation.symbolic_tail.tail_id,tail_hash=prep.rte_preparation.symbolic_tail.tail_hash,occurrence_rte_steps=n)
    circuits={'full':QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry).build_sequence(req).circuit,
              'singletons':build_singletons(t,n,choices)[0],'joint':build_joint(t,n,choices)[0]}
    reference=min((compile_reference(q) for q in circuits.values()),key=lambda q:tuple(metrics(q)))
    variants={'reference':reference}
    for mode in ('term_parity','merge','network'):variants[mode]=compile_reference(old_fused(t,n,choices,mode)[0])
    selected=min(variants,key=lambda k:tuple(metrics(variants[k])))
    return variants[selected],{k:metrics(q) for k,q in variants.items()},selected


def closed_native(t,n,records):
    ev=application_events(t,n,records);variants={}
    # Generic reference controls are unnecessary: all fallbacks below retain the
    # original full basis but use already-qualified efficient diagonal controls.
    for basis,mode in [('full','network'),('singletons','network'),('joint','term_parity'),('joint','merge'),('joint','network')]:
        q,_=build_events(ev,mode,basis);variants[basis+'_'+mode]=compile_reference(q)
    selected=min(variants,key=lambda k:tuple(metrics(variants[k])))
    return variants[selected],{k:metrics(q) for k,q in variants.items()},selected


def price_chunk(task):
    phase,t,n,rare,start,count=task;setup();rows=[]
    for j in range(start,start+count):
        target=OUT/phase/(cellkey(t,n)+f'_{int(rare)}_{j}.json.gz')
        if target.exists():
            with gzip.open(target,'rt') as f:rows.append(json.load(f))
            continue
        seed=int.from_bytes(hashlib.sha256(f'ClosedP3:{phase}:{t}:{n}:{rare}:{j}'.encode()).digest()[:8],'big')
        wall=time.monotonic();cpu=time.process_time();choices,rs,compressed,p=paired_paths(t,n,rare,seed)
        sample_cpu=time.process_time()-cpu;clock=time.process_time()
        c,cv,cc=canonical_native(t,n,choices);r,rv,rc=closed_native(t,n,rs)
        row=dict(t=t,n=n,rare=rare,j=j,seed=seed,stratum_probability=dict(canonical=p[0],closed=p[1]),canonical=metrics(c),closed=metrics(r),
            canonical_variants=cv,closed_variants=rv,canonical_choice=cc,closed_choice=rc,canonical_choices=choices,closed_choices=compressed,
            sampler_cpu_seconds=sample_cpu,build_compile_cpu_seconds=time.process_time()-clock,wall_seconds=time.monotonic()-wall,
            max_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_commit=provenance()['source_commit'])
        target.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(target,'wt',encoding='utf-8') as f:json.dump(row,f,separators=(',',':'),allow_nan=False)
        rows.append(row)
    return [str(OUT/phase/(cellkey(t,n)+f'_{int(rare)}_{j}.json.gz')) for j in range(start,start+count)]


def load_prices(phase):
    rs={}
    for p in sorted((OUT/phase).glob('*.json.gz')):
        with gzip.open(p,'rt') as f:r=json.load(f)
        rs.setdefault(cellkey(r['t'],r['n']),{}).setdefault(str(r['rare']),[]).append(r)
    for strata in rs.values():
        for rows in strata.values():rows.sort(key=lambda r:r['j'])
    return rs


def price_run(phase,selected,samples):
    tasks=[(phase,t,n,rare,j,1) for t,n in sorted(selected,key=lambda x:x[1],reverse=True) for rare in (False,True) for j in range(samples)]
    setup();start=time.monotonic();workers=min(len(tasks),len(os.sched_getaffinity(0)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed(pool.submit(price_chunk,t) for t in tasks):f.result()
    save_json(OUT/(phase.upper()+'_RUN.json'),dict(**provenance(),cells=selected,samples_per_stratum=samples,workers=workers,wall_seconds=time.monotonic()-start,compiler=PROFILE,
        pairing='same seed and corresponding underlying uniforms within each cell/stratum; independent cell/stratum/replicate seeds',
        price_scope='whole occurrence direct native compile; method-specific rare weights; CPU is separate; depth is occurrence DAG'))
    print(phase.upper()+'_COMPLETE',len(tasks),flush=True)


def qualify(selected):
    """Retain the existing uniform logical/basis/native disks conservatively.

    A m3 RA event still has <=3 applications; the existing all-context Gaussian
    and full/single/joint fallback bounds therefore apply. New central circuits
    contain native RZ/CX and unchanged CZ/P. RZ parameters pass through A, half
    angles are exact binary scaling, and the old 2u*N diagonal-sum bound still
    applies. New root and atan-to-float mean bias is added separately. Old
    canonical angle/CDF allowances are retained as conservative slack, not used
    to certify the new angles or counted as a second proposal bias.
    """
    mp.iv.prec=192;x=selected['signal'];old=selected['qualification'];oldlog=iv(0);newlog=iv(0);ebias=iv(0);growth=iv(0)
    for recipe in x['recipe']:
        t,n=recipe['time'],recipe['r'];c=certificate(t,n);z=iv(F(c['x_exact']));events=x['steps']*n
        b=mp.iv.sqrt(1+z*z)+z*z/2*mp.iv.sqrt(1+(z/3)**2)
        oldlog+=events*mp.iv.log(b);newlog+=events*iv(c['log_Wmax'][1])
        weight_round=iv(2)**-53*iv(c['B'][1])*iv(1+RHO)
        step_error=iv(c['corrected_mean_bias_step_upper'])+weight_round
        ebias+=events*step_error;growth+=events*(mp.iv.exp(z)*z**4/24+step_error)
        newlog+=events*mp.iv.log(1+iv(2)**-53)
    final_round=iv(2)**-53+iv(F(1,10**24))
    newlog+=mp.iv.log(1+final_round)
    scale=mp.iv.exp(oldlog-newlog);extra=3*ebias*mp.iv.exp(growth-newlog)+3*final_round
    base=iv(old['rho_lower'])*scale;rho=sg.lower(base-extra);ratio=extra/base;phase=sg.up(mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2)));beta=sg.lower(iv(old['beta_stat_lower'])-phase)
    assert rho>0 and beta>0 and old['closed']
    result=dict(old);result.update(rho_lower=rho,beta_stat_lower=beta,closed=True,RA_log_Wmax_upper=sg.up(newlog),canonical_log_B=interval_json(oldlog),
        RA_corrected_mean_extra_disk_normalized=sg.up(extra),RA_phase_extra_upper=phase,
        range_normalization='weighted corrected estimator divided by certified full-path Wmax; NOT an unweighted physical Hadamard mean',
        inherited_canonical_rho=old['rho_lower'],inherited_canonical_beta=old['beta_stat_lower'],
        weight_arithmetic='G7 exact rational weight rounded once to binary64; compressed directed interval product at 100 dps; final float plus certified relative interval width <=1e-24. Range and mean bias include all rounding.')
    return result


@lru_cache(None)
def fixed_costs(m):
    p=policy();r=p['rounds'][m];s=p['selected'][m];total=sum(r['shots'])
    d=np.array(r['D'],float)/total;assert np.all(d==np.round(d))
    # Exactly the same stored modular D/D junction counts; no new D policy.
    over=np.array([metrics(overhead(3,s['signal']['t'],axis)) for axis in ('X','Y')],float)
    return d,over


def points(rs,arm):
    result=[]
    for m,s in enumerate(policy()['selected']):
        d,over=fixed_costs(m);rmean=np.zeros(4)
        for recipe in s['signal']['recipe']:
            k=cellkey(recipe['time'],recipe['r'])
            for rows in rs[k].values():
                rmean+=s['signal']['steps']*rows[0]['stratum_probability'][arm]*np.mean([r[arm] for r in rows],axis=0)
        q=s['qualification'] if arm=='canonical' else qualify(s)
        result.append(dict(signal=s['signal'],qualification=q,D=d,R=rmean,overhead=over,mean=d+rmean+over))
    return result


def account(pp,ns):
    rows=[]
    for m,(p,n) in enumerate(zip(pp,ns)):
        w=int(sum(n));rows.append(dict(m=m,trials=list(map(int,n)),D=(w*p['D']).tolist(),R=(w*p['R']).tolist(),
            wrapper=(n[:,None]*p['overhead']).sum(axis=0).tolist(),total=(n[:,None]*p['mean']).sum(axis=0).tolist(),
            rho_lower=p['qualification']['rho_lower'],beta_stat_lower=p['qualification']['beta_stat_lower']))
    return dict(rounds=rows,trials_total=int(ns.sum()),expected_quantum_calls=int(ns.sum()),totals=np.sum([r['total'] for r in rows],axis=0).tolist())


def price_weights(ns):
    result={}
    for s,n in zip(policy()['selected'],ns):
        for r in s['signal']['recipe']:
            k=cellkey(r['time'],r['r']);result[k]=result.get(k,0)+int(sum(n))*s['signal']['steps']
    return result


def covariance(rs,arms_and_weights):
    total=np.zeros((4,4))
    for k,strata in rs.items():
        for rows in strata.values():
            values=sum((w.get(k,0)*rows[0]['stratum_probability'][arm]*np.array([r[arm] for r in rows],float) for arm,w in arms_and_weights),np.zeros((len(rows),4)))
            total+=np.cov(values,rowvar=False,ddof=1)/len(rows)
    return total


def freeze():
    target=OUT/'POLICY.json'
    if target.exists():raise RuntimeError('Preserve frozen policy')
    rs=load_prices('training');arms={}
    for arm in ('canonical','closed'):
        pp=points(rs,arm);alpha,ns=allocate(pp,0)
        assert sum(F.from_float(float(a)) for a in alpha.flat)<=F(1,20)
        fixed=np.array([[rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],a) for a in old] for p,old in zip(pp,policy()['alpha'])])
        arms[arm]=dict(alpha=alpha.tolist(),trials=ns.tolist(),training=account(pp,ns),old_alpha_comparison=account(pp,fixed),
            qualifications=[p['qualification'] for p in pp])
    save_json(target,dict(**provenance(),arms=arms,rule='D cases, BCH coefficients, h/resolution and times fixed. Training-only RZ alpha allocation; same engine for both arms. No retuning on holdout.',
        trial_formula='ceil(4 log(2/alpha)/(rho_scaled_weighted^2 sin(beta_stat)^2)); follows bounded Hoeffding after explicit certified Wmax scaling',
        acceptance=1,quantum_backend_shots=0,Full=policy('Full')))
    print('POLICY_FROZEN',flush=True)


def result():
    from trotterlib.rpe_four_round_phase_validation import reconstruct_rpe_phase
    frozen=json.loads((OUT/'POLICY.json').read_text());rs=load_prices('holdout');results={};ws={}
    for arm in ('canonical','closed'):
        pp=points(rs,arm);p=frozen['arms'][arm];ns=np.array(p['trials'],dtype=np.int64);out=account(pp,ns);w=price_weights(ns);ws[arm]=w
        cov=covariance(rs,[(arm,w)]);out.update(MC_SE=np.sqrt(np.diag(cov)).tolist(),covariance=cov.tolist(),alpha_sum_exact=str(sum(F.from_float(a) for r in p['alpha'] for a in r)),
            alpha=p['alpha'],qualifications=p['qualifications']);results[arm]=out
    delta=np.array(results['canonical']['totals'])-results['closed']['totals']
    pair=covariance(rs,[('canonical',ws['canonical']),('closed',{k:-v for k,v in ws['closed'].items()})])
    phases=[math.atan2(s['signal']['raw']['imag'],s['signal']['raw']['real']) for s in policy()['selected']]
    dec=reconstruct_rpe_phase(phases,[1<<m for m in range(18)]);energy=-dec['final_phase_estimate']/.02
    branch=all(abs(r['base_phase_estimate']+.02*common()['energy'])<=.4/(1<<m)+1e-10 for m,r in enumerate(dec['rounds']))
    assert branch and abs(energy-common()['energy'])<1.5936e-4
    oldns=np.array(frozen['arms']['canonical']['trials']);fixedcov=covariance(rs,[('canonical',ws['canonical']),('closed',{k:-v for k,v in ws['canonical'].items()})])
    closed_same_n=account(points(rs,'closed'),oldns)
    out=dict(**provenance(),arms=results,Full=frozen['Full'],savings=delta.tolist(),saving_fraction=(delta/np.array(results['canonical']['totals'])).tolist(),paired_saving_MC_SE=np.sqrt(np.diag(pair)).tolist(),
        fixed_canonical_trials_savings=(np.array(results['canonical']['totals'])-closed_same_n['totals']).tolist(),fixed_canonical_trials_savings_MC_SE=np.sqrt(np.diag(fixedcov)).tolist(),
        decoder=dec,mean_branch_pass=branch,mean_energy_error=abs(energy-common()['energy']),energy_statistical_bound=.4/(.02*(1<<17)),
        decoder_input='same ideal corrected P3 phase, enclosed by inherited disks plus new digital/native-angle disk; E0 not passed to decoder',
        scope='saved Hermitian H/exact ground/ideal native/fresh independent bits; state preparation excluded; no quantum backend shots',
        inherited_qualification_basis='unchanged D and <=3 signed applications/event, legacy full/single/joint bounds; central RA output native RZ/CX and legacy CZ/P; new angle/coeff bias certified separately',
        price_depth='R/D/wrapper modular serial sum is an upper bound, not full long-round DAG depth')
    save_json(OUT/'FINAL.json',out)
    with (OUT/'ROUND_RESOURCES.csv').open('w',newline='') as f:
        fields=['arm','m','trials_X','trials_Y','rho_lower','beta_stat_lower','RZ','CX','serial_depth_upper','size'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for arm,data in results.items():
            for r in data['rounds']:w.writerow(dict(arm=arm,m=r['m'],trials_X=r['trials'][0],trials_Y=r['trials'][1],rho_lower=r['rho_lower'],beta_stat_lower=r['beta_stat_lower'],**dict(zip(fields[-4:],r['total']))))
    print('FINAL_SAVINGS',out['saving_fraction'],'MC_SE',out['paired_saving_MC_SE'],flush=True)
    return out


@lru_cache(None)
def fock_component(label):
    from .track_c_bottleneck_basis import configuration
    spec=common()['preps'][3].rte_preparation.symbolic_tail.components[label]
    b=configuration()[3][spec.basis_id];z=np.ones(256)
    for q in spec.diagonal_pauli_support:z*=1-2*((np.arange(256)>>q)&1)
    return spec.coefficient_sign*(b*z[None,:])@b.conj().T


def event_matrix(e,sigma=1,sector=False):
    ops=h4_operators() if sector else None;get=lambda j:ops[j] if sector else fock_component(j)
    dim=70 if sector else 256;v=np.eye(dim,dtype=complex)
    for label in reversed(e['word']):v=get(label)@v
    theta,_=angle(e['ratio']);rot=math.cos(theta)*np.eye(dim)-1j*sigma*math.sin(theta)*get(e['child'])
    return 1j**e['phase_i_power']*rot@v


def joint_check(task):
    j,t,n,descriptors,sigma=task
    prior=OUT/'correctness'/f'joint_{j}.json'
    if prior.exists():
        r=json.loads(prior.read_text());assert r['t']==t and r['n']==n and r['sigma']==sigma and r['events']==json.loads(json.dumps(descriptors))
        return r
    g=generator(t,n);records=[g.event(gi,tuple(word),child) for gi,word,child in descriptors]
    ev=application_events(t,n,records,sigma);v=np.eye(256,dtype=complex)
    for e in records:v=event_matrix(e,sigma)@v
    expected=np.eye(512,dtype=complex);expected[256:,256:]=v;rows=[]
    for mode in ('term_parity','merge','network'):
        q,_=build_events(ev,mode,'joint');actual=Operator(q).data;native=compile_reference(q);nv=Operator(native).data
        phase=np.vdot(expected,nv);phase/=abs(phase)
        err=float(np.linalg.norm(actual-expected,'fro'));ne=float(np.linalg.norm(nv/phase-expected,'fro'))
        inactive=float(np.linalg.norm(nv[:256,:256]/phase-np.eye(256),'fro'));assert err<1e-10 and ne<1e-9 and inactive<1e-9
        init=initial_state();signal=np.vdot(init[256:] if np.any(init[256:]) else init[:256],v@(init[256:] if np.any(init[256:]) else init[:256]))
        # initial_state has ancilla zero; X/Y wrappers create the two branches.
        xy=[]
        for axis,want in [('X',signal.real),('Y',signal.imag)]:
            w=compile_reference(wrapper([q],axis));state=Statevector(init.copy()).evolve(w.remove_final_measurements(inplace=False)).data.copy()
            got=float(np.vdot(state[:256],state[:256]).real-np.vdot(state[256:],state[256:]).real);xy.append(abs(got-float(want)))
        assert max(xy)<1e-10
        rows.append(dict(mode=mode,source_Fro_error=err,native_joint_common_phase_Fro_error=ne,inactive_after_same_phase_error=inactive,XY_errors=xy,native=metrics(native)))
    record=dict(j=j,t=t,n=n,sigma=sigma,events=descriptors,checks=rows)
    save_json(OUT/'correctness'/f'joint_{j}.json',record);return record


def whole_check():
    from .track_c_d_internal import schedule, evolution
    s=policy()['selected'][0]['signal'];t=s['recipe'][0]['time'];n=s['recipe'][0]['r'];g=generator(t,n);rows=[]
    # Registered m0, n=1. Deliberately select a rare parent, then compose two
    # identical outer source steps; this checks D/R ordering and scalar too.
    e=g.event(1,(g.groups[1].first,1),2)
    tail=closed_native(t,n,[e])[0];c=common();scalar=c['preps'][3].constant_coefficient+c['preps'][3].extracted_identity_coefficient
    one=np.eye(70,dtype=complex)
    for kind,i,dt in schedule(s['case'],s['h']):one=(evolution(c['spectra'][i],dt) if kind=='D' else event_matrix(e,sector=True))@one
    one*=np.exp(-1j*s['h']*scalar)
    for steps in (1,2):
        src=assemble(s['case'],s['h'],steps,[tail]*steps);z=np.vdot(c['state'],np.linalg.matrix_power(one,steps)@c['state']);init=initial_state()
        for axis,want in [('X',z.real),('Y',z.imag)]:
            q=wrapper([src],axis);native=compile_reference(q);state=Statevector(init.copy()).evolve(native.remove_final_measurements(inplace=False)).data.copy()
            got=float(np.vdot(state[:256],state[:256]).real-np.vdot(state[256:],state[256:]).real)
            assert abs(got-want)<1e-9;rows.append(dict(steps=steps,axis=axis,independent_sector_signal=float(want),native_readout=got,error=abs(got-want),whole_native=metrics(native)))
    return rows


def bookkeeping_check():
    """Exercise the existing C1 all-trials accumulator, not a rewritten copy."""
    import ast
    from .algorithm_codesign.g6_f3_contract import Evidence
    ref='af75af03c0b334c29b5a4311bd2b4578a999460c'
    path='docs/research/track_b_c1_4_5_c1_5_technical_preflight_20261021/whole_path.py'
    text=subprocess.check_output(['git','show',ref+':'+path],cwd=ROOT,text=True);tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AllTrials')
    def require(v,reason):
        if not v:raise ValueError(reason)
    ns=dict(F=F,Evidence=Evidence,require=require);exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),ns)
    class PathFixture:
        context_id='synthetic:closedP3-integration'
        def validate(self,r):return r.read()
    def event(tid,kind,weight):return Evidence.create(dict(trial_id=tid,conditioning='NONE',kind=kind,weight=weight,spent_prefix=None))
    a=ns['AllTrials'](PathFixture(),['a','b','c'],'X');a.add(event('a','EVENT',F(3,2)),1);a.add(event('b','ZERO',F(0)))
    assert a.evidence().read()['full_trial_mean'] is None
    a.add(event('c','EVENT',F(5,4)),-1);d=a.evidence().read();assert d['full_trial_mean']==F(1,12)
    b=ns['AllTrials'](PathFixture(),['a'],'Y');b.fail('artificial technical failure');assert b.evidence().read()['full_trial_mean'] is None
    # Independent ordered finite means: exhaustive small matrices, no stochastic
    # estimate used as proof of the path-product rule.
    return dict(source_ref=ref,source_path=path,all_trial_mean=str(d['full_trial_mean']),valid_zero_count=d['valid_zeros'],incomplete_prefix_rejected=True,technical_failure_rejected=True,
        actual_closed_P3_zero_mass=0,actual_closed_P3_acceptance=1,independence='fresh bit draws between microsteps and trials is an explicit model condition')


def correctness():
    target=OUT/'CORRECTNESS.json'
    if target.exists():raise RuntimeError('Preserve completed correctness')
    setup();symbols=common()['preps'][3].rte_preparation.symbolic_tail.components;bybasis={}
    for i,s in enumerate(symbols):bybasis.setdefault(s.basis_id,i)
    t,n=cells()[0];g=generator(t,n);tasks=[]
    for j,child in enumerate(bybasis.values()):tasks.append((j,t,n,[(0,(),child)],1))
    first=next(i for i,s in enumerate(symbols) if s.coefficient_sign<0);other=next(i for i,s in enumerate(symbols) if s.basis_id!=symbols[first].basis_id)
    child=next(i for i,s in enumerate(symbols) if i!=first and s.basis_id==symbols[first].basis_id);gi=1+first
    for sigma in (-1,1):tasks.append((len(tasks),t,n,[(gi,(first,other),child),(0,(),child)],sigma))
    records=[]
    with ProcessPoolExecutor(max_workers=min(len(tasks),len(os.sched_getaffinity(0)))) as pool:
        for f in as_completed(pool.submit(joint_check,t) for t in tasks):records.append(f.result())
    whole_path=OUT/'WHOLE_CHECKS.json'
    whole=json.loads(whole_path.read_text()) if whole_path.exists() else whole_check()
    if not whole_path.exists():save_json(whole_path,whole)
    book=bookkeeping_check()
    q=[qualify(s) for s in policy()['selected']]
    save_json(target,dict(**provenance(),joint=records,whole=whole,all_trials=book,weighted_round_qualification=q,
        output_gate_families='RA central outputs native RZ/CX plus existing CZ/P; A passes RZ parameters through. Legacy Gaussian/all-context certificates are retained.',
        legacy_basis_certificate_sha256=hashlib.sha256((INPUT/'joint_certificate.json').read_bytes()).hexdigest(),all_round_budgets_positive=all(r['closed'] for r in q),passed=True))
    print('CORRECTNESS_PASS',len(records),max(v['native_joint_common_phase_Fro_error'] for r in records for v in r['checks']),flush=True)


def pilot_gate():
    rs=load_prices('pilot');m=17;s=policy()['selected'][m];d,over=fixed_costs(m);rmean={arm:np.zeros(4) for arm in ('canonical','closed')}
    for recipe in s['signal']['recipe']:
        k=cellkey(recipe['time'],recipe['r'])
        for rows in rs[k].values():
            for arm in rmean:rmean[arm]+=s['signal']['steps']*rows[0]['stratum_probability'][arm]*np.mean([r[arm] for r in rows],axis=0)
    alphas=policy()['alpha'][m];qs={'canonical':s['qualification'],'closed':qualify(s)}
    ns={arm:np.array([rigorous_shots(q['rho_lower'],q['beta_stat_lower'],a) for a in alphas]) for arm,q in qs.items()}
    costs={arm:(ns[arm][:,None]*(d+rmean[arm]+over)).sum(axis=0) for arm in rmean}
    weights={arm:{cellkey(r['time'],r['r']):sum(ns[arm])*s['signal']['steps'] for r in s['signal']['recipe']} for arm in rmean}
    cov=covariance(rs,[('canonical',weights['canonical']),('closed',{k:-v for k,v in weights['closed'].items()})]);delta=costs['canonical']-costs['closed'];se=np.sqrt(np.diag(cov));passed=bool(delta[0]>2*se[0])
    out=dict(**provenance(),m=m,canonical_trials=ns['canonical'].tolist(),closed_trials=ns['closed'].tolist(),canonical_cost=costs['canonical'].tolist(),closed_cost=costs['closed'].tolist(),
        RZ_saving=float(delta[0]),paired_RZ_MC_SE=float(se[0]),saving_fraction=float(delta[0]/costs['canonical'][0]),proceed_to_all_rounds=passed,
        rule='Correctness and positive budgets first; m17 RZ saving exceeds 2 paired MC SE. This is a pilot decision, not the independent final evidence.')
    save_json(OUT/'PILOT_GATE.json',out);print('PILOT_GATE',out,flush=True);return passed


def run(stage):
    setup()
    if stage=='correctness':correctness()
    elif stage=='pilot':
        assert json.loads((OUT/'CORRECTNESS.json').read_text())['passed']
        price_run('pilot',sorted({cells()[0],max(cells(),key=lambda x:x[1])}),8);pilot_gate()
    elif stage=='holdout':
        assert json.loads((OUT/'PILOT_GATE.json').read_text())['proceed_to_all_rounds']
        price_run('training',cells(),16);freeze();price_run('holdout',cells(),16);result()
    elif stage=='finish':result()
    else:raise ValueError(stage)
