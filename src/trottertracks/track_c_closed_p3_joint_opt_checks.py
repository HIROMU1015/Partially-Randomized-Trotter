"""Limited late-round checks, then fixed-policy decoder/native checks."""
from collections import Counter
import json
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from . import track_c_closed_p3_joint_opt as j
from .track_c_h4_price import wrapper
from .track_c_joint_resources import initial_state
from .track_c_d_internal import schedule,evolution
from .track_c_d_resources import d_piece
from .track_c_closed_p3_weights import compressed_weight


def trial_pair(s,alphas,rs):
    p=j.point(s,rs,'closed'); q=p['qualification']
    ns=[j.rigorous_shots(q['rho_lower'],q['beta_stat_lower'],a) for a in alphas]
    weights={j.cp.cellkey(r['time'],r['r']):sum(ns)*s['signal']['steps'] for r in s['signal']['recipe']}
    return dict(key=s['key'],trials=ns,qualification=q,cost=(np.array(ns)[:,None]*p['mean']).sum(axis=0).tolist()),weights


def pilot(samples=8):
    rows=[s for s in j.menu()['rows'] if s['key'][0]=='tuned_0321' and s['key'][1] in (16,17) and s['key'][2]==.32]
    cells=sorted({(r['time'],r['r']) for s in rows for r in s['signal']['recipe']})
    j.prices('pilot',cells,samples); rs=j.cr.load_prices('pilot')
    alpha=json.loads((j.OLD/'FAIR_POLICY.json').read_text())['arms']['closed']['alpha']
    fixed={m:next(s for s in j.cp.policy()['selected'] if s['signal']['m']==m)['signal'] for m in (16,17)}
    results=[]; comparisons=[]
    for m in (16,17):
        options=[s for s in rows if s['signal']['m']==m]
        base=next(s for s in options if j.identity(s['signal'])==j.identity(fixed[m])); old,ow=trial_pair(base,alpha[m],rs)
        for s in options:
            new,nw=trial_pair(s,alpha[m],rs); results.append(new)
            cv=j.cr.covariance(rs,[('closed',ow),('closed',{k:-v for k,v in nw.items()})]); difference=np.array(old['cost'])-new['cost']; se=np.sqrt(np.maximum(0,np.diag(cv)))
            comparisons.append(dict(m=m,before=old['key'],after=new['key'],saving=difference.tolist(),paired_MC_SE=se.tolist(),
                                    saving_fraction=(difference/np.array(old['cost'])).tolist()))
    gain=any(r['saving'][0]>2*r['paired_MC_SE'][0] for r in comparisons)
    j.save_json(j.OUT/'PILOT.json',dict(**j.provenance(),results=results,comparisons=comparisons,
        late_round_improvement_resolved=bool(gain),samples_per_stratum=samples,
        stage='limited m16/m17 fixed-alpha comparison; choice data only, not final holdout',
        correctness='same finite P3 and saved candidate-specific PF/native certificates; exact-microstep bit/angle certificates in STAGE1'))
    print('PILOT',comparisons,'RESOLVED',gain,flush=True)
    return gain


def native_group(x,role):
    seq=j.split(x['case'],x['h'])[role]
    mode=j.dets()[(x['name'],x['h'])]['variants'][role]['selected'];ld=12 if x['case']['kind'].startswith('full_') else 3
    q=d_piece(tuple(seq),ld) if mode=='reference' else j.df.d_fused(seq,ld,mode)[0]
    return j.compile_reference(q)


def whole_fixture(s):
    """A positive-probability, complete occurrence; no truncation of n events.

    A forced noncommuting rare prefix followed by same-basis roots keeps this
    structural check small after exact run fusion, even for n=1024.
    """
    x=s['signal']; t,n=x['recipe'][0]['time'],x['recipe'][0]['r'];g=j.cp.generator(t,n)
    symbols=j.common()['preps'][3].rte_preparation.symbolic_tail.components
    first=next(i for i,z in enumerate(symbols) if z.coefficient_sign<0)
    other=next(i for i,z in enumerate(symbols) if z.basis_id!=symbols[first].basis_id)
    child=next(i for i,z in enumerate(symbols) if i!=first and z.basis_id==symbols[first].basis_id)
    rare=g.event(1+first,(first,other),child);root=g.event(0,(),child)
    events=[rare]+[root]*(n-1);tail=j.cr.closed_native(t,n,events)[0]
    R=j.cr.event_matrix(root,sector=True); R=np.linalg.matrix_power(R,n-1)@j.cr.event_matrix(rare,sector=True)
    c=j.common();scalar=c['preps'][3].constant_coefficient+c['preps'][3].extracted_identity_coefficient
    one=np.eye(70,dtype=complex)
    for kind,i,dt in schedule(x['case'],x['h']):one=(evolution(c['spectra'][i],dt) if kind=='D' else R)@one
    one*=np.exp(-1j*x['h']*scalar)
    records=[]
    for steps in (1,2):
        core=QuantumCircuit(9);core.compose(native_group(x,'first'),inplace=True)
        for k in range(steps):
            core.compose(tail,inplace=True);core.compose(native_group(x,'last' if k==steps-1 else 'junction'),inplace=True)
        core.p(-steps*x['h']*scalar,8)
        z=np.vdot(c['state'],np.linalg.matrix_power(one,steps)@c['state'])
        w,we,wd=compressed_weight([e['weight'] for e in events]*steps)
        for axis,want in [('X',z.real),('Y',z.imag)]:
            # Wrapper H/Sdg are separately compiled; no optimization through
            # modular boundaries is used to validate the priced implementation.
            pre=QuantumCircuit(9);pre.h(8);pre=j.compile_reference(pre)
            read=QuantumCircuit(9,1)
            if axis=='Y':read.sdg(8)
            read.h(8);read.measure(8,0);read=j.compile_reference(read)
            q=QuantumCircuit(9,1);q.compose(pre,inplace=True);q.compose(core,inplace=True);q.compose(read,inplace=True)
            state=Statevector(initial_state()).evolve(q.remove_final_measurements(inplace=False)).data
            got=float(np.vdot(state[:256],state[:256]).real-np.vdot(state[256:],state[256:]).real)
            error=abs(got-want); assert error<1e-8
            records.append(dict(steps=steps,axis=axis,independent_sector_signal=float(want),native_readout=got,error=error,
                                weighted_native_readout=w*got,weighted_sector_signal=w*float(want),weight=w,weight_interval_error=str(we),weight_events=wd['events']))
    return dict(key=s['key'],microsteps_per_occurrence=n,rare_prefix=True,complete_occurrence=True,records=records)


def correctness():
    target=j.OUT/'CORRECTNESS.json'
    if target.exists():raise RuntimeError('Preserve completed correctness')
    rows=j.menu()['rows']; assert all(s['signal']['case']==s['saved_native_qualification']['case'] for s in rows)
    assert all(e==0 for d in j.dets().values() for e in d['errors'].values())
    # Validate representative new h/resolution accounting with actual source
    # controls and exact ground input; full support uses inherited interval bounds.
    keys=[('tuned_0321',17,.32,32),('tuned_0321',17,.32,64)]
    checked=[whole_fixture(next(s for s in rows if tuple(s['key'])==k)) for k in keys]
    parent=json.loads((j.OLD/'CORRECTNESS.json').read_text()); assert parent['passed']
    q=[s['closed_qualification'] for s in rows if s['family']=='PR'];assert all(z['closed'] for z in q)
    j.save_json(target,dict(**j.provenance(),whole_fixtures=checked,all_candidate_case_native_bindings=True,
        canonical_qualified_candidates=len(rows),closed_qualified_candidates=len(q),all_D_group_rational_rounding_errors_zero=True,
        inherited_full_support_correctness=dict(commit=j.BASE,file='artifacts/track_c_closed_p3/2026-11-08/CORRECTNESS.json',passed=parent['passed']),
        minimum_closed_rho=min(z['rho_lower'] for z in q),minimum_closed_beta=min(z['beta_stat_lower'] for z in q),
        conditional_success='sum alpha <=.05 and directed sufficient integer trials at each selected qualification; bounded WEIGHTED estimator, fresh independent bits',passed=True))
    print('CORRECTNESS_PASS',max(r['error'] for x in checked for r in x['records']),flush=True)


def selected_correctness():
    """Check actual selected late-round h/case, then all frozen shot budgets."""
    from fractions import Fraction as F
    f=json.loads(j.policy_path().read_text());allp={**f['methods'],**f['ablations']};checks=[]
    lookup={tuple(s['key']):s for s in j.menu()['rows']}
    keys={tuple(p['selected'][-1]['key']) for name,p in f['methods'].items() if p['family']=='PR'}
    for key in sorted(keys):checks.append(whole_fixture(lookup[key]))
    for p in allp.values():
        assert sum(F.from_float(a) for r in p['alpha'] for a in r)<=F(1,20)
        for s,aa,nn in zip(p['selected'],p['alpha'],p['trials']):
            q=s['qualification'];assert q['closed'] and q['rho_lower']>0 and q['beta_stat_lower']>0
            for a,n in zip(aa,nn):assert n>=j.rigorous_shots(q['rho_lower'],q['beta_stat_lower'],a)
    j.save_json(j.OUT/'SELECTED_CORRECTNESS.json',dict(**j.provenance(),policy_sha256=j.hashlib.sha256(j.policy_path().read_bytes()).hexdigest(),
        checks=checks,all_frozen_integer_shot_bounds=True,all_exact_alpha_sums=True,passed=True))
    print('SELECTED_CORRECTNESS_PASS',max(r['error'] for c in checks for r in c['records']),flush=True)
