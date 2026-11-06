"""Finite structural compiler audit. No price, full-round build, or new sampling."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import mpmath as mp
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    XXPlusYYGate, RZGate, RZZGate, CRZGate, PhaseGate, HGate, SdgGate,
    CZGate, CCXGate,
)
from qiskit.quantum_info import Operator
from .track_c_h4 import common, diagonal_primitives, weights, RVALUES, ROOT, save_json
from .track_c_compiler_a import compile_reference, PROFILE

BASE = 'f45e9b07bf730f7272933b28055914cfe76fdaab'
OUT = ROOT/'artifacts/track_c_compiler_a_18round/2026-11-06/run01'
CONTEXT_FAMILIES = ('xx_plus_yy', 'crz', 'ccx', 'p', 'h', 'sdg', 'cz')
IV_BITS = 192


def stored(path):
    return json.loads(subprocess.check_output(['git', 'show', BASE+':'+path], cwd=ROOT))


def key(family, params=()):
    return family+':'+','.join(float(x).hex() for x in params)


def spec(family, params=()):
    return dict(family=family, params=list(map(float, params)), key=key(family, params))


def gate(s):
    f=s['family']; p=s['params']
    return {'xx_plus_yy': lambda: XXPlusYYGate(*p), 'crz': lambda: CRZGate(*p),
            'crzz': lambda: RZZGate(*p).control(), 'p': lambda: PhaseGate(*p),
            'h': HGate, 'sdg': SdgGate, 'cz': CZGate, 'ccx': CCXGate,
            'rz': lambda: RZGate(*p)}[f]()


def add(counter, catalogue, family, params=(), count=1):
    if family in ('rz', 'cx'): return  # Native parameters are passed through.
    s=spec(family, params); catalogue[s['key']]=s; counter[s['key']]+=count


def basis_counter(operations, catalogue):
    c=Counter()
    for g, _ in operations:
        assert g.name in ('rz', 'xx_plus_yy'), g.name
        for op in (g, g.inverse()): add(c, catalogue, op.name, op.params)
    return c


def inventory():
    """Reuse preparations; instantiate only local gates, not full trajectories."""
    c=common(); catalogue={}; det={}; tails={}
    for f, ld in [('S2', 3), ('Y4', 12)]:
        for w in weights(f):
            t=float(w*.02); counts=Counter()
            for b in c['preps'][ld].deterministic_blocks:
                bc=basis_counter(b.runtime_basis_operations, catalogue)
                # Two D halves. Each half contains one B and one actual B^-1.
                counts.update(bc);counts.update(bc)
                p=diagonal_primitives(b, t/2)
                for _q, a in p.rz: add(counts, catalogue, 'crz', [a], 2)
                for _q, _r, a in p.rzz: add(counts, catalogue, 'crzz', [a], 2)
                # All P gates are covered by the symbolic P -> RZ(theta) rule.
            det[(f, t.hex())]=counts
    prep=c['preps'][3].rte_preparation
    components=prep.symbolic_tail.components
    assert len(components)==len(prep.component_specs)
    assert all(not x.is_identity and len(x.diagonal_pauli_support) in (1, 2) for x in components)
    basis=[]
    for d in prep.basis_registry._definitions.values():
        basis.append(basis_counter(d.runtime_operations, catalogue))
    for r in sorted(set(RVALUES)):
        # S2 has one stage, positive h; both signs cover coefficient signs.
        tau=c['preps'][3].exact_rte_lambda_r*.02/r
        ids=[]
        for order in (0, 2):
            a=math.atan(tau/(order+1))
            for sign in (-1, 1):
                for family in ('crz', 'crzz'):
                    s=spec(family, [sign*2*a]);catalogue[s['key']]=s;ids.append(s['key'])
        tails[r]=ids
    for f in ('h', 'sdg', 'cz', 'ccx'):
        s=spec(f);catalogue[s['key']]=s
    return dict(catalogue=catalogue,det=det,tail_basis=basis,tails=tails,
                components=len(components),component_identity_count=0)


def context_compile(request):
    """All numeric variants share a basis-search context; barriers label slices."""
    mask, catalogue=request
    context={f for j, f in enumerate(CONTEXT_FAMILIES) if mask & (1<<j)}
    selected=[s for s in catalogue.values() if s['family'] in context or
              (s['family']=='crzz' and {'crz', 'ccx'} <= context)]
    q=QuantumCircuit(3); tags=[]
    for s in selected:
        g=gate(s);q.append(g, range(g.num_qubits));q.barrier();tags.append(s)
    alpha=Parameter('phase_witness')
    if 'p' in context:
        q.p(alpha, 0);q.barrier();tags.append(dict(family='p_symbolic', key='p_symbolic',params=[]))
    # P is otherwise only symbolic; every other requested family has variants.
    q.rz(.125, 0);q.barrier();tags.append(dict(family='rz_witness',key='rz_witness',params=[.125]))
    trace={}
    def cb(**kw):
        if type(kw['pass_']).__name__=='UnrollCustomDefinitions':
            trace['post_unroll']=sorted(set(kw['dag'].count_ops())-{'rz','cx','barrier'})
    native=compile_reference(q, callback=cb)
    assert set(trace['post_unroll'])==context, (context, trace)
    slices=[]; current=[]
    for inst in native.data:
        if inst.operation.name=='barrier':slices.append(current);current=[]
        else:current.append(inst)
    assert not current and len(slices)==len(tags)
    records=[]; symbolic=False
    for s, ops in zip(tags, slices):
        if s['family']=='p_symbolic':
            assert len(ops)==1 and ops[0].operation.name=='rz' and ops[0].operation.params[0]==alpha
            symbolic=True;continue
        if s['family']=='rz_witness':continue
        n=gate(s).num_qubits
        lowering=[]
        for inst in ops:
            qs=[native.find_bit(x).index for x in inst.qubits]
            assert max(qs)<n
            lowering.append([inst.operation.name,qs,list(map(float,inst.operation.params))])
        record=dict(source=s,n=n,lowering=lowering)
        record['lowering_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
        records.append(record)
    return dict(mask=mask,context=sorted(context),post_unroll=trace['post_unroll'],
                symbolic_P_passthrough=symbolic,records=records)


def I(x=0): return mp.iv.mpf(x)
def C(x=0, y=0): return mp.iv.mpc(x,y)
def expi(x): return C(mp.iv.cos(x),mp.iv.sin(x))
def eye(n): return [[C(int(i==j)) for j in range(n)] for i in range(n)]
def upper(x): return math.nextafter(float(x.b), math.inf)
def lower(x): return math.nextafter(float(x.a), -math.inf)


def source_matrix(s):
    f=s['family'];p=list(map(I,s['params']));n=gate(s).num_qubits;d=1<<n;a=eye(d)
    if f=='xx_plus_yy':
        t,b=p;co=mp.iv.cos(t/2);si=mp.iv.sin(t/2)
        a[1][1]=a[2][2]=C(co)
        a[1][2]=-C(0,1)*si*expi(-b);a[2][1]=-C(0,1)*si*expi(b)
    elif f in ('crz','crzz'):
        for j in range(d):
            if j&1:
                z=1-2*((j>>1)&1)
                if f=='crzz':z*=1-2*((j>>2)&1)
                a[j][j]=expi(-p[0]*z/2)
    elif f=='h':
        z=1/mp.iv.sqrt(I(2));a=[[C(z),C(z)],[C(z),C(-z)]]
    elif f=='sdg':a[1][1]=C(0,-1)
    elif f=='cz':a[3][3]=C(-1)
    elif f=='ccx':a=eye(8);a[3][3]=a[7][7]=C();a[3][7]=a[7][3]=C(1)
    else:raise ValueError(f)
    return a


def native_matrix(n, lowering):
    d=1<<n;a=eye(d)
    for f, qs, p in lowering:
        if f=='rz':
            v=[expi(-I(p[0])/2),expi(I(p[0])/2)]
            a=[[x*v[(j>>qs[0])&1] for x in row] for j,row in enumerate(a)]
        elif f in ('x','cx'):
            q=qs[-1];new=[]
            for j in range(d):
                k=j^(1<<q) if f=='x' or ((j>>qs[0])&1) else j
                new.append(a[k])
            a=new
        elif f=='sx':
            q=qs[0];b=[[None]*d for _ in range(d)];v=C(I(1)/2,I(1)/2);w=C(I(1)/2,-I(1)/2)
            for j in range(d):
                if j & (1<<q):continue
                k=j|(1<<q)
                for col in range(d):b[j][col]=v*a[j][col]+w*a[k][col];b[k][col]=w*a[j][col]+v*a[k][col]
            a=b
        else:raise ValueError(f)
    return a


def interval_check(record):
    mp.iv.prec=IV_BITS
    a=source_matrix(record['source']);b=native_matrix(record['n'], record['lowering']);d=len(a)
    # A rational (binary64) angle selects a unit complex gauge. The bound
    # includes gauge-selection error; it does not assume that this fit is exact.
    approx=lambda z:complex(float(z.real.mid),float(z.imag.mid))
    pivot=max(((i,j) for i in range(d) for j in range(d)),key=lambda ij:abs(approx(a[ij[0]][ij[1]])))
    i,j=pivot;phase=float(np.angle(approx(b[i][j])/approx(a[i][j])))
    phase_iv=expi(I(-phase));ss=I(0)
    for i in range(d):
        for j in range(d):
            delta=a[i][j]-b[i][j]*phase_iv
            ss+=delta.real**2+delta.imag**2
    bound=upper(mp.iv.sqrt(ss))
    return dict(source_key=record['source']['key'],lowering_sha256=record['lowering_sha256'],
                operator_error_upper=bound,method='directed interval Frobenius bound >= spectral norm',
                gauge_angle_binary64=phase,precision_bits=IV_BITS)


def arithmetic_tests():
    """New bound machinery only; do not repeat the inherited 44-case suite."""
    mp.iv.prec=IV_BITS
    # Exact native gate and deliberately wrong tiny CRZ: cannot pass by fitting
    # a separate phase to the controlled branch.
    bad=dict(source=spec('crz',[1e-5]),n=2,lowering=[],lowering_sha256='deliberate_identity')
    r=interval_check(bad);assert r['operator_error_upper']>1e-6
    assert I(.1).a==I(.1).b
    x=I(2)**I(.5);assert x.a*x.a<=I(2) and x.b*x.b>=I(2)
    return dict(binary64_input_exact=True,deliberate_relative_phase_error_detected=True,
                wrong_identity_error_upper=r['operator_error_upper'],interval_bits=IV_BITS)


def sum_upper(values):
    mp.iv.prec=IV_BITS
    return upper(sum((I(x) for x in values),I(0)))


def phase_accumulation_bound(n):
    """No identities: at most 3n nonzero pi terms in one RTE occurrence."""
    mp.iv.prec=IV_BITS;k=3*n;u=I(2)**-53
    pi_error=abs(I(math.pi)-mp.iv.pi)
    gamma=(k*u)/(1-k*u)
    return upper(k*pi_error+gamma*k*abs(I(math.pi)))


def verify_saved_models():
    """Check the new local matrix evaluator against Qiskit, without compiling."""
    mp.iv.prec=IV_BITS
    records=json.loads((OUT/'LOCAL_LOWERINGS.json').read_text())
    maximum_source=maximum_native=0.
    as_numpy=lambda a:np.array([[complex(float(z.real.mid),float(z.imag.mid)) for z in row] for row in a])
    for r in records:
        a=as_numpy(source_matrix(r['source']))
        maximum_source=max(maximum_source,float(np.max(np.abs(a-Operator(gate(r['source'])).data))))
        q=QuantumCircuit(r['n'])
        for f,qs,p in r['lowering']:
            getattr(q,f)(*(p+qs))
        b=as_numpy(native_matrix(r['n'],r['lowering']))
        maximum_native=max(maximum_native,float(np.max(np.abs(b-Operator(q).data))))
    assert maximum_source<1e-13 and maximum_native<1e-13
    result=dict(local_source_and_native_pairs=len(records),max_source_matrix_difference=maximum_source,
        max_native_matrix_difference=maximum_native,diagnostic_tolerance=1e-13,
        scope='new interval evaluator implementation check only; not a physical compiler allowance',
        new_compiles=0,inherited_44_cases_repeated=False)
    save_json(OUT/'MATRIX_MODEL_CHECKS.json',result)
    return result


def accounting(inv, bounds):
    mp.iv.prec=IV_BITS
    def total(counts):return sum_upper([I(count)*I(bounds[k]) for k,count in counts.items()])
    maximum_basis=max(total(x) for x in inv['tail_basis'])
    results=[]
    for formula,ld,name in [('S2',3,'CN_S2_h0.02_ld3'),('Y4',12,'CN_Y4_h0.02_ld12')]:
        saved=stored('artifacts/track_c_h4_pilots/2026-11-06/run01/signals/'+name+'.json')
        rows=[]
        for old in saved['rows']:
            stage_errors=[];phase_errors=[]
            for stage in old['recipe']:
                t=stage['time'];d=total(inv['det'][(formula,t.hex())]);n=stage['r']
                if ld==3:
                    rotation=max(bounds[k] for k in inv['tails'][old['r']])
                    # K=2: <=3 basis conjugations, <=4 CZ operations and 1 rotation.
                    # basis_counter already contains B and actual B^-1.
                    event=sum_upper([I(3)*I(maximum_basis),I(4)*I(bounds[key('cz')]),rotation])
                    d=sum_upper([d,I(n)*I(event)]);phase_errors.append(phase_accumulation_bound(n))
                stage_errors.append(d)
            q=old['steps']
            core=sum_upper([I(q)*I(sum_upper(stage_errors))])
            wrapper=sum_upper([I(2)*I(bounds[key('h')]),bounds[key('sdg')]])
            compiler=sum_upper([core,wrapper])
            construction_phase=sum_upper([I(q)*I(sum_upper(phase_errors))]) if phase_errors else 0.
            disk_comp=upper(2*mp.iv.sqrt(I(2))*I(compiler))
            disk_phase=upper(2*mp.iv.sqrt(I(2))*I(construction_phase)) if construction_phase else 0.
            disk=sum_upper([disk_comp,disk_phase])
            rho=lower(I(old['radius_lower'])-I(disk))
            ratio=I(disk)/I(old['radius_lower'])
            beta_new=upper(mp.iv.atan2(ratio,mp.iv.sqrt(1-ratio**2))) if disk<old['radius_lower'] else None
            beta_stat=None if beta_new is None else lower(I(old['beta_stat'])-I(beta_new))
            feasible=bool(rho>0 and beta_stat is not None and beta_stat>0)
            shots=None
            if feasible:
                eps=I(rho)*mp.iv.sin(I(beta_stat))/mp.iv.sqrt(I(2))
                n_bound=2*mp.iv.log(I(1440))/eps**2  # alpha_axis = exact 1/720
                with mp.workprec(IV_BITS):shots=int(mp.ceil(mp.mpf(n_bound._mpi_[1])))
            rows.append(dict(m=old['m'],q=old['q'],r=0 if ld==12 else old['r'],
                inherited_r_field=old['r'],registered_r_m=RVALUES[old['m']],recipe=old['recipe'],
                compiler_operator_upper=compiler,source_phase_operator_upper=construction_phase,
                raw_compiler_disk_upper=disk_comp,raw_source_phase_disk_upper=disk_phase,
                old_raw_radius_lower=old['radius_lower'],new_raw_radius_lower=rho,
                old_beta_stat=old['beta_stat'],additional_phase_diagnostic=beta_new,
                new_beta_stat=beta_stat,shots_per_axis_conditional=shots,
                normalization='unweighted native raw trajectory mean; no extra inverse attenuation on compiler disk',
                attenuation=old['attenuation'],log_normalizer=old['log_normalizer'],
                inherited_PF_phase=old['beta_PF'],inherited_RTE_phase=old['beta_original_RTE'],
                inherited_numerical_input_phase=old['beta_numerical_input'],
                numerical_conditions_satisfied=feasible,
                full_equals_strong_error_envelope=True,
                guarantee_status='conditional on translation coverage and inherited SOURCE mean model'))
        results.append(dict(name=name,rows=rows,all18_numerical_conditions_satisfied=all(x['numerical_conditions_satisfied'] for x in rows),
            total_shots_conditional=2*sum(x['shots_per_axis_conditional'] for x in rows) if all(x['shots_per_axis_conditional'] is not None for x in rows) else None,
            inherited_alias=saved['alias'],all_round_scientific_qualification='UNQUALIFIED'))
    return results


def rpe_conditions(results):
    mp.iv.prec=IV_BITS
    energy=I('0.4')/(I('0.02')*2**17)
    result=dict(energy_error_upper_Ha=upper(energy),target_energy_error_Ha=1.5936e-4,
        energy_resolution_condition=bool(energy.b<I('0.00015936').a),
        branch_recursion_condition_3beta_lt_pi=bool((3*I('.4')).b<mp.iv.pi.a),
        base_window_status='external assumption, not verified here',alpha_axis_exact='1/720',
        alpha_total_exact='1/20',rounds=18,shot_comparison=[])
    for case in results:
        old=stored('artifacts/track_c_h4_pilots/2026-11-06/run01/signals/'+case['name']+'.json')
        old_n=old['shots'][0]['shots']
        new_n=[n for row in case['rows'] for n in [row['shots_per_axis_conditional']]*2]
        result['shot_comparison'].append(dict(name=case['name'],old_uniform_total_shots=sum(old_n),
            new_conditional_uniform_total_shots=None if None in new_n else sum(new_n),
            integer_shot_counts_unchanged=old_n==new_n))
    save_json(OUT/'RPE_CONDITIONS.json',result)
    return result


def run():
    start=time.monotonic()
    if (OUT/'STOP.json').exists():raise RuntimeError('Completed study: preserve STOP')
    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'ARITHMETIC_TESTS.json',arithmetic_tests())
    inv=inventory();save_json(OUT/'INVENTORY.json',dict(catalogue=list(inv['catalogue'].values()),
        component_count=inv['components'],identity_components=inv['component_identity_count'],
        basis_families=list(CONTEXT_FAMILIES),context_masks=128,
        support_proof='K2; at most 2 products and 1 rotation; basis conjugation for each application; optional exact reuse only reduces this envelope'))
    save_json(OUT/'STRUCTURAL_COUNTS.json',dict(
        deterministic=[dict(formula=f,time_hex=t,multiplicities=dict(v)) for (f,t),v in inv['det'].items()],
        tail_basis_conjugations=[dict(x) for x in inv['tail_basis']],tail_rotation_keys=inv['tails']))
    requests=[(mask, inv['catalogue']) for mask in range(128)]
    workers=min(len(requests),len(os.sched_getaffinity(0)))
    records={};context_rows=[]
    print('local context compiles',len(requests),'workers',workers,'gate variants',len(inv['catalogue']),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(context_compile,x) for x in requests]):
            x=future.result();context_rows.append({k:v for k,v in x.items() if k!='records'})
            for r in x['records']:records[r['lowering_sha256']]=r
    save_json(OUT/'CONTEXTS.json',sorted(context_rows,key=lambda x:x['mask']))
    save_json(OUT/'LOCAL_LOWERINGS.json',list(records.values()))
    print('unique local lowerings',len(records),flush=True)
    checks=[]
    with ProcessPoolExecutor(max_workers=min(workers,len(records))) as pool:
        for future in as_completed([pool.submit(interval_check,x) for x in records.values()]):checks.append(future.result())
    save_json(OUT/'LOCAL_INTERVAL_BOUNDS.json',checks)
    print(verify_saved_models(),flush=True)
    bounds={}
    for x in checks:bounds[x['source_key']]=max(bounds.get(x['source_key'],0.),x['operator_error_upper'])
    assert set(bounds)==set(inv['catalogue']),set(inv['catalogue'])-set(bounds)
    save_json(OUT/'PRIMITIVE_UPPERS.json',bounds)
    result=accounting(inv,bounds)
    save_json(OUT/'ROUNDS.json',result)
    rpe_conditions(result)
    summary=dict(base_commit=BASE,profile=PROFILE,local_contexts=len(context_rows),
        unique_lowerings=len(records),interval_bits=IV_BITS,
        local_max_upper=max(bounds.values()),all36_numerical_conditions_satisfied=all(x['all18_numerical_conditions_satisfied'] for x in result),
        compiler_bound_status='conditional structural envelope from interval local rules and source-basis context coverage',
        full_RPE_scientific_qualification='UNQUALIFIED',
        unresolved=['Saved PF/source mean numerical disks are conditional float64 diagnostics, not rigorous interval certificates.',
                    'External base-window/alias and original-H reference assumptions are inherited, not revalidated.',
                    'Generated SOURCE versus saved analytic mean: Gaussian basis realization, diagonal/atan parameter generation and distribution normalization are not newly certified.',
                    'Local basis-context transfer relies on the pinned BasisTranslator implementation and session equivalence graph; full-round circuits are not compiled.'],
        new_full_round_builds=0,new_full_round_simulations=0,new_sampling=0,new_price_cells=0,new_T_synthesis=0,
        old_run01_restarted=False,wall_seconds=time.monotonic()-start)
    save_json(OUT/'SUMMARY.json',summary)
    save_json(OUT/'STOP.json',dict(status='MANDATORY_STOP',full_RPE_scientific_qualification='UNQUALIFIED',
        science_workers_remaining=0,new_price_cells=0,new_T_synthesis=0,automatic_resume=False))
    print(summary,flush=True)
