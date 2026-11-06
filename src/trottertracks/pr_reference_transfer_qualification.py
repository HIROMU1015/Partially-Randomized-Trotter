"""Finite long-round transfer study. Old SOURCE and STOP are read-only."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
from functools import lru_cache
from importlib.metadata import version
import hashlib,io,json,lzma,math,os,subprocess,tarfile,time
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit,qpy
from qiskit.quantum_info import Operator,Statevector
from . import pr_reference_interference_pilot as old
from .track_c_h4 import ROOT,common,save_json,diagonal_primitives
from .track_c_compiler_a import compile_reference,PROFILE
from trotterlib.rte import _make_event,make_rte_config
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.df_partial_s2 import DFPartialS2StepRequest
BASE='79cba2fc4882bc42282f8feee7257a69f7628072'
OLD='artifacts/pr_reference_interference_pilot/2026-11-06/'
OUT=ROOT/'artifacts/pr_reference_transfer_qualification/2026-11-06'
METRICS=('rz','cx','depth','size')
# Held-out registered schedule cells. No result-driven additions.
HOLDOUTS=((4,1),(32,1),(128,1),(256,2))
N=2


def data(path):
    return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    raw=lzma.decompress(data(OLD+'common.pkl.xz'))
    path=OUT/'common.pkl'
    if not path.exists():path.write_bytes(raw)
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(path)
    common()
    save_json(OUT/'INPUTS.json',dict(base=BASE,common_pickle_sha256=hashlib.sha256(raw).hexdigest(),
        original_input_provenance=json.loads(data(OLD+'INPUTS.json')),
        new_SCF=0,new_DF_fit=0,new_ground_solve=0,old_science_runner_called=False))
    save_json(OUT/'ENVIRONMENT.json',dict(host=os.uname().nodename,packages={p:version(p) for p in ('numpy','scipy','qiskit','mpmath')},
        affinity=len(os.sched_getaffinity(0)),meminfo=Path('/proc/meminfo').read_text(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text(),
        profile=PROFILE,runtime=os.sys.executable,shared_settings_changed=False))


def read_prior_prices():
    b=data(OLD+'raw_prices.tar.xz')
    with tarfile.open(fileobj=io.BytesIO(b),mode='r:xz') as t:
        return [json.load(t.extractfile(m)) for m in t.getmembers() if m.isfile()]


def frontier_depth(q):
    """Actual wire critical path, not the sum of component depths."""
    levels=[0]*(q.num_qubits+q.num_clbits)
    for inst in q.data:
        wires=[q.find_bit(x).index for x in inst.qubits]
        wires += [q.num_qubits+q.find_bit(x).index for x in inst.clbits]
        d=max((levels[x] for x in wires),default=0)+(inst.operation.name!='barrier')
        for x in wires:levels[x]=d
    return max(levels),levels


def stage1():
    p=common()['preps'][3];raw=read_prior_prices()
    dc=[]
    for k,b in enumerate(p.deterministic_blocks):
        prim=diagonal_primitives(b,.01)
        dc.append(dict(block=k,kind=b.block_kind,rz=len(prim.rz),rzz=len(prim.rzz),
            basis_operations=dict(Counter(g.name for g,_ in b.runtime_basis_operations))))
    tail=p.rte_preparation.symbolic_tail
    total=math.fsum(c.probability for c in tail.components);basis={}
    for c in tail.components:basis[c.basis_id]=basis.get(c.basis_id,0.)+c.probability/total
    switching=1-math.fsum(x*x for x in basis.values())
    prior=json.loads(data(OLD+'A3_PRICES.json'))
    fit={};naive={}
    for arm in ('Simon_Love','reference'):
        means={x['q']:np.array(x['means'][arm]) for x in prior['rows']}
        slope=(means[8]-means[2])/6;intercept=means[2]-2*slope
        fit[arm]=dict(intercept=intercept.tolist(),slope=slope.tolist(),
            qualification='diagnostic affine r1 model only; fitted anchors, not a structural certificate')
        prediction=7*means[2]-6*means[1]
        naive[arm]=dict(q8_prediction_from_q1_q2=prediction.tolist(),
            q8_observed=means[8].tolist(),residual=(means[8]-prediction).tolist())
    quantum=[];archive=data(OLD+'native_circuits.tar.xz')
    raw_by={(x['q'],x['rare'],x['j']):x for x in raw}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:xz') as t:
        for m in t.getmembers():
            if not m.isfile():continue
            name=Path(m.name).name
            qv=int(name.split('_')[0][1:]);rare=bool(int(name.split('_')[1][4:]))
            arm=name.split('_',2)[2][:-4]
            q=qpy.load(io.BytesIO(t.extractfile(m).read()))[0]
            if arm not in ('Simon_Love','reference'):continue
            d,levels=frontier_depth(q);actual=old.metric(q)
            assert actual==raw_by[(qv,rare,0)]['values'][arm][0]
            assert d==q.depth()
            quantum.append(dict(file=m.name,arm=arm,q=qv,rare=rare,actual_native_metrics=actual,
                wire_depth_frontier=levels,critical_path=d,source='saved QPY, no new compilation'))
    rounds=json.loads(data(OLD+'A4_RPE_ACCOUNTING.json'))['rows']
    structure=[]
    for row in rounds:
        q=row['q'];r=row['r'];tau=p.exact_rte_lambda_r*.02/r
        dist=old.finite_rte_distribution(tau,2);p2=dist.order_probabilities[1]
        structure.append(dict(m=row['m'],q=q,r=r,RTE_occurrences=q,events=q*r,
            D_boundary_fusions_baseline=q-1,D_boundary_fusions_reference=max(0,q-2),
            D_factor_calls_baseline=7*q+1,D_factor_calls_reference=8 if q==1 else 7*q+2,
            source_CSWAPs=16,reference_resets=8,extra_qubits=8,
            all_support_order2_count=[0,q*r],expected_order2=q*r*p2,
            expected_RTE_basis_runs=q*(1+(r*(1+2*p2)-1)*switching)))
    paired=[]
    for q in (1,2,8):
        cohort=[x for x in raw if x['q']==q]
        cx=[x['values']['reference'][0][1]-x['values']['Simon_Love'][0][1] for x in cohort]
        paired.append(dict(q=q,observed_paired_CX_min=min(cx),observed_paired_CX_max=max(cx),
            caveat='observed 16 paths; not an all-support minimum/maximum'))
    result=dict(status='PASS_TO_FINITE_HOLDOUTS',deterministic_primitives=dc,
        tail_components=len(tail.components),support_sizes=sorted(set(len(c.diagonal_pauli_support) for c in tail.components)),
        source_basis_probabilities=basis,source_switch_probability=switching,
        exact_source_structure=structure,actual_saved_DAG_checks=quantum,
        old_prices_reused=48,new_full_compiles=0,paired_CX_diagnostics=paired,
        diagnostic_affine_r1=fit,naive_q1_q2_extrapolation=naive,
        rigorous_native_price_model=False,
        missing='all contexts/cancellation interfaces and random critical paths; r2/4/8/16/32 transfer not certified',
        interpretation='D asymptotic structure and observed q8 CX/depth leave an opportunity. Source counts are not native prices. Stage2 must test the conditional model; no total cost is extrapolated.')
    save_json(OUT/'STAGE1_STRUCTURE.json',result)
    save_json(OUT/'STAGE2_PREDECLARED.json',dict(cells=[dict(q=q,r=r,n_per_stratum=N) for q,r in HOLDOUTS],
        strata=['no_order2','at_least_one_order2'],axes=['X','Y'],arms=['Simon_Love','reference'],
        seed_namespace='newA-transfer-fixed-20261106',paired=True,
        roles={'q4_r1':'missing short round / interpolation holdout','q32_r1':'out-of-anchor growth holdout',
               'q128_r1':'last r1 schedule round','q256_r2':'first r2 schedule round'},
        r1_model=fit,model_scope='diagnostic only, not certified; no r2 estimate filled from r1',
        model_rejection='unexplained context/cancellation or critical-path discrepancy prevents claiming all18 prices',
        results_seen=False,expansion_allowed=False,T_synthesis=False))
    return result


def seed(q,r,rare,j):
    return int.from_bytes(hashlib.sha256(f'newA-transfer-fixed-20261106:{q}:{r}:{rare}:{j}'.encode()).digest()[:8],'big')


def events_for(q,r,rare,j):
    p=common()['preps'][3];tail=p.rte_preparation.symbolic_tail
    tolerance=old.finite_rte_distribution(p.exact_rte_lambda_r*.02/r,2).step_truncation_residual_bound
    cfg,dist=make_rte_config(tail,evolution_time=.02,rte_steps=r,
        truncation_tolerance=tolerance,finite_taylor_order=2,seed=seed(q,r,rare,j))
    rng=np.random.default_rng(seed(q,r,rare,j))
    flags=old.conditional_flags([dist.order_probabilities[1]]*(q*r),rare,rng)
    probs=np.array([x.probability for x in tail.components]);probs/=probs.sum()
    events=[_make_event(rng.choice(len(probs),size=3 if flag else 1,p=probs),
        tail.components,dist,1 if flag else 0) for flag in flags]
    return cfg,dist,events


def assemble(q,r,cfg,dist,events):
    p=common()['preps'][3];rp=p.rte_preparation
    factory=QiskitDFRTEEventCircuitBuilder(basis_registry=rp.basis_registry)
    steps=[];tails=[];uncontrolled=[]
    for i in range(q):
        pair={}
        for ctrl in (False,True):
            req=DFRTEEventSequenceCircuitRequest(events=tuple(events[i*r:(i+1)*r]),
                component_specs=rp.component_specs,controlled=ctrl,ancilla_qubit=8 if ctrl else None,
                tail_id=rp.symbolic_tail.tail_id,tail_hash=rp.symbolic_tail.tail_hash,occurrence_rte_steps=r)
            pair[ctrl]=factory.build_sequence(req).circuit
            if not ctrl:
                steps.append(DFPartialS2StepRequest(preparation=p,step_time=.02,
                    rte_config=cfg,rte_distribution=dist,rte_occurrence=req,controlled=False,
                    ancilla_qubit=None,seed=cfg.seed))
        tails.append(pair[True]);uncontrolled.append(pair[False])
    if q==1:
        g=old.deterministic(.01).copy()
        g.global_phase+=-.02*(p.constant_coefficient+p.extracted_identity_coefficient)
        g.compose(uncontrolled[0],inplace=True);f=old.deterministic(.01,True)
    else:
        g=old.fused_steps(steps[:q//2]);f=old.fused_steps(steps[q//2:])
    return dict(F=f,G=g,baseline=old.symmetric_fused(tails),
        phase=old.vacuum_phase(f)+old.vacuum_phase(g),first_uncontrolled_RTE=uncontrolled[0],
        first_controlled_RTE=tails[0])


def adapter_checks():
    """Literal old-r1 reconstruction; one changed r2 occurrence only."""
    p=common()['preps'][3];comps=p.rte_preparation.symbolic_tail.components
    lookup={c.component_id:i for i,c in enumerate(comps)};checks=[]
    for saved in [x for x in read_prior_prices() if x['j']==0]:
        q=saved['q'];cfg,dist=make_rte_config(p.rte_preparation.symbolic_tail,
            evolution_time=.02,rte_steps=1,
            truncation_tolerance=old.finite_rte_distribution(p.exact_rte_lambda_r*.02,2).step_truncation_residual_bound,
            finite_taylor_order=2,seed=saved['seed'])
        ev=[]
        for rec in saved['event_records']:
            indices=[lookup[rec['rotation_component_id']]]+[lookup[x] for x in rec['product_component_ids']]
            ev.append(_make_event(indices,comps,dist,0 if rec['taylor_order']==0 else 1))
        a=assemble(q,1,cfg,dist,ev);b=old.trajectory(q,saved['rare'],0)
        same=bool(a['F']==b['F'] and a['G']==b['G'] and a['baseline']==b['symmetric'])
        checks.append(dict(q=q,rare=saved['rare'],literal_same_as_fixed_source=same))
    cfg,dist,ev=events_for(256,2,False,0)
    # The first local occurrence of a preregistered realization, not a new RPE case.
    rp=p.rte_preparation;factory=QiskitDFRTEEventCircuitBuilder(basis_registry=rp.basis_registry)
    pair={}
    for ctrl in (False,True):
        req=DFRTEEventSequenceCircuitRequest(events=tuple(ev[:2]),component_specs=rp.component_specs,
            controlled=ctrl,ancilla_qubit=8 if ctrl else None,
            tail_id=rp.symbolic_tail.tail_id,tail_hash=rp.symbolic_tail.tail_hash,occurrence_rte_steps=2)
        pair[ctrl]=factory.build_sequence(req).circuit
    u=Operator(pair[False]).data;a=Operator(pair[True]).data
    expected=np.zeros((512,512),complex);expected[:256,:256]=np.eye(256);expected[256:,256:]=u
    error=float(np.linalg.norm(a-expected))
    passed=all(x['literal_same_as_fixed_source'] for x in checks) and error<1e-9
    out=dict(old_r1_reconstructions=checks,r2_local_controlled_joint_operator_error=error,
        r2_scope='first local occurrence of registered q256/r2/common/j0; no new PF or case',
        all_passed=bool(passed),old_correctness_suite_rerun=False,new_native_price_calls=0)
    save_json(OUT/'ADAPTER_CHECKS.json',out)
    if not passed:raise ArithmeticError('Changed-r adapter mismatch')
    return out


def event_descriptor(event):
    return dict(order=event.taylor_order,rotation=event.rotation_component_id,products=list(event.product_component_ids),
        phase=[event.phase.real,event.phase.imag],angle=event.rotation_angle,
        probability=event.event_probability,normalization=event.event_normalization)


def sample(request):
    q,r,rare,j=request;start=time.monotonic()
    path=OUT/'raw'/f'q{q}_r{r}_rare{int(rare)}_{j}.json'
    if path.exists():raise RuntimeError('Duplicate registered sample')
    cfg,dist,events=events_for(q,r,rare,j);a=assemble(q,r,cfg,dist,events)
    core=old.reference_core(a['F'],a['G'],a['phase'])
    values={};traces={}
    for arm,body in [('Simon_Love',a['baseline']),('reference',core)]:
        values[arm]=[];traces[arm]={}
        for axis in ('X','Y'):
            trace=[]
            def cb(**kw):
                name=type(kw['pass_']).__name__
                if name in ('InverseCancellation','UnrollCustomDefinitions','BasisTranslator'):
                    trace.append(dict(pass_name=name,operations=dict(kw['dag'].count_ops())))
            native=compile_reference(old.wrapper(body,axis),callback=cb)
            if arm=='reference':
                full=QuantumCircuit(17,1);full.reset(range(8,16));full.compose(native,inplace=True);native=full
            values[arm].append(old.metric(native));traces[arm][axis]=trace
            # Small held-out native representative only; no huge QPY bundle.
            if q==4 and j==0 and axis=='X':
                d=OUT/'circuits';d.mkdir(exist_ok=True)
                with (d/f'q4_rare{int(rare)}_{arm}.qpy').open('wb') as stream:qpy.dump(native,stream)
    p_rare=-math.expm1(q*r*math.log1p(-dist.order_probabilities[1]))
    result=dict(q=q,r=r,rare=rare,j=j,seed=seed(q,r,rare,j),probability=p_rare if rare else 1-p_rare,
        values=values,traces=traces,phase=a['phase'],events=[event_descriptor(x) for x in events],
        source_CSWAPs=16,reference_resets=8,extra_qubits=8,T=None,
        wall_seconds=time.monotonic()-start)
    save_json(path,result);return result


def stage2_prices():
    if not json.loads((OUT/'STAGE1_STRUCTURE.json').read_text())['status']=='PASS_TO_FINITE_HOLDOUTS':
        raise RuntimeError('Stage1 not passed')
    if not json.loads((OUT/'ADAPTER_CHECKS.json').read_text())['all_passed']:
        raise RuntimeError('Adapter not passed')
    if (OUT/'raw').exists():raise RuntimeError('Cohort already started; no automatic reprice')
    requests=[(q,r,rare,j) for q,r in HOLDOUTS for rare in (False,True) for j in range(N)]
    workers=min(len(requests),len(os.sched_getaffinity(0)))
    save_json(OUT/'EXECUTION.json',dict(workers=workers,requests=requests,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preregistration_sha256=hashlib.sha256((OUT/'STAGE2_PREDECLARED.json').read_bytes()).hexdigest(),
        old_price_reacquisitions=0))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fs=[pool.submit(sample,x) for x in requests]
        for f in as_completed(fs):
            x=f.result();print('priced',x['q'],x['r'],x['rare'],x['j'],round(x['wall_seconds'],2),flush=True)
