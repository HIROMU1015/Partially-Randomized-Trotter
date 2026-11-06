"""Fixed new-A B-prime validation; historical SOURCE/results remain read-only."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib, io, json, lzma, math, os, platform, subprocess, sys, tarfile
from importlib.metadata import distribution, version
import numpy as np
from collections import Counter
import time
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator
from .track_c_h4 import ROOT, common, save_json, diagonal_primitives
from . import pr_reference_interference_pilot as pilot
from . import pr_reference_transfer_qualification as transfer
from .track_c_compiler_a import compile_reference, PROFILE
from .pr_reference_transfer_catalogue import chunks, fence
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
from trotterlib.rte import make_rte_config, _make_event

BASE='da9fceb944dae27f397e48734be30c6864e15f44'
OLD='artifacts/pr_reference_transfer_qualification/2026-11-06/'
PILOT='artifacts/pr_reference_interference_pilot/2026-11-06/'
OUT=ROOT/'artifacts/pr_reference_bprime/2026-11-07'

def data(path):
    return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT)

def initialize():
    if OUT.exists():raise RuntimeError('Fresh B-prime output required')
    OUT.mkdir(parents=True)
    raw=lzma.decompress(data(PILOT+'common.pkl.xz'))
    (OUT/'common.pkl').write_bytes(raw)
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(OUT/'common.pkl')
    save_json(OUT/'INPUTS.json',dict(base=BASE,common_sha256=hashlib.sha256(raw).hexdigest(),
        inherited=json.loads(data(OLD+'INPUTS.json')),new_SCF_DF_ground_solves=0))
    wheel=distribution('qiskit')
    save_json(OUT/'ENVIRONMENT.json',dict(host=platform.node(),platform=platform.platform(),
        machine=platform.machine(),libc=platform.libc_ver(),python=sys.version,executable=sys.executable,
        byteorder=sys.byteorder,float_info=str(sys.float_info),qiskit_WHEEL=wheel.read_text('WHEEL'),
        qiskit_metadata=wheel.metadata.get('Version'),packages={p:version(p) for p in ('qiskit','numpy','scipy','mpmath')},
        affinity=len(os.sched_getaffinity(0)),memory=Path('/proc/meminfo').read_text(),
        cpu_pressure=Path('/proc/pressure/cpu').read_text(),memory_pressure=Path('/proc/pressure/memory').read_text(),
        disk=tuple(os.statvfs(OUT)),profile=PROFILE,shared_settings_changed=False))

def attach_cache():
    if (OUT/'STOP.json').exists():raise RuntimeError('Mandatory STOP; no further scientific execution')
    os.environ['TRACK_C_H4_COMMON_CACHE']=str(OUT/'common.pkl')

@lru_cache(None)
def saved_samples():
    rows=[]
    for path in (PILOT+'raw_prices.tar.xz',OLD+'raw_prices.tar.xz'):
        with tarfile.open(fileobj=io.BytesIO(data(path)),mode='r:xz') as t:
            for m in t.getmembers():
                if m.isfile():
                    row=json.load(t.extractfile(m));row.setdefault('r',1)
                    row['origin_archive']=path;rows.append(row)
    assert len(rows)==64
    return rows

def reconstructed(row):
    p=common()['preps'][3];tail=p.rte_preparation.symbolic_tail;r=row['r']
    cfg,dist=make_rte_config(tail,evolution_time=.02,rte_steps=r,
        truncation_tolerance=pilot.finite_rte_distribution(p.exact_rte_lambda_r*.02/r,2).step_truncation_residual_bound,
        finite_taylor_order=2,seed=row['seed'])
    lookup={c.component_id:i for i,c in enumerate(tail.components)}
    events=[]
    for e in row.get('events',row.get('event_records')):
        ids=[e.get('rotation',e.get('rotation_component_id')),*e.get('products',e.get('product_component_ids'))]
        order=e.get('order',e.get('taylor_order'))
        event=_make_event([lookup[c] for c in ids],tail.components,dist,0 if order==0 else 1)
        if 'angle' in e:assert event.rotation_angle==e['angle']
        events.append(event)
    return cfg,dist,events

def rewrite_rte(tail):
    """Only central controlled-RZZ nodes of a supplied RTE occurrence."""
    q=tail.copy_empty_like();replaced=[]
    for inst in tail.data:
        op=inst.operation;qs=[tail.find_bit(x).index for x in inst.qubits]
        if op.name=='crzz':
            assert op.num_ctrl_qubits==1 and op.base_gate.name=='rzz' and op.ctrl_state==1
            c,a,b=qs;theta=float(op.params[0])
            q.cx(a,b);q.crz(theta,c,b);q.cx(a,b)
            replaced.append(dict(wires=qs,theta=theta))
        else:q.append(op,qs,[tail.find_bit(x).index for x in inst.clbits])
    assert q.global_phase==tail.global_phase
    return q,replaced

def build_saved(row):
    cfg,dist,events=reconstructed(row);p=common()['preps'][3];rp=p.rte_preparation
    builder=QiskitDFRTEEventCircuitBuilder(basis_registry=rp.basis_registry)
    tails=[];changes=[]
    for i in range(row['q']):
        req=DFRTEEventSequenceCircuitRequest(events=tuple(events[i*row['r']:(i+1)*row['r']]),
            component_specs=rp.component_specs,controlled=True,ancilla_qubit=8,
            tail_id=rp.symbolic_tail.tail_id,tail_hash=rp.symbolic_tail.tail_hash,occurrence_rte_steps=row['r'])
        tail=builder.build_sequence(req).circuit
        q,replacements=rewrite_rte(tail);tails.append(q);changes.extend(replacements)
    return pilot.symmetric_fused(tails),events,changes

def v2_inputs_and_constants():
    """Check fixed structural ceilings; never increase historical constants."""
    p=common()['preps'][3];tail=p.rte_preparation.symbolic_tail
    diagonal=[]
    for i,b in enumerate(p.deterministic_blocks):
        prim=diagonal_primitives(b,.01)
        basis_rz=sum(g.name=='rz' for g,_ in b.runtime_basis_operations)
        diagonal.append(dict(block=i,basis_RZ=basis_rz,diagonal_RZ_RZZ=len(prim.rz)+len(prim.rzz),
            full_frame_RZ=2*basis_rz+len(prim.rz)+len(prim.rzz)))
    d0=diagonal[0]['full_frame_RZ'];slope=d0+2*sum(x['full_frame_RZ'] for x in diagonal[1:])
    tail_basis=max(sum(g.name=='rz' for g,_ in d.runtime_operations) for d in p.rte_preparation.basis_registry._definitions.values())
    # Boundary-optimized lane: 7n+1 deterministic append_diagonal calls,
    # n RTE compositions, one initial aggregated scalar update. q1 is covered
    # by the same n=1 envelope; temporary unused additive circuits do not
    # contribute to the resulting lane global phase.
    passed=slope<=336 and d0<=24 and 6*tail_basis+1<=49 and 9<=20
    angles=[]
    for r in (1,2,4,8,16,32):
        for order in (0,2):
            cfg,dist=make_rte_config(tail,evolution_time=.02,rte_steps=r,
                truncation_tolerance=pilot.finite_rte_distribution(p.exact_rte_lambda_r*.02/r,2).step_truncation_residual_bound,
                finite_taylor_order=2)
            theta_actual=2*math.atan(dist.dimensionless_step_time/(order+1))
            theta_old=2*math.atan(p.exact_rte_lambda_r*.02/r/(order+1))
            angles.append(dict(r=r,order=order,actual_hex=theta_actual.hex(),old_interval_hex=theta_old.hex(),
                absolute_angle_difference=abs(theta_actual-theta_old),exact_equal=theta_actual==theta_old))
    result=dict(status='STRUCTURAL_CONSTANTS_PASS' if passed else 'UNQUALIFIED_CONSTANTS_BROKEN',
        deterministic=diagonal,D_slope=slope,D_endpoint=d0,tail_basis_RZ_max=tail_basis,
        per_event_RZ_bound=6*tail_basis+1,global_phase_updates_per_lane='(7n+1) D updates + n compose updates + 1 scalar = 8n+2 <=20n',
        constants_unchanged=[336,49,24,20],lambda_prep_hex=p.exact_rte_lambda_r.hex(),
        lambda_tail_hex=tail.lambda_r.hex(),angles=angles,all_angles_equal=all(x['exact_equal'] for x in angles))
    save_json(OUT/'V2_INPUTS_CONSTANTS.json',result)
    if not passed:raise ArithmeticError('Mandatory STOP: historical structural constants fail')
    return result

def bprime_kernel(theta):
    q=QuantumCircuit(3);q.cx(1,2);q.crz(theta,0,2);q.cx(1,2)
    return q

def correctness():
    from qiskit.circuit.library import RZZGate
    p=common()['preps'][3];rows=[]
    for r in (1,2,4,8,16,32):
        _,dist=make_rte_config(p.rte_preparation.symbolic_tail,evolution_time=.02,rte_steps=r,
            finite_taylor_order=2,truncation_tolerance=pilot.finite_rte_distribution(p.exact_rte_lambda_r*.02/r,2).step_truncation_residual_bound)
        for order in (0,2):
            for sign in (-1,1):
                theta=sign*2*math.atan(dist.dimensionless_step_time/(order+1));q=bprime_kernel(theta)
                target=Operator(RZZGate(theta).control()).data;actual=Operator(q).data
                inverse=Operator(q.inverse()).data
                err=float(np.linalg.norm(actual-target));inv=float(np.linalg.norm(inverse-target.conj().T))
                native=compile_reference(q);ops=dict(native.count_ops())
                expected=[['cx',[1,2],[]],['rz',[2],[theta/2]],['cx',[0,2],[]],['rz',[2],[-theta/2]],['cx',[0,2],[]],['cx',[1,2],[]]]
                lowered=instructions(native)
                passed=err<1e-13 and inv<1e-13 and ops=={'cx':4,'rz':2} and lowered==expected
                rows.append(dict(r=r,order=order,sign=sign,theta=theta,theta_hex=theta.hex(),
                    joint_error_no_phase_fit=err,inverse_error_no_phase_fit=inv,ops=ops,lowering=lowered,passed=passed))
    # Reuse the changed-r occurrence from the previous registered q256 path.
    row=next(x for x in saved_samples() if x['q']==256 and not x['rare'] and x['j']==0)
    _,_,events=reconstructed(row);rp=p.rte_preparation
    builder=QiskitDFRTEEventCircuitBuilder(basis_registry=rp.basis_registry);pair={}
    for ctrl in (False,True):
        req=DFRTEEventSequenceCircuitRequest(events=tuple(events[:2]),component_specs=rp.component_specs,
            controlled=ctrl,ancilla_qubit=8 if ctrl else None,tail_id=rp.symbolic_tail.tail_id,
            tail_hash=rp.symbolic_tail.tail_hash,occurrence_rte_steps=2)
        pair[ctrl]=builder.build_sequence(req).circuit
    bp,changes=rewrite_rte(pair[True]);u=Operator(pair[False]).data;a=Operator(bp).data
    goal=np.zeros((512,512),complex);goal[:256,:256]=np.eye(256);goal[256:,256:]=u
    joint=float(np.linalg.norm(a-goal));inverse=float(np.linalg.norm(Operator(bp.inverse()).data-goal.conj().T))
    passed=all(x['passed'] for x in rows) and joint<1e-9 and inverse<1e-9
    result=dict(status='PASS' if passed else 'UNQUALIFIED',local_angles=rows,
        r2_occurrence_joint_error=joint,r2_actual_inverse_error=inverse,replacements=changes,
        independent_branch_phase_fits=False,old_suite_rerun=False,passed=passed)
    save_json(OUT/'CORRECTNESS.json',result)
    if not passed:raise ArithmeticError('Mandatory STOP: B-prime correctness failed')
    return result

def instructions(q):
    return [[i.operation.name,[q.find_bit(x).index for x in i.qubits],
        [str(v) if getattr(v,'parameters',None) else float(v) for v in i.operation.params]] for i in q.data]

def catalogue_one(request):
    rare,y=request;q=QuantumCircuit(9);tags=[]
    for name,c,meta in chunks():
        if meta.get('family') in ('crzz','cswap','z'):continue
        if not rare and meta.get('family')=='cz':continue
        if not y and meta.get('family')=='sdg':continue
        q.compose(c,qubits=range(c.num_qubits),inplace=True);q.barrier();tags.append((name,c.num_qubits,meta))
    for rec in json.loads((OUT/'CORRECTNESS.json').read_text())['local_angles']:
        c=bprime_kernel(rec['theta']);name='bprime:'+rec['theta_hex']
        q.compose(c,qubits=range(3),inplace=True);q.barrier();tags.append((name,3,dict(kind='bprime',theta=rec['theta'])))
    from qiskit.circuit import Parameter
    q.p(Parameter('symbolic_correction'),0);q.barrier();tags.append(('p_symbolic',1,dict(kind='symbolic')))
    traces={}
    def callback(**kw):
        name=type(kw['pass_']).__name__
        if name=='UnrollCustomDefinitions':traces['post_unroll']=sorted(set(kw['dag'].count_ops())-{'barrier'})
        if name=='BasisTranslator':
            dag=kw['dag'];slices=[];current=[]
            for node in dag.topological_op_nodes():
                if node.op.name=='barrier':slices.append(current);current=[]
                else:current.append([node.op.name,[dag.find_bit(z).index for z in node.qargs],[]])
            assert len(slices)==len(tags) and not current
            traces['pre_last_cancel']=[fence(s,w) for s,(_,w,_) in zip(slices,tags)]
    native=compile_reference(q,callback=callback);slices=[];current=[]
    for i in instructions(native):
        if i[0]=='barrier':slices.append(current);current=[]
        else:current.append(i)
    assert len(slices)==len(tags) and not current
    records=[]
    for (name,n,meta),items in zip(tags,slices):
        count={}
        for op,_,_ in items:count[op]=count.get(op,0)+1
        records.append(dict(name=name,n=n,meta=meta,ops=count,lowering=items,fence=fence(items,n)))
    result=dict(request=request,records=records,**traces)
    save_json(OUT/'catalogue'/f'rare{int(rare)}_Y{int(y)}.json',result)
    return result

def local_catalogue():
    requests=[(z,y) for z in (False,True) for y in (False,True)]
    save_json(OUT/'LOCAL_SCOPE.json',dict(contexts=requests,
        kernel_angles='24 saved r/order/sign angles; no new candidates',
        whole_cohort='64 saved paths x X/Y; only B-prime repriced',new_sampling=False))
    chunks()
    with ProcessPoolExecutor(max_workers=min(len(requests),len(os.sched_getaffinity(0)))) as pool:
        rows=list(pool.map(catalogue_one,requests))
    for row in rows:
        for rec,pre in zip(row['records'],row['pre_last_cancel']):
            if rec['meta']['kind']=='bprime':assert rec['ops']=={'cx':4,'rz':2}
            if rec['meta']['kind'] in ('basis','basis_inverse','baseline_D'):
                assert not pre['exposed_cancelable_prefix'] and not pre['exposed_cancelable_suffix']
    save_json(OUT/'CATALOGUE_SUMMARY.json',dict(contexts=4,frame_D_both_sides_fenced=True,kernel_exact_CX=4,kernel_exact_RZ=2,
        kernel_prefix_suffix_exposed=True,bprime_requires_new_cancellation_bound=True))
    return rows

def params():
    rows=[json.loads(p.read_text()) for p in sorted((OUT/'catalogue').glob('*.json'))]
    assert len(rows)==4
    records=rows[0]['records'];c={x['name']:x['ops'] for x in records}
    fixed={k:c[k].get('cx',0) for k in ('left_all','left_no0','right_all','right_no0','boundary_control')}
    p=common()['preps'][3];comp={x.component_id:x for x in p.rte_preparation.symbolic_tail.components}
    basis={v.basis_id:c['basis:'+v.basis_id].get('cx',0)+c['basis_inverse:'+v.basis_id].get('cx',0) for v in comp.values()}
    ref={k:c[f'block:{k}:half'].get('cx',0) for k in range(4)}
    return fixed,basis,ref,c['block:0:full'].get('cx',0),comp

def path_bounds(row,events):
    from .pr_reference_transfer_cost_bounds import path_prices
    descriptors=[dict(rotation=e.rotation_component_id,products=e.product_component_ids) for e in events]
    old_b,ref=path_prices(row['q'],row['r'],descriptors,params())
    comp=params()[-1];zz=sum(len(comp[e.rotation_component_id].diagonal_pauli_support)==2 for e in events)
    return dict(old_naive=old_b,bprime_CX_upper=old_b-10*zz,bprime_CX_lower=old_b-12*zz,
        reference_CX_upper=ref,ZZ_rotations=zz,
        lower_rule='subtract ALL two parity CX per ZZ rotation; protect native CRZ/D/frame/product cores')

def price_one(index):
    row=saved_samples()[index];start=time.monotonic();body,events,changes=build_saved(row)
    predicted=path_bounds(row,events);values=[];traces={}
    for axis in ('X','Y'):
        source=pilot.wrapper(body,axis);trace=[];before={};pretranslate={}
        def cx_nodes(dag):
            return {node._node_id:[dag.find_bit(q).index for q in node.qargs] for node in dag.op_nodes() if node.op.name=='cx'}
        def callback(**kw):
            dag=kw['dag'];name=type(kw['pass_']).__name__
            if name=='RejectOpaqueUnitary' and not before:
                before.update(cx_nodes(dag))
            rec=dict(pass_name=name,ops=dict(dag.count_ops()))
            if name=='InverseCancellation':
                pre=pretranslate if pretranslate else before
                after=cx_nodes(dag);removed=[dict(node_id=k,wires=v) for k,v in pre.items() if k not in after]
                rec['removed_CX_nodes']=removed
                rec['removed_by_wire']={str(k):v for k,v in Counter(tuple(x['wires']) for x in removed).items()}
                if not pretranslate:
                    rec['parity_CX_removed']=sum(all(q<8 for q in x['wires']) for x in removed)
            if name=='BasisTranslator':pretranslate.update(cx_nodes(dag))
            if name in ('InverseCancellation','UnrollCustomDefinitions','BasisTranslator'):trace.append(rec)
        native=compile_reference(source,callback=callback)
        metric=pilot.metric(native);values.append(metric);traces[axis]=trace
        assert predicted['bprime_CX_lower']<=metric[1]<=predicted['bprime_CX_upper']
        if row['q']==4 and row['j']==0 and axis=='X':
            dest=OUT/'circuits';dest.mkdir(exist_ok=True)
            with (dest/f'q4_rare{int(row["rare"])}_Bprime.qpy').open('wb') as f:qpy.dump(native,f)
    result=dict(q=row['q'],r=row['r'],rare=row['rare'],j=row['j'],seed=row['seed'],
        origin_archive=row['origin_archive'],probability=row['probability'],Bprime_values=values,
        saved_values=row['values'],traces=traces,structural_bounds=predicted,
        replacements=len(changes),events_sha256=hashlib.sha256(json.dumps(row.get('events',row.get('event_records')),sort_keys=True).encode()).hexdigest(),
        wall_seconds=time.monotonic()-start)
    save_json(OUT/'raw'/f'path{index:02d}.json',result)
    return index,row['q'],row['r'],result['wall_seconds']

def whole_prices():
    if (OUT/'raw').exists():raise RuntimeError('Cohort already started; no automatic duplicate compile')
    assert json.loads((OUT/'CORRECTNESS.json').read_text())['passed']
    assert json.loads((OUT/'CATALOGUE_SUMMARY.json').read_text())['frame_D_both_sides_fenced']
    tasks=list(range(len(saved_samples())));workers=min(len(tasks),len(os.sched_getaffinity(0)))
    save_json(OUT/'PRICE_EXECUTION.json',dict(tasks=tasks,workers=workers,whole_compiles=128,
        old_reference_and_baseline_recompiles=0,new_samples=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(price_one,i) for i in tasks]):print('priced Bprime',f.result(),flush=True)

def boundary_checks():
    theta=json.loads((OUT/'CORRECTNESS.json').read_text())['local_angles'][0]['theta']
    requests=[('same',None,(0,1)),('overlap_a',None,(0,2)),('overlap_b',None,(2,1)),
        ('disjoint',None,(2,3)),('same_CZ_block','cz_block',(0,1)),
        ('same_CZ_disjoint','cz_disjoint',(0,1)),('same_P','p',(0,1)),
        ('same_CRZ_block','crz_block',(0,1)),('same_CRZ_disjoint','crz_disjoint',(0,1))]
    results=[]
    for name,middle,pair in requests:
        q=QuantumCircuit(9);q.compose(bprime_kernel(theta),[8,0,1],inplace=True)
        if middle=='cz_block':q.cz(8,1)
        if middle=='cz_disjoint':q.cz(8,3)
        if middle=='p':q.p(math.pi,8)
        if middle=='crz_block':q.crz(theta,8,1)
        if middle=='crz_disjoint':q.crz(theta,8,3)
        q.compose(bprime_kernel(-theta),[8,*pair],inplace=True)
        trace=[]
        def cb(**kw):
            if type(kw['pass_']).__name__ in ('InverseCancellation','BasisTranslator'):
                trace.append(dict(stage=type(kw['pass_']).__name__,ops=dict(kw['dag'].count_ops())))
        native=compile_reference(q,callback=cb)
        # The two protected CRZ kernels retain four control CX altogether.
        controls=sum(i.operation.name=='cx' and native.find_bit(i.qubits[0]).index==8 for i in native.data)
        assert controls>=4
        results.append(dict(name=name,source=instructions(q),native=instructions(native),trace=trace,
            protected_control_CX=controls))
    oldrows=[]
    with tarfile.open(fileobj=io.BytesIO(data(OLD+'local_catalogue_boundary.tar.xz')),mode='r:xz') as t:
        oldrows=[json.load(t.extractfile(m)) for m in t.getmembers() if m.isfile()]
    unchanged=[]
    for p in (OUT/'catalogue').glob('*.json'):
        row=json.loads(p.read_text());z,y=row['request']
        old=next(x for x in oldrows if x['request']==['Simon_Love',False,z,y])
        before={r['name']:r for r in old['records']}
        for r in row['records']:
            if r['meta']['kind'] in ('bprime','symbolic'):continue
            assert r['lowering']==before[r['name']]['lowering'],(p,r['name'])
            unchanged.append([p.name,r['name']])
    save_json(OUT/'BOUNDARY_CHECKS.json',dict(cases=results,unchanged_inherited_local_lowerings=unchanged,
        empirical_only='boundary examples test implementation; all-support lower bound discards ALL exposed parity CX',
        lower_bound='D/frame chunks fenced on both sides; CRZ CXs protected by RZ target predecessors; CZ/P have no exposed CX prefix; only parity CX may disappear across chunks',
        passed=True))
    return results
