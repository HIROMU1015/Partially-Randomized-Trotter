"""Fuse logical commuting diagonal words inside fixed source-basis runs.

Reuse the existing central-control emitter and Qiskit 1.3 GraySynth topology.
GraySynth receives T placeholders: its numeric-angle path uses modulo pi.
Walk its parity network and insert native RZ angles directly, without modulo.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
from functools import lru_cache
import json
import math
import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.synthesis import synth_cnot_phase_aam
from qiskit.quantum_info import Operator
from .track_c_h4 import ROOT,common,save_json,diagonal_primitives
from .track_c_joint_isometry import runs,catalogue,key,decode,circuit,configuration,source
from .track_c_joint_resources import initial_state
from .track_c_d_resources import metrics,split,d_piece
from .track_c_compiler_a import compile_reference
from trotterlib.h4_d2_pauli_control import _rotation
from trotterlib.df_rte_qiskit import _event_phase_angle

OUT=ROOT/'artifacts/track_c_diagonal_fusion/2026-11-08'
PRIOR=ROOT/'.venv/joint'
MODES=('reference','term_parity','merge','network')


@lru_cache(None)
def network(masks):
    """Return a verified identity-ending parity schedule from existing GraySynth."""
    active=sorted({q for mask in masks for q in range(9) if mask>>q&1});n=len(active)
    local=[sum(((mask>>q)&1)<<j for j,q in enumerate(active)) for mask in masks]
    cnots=[[(mask>>i)&1 for mask in local] for i in range(n)]
    template=synth_cnot_phase_aam(cnots,['t']*len(local),section_size=1)
    state=[1<<i for i in range(n)];emitted=[];seen=[]
    for inst in template.data:
        wires=[template.find_bit(q).index for q in inst.qubits]
        if inst.operation.name=='cx':
            a,b=wires;state[b]^=state[a];emitted.append(('cx',active[a],active[b]))
        else:
            assert inst.operation.name=='t'
            mask=sum(((state[wires[0]]>>j)&1)<<q for j,q in enumerate(active))
            emitted.append(('rz',active[wires[0]],mask));seen.append(mask)
    assert sorted(seen)==sorted(masks) and state==[1<<i for i in range(n)]
    return tuple(emitted)


def emit(q,rotations,products=(),*,mode='network'):
    """Append diag(I, product Z * product exp(-i angle Z_S/2)).

    All incoming angles are exact binary64 source parameters. Aggregate with
    rational arithmetic, round once, record a spectral error upper bound.
    No global phase from a signed product is inferred here: caller owns it.
    """
    rotations=[(tuple(s),float(a)) for s,a in rotations];products=[tuple(s) for s in products]
    error=Fraction(0);groups=defaultdict(Fraction);parity=set()
    for support,angle in rotations:groups[support]+=Fraction.from_float(angle)
    for support in products:
        for i in support:
            if i in parity:parity.remove(i)
            else:parity.add(i)
    if mode in ('term_parity','merge'):
        terms=rotations if mode=='term_parity' else [(s,float(a)) for s,a in sorted(groups.items()) if a]
        if mode=='merge':error=sum((abs(a-Fraction.from_float(float(a)))/2 for a in groups.values()),Fraction(0))
        temp=QuantumCircuit(9)
        for support,angle in terms:_rotation(temp,'Z'*len(support),angle,support,8)
        # Reuse CRZ's own exact RZ(a/2),CX,RZ(-a/2),CX definition. It avoids
        # context-dependent CRZZ/CCX lowering and introduces no new gate family.
        temp=temp.decompose(gates_to_decompose=['crz']);assert set(temp.count_ops())<={'rz','cx'}
        q.compose(temp,inplace=True)
    elif mode=='network':
        polynomial=defaultdict(Fraction)
        for support,angle in rotations:
            mask=sum(1<<i for i in support);value=Fraction.from_float(angle)/2
            polynomial[mask]+=value;polynomial[mask|(1<<8)]-=value
        polynomial={m:a for m,a in polynomial.items() if a}
        rounded={m:float(a) for m,a in polynomial.items()};error=sum((abs(a-Fraction.from_float(rounded[m]))/2 for m,a in polynomial.items()),Fraction(0))
        if polynomial:
            for op,a,b in network(tuple(sorted(polynomial))):
                if op=='cx':q.cx(a,b)
                else:q.rz(rounded[b],a)
    else:raise ValueError(mode)
    if mode=='term_parity':
        for support in products:
            for i in support:q.cz(8,i)
    else:
        for i in sorted(parity):q.cz(8,i)
    # nextafter prevents a downward float conversion of the exact error sum.
    upper=math.nextafter(float(error),math.inf) if error else 0.
    return dict(rotations=len(rotations),unique_rotations=len(groups),products=sum(map(len,products)),odd_product_wires=len(parity),rounding_error_upper=upper)


def diagonal_word(flat,mapping=None):
    rotations=[];products=[];phase=0.
    for e,a in flat:
        support=tuple(sorted(mapping[i] for i in a.diagonal_pauli_support)) if mapping is not None else a.diagonal_pauli_support
        if a.role=='product':
            products.append(support)
            if a.coefficient_sign==-1:phase+=math.pi
        else:rotations.append((support,2*e.unsigned_rotation_angle))
    return rotations,products,phase


def build(t,n,choices,mode):
    ev,flat,groups=runs(t,n,choices);q=QuantumCircuit(9);phase=sum(_event_phase_angle(e.phase) for e in ev);stats=[]
    for start,stop,basis,union in groups:
        entry=catalogue().get(key(basis,union));partial=entry is not None and entry['adopt']
        ops=decode(entry['operations']) if partial else common()['preps'][3].rte_preparation.basis_registry.definition(basis).runtime_operations
        mapping=dict(zip(entry['support'],entry['targets'])) if partial else None
        rotations,products,p=diagonal_word(flat[start:stop],mapping);phase+=p
        q.compose(circuit(ops,9,True),inplace=True);s=emit(q,rotations,products,mode=mode);q.compose(circuit(ops,9),inplace=True)
        stats.append(dict(start=start,stop=stop,basis=basis,columns=len(union),**s))
    if phase:q.p(phase,8)
    return q,stats


def d_fused(seq,ld,mode):
    q=QuantumCircuit(9);error=0.
    for kind,index,t in seq:
        assert kind=='D';block=common()['preps'][ld].deterministic_blocks[index];p=diagonal_primitives(block,t)
        rotations=[((i,),a) for i,a in p.rz]+[((i,j),a) for i,j,a in p.rzz if a!=0]
        q.compose(circuit(block.runtime_basis_operations,9,True),inplace=True)
        error+=emit(q,rotations,mode=mode)['rounding_error_upper']
        if p.global_phase:q.p(p.global_phase,8)
        q.compose(circuit(block.runtime_basis_operations,9),inplace=True)
    return q,error


def best(vectors):return min(vectors,key=lambda k:tuple(vectors[k]))


def training(path):
    r=json.loads(path.read_text());vectors={'reference':r['selected']};stats={}
    for mode in MODES[1:]:
        q,s=build(r['t'],r['n'],r['choices'],mode);vectors[mode]=metrics(compile_reference(q));stats[mode]=s
    chosen=best(vectors);row={**r,'baseline':r['selected'],'variants':vectors,'selected':vectors[chosen],'fusion_choice':chosen,'stats':stats['network']}
    save_json(OUT/'training_prices'/path.name,row);return row


def det(task):
    case,h=task;ld=12 if case['kind'].startswith('full_') else 3;g=split(case,h);groups={};errors={};variants={}
    for role,seq in g.items():
        if role=='full':continue
        vs={'reference':metrics(compile_reference(d_piece(tuple(seq),ld)))};ee={'reference':0.}
        for mode in MODES[1:]:
            q,e=d_fused(seq,ld,mode);vs[mode]=metrics(compile_reference(q));ee[mode]=e
        selected=best(vs);groups[role]=vs[selected];errors[role]=ee[selected];variants[role]=dict(metrics=vs,selected=selected)
    row=dict(case=case,h=h,groups=groups,errors=errors,variants=variants)
    save_json(OUT/'det_prices'/f'{case["name"]}_{h}.json',row);return row


def representative(task):
    path,start,stop=task;r=json.loads(path.read_text());_,flat,groups=runs(r['t'],r['n'],r['choices'])
    _,_,basis,union=next(g for g in groups if g[:2]==(start,stop));entry=catalogue().get(key(basis,union));partial=entry is not None and entry['adopt']
    ops=decode(entry['operations']) if partial else common()['preps'][3].rte_preparation.basis_registry.definition(basis).runtime_operations
    mapping=dict(zip(entry['support'],entry['targets'])) if partial else None
    from .track_c_joint_isometry import append_run
    old=QuantumCircuit(9);p=append_run(old,flat[start:stop],ops,mapping);old.p(p+math.pi,8)
    target=Operator(old).data;rows=[]
    for mode in MODES[1:]:
        rotations,products,phase=diagonal_word(flat[start:stop],mapping);q=QuantumCircuit(9);q.compose(circuit(ops,9,True),inplace=True)
        s=emit(q,rotations,products,mode=mode);q.p(phase+math.pi,8);q.compose(circuit(ops,9),inplace=True)
        native=compile_reference(q);a,b=Operator(q).data,Operator(native).data;g=np.vdot(a,b);g/=abs(g)
        error=float(np.linalg.norm(target-a,'fro'));native_error=float(np.linalg.norm(a-b/g,'fro'));assert max(error,native_error)<1e-9
        from .track_c_h4_price import wrapper
        from qiskit.quantum_info import Statevector
        means=[]
        for axis in ('X','Y'):
            vv=[Statevector(initial_state()).evolve(wrapper([v],axis).remove_final_measurements(inplace=False)).data for v in (old,native)]
            f=lambda x:float(np.vdot(x[:256],x[:256]).real-np.vdot(x[256:],x[256:]).real)
            means.append(abs(f(vv[0])-f(vv[1])))
        assert max(means)<1e-9
        rows.append(dict(mode=mode,source_Fro_error=error,native_projective_Fro_error=native_error,XY_errors=means,metrics=metrics(native),**s))
    return dict(file=path.name,start=start,stop=stop,reference=metrics(compile_reference(old)),rows=rows)


def pilot():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'training_prices').mkdir();(OUT/'det_prices').mkdir();configuration();catalogue()
    paths=sorted((PRIOR/'holdout_prices').glob('*.json'));checks=[];prices=[];dets=[]
    # Use previously selected representative runs; re-use their trajectories.
    reps=json.loads((PRIOR/'REPRESENTATIVES.json').read_text())['rows']
    # Those representatives reference the bottleneck dataset, not joint holdout.
    tasks=[(ROOT/'.venv/bottleneck/holdout_prices'/r['file'],r['start'],r['stop']) for r in reps]
    previous=json.loads((ROOT/'artifacts/track_c_joint_isometry/2026-11-08/POLICY.json').read_text())
    policies=previous['policies']+[x for x in previous['old_policies'] if x['family']=='Full']
    cells={(s['signal']['name'],s['signal']['h']):(s['signal']['case'],s['signal']['h']) for p in policies for s in p['selected']}
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(representative,t) for t in tasks):checks.append(f.result())
        save_json(OUT/'CORRECTNESS.json',dict(source_commit=source(),rows=checks))
        for f in as_completed(pool.submit(training,p) for p in paths):prices.append(f.result())
        for f in as_completed(pool.submit(det,t) for t in cells.values()):dets.append(f.result())
    save_json(OUT/'PILOT.json',dict(source_commit=source(),training_samples=len(prices),deterministic_cells=len(dets),
        methodology='Previous independent joint-isometry holdout is training here. New independent holdout required. Same logical run boundaries and frozen isometry catalogue.',
        ablations='reference includes all prior compiler cancellation and isometry; term_parity reuses known central-control code; merge adds equal-string sums/CZ parity cancellation; network reuses GraySynth for joint parities.'))
    print('DIAGONAL_PILOT_COMPLETE',len(prices),len(dets),flush=True)


if __name__=='__main__':pilot()
