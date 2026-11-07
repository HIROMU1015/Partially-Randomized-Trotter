"""Joint column isometries inside maximal same-source-basis RTE runs.

Reuse stable partial QR and the production central-application builder. The
legacy basis-plan API keeps support labels fixed; this adapter explicitly
maps the union's original columns to leading modes. Logical events stay intact.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, XXPlusYYGate
from qiskit.quantum_info import Operator, Statevector
from .track_c_h4 import ROOT, common, save_json
from .track_c_bottleneck_basis import partial_ops, circuit, configuration, basis_price
from .track_c_bottleneck_runwise import events, build_singletons
from .track_c_d_resources import metrics, r_part, r_key
from .track_c_compiler_a import compile_reference
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder as Builder, _event_phase_angle

OUT=ROOT/'artifacts/track_c_joint_isometry/2026-11-08'
PRIOR=ROOT/'.venv/bottleneck'
PUBLIC=ROOT/'artifacts/track_c_bottleneck/2026-11-08'


def source(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()


def key(basis, support): return basis+'|'+','.join(map(str,support))


def runs(t,n,choices):
    ev=events(t,n,choices); flat=[(e,a) for e in ev for a in e.application_sequence]
    result=[];cursor=0
    while cursor<len(flat):
        stop=cursor+1;basis=flat[cursor][1].basis_id
        while stop<len(flat) and flat[stop][1].basis_id==basis:stop+=1
        union=tuple(sorted({i for _,a in flat[cursor:stop] for i in a.diagonal_pauli_support}))
        result.append((cursor,stop,basis,union));cursor=stop
    return ev,flat,result


@lru_cache(None)
def operations(basis,support):
    b=configuration()[3][basis];idx=[1<<i for i in range(8)]
    u=b[np.ix_(idx,idx)]/b[0,0]
    return partial_ops(u,support)


def encode(ops):return [[g.name,list(w),[float(x) for x in g.params]] for g,w in ops]


def decode(ops):
    return tuple((RZGate(*p) if name=='rz' else XXPlusYYGate(*p),tuple(w)) for name,w,p in ops)


def candidate(task):
    basis,support=task;ops=operations(basis,tuple(support))
    after=sum((np.array(metrics(compile_reference(circuit(ops,9,inverse)))) for inverse in (True,False)),np.zeros(4))
    before=basis_price(basis,True)+basis_price(basis,False)
    return dict(key=key(basis,support),basis=basis,support=list(support),operations=encode(ops),
                full_basis_pair=before.tolist(),joint_basis_pair=after.tolist(),adopt=bool(after[0]<before[0]),
                reason='Strictly lower actual isolated native RZ; all other metrics retained. No column cutoff or per-event split.')


@lru_cache(None)
def catalogue():
    path=OUT/'CATALOGUE_FINAL.json'
    return json.loads((path if path.exists() else OUT/'CATALOGUE.json').read_text())['entries']


def append_run(q,flat,ops,mapping=None):
    q.compose(circuit(ops,9,True),inplace=True);phase=0.
    for e,a in flat:
        used=a if mapping is None else replace(a,diagonal_pauli_support=tuple(mapping[i] for i in a.diagonal_pauli_support))
        phase+=Builder._append_central_application(q,e,used,controlled=True,ancilla_qubit=8)
    q.compose(circuit(ops,9),inplace=True);return phase


def build_joint(t,n,choices):
    ev,flat,groups=runs(t,n,choices);q=QuantumCircuit(9);phase=sum(_event_phase_angle(e.phase) for e in ev);used=[]
    for start,stop,basis,union in groups:
        entry=catalogue().get(key(basis,union)); partial=entry is not None and entry['adopt']
        ops=decode(entry['operations']) if partial else common()['preps'][3].rte_preparation.basis_registry.definition(basis).runtime_operations
        mapping=dict(zip(entry['support'],entry.get('targets',range(len(union))))) if partial else None
        phase+=append_run(q,flat[start:stop],ops,mapping)
        used.append(dict(start=start,stop=stop,key=key(basis,union),joint=bool(partial)))
    if phase:q.p(phase,8)
    return q,used


def scan(path):
    r=json.loads(path.read_text());_,flat,groups=runs(r['t'],r['n'],r['choices'])
    rows=[]
    for start,stop,basis,union in groups:
        rows.append(dict(length=stop-start,key=key(basis,union),basis=basis,support=list(union),
                         product_count=sum(a.role=='product' for _,a in flat[start:stop])))
    return dict(file=path.name,t=r['t'],n=r['n'],rare=r['rare'],j=r['j'],probability=r['probability'],runs=rows)


def fock_check(entry):
    basis=entry['basis'];support=entry['support'];b=configuration()[3][basis];v=Operator(circuit(decode(entry['operations']))).data
    states=np.arange(256);errors=[]
    for j,i in zip(entry.get('targets',range(len(support))),support):
        old=(b*(1-2*((states>>i)&1)))@b.conj().T
        new=(v*(1-2*((states>>j)&1)))@v.conj().T
        errors.append(float(np.linalg.norm(old-new,'fro')))
    unitary=float(np.linalg.norm(v.conj().T@v-np.eye(256),'fro'))
    assert max(errors+[unitary])<1e-10
    # Every requested ZZ is a product of these verified full-Fock involutions.
    return dict(key=entry['key'],Z_full_Fock_Fro_errors=errors,unitarity_Fro_error=unitary,
                proof='Every individual occupation projector is preserved; Z and ZZ products telescope on full Fock space. No subspace-only identification.')


def price_saved(path):
    r=json.loads(path.read_text());start=time.monotonic();q,used=build_joint(r['t'],r['n'],r['choices']);proposal=metrics(compile_reference(q))
    # The old full/singleton choice is retained as a whole-circuit fallback.
    selected=proposal if proposal[0]<r['selected'][0] else r['selected']
    result={**r,'baseline':r['selected'],'joint_proposal':proposal,'selected':selected,'choice':'joint' if selected is proposal else 'baseline',
            'used_runs':used,'wall_seconds':time.monotonic()-start}
    save_json(OUT/'training_prices'/path.name,result);return result


def pilot():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'training_prices').mkdir();scans=[];entries=[];checks=[];prices=[]
    configuration()  # Share immutable basis matrices and singleton QR cache on fork.
    paths=sorted((PRIOR/'holdout_prices').glob('*.json'))
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(scan,p) for p in paths):scans.append(f.result())
        subsets={(r['basis'],tuple(r['support'])) for s in scans for r in s['runs']}
        # Include every singleton's known support, so unseen singleton draws
        # retain their old implementation. Other unseen unions use full basis.
        subsets.update((s.basis_id,tuple(s.diagonal_pauli_support)) for s in common()['preps'][3].rte_preparation.component_specs)
        for f in as_completed(pool.submit(candidate,t) for t in sorted(subsets)):entries.append(f.result())
        save_json(OUT/'CATALOGUE.json',dict(source_commit=source(),entries={r['key']:r for r in entries},
             rule='Frozen observed union sets plus all singleton supports; adopt only strictly cheaper native basis RZ; unseen union full fallback. No holdout-based catalogue extension.'))
        for f in as_completed(pool.submit(fock_check,r) for r in entries if r['adopt']):checks.append(f.result())
    # Fork again after writing the catalogue: no stale worker cache.
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(price_saved,p) for p in paths):prices.append(f.result())
    histogram=Counter((r['length'],len(r['support'])) for s in scans for r in s['runs'])
    weighted={}
    for s in scans:
        k=r_key(s['t'],s['n']);a=weighted.setdefault(k,dict(full=0.,joint=0.,runs=0.,applications=0.))
        for run in s['runs']:
            e=next(x for x in entries if x['key']==run['key']);weight=s['probability']/16
            a['full']+=weight*e['full_basis_pair'][0]
            a['joint']+=weight*(e['joint_basis_pair'][0] if e['adopt'] else e['full_basis_pair'][0])
            a['runs']+=weight;a['applications']+=weight*run['length']
    save_json(OUT/'PILOT.json',dict(source_commit=source(),training_origin='Prior study holdout is training in this new study. New independent holdout required.',
             scans=scans,histogram=[dict(length=k[0],columns=k[1],count=v) for k,v in sorted(histogram.items())],weighted_basis=weighted,
             catalogue_count=len(entries),adopted_count=sum(r['adopt'] for r in entries),full_Fock_checks=checks,paired_prices=len(prices),
             unchanged='Logical events, component law, order, signs, normalization, controlled phases, X/Y meaning; no change to D/PF steps or resolution.'))
    print('JOINT_ISOMETRY_PILOT_COMPLETE',len(entries),sum(r['adopt'] for r in entries),flush=True)


if __name__=='__main__':pilot()
