"""Attribute gains to shared unions versus layout changes on singletons.

Post-freeze diagnostic on the same holdout; never selects a new policy.
Individual partial circuits are priced on representative runs only, reusing
the prior negative always-partial result instead of a redundant large MC.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
import os
import numpy as np
from qiskit import QuantumCircuit
from .track_c_joint_isometry import OUT,PRIOR,source,catalogue,runs,append_run,decode,configuration
from .track_c_joint_resources import weights,covariance,records
from .track_c_bottleneck_runwise import events
from .track_c_h4 import common,save_json
from .track_c_compiler_a import compile_reference
from .track_c_d_resources import metrics,r_key
from trotterlib.df_rte_qiskit import _event_phase_angle


def singleton_layout(raw):
    ev,flat,groups=runs(raw['t'],raw['n'],raw['choices']);q=QuantumCircuit(9);phase=sum(_event_phase_angle(e.phase) for e in ev)
    for start,stop,basis,union in groups:
        entry=catalogue().get(basis+'|'+','.join(map(str,union)));use=stop==start+1 and entry is not None and entry['adopt']
        ops=decode(entry['operations']) if use else common()['preps'][3].rte_preparation.basis_registry.definition(basis).runtime_operations
        mapping=dict(zip(entry['support'],entry['targets'])) if use else None
        phase+=append_run(q,flat[start:stop],ops,mapping)
    if phase:q.p(phase,8)
    return q


def price(path):
    raw=json.loads(path.read_text());vector=metrics(compile_reference(singleton_layout(raw)))
    value={**raw,'singleton_layout':vector if vector[0]<raw['baseline'][0] else raw['baseline']}
    save_json(OUT/'ablation_prices'/path.name,value);return value


def per_run(rep):
    raw=json.loads((PRIOR/'holdout_prices'/rep['file']).read_text());_,flat,groups=runs(raw['t'],raw['n'],raw['choices'])
    start,stop=rep['start'],rep['stop'];_,_,basis,union=next(g for g in groups if g[:2]==(start,stop))
    entry=catalogue()[basis+'|'+','.join(map(str,union))];oldops=common()['preps'][3].rte_preparation.basis_registry.definition(basis).runtime_operations
    prices=[]
    for role in ('full_shared','individual_partial','joint_shared'):
        q=QuantumCircuit(9);phase=0.
        if role=='full_shared':phase=append_run(q,flat[start:stop],oldops)
        elif role=='joint_shared':phase=append_run(q,flat[start:stop],decode(entry['operations']),dict(zip(entry['support'],entry['targets'])))
        else:
            registry,_,lookup,_=configuration()
            for e,a in flat[start:stop]:
                spec=lookup[a.component_id];d=registry.definition(spec.basis_id)
                phase+=append_run(q,[(e,a)],d.runtime_operations,dict(zip(a.diagonal_pauli_support,spec.diagonal_pauli_support)))
        if phase:q.p(phase,8)
        prices.append(dict(role=role,metrics=metrics(compile_reference(q))))
    return dict(file=rep['file'],start=start,stop=stop,columns=len(union),length=stop-start,prices=prices)


def run():
    if (OUT/'ABLATION.json').exists():raise RuntimeError('Do not overwrite ablation')
    (OUT/'ablation_prices').mkdir(exist_ok=False);configuration();catalogue();rows=[];examples=[]
    with ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:
        for f in as_completed(pool.submit(price,p) for p in sorted((OUT/'holdout_prices').glob('*.json'))):rows.append(f.result())
        for f in as_completed(pool.submit(per_run,r) for r in json.loads((OUT/'REPRESENTATIVES.json').read_text())['rows']):examples.append(f.result())
    rs=records(OUT/'ablation_prices');results=[]
    for p in json.loads((OUT/'POLICY.json').read_text())['policies']:
        w=weights(p);gains={};se={}
        for label,a,b in [('singleton_layout','baseline','singleton_layout'),('joint_sharing','singleton_layout','selected'),('total_basis','baseline','selected')]:
            gain=sum((w.get(k,0)*r[0]['probability']*np.mean([np.array(x[a])-x[b] for x in r],axis=0) for k,s in rs.items() for r in s.values()),np.zeros(4))
            cov=covariance(rs,[(a,w),(b,{k:-v for k,v in w.items()})]);gains[label]=gain.tolist();se[label]=np.sqrt(np.diag(cov)).tolist()
        results.append(dict(primary=p['primary'],fixed_new_shots_savings=gains,paired_MC_SE=se))
    save_json(OUT/'ABLATION.json',dict(source_commit=source(),results=results,representative_run_prices=examples,
        scope='Fixed already frozen new shots/alpha and same holdout trajectories. Separates shared multi-application unions from singleton layout optimization; no policy chosen using these diagnostics. Metrics RZ,CX,DAG occurrence depth,size; total depth effect is a modular serial-bound difference.'))
    print('JOINT_ABLATION_COMPLETE',flush=True)


if __name__=='__main__':run()
