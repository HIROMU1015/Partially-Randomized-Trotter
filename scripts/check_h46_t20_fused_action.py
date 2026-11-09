"""Benchmark exact gate fusion on already priced H6 control wrapper artifacts."""
import gzip
import io
import time
from functools import partial
import numpy as np
from qiskit import qpy
from qiskit_aer import AerSimulator
import h46_full_h_t20_aer as verifier
import h46_full_h_t20_opt as study

# Action-only exact gate fusion; no approximation, compiler or circuit change.
verifier.AerSimulator=partial(AerSimulator,fusion_enable=True,fusion_threshold=0,fusion_max_qubit=4)
study.aer_statevector=verifier.simulate_statevector

if __name__=='__main__':
    study.setup();ctx=study.context('H6')
    r=study.row('H6',32,32,6,'INITIAL_B3_GRID')
    r=next(v['candidate'] for v in study.read(study.PUBLIC/'signal_records.json.gz').values() if v['candidate']['id']==r['id'])
    native=study.setup()
    requests,_,_=native.sample_requests(r,ctx.preparations[0],'FULL_H_T20_EXPLORATION_V2',0)
    expected=ctx.oracle.trajectory(requests)
    low=np.r_[ctx.qstate,np.zeros_like(ctx.qstate)]
    checked=[]
    for path in (study.PRIVATE/'compiled_cache').glob('*.json'):
        meta=study.read(path)
        if meta['metrics']['rz_count']!=495926:continue
        stem=path.stem
        with gzip.open(path.with_name(stem+'_wrapper.qpy.gz'),'rb') as f:wrapper=qpy.load(f)[0]
        with gzip.open(path.with_name(stem+'_compiled.qpy.gz'),'rb') as f:compiled=qpy.load(f)[0]
        names=[i.operation.name for i in wrapper.remove_final_measurements(inplace=False).data[-3:]]
        sine='sdg' in names
        phase=-1j if sine else 1.
        independent=np.r_[(ctx.qstate+phase*expected)/2,(ctx.qstate-phase*expected)/2]
        started=time.monotonic()
        before=verifier.simulate_statevector(wrapper.remove_final_measurements(inplace=False),low)
        after=verifier.simulate_statevector(compiled.remove_final_measurements(inplace=False),low)
        errors=dict(native=float(np.linalg.norm(before-independent)),compiled=float(np.linalg.norm(after-before)))
        study.full.require(errors['native']<1e-9 and errors['compiled']<1e-7,'FUSED_ACTION_CORRECTNESS')
        checked.append(dict(fingerprint=stem,axis='sine' if sine else 'cosine',errors=errors,wall_seconds=time.monotonic()-started,price_changed=False))
    study.full.require(len(checked)==2,'BOTH_AXES')
    study.save(study.PUBLIC/'fused_action_validation.json',dict(status='PASS',checks=checked,statistical_samples=0,
        existing_priced_candidate=r,method='Aer CPU double; exact fusion threshold0/max_qubit4',input_circuit_unchanged=True))
    print(checked)
