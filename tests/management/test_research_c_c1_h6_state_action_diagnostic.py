"""Artificial saved13-qubit circuits; no molecular data or compilation."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import pytest
from qiskit import QuantumCircuit,qpy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))


@pytest.mark.parametrize('changed',[False,True])
def test_independent_diagnostic_actions_and_relative_phase(tmp_path,monkeypatch,changed):
    from trottertracks.c1_stage1_execution import sha
    spec=importlib.util.spec_from_file_location('diagnostic',ROOT/'scripts/diagnose_research_c_c1_h6_native_state_action.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);monkeypatch.setattr(module,'ROOT',tmp_path)
    raw=QuantumCircuit(13);raw.global_phase=.37;raw.h(0);raw.cx(0,1)
    compiled=raw.copy()
    if changed:compiled.p(.01,12)
    for name,qc in [('raw',raw),('compiled',compiled)]:
        with (tmp_path/(name+'.qpy')).open('wb') as f:qpy.dump(qc,f)
    ground=np.zeros(400);ground[0]=1
    np.savez(tmp_path/'ground.npz',state=ground);np.savez(tmp_path/'kernel.npz',pf=np.eye(400))
    (tmp_path/'SOURCE.json').write_text(json.dumps(dict(files={})))
    c=dict(scope='C1_H6_SAVED_NATIVE_STATE_ACTION_DIAGNOSTIC_V1',source_manifest='SOURCE.json',source_manifest_sha256=sha(tmp_path/'SOURCE.json'),
        raw='raw.qpy',compiled='compiled.qpy',ground='ground.npz',kernel='kernel.npz',evaluation_packages={},
        saved_inputs={p.name:sha(p) for p in tmp_path.iterdir() if p.suffix in ('.qpy','.npz')},
        fresh_output='out',call_limits=dict(saved_probe_load=1,state_action=4))
    p=tmp_path/'contract.json';p.write_text(json.dumps(c))
    monkeypatch.setattr(sys,'argv',['diagnostic','--contract',str(p),'--execute','--output',str(tmp_path/'out')])
    module.main();r=json.loads((tmp_path/'out/result.json').read_text())
    assert r['scientific_units']==5
    assert r['comparisons']['compiled_Aer_Qiskit']['max_entry']<1e-14
    difference=r['comparisons']['compiled_Qiskit_raw_Qiskit']['max_entry']
    assert difference>1e-4 if changed else difference<1e-14
    assert json.loads((tmp_path/'out/STOP.json').read_text())['mandatory_stop']
