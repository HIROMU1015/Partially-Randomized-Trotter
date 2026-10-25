"""Artificial saved P0 circuits/full-column runner; real H6 files never loaded."""
from pathlib import Path
import sys,json
from unittest.mock import patch,Mock
import numpy as np
import pytest
from qiskit import QuantumCircuit,qpy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h6_synthesis_bound as m


def artificial(root,fault=None):
    shell={k:dict(unitary_distance_pilot=1e-15) for k in ['head','X','Y']}
    m.io.save(root/'native.json',dict(status='H6_NATIVE_FINITE_PROBE_COST_COMPLETE',shells=shell))
    pf=dict(status='H6_SAVED_RANK14_SECTOR_PF_COMPLETE',candidates={});kernels={};circuits={}
    for arm,s in m.native.saved.CANDIDATES:
        key=f'{arm}:s{s}';theta=(.02 if arm=='CDF' else .03)/s
        w=np.exp(1j*theta)*np.eye(4,dtype=complex)
        if fault=='reference':w[0,0]=2.
        f=f'{arm}_s{s}.npz';np.savez(root/f,pf=w);kernels[key]=f
        pf['candidates'][key]=dict(rounds=[dict(m=i,pf_operator_error=0.) for i in range(13)])
        for route in ['literal','shared']:
            label=f'{arm}_s{s}_{route}';qc=QuantumCircuit(5,name=label);qc.rz(theta,4);qc.global_phase=theta/2
            if fault=='phase':qc.rz(.001,4)
            if fault=='nonunitary':qc.reset(0)
            f=label+'.qpy'
            with (root/f).open('wb') as stream:qpy.dump(qc,stream)
            circuits[label]=f
    m.io.save(root/'pf.json',pf)
    paths=['native.json','pf.json',*kernels.values(),*circuits.values()]
    return dict(native_result='native.json',PF_result='pf.json',PF_kernels=kernels,compiled_circuits=circuits,
        saved_inputs={f:m.io.sha(root/f) for f in paths},
        conditions=dict(m.native.saved.CONDITIONS,atoms=2,Nalpha=1,Nbeta=1,sector_dimension=4))


@pytest.mark.parametrize('fault',[None,'phase','nonunitary','reference','resources'])
def test_artificial_complete_basis_runner_dedup_and_STOP(tmp_path,fault):
    c=artificial(tmp_path,fault);out=tmp_path/'out';forbidden=Mock(side_effect=AssertionError('NO_NEW_FIT_PF_BUILD_COMPILE_OPERATOR'))
    from trottertracks.pr_dictionary_validation import gates
    with patch.object(m,'preflight',return_value=c), \
            patch.object(m.native.native.NativeBuilder,'ordered',forbidden), \
            patch.object(gates,'compiled_wrapper',forbidden), \
            patch.object(m.native.saved.molecule,'PF_rounds',forbidden):
        if fault=='resources':
            plan=dict(workers=1,available_affinity_cpus=1,quota_cpus=None,
                available_memory_bytes=0,estimated_bytes_per_candidate=1)
            with patch.object(m.native.saved.molecule.integrated,'resource_plan',return_value=plan), \
                    pytest.raises(ValueError,match='INSUFFICIENT_AVAILABLE_MEMORY'):
                m.run(tmp_path,tmp_path/'unused',out)
        elif fault:
            with pytest.raises((ValueError,RuntimeError)):m.run(tmp_path,tmp_path/'unused',out)
        else:m.run(tmp_path,tmp_path/'unused',out)
    forbidden.assert_not_called();stop=m.io.read(out/'STOP.json');assert stop['failed']==bool(fault)
    assert stop['retry']==stop['resume']==0 and stop['scientific_units']<=m.MAX_CALLS
    if fault:
        assert (out/'failure.json').exists() and not (out/'result.json').exists()
        events=[json.loads(s) for s in (out/'science_calls.jsonl').read_text().splitlines()]
        if fault=='resources':assert stop['call_counts'].get('state_action_columns',0)==0
        else:
            first=next(i for i,x in enumerate(events) if x['event']=='FAIL')
            assert not any(x['event']=='attempt' for x in events[first+1:])
    else:
        r=m.io.read(out/'result.json');assert r['evaluated_groups']==4 and len(r['routes'])==8
        assert stop['call_counts']['state_action_columns']==4*8
        assert stop['scientific_units']==135+32
        assert r['evidence_level']==m.bounds.LEVEL and r['rigorous_enclosure'] is None
        assert all(v['kernel']['covered_columns']==8 for v in r['routes'].values())
        assert all(v['CA_achieved'] is v['shots'] is v['total_cost'] is None for v in r['routes'].values())
        assert sum(len(v['rounds']) for v in r['routes'].values())==104
    previous=(out/'STOP.json').read_bytes()
    with patch.object(m,'preflight',return_value=c),pytest.raises(FileExistsError):m.run(tmp_path,tmp_path/'unused',out)
    assert (out/'STOP.json').read_bytes()==previous


def test_gate_identity_ignores_names_but_keeps_phase_angles_qubits():
    q=QuantumCircuit(3,name='one');q.rz(.2,0);q.cx(0,2);q.sx(1)
    original=m.circuit_identity(q)['signature'];copy=q.copy(name='two')
    assert original==m.circuit_identity(copy)['signature']
    phase=q.copy();phase.global_phase=.01
    angle=q.copy();angle.rz(.00001,0)
    position=QuantumCircuit(3);position.rz(.2,1);position.cx(1,2);position.sx(0)
    assert all(m.circuit_identity(x)['signature']!=original for x in [phase,angle,position])


@pytest.mark.parametrize('kind',['reset','unitary_named_rz','unbound','nonfinite'])
def test_invalid_structural_gate_is_rejected(kind):
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import UnitaryGate
    q=QuantumCircuit(1)
    if kind=='reset':q.reset(0)
    elif kind=='unitary_named_rz':
        gate=UnitaryGate(np.eye(2));gate.name='rz';q.append(gate,[0])
    elif kind=='unbound':q.rz(Parameter('angle'),0)
    else:q.rz(float('nan'),0)
    with pytest.raises(ValueError):m.circuit_identity(q)


def contract(root):
    (root/'source').write_text('artificial source');(root/'input').write_text('hash only')
    m.io.save(root/'SOURCE.json',dict(files={'source':m.io.sha(root/'source')}))
    return dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT',conditions=m.native.saved.CONDITIONS,
        allowances=m.ALLOWANCES,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,method=m.bounds.VERSION,
        candidate_order=[f'{a}:s{s}' for a,s in m.native.saved.CANDIDATES],routes=list(m.native.ROUTES),
        **{k:0 for k in ['new_fit','optimizer','DF_recomputation','PF_recomputation','build_compile','CA_evaluation','shots_evaluation','total_cost_evaluation','retry','resume']},
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        saved_inputs={'input':m.io.sha(root/'input')},native_result='input',PF_result='input',
        PF_kernels={f'{a}:s{s}':'input' for a,s in m.native.saved.CANDIDATES},
        compiled_circuits={f'{a}_s{s}_{r}':'input' for a,s,r in m.JOBS},evaluation_packages={})


@pytest.mark.parametrize('fault',[None,'source','input','conditions','allowance','calls','scope','missing_route'])
def test_passive_preflight_without_molecular_decode(tmp_path,fault):
    c=contract(tmp_path)
    if fault=='source':(tmp_path/'source').write_text('changed')
    if fault=='input':(tmp_path/'input').write_text('changed')
    if fault=='conditions':c['conditions']=dict(c['conditions'],rank=18)
    if fault=='allowance':c['allowances']=dict(c['allowances'],state_component=1e-9)
    if fault=='calls':c['maximum_logical_units']+=1
    if fault=='scope':c['build_compile']=1
    if fault=='missing_route':c['compiled_circuits'].pop('CDF_s1_shared')
    p=tmp_path/'contract.json';m.io.save(p,c)
    with patch.object(np,'load',side_effect=AssertionError('MOLECULE_DECODE_FORBIDDEN')):
        if fault:
            with pytest.raises(ValueError):m.preflight(tmp_path,p)
        else:assert m.preflight(tmp_path,p)==c
