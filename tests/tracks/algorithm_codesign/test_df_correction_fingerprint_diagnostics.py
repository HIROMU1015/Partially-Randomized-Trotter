"""Observation/off-domain controls only; registered E recipes are forbidden."""
import ast
from dataclasses import fields
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.algorithm_codesign.df_correction_comparison import fingerprint_diagnostics as diag
from trottertracks.algorithm_codesign.df_correction_comparison import certification as cert
from trottertracks.algorithm_codesign.df_correction_comparison import native_evaluation as ne
from trottertracks.algorithm_codesign.df_correction_comparison.storage import dump_json, load_json


def compiler():
    import qiskit
    from trotterlib.rte import CompilerSettings
    return CompilerSettings(basis_gates=('rz','sx','x','cx'), backend_name=None,
        coupling_map=None, optimization_level=0, layout_method=None, routing_method=None,
        transpiler_seed=20261011, qiskit_version=qiskit.__version__)


def small_circuit():
    from qiskit import QuantumCircuit
    q = QuantumCircuit(1); q.rz(0.173, 0)
    return q


def snapshots():
    records = []
    def save(record):
        records.append(json.loads(json.dumps(record)))
    return records, save


def small_frames(case=0):
    # Analytic 2-mode rank1 controls: NOT registered r3/r6, no RNG or recipe.
    triples = (((3/5,4/5,0.0), (5/13,12/13,0.37), (8/17,15/17,-0.43)),
               ((7/25,24/25,0.21), (20/29,21/29,-0.31), (9/41,40/41,0.57)))
    out = []
    for a,b,phase in triples[case]:
        z = b*complex(math.cos(phase), math.sin(phase))
        square = np.array([[a,-z.conjugate()],[z,a]],complex)
        out.append(square[:,:1])  # preserve old sliced-buffer construction
    return out


@pytest.mark.parametrize('layout', ['C','F','slice'])
def test_lossless_matrix_bytes_and_original_layout(layout):
    base=np.array([[1+2j,3-4j],[5+6j,7-8j]],complex)
    a={'C':base,'F':np.asfortranarray(base),'slice':base[:,::2]}[layout]
    before=a.tobytes();strides=a.strides
    row=diag.array_record(a)
    raw=bytes.fromhex(row['raw_bytes_hex'])
    assert raw==before and hashlib.sha256(raw).hexdigest()==row['sha256']
    assert row['strides']==list(strides) and a.strides==strides and a.tobytes()==before
    assert np.array_equal(np.frombuffer(raw,dtype=row['dtype_str']).reshape(row['shape']),a)


@pytest.mark.parametrize('a',[np.ones((1,),dtype=object),np.zeros((129,129),complex)])
def test_diagnostic_bounds_reject_unsupported_array(a):
    with pytest.raises(ValueError):diag.array_record(a)


@pytest.mark.parametrize('error',[TypeError('bad parameter'),ValueError('recursive definition')])
def test_None_fallback_saved_with_original_exception_and_no_fingerprint_retry(monkeypatch,error):
    from trotterlib import rte_compiled_cost as shared
    records,save=snapshots();calls=[]
    def fail(circuit):calls.append('fingerprint');raise error
    def compile_once(circuit,**kw):calls.append('compile');return circuit.copy()
    monkeypatch.setattr(shared,'_circuit_payload',fail)
    monkeypatch.setattr(shared,'transpile',compile_once)
    cost=diag.diagnostic_transpile_and_measure_cost(small_circuit(),compiler(),expected='fixed',
        arrays={'target':np.eye(2)},context={'off_domain':True},save=save)
    assert calls==['fingerprint','compile'] and cost.actual_circuit_fingerprint is None
    assert cost.circuit_fingerprint=='uncacheable-circuit'
    assert records[-1]['actual_fingerprint'] is None and not records[-1]['matches_expected']
    assert records[-1]['fingerprint_exception']['message']==str(error)
    assert records[-1]['recursive_definition_status']=='INCOMPLETE_NOT_REEVALUATED'
    assert records[-1]['fingerprint_attempts']==records[-1]['transpile_attempts']==1


def test_unexpected_fingerprint_exception_is_saved_and_not_swallowed(monkeypatch):
    from trotterlib import rte_compiled_cost as shared
    records,save=snapshots()
    def fail(q):raise RuntimeError('unexpected')
    monkeypatch.setattr(shared,'_circuit_payload',fail)
    monkeypatch.setattr(shared,'transpile',lambda *a,**k:pytest.fail('must not compile'))
    with pytest.raises(RuntimeError,match='unexpected'):
        diag.diagnostic_transpile_and_measure_cost(small_circuit(),compiler(),expected='fixed',
            arrays={},context={},save=save)
    assert records[-1]['phase']=='FINGERPRINT_EXCEPTION'
    assert records[-1]['transpile_attempts']==0


def test_transpile_failure_preserves_already_captured_payload(monkeypatch):
    from trotterlib import rte_compiled_cost as shared
    records,save=snapshots()
    def fail(*a,**k):raise RuntimeError('compiler error')
    monkeypatch.setattr(shared,'transpile',fail)
    with pytest.raises(RuntimeError,match='compiler error'):
        diag.diagnostic_transpile_and_measure_cost(small_circuit(),compiler(),expected='fixed',
            arrays={},context={},save=save)
    assert records[-1]['phase']=='TRANSPILE_OR_METRIC_EXCEPTION'
    assert records[-1]['actual_fingerprint']==diag.payload_digest(records[-1]['recursive_payload'])


def test_save_failure_aborts_before_transpile(monkeypatch):
    from trotterlib import rte_compiled_cost as shared
    monkeypatch.setattr(shared,'transpile',lambda *a,**k:pytest.fail('must not compile'))
    def fail_save(record):raise OSError('bounded output unavailable')
    with pytest.raises(OSError,match='bounded output'):
        diag.diagnostic_transpile_and_measure_cost(small_circuit(),compiler(),expected='fixed',
            arrays={},context={},save=fail_save)


def test_real_adapter_matches_shared_no_backend_cost_and_settings():
    from trotterlib import rte_compiled_cost as shared
    q=small_circuit();settings=compiler();records,save=snapshots()
    reference=shared.transpile_and_measure_cost(q,settings)
    observed=diag.diagnostic_transpile_and_measure_cost(q,settings,
        expected=reference.actual_circuit_fingerprint,arrays={},context={'off_domain':True},save=save)
    for field in fields(reference):
        if field.name!='transpiled_circuit':assert getattr(observed,field.name)==getattr(reference,field.name)
    assert observed.transpiled_circuit==reference.transpiled_circuit
    assert records[-1]['matches_expected'] and records[-1]['fingerprint_exception'] is None


def test_recursive_payload_exact_hash_and_global_phase_are_retained():
    from qiskit import QuantumCircuit
    from trotterlib.rte_compiled_cost import _circuit_payload,canonical_qiskit_circuit_fingerprint
    inner=small_circuit();outer=QuantumCircuit(1);outer.append(inner.to_gate(),[0])
    p=_circuit_payload(outer)
    assert diag.payload_digest(p)==canonical_qiskit_circuit_fingerprint(outer)
    index=diag._definition_index(p)
    assert index and index[0]['instruction_count']==1
    outer.global_phase=0.123
    assert canonical_qiskit_circuit_fingerprint(outer)!=diag.payload_digest(p)


def test_large_diagnostic_is_chunked_lossless_and_cannot_overwrite_prior(tmp_path):
    a=np.arange(128*128,dtype=float).reshape(128,128).astype(complex)
    row={'input_arrays':{'target':diag.array_record(a)},'recursive_payload':None}
    p=tmp_path/'off-domain-diagnostic.json'
    receipt=dump_json(p,row)
    assert receipt['storage']=='chunks' and load_json(p)==row
    assert all(f.stat().st_size<=512*1024 for f in tmp_path.rglob('*') if f.is_file())
    with pytest.raises(FileExistsError):dump_json(p,row)
    part=next(tmp_path.rglob('chunk-*.txt'));part.write_bytes(b'corrupted')
    with pytest.raises(ValueError):load_json(p)


def test_native_mismatch_saves_observed_actual_before_unchanged_guard(monkeypatch,tmp_path):
    from types import SimpleNamespace
    calls=[];saved=[]
    def mock_cost(q,compiler,**kw):
        record={'actual_fingerprint':None,'expected_fingerprint':kw['expected'],
            'fingerprint_exception':{'type':'builtins.ValueError','message':'control exception'}}
        kw['save'](record);calls.append('compile-mock')
        return SimpleNamespace(actual_circuit_fingerprint=None,
                               transpiled_circuit=SimpleNamespace(data=[]))
    monkeypatch.setattr(diag,'diagnostic_transpile_and_measure_cost',mock_cost)
    def save(controlled,record):
        saved.append((controlled,record));dump_json(tmp_path/'saved.json',record)
    prior=[{'active_modes':2,'controlled':False,'circuit_fingerprint':'fixed-unchanged'}]
    with pytest.raises(ArithmeticError,match='Frozen E whole fingerprint mismatch; no synthesis'):
        ne.native_fixture(2,small_frames(),prior,lambda p:None,on_diagnostic=save)
    assert len(calls)==1 and len(saved)==1 and saved[0][0] is False
    assert load_json(tmp_path/'saved.json')['actual_fingerprint'] is None


@pytest.mark.parametrize('case',[0,1])
def test_full_old_new_off_domain_path_records_identity_without_adopting_repair(case,tmp_path):
    from scipy.linalg import expm,block_diag
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator
    from trotterlib.rte_compiled_cost import _circuit_payload
    source=ROOT/'scripts/tracks/algorithm_codesign/audit_df_sparse_correction.py'
    tree=ast.parse(source.read_bytes())
    # Execute ONLY the unchanged named helper definitions, never original main/recipes.
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name in {'primitive','jw_annihilate','comm'}]
    scope={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    frames=small_frames(case);received=ne.deserialize_frames(ne.serialize_frames(frames))
    old_q=[scope['primitive'](v,s) for v,s in zip(frames,(1,-1,1))]
    new_q=[cert.primitive(v,s)[0] for v,s in zip(received,(1,-1,1))]
    old_t=scope['comm'](old_q[0],scope['comm'](old_q[1],old_q[2]))
    new_t=cert.comm(new_q[0],cert.comm(new_q[1],new_q[2]))
    theta=F(1,19) if case==0 else F(1,23)  # neither registered local .03125
    old_u=expm(1j*float(theta)*old_t/4);new_u=expm(1j*float(theta)*new_t.value/4)
    assert np.linalg.norm(old_t,'fro')>1e-4
    assert np.linalg.norm(old_t-new_t.value,'fro')<1e-12
    stages={'old_frames':frames,'received_frames':received,'old_Q':old_q,
            'new_Q':[q.value for q in new_q],'old_T':[old_t],'new_T':[new_t.value],
            'old_U':[old_u],'new_U':[new_u]}
    report={'kind':'OFF_DOMAIN_FULL_PATH_CONTROL','case':case,'modes':2,'rank':[1,1,1],
        'theta_raw_fraction':str(theta),'registered_frame_access':0,'registered_compile_calls':0,
        'Clifford_T_synthesis_calls':0,'helper_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'stage_arrays':{name:[diag.array_record(a) for a in arrays] for name,arrays in stages.items()},
        'Q_bytes_equal':[a.tobytes()==b.value.tobytes() for a,b in zip(old_q,new_q)],
        'T_bytes_equal':old_t.tobytes()==new_t.value.tobytes(),
        'U_bytes_equal':old_u.tobytes()==new_u.tobytes(),'new_T_assembly_radius':str(new_t.error),
        'T_difference_frobenius_diagnostic':float(np.linalg.norm(old_t-new_t.value,'fro')),'circuits':[]}
    for controlled in (False,True):
        entries=[]
        for label,u in [('old',old_u),('new',new_u)]:
            target=block_diag(np.eye(4),u) if controlled else u
            qc=QuantumCircuit(2+int(controlled));qc.append(UnitaryGate(target),qc.qubits)
            # No native transpile; this is the canonical input definition evaluation.
            payload=_circuit_payload(qc);actual=Operator(qc).data
            residual=float(np.linalg.norm(actual-target,'fro'))
            assert residual<1e-12
            if controlled:
                assert np.linalg.norm(actual[:4,:4]-np.eye(4),'fro')<1e-12
                assert np.linalg.norm(actual[4:,4:]-u,'fro')<1e-12
                wrong=block_diag(np.eye(4),-u)
                assert np.linalg.norm(actual-wrong,'fro')>1  # relative phase cannot be ignored
            entries.append({'path':label,'payload':payload,'fingerprint':diag.payload_digest(payload),
                'recursive_definition_index':diag._definition_index(payload),
                'target':diag.array_record(target),'strict_operator_residual_diagnostic':residual})
        report['circuits'].append({'controlled':controlled,'paths':entries,
            'fingerprint_equal':entries[0]['fingerprint']==entries[1]['fingerprint']})
    destination=Path(os.environ.get('DF_DIAGNOSTIC_CONTROL_OUTPUT',str(tmp_path)))
    destination.mkdir(parents=True,exist_ok=True)
    dump_json(destination/f'full_path_control_{case}.json',report)
    assert load_json(destination/f'full_path_control_{case}.json')==report
