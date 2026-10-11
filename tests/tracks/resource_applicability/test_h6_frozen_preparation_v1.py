"""Small artificial DF/codec fixtures only; no NPZ, signal or transpile."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pytest
import numpy as np
from qiskit.circuit.library import RZGate, XXPlusYYGate
from qiskit.quantum_info import Operator
from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
from trotterlib.df_rte_tail import describe_basis_change_operations
from trottertracks.resource_applicability import h6_frozen_preparation_v1 as f


@pytest.fixture(autouse=True)
def no_npz(monkeypatch):
    original = Path.open
    def guard(path, *args, **kwargs):
        if path.suffix == '.npz':
            raise AssertionError('NO_REAL_INPUT')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)


@pytest.fixture
def fixture():
    ham = DFHamiltonian(.13, np.asarray([[.2, .03], [.03, -.1]], dtype=np.complex128),
                        np.asarray([.2, -.3]),
                        (np.asarray([[1., .1], [.1, .4]], dtype=np.complex128),
                         np.asarray([[.2, .04], [.04, 2.]], dtype=np.complex128)),
                        {'name': 'ARTIFICIAL_TWO_MODE_ARCHIVE'})
    p = _prepare(ham, 1)
    return ham, {f.key(False, 1): p, f.key(True, 1): _prepare_discard(ham, 1)}


def expected(preparations):
    return {k: dict(preparation_hash=p.preparation_hash, partition_hash=p.partition_hash,
                    lambda_r=p.exact_rte_lambda_r,
                    extracted_identity=p.extracted_identity_coefficient,
                    constant=p.constant_coefficient) for k, p in preparations.items()}


def test_roundtrip_keeps_all_basis_coefficients_order_and_preparation_hash(fixture):
    ham, preparations = fixture
    codec = f.encode_preparations(preparations)
    restored = f.decode_preparations(json.loads(json.dumps(codec)))
    f.verify_preparations(restored, expected(preparations), next(iter(preparations.values())).hamiltonian_hash)
    for name, original in preparations.items():
        p = restored[name]
        assert p.deterministic_blocks == original.deterministic_blocks
        assert p.tail_extraction == original.tail_extraction
        assert p.rte_preparation.component_specs == original.rte_preparation.component_specs
        assert p.rte_preparation.basis_registry.metadata() == original.rte_preparation.basis_registry.metadata()
        assert describe_basis_change_operations(p.deterministic_blocks[0].runtime_basis_operations) == original.deterministic_blocks[0].basis_change_operations
        assert p.rte_preparation.symbolic_tail.components == original.rte_preparation.symbolic_tail.components


def test_gate_bits_signed_zero_and_operation_order_survive():
    operations = ((RZGate(-0.), (0,)), (XXPlusYYGate(.2, -0.), (0, 1)))
    codec = f.encode_preparations({'ops': operations})
    restored = f.decode_preparations(codec)['ops']
    assert restored[0][0].params[0].hex() == '-0x0.0p+0'
    assert restored[1][0].params[1].hex() == '-0x0.0p+0'
    assert describe_basis_change_operations(restored) == describe_basis_change_operations(operations)
    for (gate, qubits), (old_gate, old_qubits) in zip(restored, operations):
        assert qubits == old_qubits
        assert Operator(gate).data.tobytes() == Operator(old_gate).data.tobytes()


@pytest.mark.parametrize('kind', ['hash', 'unknown_type', 'gate_matrix', 'extra_field', 'nonfinite', 'unknown_reference'])
def test_malformed_or_changed_archive_nodes_fail_closed(kind):
    codec = f.encode_preparations({'ops': ((RZGate(.1), (0,)),)})
    gate_sha = next(k for k, v in codec['nodes'].items() if v['type'] == 'gate')
    record = codec['nodes'].pop(gate_sha)
    if kind == 'hash':
        record['name'] = 'changed'; codec['nodes'][gate_sha] = record
    else:
        if kind == 'unknown_type': record['type'] = 'UnsafePythonObject'
        if kind == 'gate_matrix': record['matrix_sha256'] = '0' * 64
        if kind == 'extra_field': record['unexpected'] = True
        if kind == 'nonfinite': record['parameters'] = [{'float_hex': 'inf'}]
        if kind == 'unknown_reference': record['parameters'] = [{'ref': '0' * 64}]
        new_sha = f.digest(record); codec['nodes'][new_sha] = record
        for sha, row in list(codec['nodes'].items()):
            if row['type'] == 'tuple' and any(v == {'ref': gate_sha} for v in row['items']):
                row['items'] = [{'ref': new_sha} if v == {'ref': gate_sha} else v for v in row['items']]
                # A stale enclosing hash is independently rejected; gate decode
                # is also called as the root so its own validation is exercised.
        codec['nodes'] = {new_sha: record}; codec['roots'] = {'ops': {'ref': new_sha}}
    with pytest.raises(ValueError): f.decode_preparations(codec)


@pytest.mark.parametrize('kind', ['old_hash', 'hamiltonian', 'set', 'constant'])
def test_old_metadata_and_input_must_match(fixture, kind):
    _, preparations = fixture; p = next(iter(preparations.values()))
    old = expected(preparations); ham_hash = p.hamiltonian_hash
    if kind == 'old_hash': old[f.key(False, 1)]['preparation_hash'] = '0' * 64
    if kind == 'hamiltonian': ham_hash = '0' * 64
    if kind == 'set': old = {}
    if kind == 'constant':
        preparations=dict(preparations)
        preparations[f.key(False,1)]=replace(p,constant_coefficient=p.constant_coefficient+.1)
    with pytest.raises(ValueError): f.verify_preparations(preparations, old, ham_hash)


def test_adapter_restores_port_on_failure_and_rejects_wrong_input(fixture):
    from trottertracks.resource_applicability import h6_matched_cost_port_v3 as port
    ham, preparations = fixture; before = port._prepare, port._prepare_discard
    with pytest.raises(RuntimeError):
        with f.use_frozen_preparations(preparations):
            assert port._prepare(ham, 1) is preparations[f.key(False, 1)]
            assert port._prepare_discard(ham, 1) is preparations[f.key(True, 1)]
            wrong = replace(ham, constant=ham.constant+.1)
            with pytest.raises(ValueError): port._prepare(wrong, 1)
            raise RuntimeError('ARTIFICIAL_COST_FAILURE')
    assert (port._prepare, port._prepare_discard) == before


@pytest.mark.parametrize('case', ['wrong_sha', 'duplicate_key', 'symlink', 'nonfinite'])
def test_archive_transport_validation(tmp_path, case):
    path = tmp_path / 'archive.json'
    data = dict(schema=f.SCHEMA, binding={}, codec={}, codec_digest=f.digest({}), export_receipt={})
    payload = f.canonical(data)
    if case == 'duplicate_key': payload = b'{"schema":1,"schema":2}'
    if case == 'nonfinite': payload = b'{"value":NaN}'
    path.write_bytes(payload)
    if case == 'symlink':
        alias = tmp_path / 'alias.json'; alias.symlink_to(path); path = alias
    sha = '0' * 64 if case == 'wrong_sha' else f.file_hash(path)
    with pytest.raises(ValueError): f.read_archive(path, sha256=sha)


def test_same_seed_events_are_exact_after_codec_roundtrip(fixture):
    from trotterlib.rte import finite_rte_distribution
    _, preparations = fixture
    p = preparations[f.key(False,1)]
    restored = f.decode_preparations(f.encode_preparations(preparations))[f.key(False,1)]
    distribution = finite_rte_distribution(.1, 2)
    a = [x.to_dict() for x in p.rte_preparation.sample_events(distribution, sample_count=8, seed=12345)]
    b = [x.to_dict() for x in restored.rte_preparation.sample_events(distribution, sample_count=8, seed=12345)]
    assert a == b and f.digest(a) == f.digest(b)


@pytest.mark.parametrize('match',[True,False])
def test_phase_a_costs_cannot_mix_unmatched_full_wrappers(tmp_path,monkeypatch,match):
    from types import SimpleNamespace
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native
    from trottertracks.resource_applicability import ax2b_stream_fingerprint_v5 as stream
    run=tmp_path/'cost_exploration_0447';run.mkdir()
    for axis in ('cosine','sine'):
        (run/('wrapper_symmetric_directional_'+axis+'.json')).write_text(json.dumps({'fingerprint':axis}))
    called=[]
    def build(*args,**kwargs):called.append(kwargs);return object()
    monkeypatch.setattr(native,'build_deterministic_native',build)
    monkeypatch.setattr(native,'build_native_hadamard_wrapper',lambda e,**kwargs:kwargs['axis'])
    monkeypatch.setattr(stream,'canonical_qiskit_circuit_fingerprint',lambda axis:axis if match else 'DIFFERENT')
    p=SimpleNamespace(deterministic_blocks=(),num_system_qubits=12,constant_coefficient=.1)
    base=dict(exclusive_output={'absolute_path':str(tmp_path)},phase_scope={'task':{'cell':{'q':8,'formula':'4th'}}},
        plan={'T':.8,'caps_proposed':{'untranspiled_instructions':1000000}})
    comparison=f.verify_phase_a_wrappers({f.key(False,19):p},base)
    assert comparison['exact'] is match and comparison['compile_calls']==0
    assert called[0]['q']==8 and called[0]['formula']=='4th'
