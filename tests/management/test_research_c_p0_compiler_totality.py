"""Invented small circuits/events and passive JSON only; no real H4 build."""
from dataclasses import replace
from fractions import Fraction as F
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import RZZGate, XXPlusYYGate
from qiskit.quantum_info import Operator

from trottertracks import c_p0_compiler_totality as m
from trottertracks.c_pr_aware_df_stage_b2_compat import SharedQiskitCalls
from trottertracks.pr_dictionary_validation.fixture import CompilerProfile

ROOT = Path(__file__).resolve().parents[2]


def compile_toy(q):
    shared = SharedQiskitCalls(lambda kind, label, fn: fn())
    return m.compile_once(q, CompilerProfile(), call=shared)


def test_fixed_length_word_grid_support():
    assert m.check_word(['a']*32, ['a', 'b'], [1, (1 << 53)-1])['support_size'] == 2
    assert not m.check_word(['a']*32, ['a', 'b'], [1 << 53, 0])['all_q_words_compile_success_qualified']


def test_saved_grid_support_metadata_matches_stage_R_without_sampling():
    p = ROOT/'artifacts/research_c_iid_grid_all_law/2026-11-03/stageA-run01'
    table = json.loads((p/'categorical_table.json').read_text())
    atoms = json.loads((p/'atoms.json').read_text())['atoms']
    counts = table['grid']['counts']
    assert len(atoms) == len(counts) == 216
    assert all(type(c) is int and c > 0 for c in counts)
    assert sum(counts) == 1 << 53
    assert len({a['atom_id'] for a in atoms}) == 216
    # Existing right-boundary implementation reused, no random-bit generator.
    from trottertracks.c_iid_grid_expected_bias import grid_index
    assert grid_index(0, [0., .5, 1.]) == 1
    assert grid_index(1 << 52, [0., .5, 1.]) == 2
    assert grid_index((1 << 53)-1, [0., .5, 1.]) == 2


@pytest.mark.parametrize('word,ids,counts', [(['a']*31, ['a'], [1 << 53]),
    (['b']*32, ['a', 'b'], [1 << 53, 0]), (['c']*32, ['a'], [1 << 53]),
    (['a']*32, ['a', 'a'], [1, (1 << 53)-1]),
    (['a']*32, ['a'], [(1 << 53)-1]), (['a']*32, ['a'], [float(1 << 53)])])
def test_illegal_word_or_inconsistent_support(word, ids, counts):
    with pytest.raises(ValueError):
        m.check_word(word, ids, counts)


@pytest.mark.parametrize('theta', [0., -0., 1e-300, 1e-12,
    math.nextafter(math.pi, 0), math.pi, -math.pi, 2*math.pi])
def test_angle_boundaries_all_raw_primitives_and_phase(theta):
    q = QuantumCircuit(3)
    q.rz(theta, 0); q.append(XXPlusYYGate(theta, -.19), [0, 1])
    q.crz(theta, 2, 0); q.append(RZZGate(theta).control(), [2, 0, 1])
    q.p(-theta, 2); q.global_phase = .29
    c = compile_toy(q)
    assert m.check_circuit(c, qubits=3, native=True)['finite_parameters']
    assert np.max(abs(Operator(c).data-Operator(q).data)) < 2e-12


@pytest.mark.parametrize('word', [['a']*32, ['a', 'b']*16, ['a', 'c']*16])
def test_existing_event_sharing_gaussian_and_signed_support(word):
    # Two invented 2-mode frames; no endpoint arrays, sampling, or H4 input.
    from trottertracks.pr_dictionary_validation.prototype import MixedAtom, make_tail
    from trottertracks.pr_dictionary_validation.circuits import BasisProvider, build_event_sequence
    from trottertracks.c1_h6_native_precision import gaussian
    from trotterlib.rte import make_rte_config, iter_rte_events
    atoms = (MixedAtom('a', 'FRAME_Z', F(1), ('toy',), ('Z', 'I'), 'A', 'A'),
             MixedAtom('b', 'FRAME_Z', F(-2), ('toy',), ('Z', 'Z'), 'A', 'A'),
             MixedAtom('c', 'FRAME_Z', F(3), ('toy',), ('I', 'Z'), 'B', 'B'))
    tail = make_tail(atoms, 'invented')
    config, dist = make_rte_config(tail, evolution_time=.1, rte_steps=32,
                                  truncation_tolerance=.1, finite_taylor_order=0)
    catalog = {e.rotation_component_id:e for e in iter_rte_events(tail.components, dist)}
    frames = {'A':np.eye(2), 'B':np.array([[.6, -.8], [.8, .6]])}
    class Provider(BasisProvider):
        def basis(self, atom, singleton=False):
            return gaussian(self.frames[atom.frame_id])[0]
    q = build_event_sequence([catalog[k] for k in word], atoms, Provider(2, frames), optimized=True)
    c = compile_toy(q)
    assert c.num_qubits == 3 and config.rte_steps == 32
    assert np.max(abs(Operator(c).data-Operator(q).data)) < 2e-12


def test_relative_transition_reuses_existing_builder_and_frame_order():
    from types import SimpleNamespace
    from trottertracks.c1_h4_strong_baseline import StrongBuilder
    from trottertracks.c1_stage2_native import NativeBlock
    rotations = [((1.+0j, 0j), (0j, 1.+0j)), ((.6+0j, -.8+0j), (.8+0j, .6+0j))]
    blocks = [NativeBlock(str(i), r, ()) for i, r in enumerate(rotations)]
    model = SimpleNamespace(label='invented', source_identity='invented', input_identity='toy')
    builder = StrongBuilder(relative_fusion=True)
    for a, b in (blocks, blocks[::-1]):
        t = builder.transition(model, a, b)
        expected = Operator(builder._basis(model, b)).data.conj().T @ Operator(builder._basis(model, a)).data
        assert np.max(abs(Operator(t).data-expected)) < 1e-12
        assert np.max(abs(Operator(compile_toy(t)).data-expected)) < 1e-12


@pytest.mark.parametrize('frame', [np.array([[1., 0.], [0., 2.]]),
    np.array([[float('nan')]]), np.ones((2,3))])
def test_invalid_frame_stops_in_existing_gaussian(frame):
    from trottertracks.c1_h6_native_precision import gaussian
    with pytest.raises(ValueError): gaussian(frame)


@pytest.mark.parametrize('word', [('Z','Z','Z'), ('X','I')])
def test_invalid_frame_atom_support_rejected_by_existing_type(word):
    from trottertracks.pr_dictionary_validation.prototype import MixedAtom
    with pytest.raises(ValueError):
        MixedAtom('invented', 'FRAME_Z', F(1), ('toy',), word, 'frame', 'frame')


def test_cross_event_optimization_and_trace_reuse():
    from trottertracks.c_pr_aware_df_c2a import trace_artificial
    a = QuantumCircuit(2); a.rz(.21, 0); a.rz(.13, 0)
    q = a.compose(a.inverse()); q.p(.19, 1)
    compiled, trace = trace_artificial(q)
    assert compiled.size() < q.size()
    assert any(max(r['run_lengths'], default=0) >= 4 for r in trace['passes'])
    assert not trace['all_law_proof']


def test_real_controller_guard_stops_and_fixed_point_returns():
    from qiskit.passmanager import DoWhileController
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Depth
    from qiskit.passmanager.exceptions import PassManagerError
    # Test only controller behavior; never change production P0's 1000 guard.
    q = QuantumCircuit(1); q.x(0)
    with pytest.raises(PassManagerError, match='Maximum iteration reached'):
        PassManager(DoWhileController([Depth()], do_while=lambda _:True, options={'max_iteration':2})).run(q)
    assert PassManager(DoWhileController([Depth()], do_while=lambda _:False)).run(q).size() == 1
    r = m.readiness()
    assert r['monotone_iterations_upper'] > r['fixed_controller_iterations']
    assert not r['all_q_words_compile_success_qualified']


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), Parameter('unbound')])
def test_invalid_parameter_stops_before_compiler(value):
    q = QuantumCircuit(1); q.rz(value, 0)
    with pytest.raises(ValueError):
        m.compile_once(q, CompilerProfile(), call=lambda *args:pytest.fail('compiler called'))


@pytest.mark.parametrize('fault', ['opaque', 'measure', 'width', 'profile', 'condition'])
def test_unsupported_path_stops(fault):
    q = QuantumCircuit(1)
    if fault == 'opaque': q.append(Gate('rz', 1, [.1]), [0])
    if fault == 'measure': q.measure_all()
    if fault == 'condition': q = QuantumCircuit(1, 1); q.x(0).c_if(q.cregs[0], 1)
    if fault == 'width':
        with pytest.raises(ValueError): m.check_circuit(q, qubits=2)
    elif fault == 'profile':
        with pytest.raises(ValueError): m.compile_once(q, replace(CompilerProfile(), seed_transpiler=18), call=None)
    else:
        with pytest.raises(ValueError): compile_toy(q)


def test_compiler_exception_preserved_no_retry_or_next_word(monkeypatch):
    from trottertracks import c1_h4_native_fair as fair
    attempts = []
    def fail(q, p):
        attempts.append(q)
        raise RuntimeError('invented compiler failure')
    monkeypatch.setattr(fair, 'compile_checked', fail)
    with pytest.raises(RuntimeError, match='invented compiler failure'):
        for _ in range(3): compile_toy(QuantumCircuit(1))
    assert len(attempts) == 1


def test_nonfinite_native_output_is_rejected(monkeypatch):
    from trottertracks import c1_h4_native_fair as fair
    q = QuantumCircuit(1); q.rz(float('nan'), 0)
    monkeypatch.setattr(fair, 'compile_checked', lambda *args:q)
    with pytest.raises(ValueError, match='NONFINITE'): compile_toy(QuantumCircuit(1))


@pytest.mark.parametrize('rows,N', [(['SUCCESS','FAIL'], 2), (['SUCCESS'],2), (['UNKNOWN'],1)])
def test_failure_cannot_be_discarded(rows, N):
    with pytest.raises(ValueError): m.require_complete_sample(rows, N)


def test_finite_pass_is_never_all_word_guarantee():
    r = m.require_complete_sample(['SUCCESS']*4, 4)
    assert r['population_failure_probability'] is None
    assert not r['all_q_words_compile_success_qualified']


def test_zero_failure_bound_reuses_existing_exact_binomial():
    from trottertracks.c_pr_aware_df_guarantee_strategy import zero_failure_tail_limit
    r = zero_failure_tail_limit(10, F(1,20), iid=True, fresh_holdout=True, certified_decisions=True)
    assert 0 < F(r['tail_probability_upper_exact']) < 1
    assert r['not_uniform']


def test_passive_contract_and_stage_P_scope_independence(tmp_path):
    c = json.loads((ROOT/m.CONTRACT).read_text())
    assert m.preflight(ROOT, ROOT/m.CONTRACT)['Stage_P_calls'] == 0
    c['scope'] = 'C_PR_R32_LITERAL_PER_WORD_STAGE_P_V1'
    p = tmp_path/'stageP.json'; p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='FIXED_NONEXECUTABLE'): m.preflight(ROOT, p)


def test_technical_source_drift_is_not_stage_P_authorization(tmp_path):
    c = json.loads((ROOT/m.CONTRACT).read_text())
    c['technical_source_files']['src/trottertracks/c_p0_compiler_totality.py'] = '0'*64
    p = tmp_path/'changed-source.json'; p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='TECHNICAL_SOURCE'): m.preflight(ROOT, p)


def test_passive_preflight_without_scientific_imports():
    code = '''
import sys
class Block:
 def find_spec(self, name, *args):
  if name.split('.')[0] in ('numpy','scipy','qiskit','pyscf','openfermion','ffsim','flint'):
   raise AssertionError(name)
sys.meta_path.insert(0,Block())
from trottertracks import c_p0_compiler_totality as m
from pathlib import Path
assert m.preflight(Path.cwd(),m.CONTRACT)['scientific_calls']==0
'''
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


def test_unapproved_science_rejected_even_with_flag():
    with pytest.raises(PermissionError): m.run_science()
    with pytest.raises(NotImplementedError): m.run_science(execute_science=True)
