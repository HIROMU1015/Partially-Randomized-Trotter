"""Synthetic contracts, toy matrices and non-science subprocesses only."""
import ast
import builtins
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.pr2_matched_accuracy_m1_execution import _eigendecomposition
from trottertracks.resource_applicability import ax2b_h4_contract_v5 as contract
from trottertracks.resource_applicability import ax2b_h4_science_v5 as science
from trottertracks.resource_applicability.ax2a_state_action import ActionBudget, partial_s2_signal
from trottertracks.resource_applicability.ax2b_limits import CallBudget, supervise

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def no_scientific_io_sampling_compile(monkeypatch):
    original, path_open = builtins.open, Path.open
    def check(path):
        if isinstance(path, (str, Path)) and ('/artifacts/' in str(path) or '/.runtime/' in str(path)):
            raise AssertionError('Scientific artifact access forbidden')
    def guarded(path, *args, **kwargs):
        check(path)
        return original(path, *args, **kwargs)
    def guarded_path(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Scientific loading/sampling/compile forbidden')
    monkeypatch.setattr(builtins, 'open', guarded)
    monkeypatch.setattr(Path, 'open', guarded_path)
    monkeypatch.setattr(np, 'load', forbidden)
    import qiskit
    import trotterlib.rte_compiled_cost as cost
    import trotterlib.df_rte_circuit as rte_circuit
    import trotterlib.rte as rte
    monkeypatch.setattr(qiskit, 'transpile', forbidden)
    monkeypatch.setattr(cost, 'transpile', forbidden)
    monkeypatch.setattr(science, 'transpile_and_measure_cost', forbidden)
    monkeypatch.setattr(science, 'make_df_partial_s2_repeated_request', forbidden)
    monkeypatch.setattr(rte_circuit.DFRTEEventPreparation, 'sample_occurrence_request', forbidden)
    monkeypatch.setattr(rte, 'sample_rte_events', forbidden)


def toy_metadata_zip(path, *, metadata=None, shape=(), dtype='<U', duplicate=False):
    text = json.dumps({'toy': '合成'} if metadata is None else metadata)
    descriptor = dtype + str(len(text))
    header = repr({'descr': descriptor, 'fortran_order': False, 'shape': shape}).encode() + b'\n'
    payload = b'\x93NUMPY' + b'\x01\x00' + struct.pack('<H', len(header)) + header
    payload += text.encode('utf-32-le' if dtype[0] == '<' else 'utf-32-be')
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('metadata_json.npy', payload)
        archive.writestr('scientific_array.npy', b'never read')
        if duplicate:
            archive.writestr('metadata_json.npy', payload)


@pytest.mark.parametrize('dtype', ['<U', '>U'])
def test_metadata_parser_reads_only_unicode_scalar(tmp_path, dtype):
    path = tmp_path / 'synthetic.npz'
    toy_metadata_zip(path, dtype=dtype)
    assert contract.metadata_only(path) == {'toy': '合成'}


@pytest.mark.parametrize('shape,dtype', [((1,), '<U'), ((), '|O')])
def test_metadata_parser_rejects_array_or_object_layout(tmp_path, shape, dtype):
    path = tmp_path / 'bad.npz'
    toy_metadata_zip(path, shape=shape, dtype=dtype)
    with pytest.raises(ValueError):
        contract.metadata_only(path)


def test_metadata_parser_rejects_duplicate_entry(tmp_path):
    path = tmp_path / 'bad.npz'
    with pytest.warns(UserWarning):
        toy_metadata_zip(path, duplicate=True)
    with pytest.raises(ValueError, match='LAYOUT'):
        contract.metadata_only(path)


def test_plan_h4_only_and_paired_tasks():
    plan = contract.h4_plan()
    assert len(plan['correctness_cells']) == 8 and len(plan['wrapper_tasks']) == 28
    assert not plan['H6_tasks'] and not plan['H8_tasks'] and plan['gpu'] is False
    random = [t for t in plan['wrapper_tasks'] if t['trajectory_seed'] is not None]
    assert len({t['trajectory_seed'] for t in random}) == 4
    assert len(random) == 16
    assert plan['caps']['compile_calls'] == 28
    assert plan['caps']['trajectory_samples'] == 4 and plan['caps']['occurrence_samples'] == 16
    assert plan['retry'] is False and plan['resume'] is False


def launch_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(contract, 'source_hashes', lambda root: {'toy.py': 'synthetic'})
    monkeypatch.setattr(contract, 'environment_identity', lambda: {'python': 'synthetic'})
    monkeypatch.setattr(contract, 'snapshot_binding', lambda root: {'path': 'synthetic', 'sha256': 'toy'})
    manifest = contract.preparation(tmp_path)
    output = tmp_path / 'exclusive'
    auth = {'schema': 'track_a_ax2b_h4_authorization_v5', 'approved_by_user': True,
            'manifest_digest': contract.digest(manifest), 'assigned_cpu': 0,
            'exclusive_output': str(output.resolve())}
    return manifest, auth, output


def test_preparation_never_authorizes_launch(monkeypatch, tmp_path):
    manifest, _, _ = launch_fixture(monkeypatch, tmp_path)
    assert manifest['science_authorized'] is False and manifest['launch_allowed'] is False
    assert manifest['mandatory_stop'] is True and manifest['assigned_cpu'] is None


@pytest.mark.parametrize('requested,approved', [(False, True), (True, False), (True, 'true'), (1, True)])
def test_launch_requires_explicit_boolean_authorization_before_identity_reads(monkeypatch, tmp_path, requested, approved):
    _, auth, output = launch_fixture(monkeypatch, tmp_path)
    auth['approved_by_user'] = approved
    monkeypatch.setattr(contract, 'source_hashes', lambda root: pytest.fail('identity read before authorization'))
    with pytest.raises(ValueError, match='EXPLICIT'):
        contract.validate_launch(tmp_path, {}, auth, requested=requested, output=output)


@pytest.mark.parametrize('mutation', ['digest', 'plan', 'source', 'environment', 'snapshot', 'cpu', 'output', 'existing'])
def test_launch_rejects_each_binding_drift(monkeypatch, tmp_path, mutation):
    manifest, auth, output = launch_fixture(monkeypatch, tmp_path)
    if mutation == 'digest': auth['manifest_digest'] = 'wrong'
    elif mutation == 'plan': manifest['plan']['T'] = .9
    elif mutation == 'source': manifest['source_hashes'] = {}
    elif mutation == 'environment': manifest['environment'] = {}
    elif mutation == 'snapshot': manifest['snapshot'] = {}
    elif mutation == 'cpu': auth['assigned_cpu'] = True
    elif mutation == 'output': auth['exclusive_output'] = 'wrong'
    elif mutation == 'existing': output.mkdir()
    if mutation != 'digest': auth['manifest_digest'] = contract.digest(manifest)
    with pytest.raises(ValueError):
        contract.validate_launch(tmp_path, manifest, auth, requested=True, output=output)


def test_worker_rechecks_parent_binding(monkeypatch, tmp_path):
    manifest, auth, output = launch_fixture(monkeypatch, tmp_path)
    assert contract.validate_launch(tmp_path, manifest, auth, requested=True, output=output) == 0
    output.mkdir()
    marker = {'manifest_digest': contract.digest(manifest), 'authorization_digest': contract.digest(auth)}
    (output / 'launch_binding.json').write_text(json.dumps(marker))
    assert contract.validate_launch(tmp_path, manifest, auth, requested=True, output=output, worker=True) == 0
    (output / 'launch_binding.json').write_text('{}')
    with pytest.raises(ValueError, match='WORKER'):
        contract.validate_launch(tmp_path, manifest, auth, requested=True, output=output, worker=True)


def test_entrypoint_has_no_scientific_import_before_authorization():
    path = ROOT / 'scripts/resource_applicability/run_track_a_ax2b_h4_v5.py'
    tree = ast.parse(path.read_text())
    assert not any(isinstance(n, ast.ImportFrom) and n.module.endswith('ax2b_h4_science_v5') for n in tree.body)
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module.endswith('ax2b_h4_science_v5')]
    assert len(imports) == 1
    text = path.read_text()
    assert text.index('install_worker_limits(cpu') < text.index('from trottertracks.resource_applicability.ax2b_h4_science_v5')


def test_unauthorized_cli_refuses_before_scientific_imports_or_output(tmp_path):
    runner = ROOT / 'scripts/resource_applicability/run_track_a_ax2b_h4_v5.py'
    output = tmp_path / 'must_not_exist'
    code = ('import importlib.abc, runpy, sys; '
            'guard=type("Guard",(importlib.abc.MetaPathFinder,),'
            '{"find_spec":lambda self,fullname,*args: (_ for _ in ()).throw(AssertionError("scientific import")) '
            'if fullname.split(".")[0] in ("numpy","scipy","qiskit","openfermion") else None})(); '
            'sys.meta_path.insert(0,guard); '
            f'sys.argv=[{str(runner)!r},"--execute","--output",{str(output)!r}]; '
            f'runpy.run_path({str(runner)!r},run_name="__main__")')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and 'authorization' in result.stderr
    assert 'scientific import' not in result.stderr and not output.exists()


def test_call_counters_fail_before_extra_action():
    budget = CallBudget(compile=2, trajectory=1)
    budget.take('compile', 2)
    with pytest.raises(RuntimeError, match='compile'):
        budget.take('compile')
    assert budget.used['compile'] == 2 and budget.used['trajectory'] == 0


@pytest.mark.parametrize('count', [True, -1, 1.5])
def test_invalid_counter_requests(count):
    with pytest.raises(ValueError): CallBudget(compile=count)
    with pytest.raises(ValueError): CallBudget(compile=2).take('compile', count)


def toy_ham(one=None):
    matrix = np.array([[.3, .1j], [-.1j, -.2]]) if one is None else np.asarray(one, dtype=complex)
    return DFHamiltonian(.17, matrix, np.array([.23]), (np.diag([.7, -.4]).astype(complex),), {})


def test_number_sector_and_complex_primitive_certificate():
    ham = toy_ham()
    sector = PhysicalSector.number_sector(n_qubits=2, n_electrons=1)
    report = science.primitive_sector_certificate(ham, sector)
    assert report['matrix_count'] == 2 and report['sector_dimension'] == 2


@pytest.mark.parametrize('coupling', [.1, 1e-30])
def test_spin_sector_rejects_even_tiny_cross_spin_coupling(coupling):
    ham = toy_ham([[.3, coupling], [coupling, -.2]])
    sector = PhysicalSector.spin_sector(n_qubits=2, nelec_alpha=1, nelec_beta=0)
    with pytest.raises(ValueError, match='SPIN_SECTOR'):
        science.primitive_sector_certificate(ham, sector)


def test_sector_certificate_rejects_incomplete_saved_basis():
    ham = toy_ham()
    sector = PhysicalSector(2, np.array([1]), n_electrons=1)
    with pytest.raises(ValueError, match='INCOMPLETE'):
        science.primitive_sector_certificate(ham, sector)


def test_saved_state_bridge_retains_complex_phase_and_never_renormalizes():
    ham = toy_ham()
    sector = PhysicalSector.number_sector(n_qubits=2, n_electrons=1)
    state = np.array([.6, .8j])
    full = np.array([0, .6, .8j, 0])
    indices, qstate = science.checked_basis_bridge(ham, sector, full, state, 1e-12)
    assert indices == (2, 1)
    np.testing.assert_allclose(qstate, [0, .8j, .6, 0])
    with pytest.raises(ValueError, match='norm'):
        science.checked_basis_bridge(ham, sector, 2 * full, state, 1e-12)


def test_dense_reference_matvec_accounting_and_full_sector():
    ham = toy_ham()
    budget = CallBudget(reference_matvecs=4)
    matrix = science.dense_df_qiskit(ham, budget, 4)
    assert budget.used['reference_matvecs'] == 4
    # Independently form a†Ga and its square using Qiskit occupations and
    # Jordan-Wigner parity for the single off-diagonal term.
    one = np.array([[0, 0, 0, 0], [0, .3, .1j, 0], [0, -.1j, -.2, 0], [0, 0, 0, .1]])
    diagonal = np.array([0, .7, -.4, .3])
    expected = .17 * np.eye(4) + one + .23 * np.diag(diagonal ** 2)
    np.testing.assert_allclose(matrix, expected, atol=1e-14)
    with pytest.raises(RuntimeError, match='CALL_BUDGET'):
        science.dense_df_qiskit(ham, budget, 4)
    assert budget.used['reference_matvecs'] == 4


def test_dense_reference_refuses_oversize_before_allocation():
    ham = DFHamiltonian(0, np.eye(9), np.array([]), (), {})
    with pytest.raises(ValueError, match='H4_ONLY'):
        science.dense_df_qiskit(ham, CallBudget(reference_matvecs=0), 20000)
    with pytest.raises(RuntimeError, match='ACTION_CAP'):
        science.dense_df_qiskit(toy_ham(), CallBudget(reference_matvecs=4), 3)


@pytest.mark.parametrize('formula', ['2nd', '4th'])
@pytest.mark.parametrize('T', [.8, -.8])
def test_independent_dense_global_composition(formula, T):
    matrices = [np.array([[.3, .2j], [-.2j, -.1]]), np.array([[.4, .1], [.1, .2]])]
    eigensystems = [_eigendecomposition(m) for m in matrices]
    initial = np.array([.6, .8j])
    actual = science.dense_global_state(initial, eigensystems, T=T, q=2, formula=formula, scalar=.17)
    w = 1 / (2 - 2 ** (1 / 3))
    expected = initial.copy()
    for _ in range(2):
        for value in ((1,) if formula == '2nd' else (w, 1 - 2 * w, w)):
            for m in matrices + matrices[::-1]: expected = expm(-1j * value * T / 4 * m) @ expected
    expected *= np.exp(-1j * .17 * T)
    np.testing.assert_allclose(actual, expected, atol=2e-14)


@pytest.mark.parametrize('K', [2, 4, 6])
def test_spectral_oracle_matches_horner_corrected_raw_and_pf_tail(K):
    matrix = np.array([[.2, .31j], [-.31j, -.17]])
    deterministic_matrix = np.array([[.4, .07], [.07, -.13]])
    initial = np.array([.6, .8j])
    spectral = [('toy', _eigendecomposition(deterministic_matrix))]
    lam, q, r, T, scalar = .73, 3, 2, .8, .17
    (corrected, raw, pf), log_B = science.dense_partial_signals(initial, spectral, _eigendecomposition(matrix),
                T=T, q=q, r=r, K=K, lambda_r=lam, scalar=scalar)
    action = lambda v, t: expm(-1j * t * deterministic_matrix) @ v
    result = partial_s2_signal(initial, [action], lambda v: matrix @ v, lambda_r=lam, T=T,
               q=q, r=r, K=K, phase_energy=scalar, budget=ActionBudget(200, 200))
    assert abs(corrected - result.corrected) < 3e-14
    assert abs(raw - result.raw) < 3e-14 and abs(log_B - result.log_normalization) < 1e-14
    exact = initial.copy()
    for _ in range(q):
        exact = action(exact, T / (2 * q))
        exact = expm(-1j * lam * T / q * matrix) @ exact
        exact = action(exact, T / (2 * q)) * np.exp(-1j * scalar * T / q)
    assert abs(pf - np.vdot(initial, exact)) < 3e-14


def fake_worker(tmp_path, code, **limits):
    output = tmp_path / 'fake_output'
    output.mkdir()
    command = [sys.executable, '-c', code, str(output)]
    return supervise(command, output, total_wall_seconds=limits.get('total', 2),
                     phase_wall_seconds=limits.get('phase', 2), output_bytes=limits.get('disk', 131072),
                     poll_seconds=.01), output


def test_watchdog_non_science_success(tmp_path):
    code = 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); (p/"worker_terminal.json").write_text(json.dumps({"status":"H4_TECHNICAL_PILOT_COMPLETE","completed_correctness_cells":8,"compiled_wrappers":28}))'
    report, output = fake_worker(tmp_path, code)
    assert report['status'] == 'H4_TECHNICAL_PILOT_COMPLETE'
    assert report['mandatory_stop'] is True and report['next_stage_authorized'] is False
    assert (output / 'terminal_status.json').exists()


@pytest.mark.parametrize('kind', ['phase', 'total'])
def test_watchdog_kills_sleeping_native_like_process_and_keeps_partial(tmp_path, kind):
    code = 'import pathlib, sys, time; p=pathlib.Path(sys.argv[1]); (p/"partial.json").write_text("{}"); time.sleep(10)'
    limits = {'phase': .1, 'total': 2} if kind == 'phase' else {'phase': 2, 'total': .1}
    report, output = fake_worker(tmp_path, code, **limits)
    assert report['reason'] == ('PHASE_WALL_CAP' if kind == 'phase' else 'TOTAL_WALL_CAP')
    assert report['worker_exit_code'] < 0 and (output / 'partial.json').exists()
    assert report['retry'] is False


def test_watchdog_aggregate_output_cap(tmp_path):
    code = 'import pathlib, sys, time; p=pathlib.Path(sys.argv[1]); (p/"oversize.bin").write_bytes(b"x"*60000); time.sleep(10)'
    report, _ = fake_worker(tmp_path, code, disk=65536)
    assert report['reason'] == 'OUTPUT_CAP'


def test_watchdog_bounded_log_stops_noisy_non_science_process(tmp_path):
    report, output = fake_worker(tmp_path, 'import sys; sys.stdout.buffer.write(b"x"*1000000)')
    assert report['reason'] == 'WORKER_LOG_CAP'
    assert (output / 'worker.log').stat().st_size <= 65536


def test_watchdog_records_start_failure_without_retry(tmp_path):
    output = tmp_path / 'fake_output'
    output.mkdir()
    report = supervise([str(tmp_path / 'nonexistent_executable')], output,
                       total_wall_seconds=1, phase_wall_seconds=1, output_bytes=131072)
    assert report['reason'] == 'WORKER_START_FAILED' and report['worker_exit_code'] is None
    assert (output / 'terminal_status.json').exists()


@pytest.mark.parametrize('code', [
    'import sys; sys.exit(3)',
    'import pathlib,sys; (pathlib.Path(sys.argv[1])/"worker_terminal.json").write_text("broken")',
    'import pathlib,sys; (pathlib.Path(sys.argv[1])/"worker_terminal.json").write_text("{}")',
])
def test_watchdog_missing_or_malformed_terminal_is_stop(tmp_path, code):
    report, _ = fake_worker(tmp_path, code)
    assert report['status'] == 'H4_TECHNICAL_PILOT_STOP'


def test_watchdog_rejects_skipped_phase(tmp_path):
    code = 'import pathlib,sys,time; (pathlib.Path(sys.argv[1])/"phase_wrapper_cost.json").write_text("{}"); time.sleep(10)'
    report, _ = fake_worker(tmp_path, code)
    assert report['reason'] == 'INVALID_PHASE_ORDER'


def test_worker_limits_apply_before_numerical_imports(tmp_path):
    cpu = min(os.sched_getaffinity(0))
    code = ('import json, pathlib, sys, os, resource; '
            'from trottertracks.resource_applicability.ax2b_limits import install_worker_limits; '
            f'install_worker_limits({cpu}, 134217728, 131072); '
            'p=pathlib.Path(sys.argv[1]); '
            '(p/"limits.json").write_text(json.dumps({"cpu":sorted(os.sched_getaffinity(0)), '
            '"as":resource.getrlimit(resource.RLIMIT_AS), "fsize":resource.getrlimit(resource.RLIMIT_FSIZE), '
            '"numpy_loaded":"numpy" in sys.modules})); sys.exit(2)')
    report, output = fake_worker(tmp_path, code)
    record = json.loads((output / 'limits.json').read_text())
    assert record == {'cpu': [cpu], 'as': [134217728, 134217728],
                      'fsize': [131072, 131072], 'numpy_loaded': False}
    assert report['worker_exit_code'] == 2


def test_worker_address_space_exhaustion_is_stop(tmp_path):
    cpu = min(os.sched_getaffinity(0))
    code = ('import sys; from trottertracks.resource_applicability.ax2b_limits import install_worker_limits; '
            f'install_worker_limits({cpu}, 67108864, 131072); value=bytearray(134217728)')
    report, output = fake_worker(tmp_path, code)
    assert report['status'] == 'H4_TECHNICAL_PILOT_STOP'
    assert b'MemoryError' in (output / 'worker.log').read_bytes()


def test_cost_phase_cannot_run_before_eight_completed_cells(tmp_path):
    pilot = science.Pilot(tmp_path, tmp_path, {'plan': contract.h4_plan()})
    with pytest.raises(ValueError, match='CORRECTNESS_REQUIRED'):
        pilot.costs()
    assert pilot.calls.used['compile'] == 0 and pilot.calls.used['trajectory'] == 0


def test_partial_worker_failure_records_successful_counts_not_attempts(tmp_path, monkeypatch):
    pilot = science.Pilot(tmp_path, tmp_path, {'plan': contract.h4_plan()})
    def fail():
        pilot.calls.take('compile')
        raise ValueError('synthetic setup failure')
    monkeypatch.setattr(pilot, 'setup', fail)
    assert pilot.run() is False
    terminal = json.loads((tmp_path / 'worker_terminal.json').read_text())
    assert terminal['compiled_wrappers'] == 0 and terminal['calls']['compile'] == 1
    assert terminal['accuracy_eligibility'] == 'UNDETERMINED'


def test_correctness_pipeline_on_fixed_two_orbital_fixture(tmp_path):
    """Exercise genuine DF preparation/actions, never the molecular loader."""
    ham = DFHamiltonian(.17, np.array([[.3, .1j], [-.1j, -.2]]), np.array([.23, -.11]),
                         (np.diag([.7, -.4]).astype(complex), np.array([[.2, .07j], [-.07j, .4]])), {})
    sector = PhysicalSector.number_sector(n_qubits=2, n_electrons=1)
    state = np.array([.6, .8j])
    qindices, qstate = science.checked_basis_bridge(ham, sector, np.array([0, .6, .8j, 0]), state, 1e-12)
    plan = contract.h4_plan()
    for cell in plan['correctness_cells']:
        cell['prefix'] = 2 if cell['method'] == 'B1' else 0 if cell['method'] == 'B3' else 1
        cell['id'] = 'TOY_' + cell['id']
    pilot = science.Pilot(tmp_path, tmp_path, {'plan': plan})
    pilot.ham, pilot.sector, pilot.state, pilot.qindices, pilot.qstate = ham, sector, state, qindices, qstate
    budget = CallBudget(reference_matvecs=20)
    pilot.one_matrix = science.dense_df_qiskit(DFHamiltonian(0, ham.one_body, np.array([]), (), {}), budget, 4)
    pilot.fragments = tuple(science.dense_df_qiskit(DFHamiltonian(0, np.zeros((2, 2)), ham.lambdas[i:i+1],
                         (ham.g_matrices[i],), {}), budget, 4) for i in range(2))
    matrix = .17 * np.eye(4) + pilot.one_matrix + sum(pilot.fragments)
    pilot.target = np.vdot(qstate, expm(-1j * .8 * matrix) @ qstate)
    pilot.preparations = {}
    pilot.eigensystems = {'one_body': _eigendecomposition(pilot.one_matrix),
                          **{f'fragment_{i}': _eigendecomposition(m) for i, m in enumerate(pilot.fragments)}}
    pilot.correctness()
    assert pilot.completed == 8
    assert pilot.calls.used['compile'] == 0 and pilot.calls.used['trajectory'] == 0
    records = list(tmp_path.glob('TOY_*_correctness.json'))
    assert len(records) == 8
    for path in records:
        value = json.loads(path.read_text())
        assert value['accuracy_eligibility'] == 'UNDETERMINED'
        assert value['numerical_allowance_certified'] is False
        if value['cell']['method'] == 'B0':
            total, discard, pf = (value['signed_total_error'], value['signed_discard_error'], value['signed_PF_error'])
            assert abs(total['real'] - discard['real'] - pf['real']) < 1e-14
            assert abs(total['imag'] - discard['imag'] - pf['imag']) < 1e-14


def cost_plumbing_fixture(tmp_path, monkeypatch, fail_on=None):
    from qiskit import QuantumCircuit
    from trottertracks.resource_applicability.ax2b_native_df_v5 import NativeEvolution
    pilot = science.Pilot(tmp_path, tmp_path, {'plan': contract.h4_plan()})
    pilot.completed = 8
    pilot.ham = toy_ham(np.diag([.3, -.2]))
    pilot.qindices, pilot.qstate = (2, 1), np.array([0, .8j, .6, 0])
    monkeypatch.setattr(pilot, 'preparation_for', lambda cell: SimpleNamespace(
        deterministic_blocks=(), constant_coefficient=.17))
    def evolution():
        circuit = QuantumCircuit(3)
        circuit.p(-.17 * .8, 2)
        return NativeEvolution(circuit, 2, .8, 4, 'synthetic', 'ordinary', 1,
                               science.canonical_qiskit_circuit_fingerprint(circuit))
    def fake_trajectory(cell, seed):
        pilot.calls.take('trajectory')
        pilot.calls.take('occurrence', cell['q'])
        return object(), (), [[{'synthetic_fixture': True, 'seed': seed}]]
    monkeypatch.setattr(pilot, 'trajectory', fake_trajectory)
    monkeypatch.setattr(science, 'partial_native_from_step_requests', lambda *args, **kw: evolution())
    monkeypatch.setattr(science.QiskitDFPartialS2RepeatedCircuitBuilder, 'build',
                        lambda *args, **kw: SimpleNamespace(circuit=evolution().circuit))
    seen = []
    def fake_compile(wrapper, compiler, **kwargs):
        seen.append(wrapper)
        if fail_on == len(seen):
            raise ValueError('synthetic compiler failure')
        assert wrapper.num_qubits == 3 and wrapper.num_clbits == 1
        return SimpleNamespace(transpiled_circuit=wrapper, actual_circuit_fingerprint=kwargs['actual_circuit_fingerprint'],
            compiler_settings_hash='synthetic compiler', rz_count=1, rz_depth=1, cx_count=2, cx_depth=2,
            total_depth=3, circuit_size=4)
    monkeypatch.setattr(science, 'transpile_and_measure_cost', fake_compile)
    return pilot, seen


def test_complete_wrapper_plumbing_reuses_four_synthetic_trajectories(tmp_path, monkeypatch):
    pilot, seen = cost_plumbing_fixture(tmp_path, monkeypatch)
    pilot.costs()
    assert len(seen) == 28 and pilot.compiled == 28
    assert pilot.calls.used['compile'] == 28 and pilot.calls.used['trajectory'] == 4
    assert pilot.calls.used['occurrence'] == 16
    records = [json.loads(p.read_text()) for p in tmp_path.glob('H4_*_cost.json')]
    assert len(records) == 28
    for record in records:
        paired = [r for r in records if r['task']['cell']['id'] == record['task']['cell']['id']
                  and r['task']['replica'] == record['task']['replica']]
        assert len(paired) == 4 and len({r['event_digest'] for r in paired}) == 1
        assert record['winner_claim'] is False and record['shot_estimate_performed'] is False
    summary = json.loads((tmp_path / 'wrapper_cost_summary.json').read_text())
    assert len(summary['groups']) == 20
    assert {r['n'] for r in summary['groups']} == {1, 2}


def test_wrapper_plumbing_stops_on_first_synthetic_compiler_failure(tmp_path, monkeypatch):
    pilot, seen = cost_plumbing_fixture(tmp_path, monkeypatch, fail_on=3)
    with pytest.raises(ValueError, match='synthetic compiler'):
        pilot.costs()
    assert len(seen) == 3 and pilot.compiled == 2
    assert pilot.calls.used['compile'] == 3
    assert len(list(tmp_path.glob('H4_*_cost.json'))) == 2


def test_v3_authorization_is_not_valid_for_v5(monkeypatch, tmp_path):
    manifest, auth, output = launch_fixture(monkeypatch, tmp_path)
    auth['schema'] = 'track_a_ax2b_h4_authorization_v3'
    with pytest.raises(ValueError, match='AUTHORIZATION_SCHEMA'):
        contract.validate_launch(tmp_path, manifest, auth, requested=True, output=output)


def test_v4_authorization_is_not_valid_for_v5(monkeypatch, tmp_path):
    manifest, auth, output = launch_fixture(monkeypatch, tmp_path)
    auth['schema'] = 'track_a_ax2b_h4_authorization_v4'
    with pytest.raises(ValueError, match='AUTHORIZATION_SCHEMA'):
        contract.validate_launch(tmp_path, manifest, auth, requested=True, output=output)


def test_v4_scientific_plan_is_preserved_with_new_implementation_metadata():
    from trottertracks.resource_applicability.ax2b_h4_contract_v4 import h4_plan as previous
    old, new = previous(), contract.h4_plan()
    for plan in (old, new):
        plan.pop('schema'); plan.pop('implementation')
    assert new == old


def test_v5_retains_registered_science_tasks_seeds_compiler_gates_and_caps():
    from trottertracks.resource_applicability.ax2b_h4_contract import h4_plan
    old, new = h4_plan(), contract.h4_plan()
    assert {k: v for k, v in new.items() if k not in ('schema', 'implementation')} == {
        k: v for k, v in old.items() if k != 'schema'
    }


def test_completed_evolution_and_wrapper_objects_do_not_survive_into_next_group(tmp_path, monkeypatch):
    import gc
    import weakref
    pilot, _ = cost_plumbing_fixture(tmp_path, monkeypatch)
    evolution_refs, wrapper_refs, build_count = [], [], []
    def track(builder):
        def tracked(*args, **kwargs):
            gc.collect()
            # At most the first control policy of THIS group may remain.
            assert sum(ref() is not None for ref in evolution_refs) <= 1
            result = builder(*args, **kwargs)
            evolution_refs.append(weakref.ref(result.circuit))
            build_count.append(None)
            return result
        return tracked
    monkeypatch.setattr(science, 'build_deterministic_native', track(science.build_deterministic_native))
    monkeypatch.setattr(science, 'partial_native_from_step_requests', track(science.partial_native_from_step_requests))
    def compile_stub(wrapper, compiler, **kwargs):
        gc.collect()
        assert not any(ref() is not None for ref in wrapper_refs)
        wrapper_refs.append(weakref.ref(wrapper))
        return SimpleNamespace(transpiled_circuit=wrapper, actual_circuit_fingerprint=kwargs['actual_circuit_fingerprint'],
            compiler_settings_hash='synthetic compiler', rz_count=1, rz_depth=1, cx_count=2, cx_depth=2,
            total_depth=3, circuit_size=4)
    monkeypatch.setattr(science, 'transpile_and_measure_cost', compile_stub)
    pilot.costs(); gc.collect()
    assert len(build_count) == 14 and len(wrapper_refs) == 28
    assert not any(ref() is not None for ref in evolution_refs + wrapper_refs)
    released = [json.loads(p.read_text()) for p in tmp_path.glob('diagnostic_*.json')]
    assert sum(r['kind'] == 'group_released' for r in released) == 7


def test_failure_at_final_group_keeps_24_costs_and_reports_exact_stage_and_traceback(tmp_path, monkeypatch):
    pilot, _ = cost_plumbing_fixture(tmp_path, monkeypatch)
    original_builder = science.build_deterministic_native
    def controlled_failure(*args, **kwargs):
        if kwargs['formula'] == '4th' and kwargs['q'] == 4:
            raise MemoryError('synthetic native allocation')
        return original_builder(*args, **kwargs)
    monkeypatch.setattr(science, 'build_deterministic_native', controlled_failure)
    monkeypatch.setattr(pilot, 'setup', lambda: None)
    monkeypatch.setattr(pilot, 'correctness', lambda: None)
    assert pilot.run() is False
    terminal = json.loads((tmp_path / 'worker_terminal.json').read_text())
    assert terminal['compiled_wrappers'] == terminal['calls']['compile'] == 24
    assert terminal['completed_correctness_cells'] == 8
    assert terminal['reason'] == 'MemoryError:synthetic native allocation'
    failure = json.loads((tmp_path / 'failure_diagnostics.json').read_text())
    assert failure['context']['stage'] == 'native_build'
    assert failure['context']['cell_id'] == 'H4_B1_S4_q4'
    assert failure['context']['control_policy'] == 'ordinary'
    assert failure['traceback_frames'][-1]['function'] == 'controlled_failure'
    assert failure['locals_recorded'] is False
    assert len(list(tmp_path.glob('H4_*_cost.json'))) == 24
    assert pilot._failure_reserve is None


def test_diagnostic_write_failure_still_reports_original_exception(tmp_path, monkeypatch):
    pilot = science.Pilot(tmp_path, tmp_path, {'plan': contract.h4_plan()})
    def fail(): raise MemoryError('primary')
    monkeypatch.setattr(pilot, 'setup', fail)
    monkeypatch.setattr(pilot.trace, 'failure', lambda error: (_ for _ in ()).throw(OSError('secondary')))
    assert pilot.run() is False
    terminal = json.loads((tmp_path / 'worker_terminal.json').read_text())
    assert terminal['reason'] == 'MemoryError:primary'
    assert terminal['failure_diagnostics'] == {'diagnostic_error': 'OSError'}


def test_nested_fingerprint_failure_context_wins_over_outer_stage(tmp_path):
    from trottertracks.resource_applicability.ax2b_diagnostics_v4 import ResourceTrace, fingerprint_stage
    from qiskit import QuantumCircuit
    trace = ResourceTrace(lambda name, payload: (tmp_path / name).write_text(json.dumps(payload)), lambda: 0.)
    try:
        with trace.bind(), trace.stage('native_build', cell_id='synthetic'):
            with fingerprint_stage(QuantumCircuit(1)):
                raise MemoryError('fingerprint')
    except MemoryError as error:
        trace.failure(error)
    failure = json.loads((tmp_path / 'failure_diagnostics.json').read_text())
    assert failure['context'] == {'stage': 'numeric_fingerprint', 'cell_id': 'synthetic', 'instructions': 0}
    assert 'VmSize_bytes' in failure['memory_after_unwind']


def test_diagnostic_record_cap_fails_before_an_extra_write():
    from trottertracks.resource_applicability.ax2b_diagnostics_v4 import ResourceTrace
    trace = ResourceTrace(lambda *args: pytest.fail('extra diagnostic write'), lambda: 0.)
    trace.sequence = 1024
    with pytest.raises(RuntimeError, match='DIAGNOSTIC_RECORD_CAP'):
        trace.event('begin')


def test_real_address_space_exhaustion_preserves_trace_without_scientific_imports(tmp_path):
    cpu = min(os.sched_getaffinity(0))
    code = '''import pathlib, sys, json
from trottertracks.resource_applicability.ax2b_limits import install_worker_limits, exclusive_json
from trottertracks.resource_applicability.ax2b_diagnostics_v4 import ResourceTrace
install_worker_limits(CPU_VALUE, 67108864, 131072)
output = pathlib.Path(sys.argv[1])
reserve = bytearray(1048576)
trace = ResourceTrace(lambda name, value: exclusive_json(output / name, value), lambda: 0.)
try:
    with trace.stage('native_build', cell_id='stdlib_fixture'):
        value = bytearray(134217728)
except MemoryError as error:
    reserve = None
    detail = trace.failure(error)
    exclusive_json(output / 'worker_terminal.json', {'status':'H4_TECHNICAL_PILOT_STOP',
        'completed_correctness_cells':0, 'compiled_wrappers':0,
        'failure_diagnostics':detail,
        'scientific_modules_loaded':any(n in sys.modules for n in ('numpy','scipy','qiskit','openfermion'))})
'''.replace('CPU_VALUE', str(cpu))
    report, output = fake_worker(tmp_path, code)
    assert report['status'] == 'H4_TECHNICAL_PILOT_STOP'
    worker = json.loads((output / 'worker_terminal.json').read_text())
    assert worker['scientific_modules_loaded'] is False
    failure = json.loads((output / 'failure_diagnostics.json').read_text())
    assert failure['exception_type'] == 'MemoryError'
    assert failure['context']['stage'] == 'native_build'
    assert failure['traceback_frames'] and failure['locals_recorded'] is False
