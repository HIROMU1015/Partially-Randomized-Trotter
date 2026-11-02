"""Additional frozen-source tests: off-domain arrays, mocked runner, metadata.

Known-bug reproduction tests assert the observed behavior, not contract fitness.
They do not execute the registered runner/worker or create an authorization commit.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.algorithm_codesign.e_r3_identity_diagnostic import evaluation as ev
from trottertracks.algorithm_codesign.e_r3_identity_diagnostic import launch
from trottertracks.algorithm_codesign.df_correction_comparison import storage
from trottertracks.algorithm_codesign.df_correction_comparison import guard
from trottertracks.algorithm_codesign.df_correction_comparison.intervals import configure


def frames():
    return [np.array([[3/5], [4j/5]], complex),
            np.array([[5/13], [12/13]], complex),
            np.array([[8/17], [15j/17]], complex)]


def record_evidence(name, value):
    folder = Path(os.environ['E_R3_FINAL_AUDIT_TEST_OUTPUT'])
    storage.dump_json(folder / (name + '.json'), value)


def observe_mock_payload(tmp_path, monkeypatch, action, save=None):
    from trotterlib import rte_compiled_cost as shared
    configure()
    calls = []
    def payload(q):
        calls.append(1)
        return action(q)
    monkeypatch.setattr(shared, '_circuit_payload', payload)
    helper = ev.legacy_helpers(ROOT / 'scripts/tracks/algorithm_codesign/audit_df_sparse_correction.py')
    path = tmp_path / 'off-domain.json'
    if save is None:
        def save(record):
            storage.dump_json(path, record, replace_existing=path.exists())
    result = ev.observe_path('legacy', frames(), 'fixed-nonmatching-placeholder', helper, save, {})
    return result, calls


def test_nonnull_mismatch_preserves_expected_and_single_attempt(tmp_path, monkeypatch):
    payload = {'instructions': [], 'global_phase': 0, 'diagnostic_mock': True}
    result, calls = observe_mock_payload(tmp_path, monkeypatch, lambda q: payload)
    assert calls == [1]
    assert result['actual_fingerprint'] == ev.payload_digest(payload)
    assert result['historical_expected_fingerprint'] == 'fixed-nonmatching-placeholder'
    assert result['observation'] == 'MISMATCH' and result['fingerprint_exception'] is None


@pytest.mark.parametrize('error', [TypeError('mock parameter'), ValueError('mock definition')])
def test_none_fallback_records_exception_and_attempt_once(tmp_path, monkeypatch, error):
    def fail(q):
        raise error
    result, calls = observe_mock_payload(tmp_path, monkeypatch, fail)
    assert calls == [1] and result['actual_fingerprint'] is None
    assert result['shared_None_fallback'] and result['observation'] == 'EXCEPTION'
    assert result['fingerprint_exception']['message'] == str(error)


def minimal_observation():
    names = ['frame_0', 'frame_1', 'frame_2', 'Q_0', 'Q_1', 'Q_2', 'T', 'U', 'input_target', 'actual_gate_parameter']
    return {'input_arrays': {k: {'dtype_str': '<c16', 'shape': [2,2], 'raw_bytes_hex': 'aa'} for k in names},
            'recursive_payload': {'instructions': [{'operation': {'name': 'circuit-11'}}]}}


@pytest.mark.parametrize('difference', ['name', 'matrix', 'both'])
def test_matrix_bytes_and_auto_name_categories_remain_separate(difference):
    a = minimal_observation(); b = copy.deepcopy(a)
    if difference in {'name', 'both'}:
        b['recursive_payload']['instructions'][0]['operation']['name'] = 'circuit-19'
    if difference in {'matrix', 'both'}:
        b['input_arrays']['Q_0']['raw_bytes_hex'] = 'bb'
    result = ev.compare_observations(a,b)
    assert result['first_different_stage'] == ('Q_0' if difference in {'matrix', 'both'} else None)
    assert result['automatic_names_only'] == (difference in {'name', 'both'})
    assert result['scientific_winner'] is None and not result['acceptance_guard_changed']
    # automatic_names_only refers to PAYLOAD differences; matrix comparison is independent.
    if difference == 'both':
        record_evidence('matrix_and_name_independent_fields', result)


def test_utf8_chunks_restore_exact_serialized_bytes(tmp_path):
    value = {'synthetic_text': '軌道🙂' * 90000}
    p = tmp_path / 'chunked.json'
    receipt = storage.dump_json(p, value, ensure_ascii=False)
    expected = (json.dumps(value, default=str, indent=2, ensure_ascii=False) + '\n').encode()
    assert receipt['storage'] == 'chunks'
    assert b''.join(storage.iter_verified_bytes(p)) == expected
    assert receipt['original_sha256'] == hashlib.sha256(expected).hexdigest()
    assert storage.load_json(p) == value
    assert all((tmp_path / r['path']).stat().st_size <= 512*1024 for r in receipt['chunks'])


@pytest.mark.parametrize('corruption', ['bytes', 'missing', 'order', 'whole_hash', 'escape'])
def test_chunk_corruption_is_rejected_before_value_return(tmp_path, corruption):
    p = tmp_path / 'record.json'
    storage.dump_json(p, {'synthetic': 'x' * 600000})
    index = storage.index_path(p); receipt = json.loads(index.read_bytes())
    chunk = tmp_path / receipt['chunks'][0]['path']
    if corruption == 'bytes': chunk.write_bytes(chunk.read_bytes() + b'x')
    elif corruption == 'missing': chunk.unlink()
    elif corruption == 'order': receipt['chunks'][0]['order'] = 9
    elif corruption == 'whole_hash': receipt['original_sha256'] = '0'*64
    elif corruption == 'escape': receipt['chunks'][0]['path'] = '../untrusted.txt'
    if corruption in {'order', 'whole_hash', 'escape'}:
        index.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, FileNotFoundError)):
        storage.load_json(p)


def test_failed_atomic_replacement_leaves_old_complete_generation(tmp_path, monkeypatch):
    p = tmp_path / 'prefix.json'; storage.dump_json(p, {'phase': 'OLD_COMPLETE'})
    original = p.read_bytes()
    monkeypatch.setattr(storage.os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('mock disk')))
    with pytest.raises(OSError, match='mock disk'):
        storage.dump_json(p, {'synthetic': 'x'*600000}, replace_existing=True)
    assert p.read_bytes() == original
    assert storage.load_json(p) == {'phase': 'OLD_COMPLETE'}


def test_late_observation_save_failure_propagates(tmp_path, monkeypatch):
    def save(record):
        if record['phase'] == 'IDENTITY_OBSERVATION_SAVED': raise OSError('mock final save')
    with pytest.raises(OSError, match='mock final save'):
        observe_mock_payload(tmp_path, monkeypatch, lambda q: {'instructions': []}, save)


def test_prohibited_native_entry_becomes_technical_stop(tmp_path, monkeypatch):
    import qiskit
    def forbidden(q): return qiskit.transpile(q)
    with pytest.raises(PermissionError, match='technical STOP'):
        observe_mock_payload(tmp_path, monkeypatch, forbidden)
    result = storage.load_json(tmp_path / 'off-domain.json')
    assert result['definition_work']['native_transpile_blocked_entries'] == 1
    assert result['actual_fingerprint'] is None
    assert sys.getprofile() is None


@pytest.mark.parametrize('failure', ['install', 'restore'])
def test_reproduce_profile_lifecycle_failure_misclassified_as_observation(tmp_path, monkeypatch, failure):
    state = {'profile': None, 'set_calls': []}
    def set_profile(value):
        state['set_calls'].append('install' if value is not None else 'restore')
        if (failure == 'install' and value is not None) or (failure == 'restore' and value is None):
            raise RuntimeError('mock profile ' + failure + ' failed')
        state['profile'] = value
    monkeypatch.setattr(sys, 'getprofile', lambda: state['profile'])
    monkeypatch.setattr(sys, 'setprofile', set_profile)
    result, calls = observe_mock_payload(tmp_path, monkeypatch, lambda q: {'instructions': []})
    assert result['observation'] == 'EXCEPTION' and result['actual_fingerprint'] is None
    assert result['phase'] == 'IDENTITY_OBSERVATION_SAVED'
    assert len(calls) == (0 if failure == 'install' else 1)
    assert result['fingerprint_exception']['message'] == 'mock profile ' + failure + ' failed'
    if failure == 'restore': assert state['profile'] is not None
    # Passing this reproduction confirms a blocker. Production SHOULD technical-STOP.
    record_evidence('known_bug_profile_' + failure, {
        'source': '8377c1d12439cabab3b97bcc1f57ac030a479785',
        'fault': failure, 'observed_path_returned': True, 'observed_phase': result['phase'],
        'observed_observation': result['observation'], 'actual': result['actual_fingerprint'],
        'payload_calls': len(calls), 'profile_left_installed': state['profile'] is not None,
        'exception': result['fingerprint_exception'], 'required_outcome': 'TECHNICAL_INCONCLUSIVE',
        'production_fault_observed': False, 'synthetic_fault_injection': True,
        'contract_fitness': False})


@pytest.mark.parametrize('kind', ['output', 'log', 'cpu', 'rss'])
def test_fixed_supervisor_rejects_mock_cap_breach(tmp_path, monkeypatch, kind):
    supervisor = guard.Supervisor(guard.Limits(float('inf'), 120, 768, 512), tmp_path, 8388608-262144, 1048576)
    if kind == 'output': (tmp_path/'synthetic.bin').write_bytes(b'x'*(8388608-262144+1))
    elif kind == 'log': (tmp_path/'synthetic.log').write_bytes(b'x'*(1048576+1))
    elif kind == 'cpu': monkeypatch.setattr(supervisor, 'cpu', lambda: supervisor.cpu_start+121)
    elif kind == 'rss':
        monkeypatch.setattr(guard, 'proc_snapshot', lambda: {os.getpid(): {'ppid':0,'state':'R','cpu':0,'rss':769}})
    with pytest.raises(guard.CapFailure): supervisor.check_total()


def mock_runner(tmp_path, monkeypatch, mode):
    loaded = runpy.run_path(str(ROOT/'scripts/tracks/algorithm_codesign/run_e_r3_identity_diagnostic.py'), run_name='audit_import_only')
    main = loaded['main']; g = main.__globals__
    from trottertracks.algorithm_codesign.df_correction_comparison import worker as prior_worker
    monkeypatch.setattr(prior_worker, 'runtime_check', lambda *args: None)
    art = tmp_path / 'mock-preparation'; art.mkdir()
    contract = json.loads((ROOT/launch.ART/'contract_v1.json').read_bytes())
    (art/'contract_v1.json').write_text(json.dumps(contract))
    out = tmp_path / 'mock-result'
    provenance = {'source':'off-domain-mock', 'authorization':'NO_COMMIT_CREATED',
                  'branch':'mock', 'remote_SHA':'mock'}
    saved = {'status':'E_R3_IDENTITY_DIAGNOSTIC_COMPLETE', 'path_order':['current','legacy'],
             'provenance':{}, 'path_observations':{'current':'MISMATCH','legacy':'EXCEPTION'}}
    class FakeSupervisor:
        def __init__(self, *a, **kw): pass
        def check_total(self): pass
        def phase(self, *a, **kw):
            if mode == 'cap': raise guard.CapFailure('mock worker RSS cap')
            if mode == 'interrupted': raise KeyboardInterrupt('mock interrupt')
            if mode == 'missing': return
            saved['provenance'] = dict(provenance, marker_sha256=g['sha'](out/'one_shot_marker.json'))
            if mode == 'partial': saved['status'] = 'PARTIAL_PREFIX'
            if mode == 'identity': saved['provenance']['source'] = 'tampered'
            storage.dump_json(out/'identity_result.json',saved)
        def receipt(self): return {'mocked':True,'retries':0}
    for k,v in {'ROOT':tmp_path, 'ART':Path('mock-preparation'), 'RESULT':Path('mock-result'),
                'verify_launch':lambda root:dict(provenance),'Supervisor':FakeSupervisor}.items():
        monkeypatch.setitem(g,k,v)
    monkeypatch.setattr(sys,'argv',['mock-runner'])
    return main,out


@pytest.mark.parametrize('mode', ['cap','interrupted','missing','partial','identity','complete'])
def test_mocked_runner_failure_and_completion_terminal_STOP(tmp_path, monkeypatch, mode):
    main,out = mock_runner(tmp_path,monkeypatch,mode)
    if mode == 'complete': main()
    else:
        with pytest.raises(SystemExit) as caught: main()
        assert caught.value.code == 2
    terminal = storage.load_json(out/'terminal_status.json')
    assert terminal['mandatory_STOP'] and terminal['retries'] == 0
    assert not terminal['historical_cause_unique']
    assert terminal['status'] == ('E_R3_IDENTITY_DIAGNOSTIC_COMPLETE' if mode == 'complete'
                                 else 'E_R3_IDENTITY_DIAGNOSTIC_TECHNICAL_INCONCLUSIVE')
    if mode != 'complete': assert terminal['path_observations'] is None
    assert (out/'one_shot_marker.json').is_file()


def test_mocked_terminal_storage_failure_keeps_consumed_marker(tmp_path,monkeypatch):
    main,out = mock_runner(tmp_path,monkeypatch,'cap')
    original = main.__globals__['dump_json']
    def fail(path,data,*a,**kw):
        if Path(path).name=='terminal_status.json': raise OSError('mock unavailable storage')
        return original(path,data,*a,**kw)
    monkeypatch.setitem(main.__globals__,'dump_json',fail)
    with pytest.raises(OSError): main()
    assert (out/'one_shot_marker.json').exists() and not (out/'terminal_status.json').exists()

