"""Observer lifecycle regression controls; no registered diagnostic input."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.e_r3_identity_diagnostic import evaluation as ev
from trottertracks.algorithm_codesign.e_r3_identity_diagnostic import launch,worker
from trottertracks.algorithm_codesign.df_correction_comparison.storage import dump_json,load_json
from trottertracks.algorithm_codesign.df_correction_comparison.intervals import configure


def frames():
    return [np.array([[3/5],[4j/5]],complex),np.array([[5/13],[12/13]],complex),np.array([[8/17],[15j/17]],complex)]


def mocked_profile(monkeypatch,mode):
    state={'hook':None,'sets':[],'queries':0}
    def get():
        state['queries']+=1
        if mode=='query_error' and state['queries']==1:raise RuntimeError('mock query error')
        return state['hook']
    def set_(hook):
        state['sets'].append('install' if hook is not None else 'restore')
        if mode=='install_error' and hook is not None:raise RuntimeError('mock install error')
        if mode=='install_partial_error' and hook is not None:
            state['hook']=hook;raise RuntimeError('mock partially installed hook')
        if mode=='install_silent' and hook is not None:return
        if mode=='restore_error' and hook is None:raise RuntimeError('mock restore error')
        if mode=='restore_silent' and hook is None:return
        if mode=='both_errors':raise RuntimeError('mock install and cleanup errors')
        state['hook']=hook
    monkeypatch.setattr(sys,'getprofile',get);monkeypatch.setattr(sys,'setprofile',set_)
    return state


@pytest.mark.parametrize('mode,stage',[
    ('query_error','context_read'),('install_error','installation'),
    ('install_partial_error','installation'),('install_silent','installation'),
    ('restore_error','restoration'),('restore_silent','restoration'),('both_errors','installation')])
def test_observer_lifecycle_failures_are_technical_and_cleanup_is_not_retried(monkeypatch,mode,stage):
    state=mocked_profile(monkeypatch,mode);counter=ev.DefinitionCalls();body=[]
    with pytest.raises(ev.ObserverFailure,match='technical STOP'):
        with counter:body.append('payload-placeholder')
    assert stage in counter.lifecycle['exceptions']
    assert state['sets'].count('restore')<=1
    assert len(body)==(1 if mode.startswith('restore') else 0)
    if mode.startswith('install'):
        assert counter.lifecycle['restoration_verified'] and state['hook'] is None
    if mode=='both_errors':assert set(counter.lifecycle['exceptions'])=={'installation','restoration'}


def test_existing_profiler_is_never_removed(monkeypatch):
    state=mocked_profile(monkeypatch,'normal');old=object();state['hook']=old
    counter=ev.DefinitionCalls()
    with pytest.raises(ev.ObserverFailure,match='Fresh unprofiled'):
        with counter:pytest.fail('must not acquire payload')
    assert state['hook'] is old and state['sets']==[]
    assert counter.counts['recursive_payload_attempts']==0


@pytest.mark.parametrize('mode', ['install_error','install_silent','restore_error','restore_silent'])
def test_observe_saves_lifecycle_failure_preserves_acquired_payload_then_stops(tmp_path,monkeypatch,mode):
    from trotterlib import rte_compiled_cost as shared
    configure();state=mocked_profile(monkeypatch,mode);calls=[]
    payload={'instructions':[],'mock_off_domain':True,'global_phase':0}
    def acquire(q):calls.append(1);return payload
    monkeypatch.setattr(shared,'_circuit_payload',acquire)
    helper=ev.legacy_helpers(ROOT/'scripts/tracks/algorithm_codesign/audit_df_sparse_correction.py')
    path=tmp_path/'off-domain.json'
    def save(record):dump_json(path,record,replace_existing=path.exists())
    with pytest.raises(ev.ObserverFailure,match='technical STOP'):
        ev.observe_path('legacy',frames(),'fixed-placeholder',helper,save,{'off_domain':True})
    saved=load_json(path);attempts=int(mode.startswith('restore'))
    assert len(calls)==attempts==saved['definition_work']['recursive_payload_attempts']
    assert saved['observation']=='TECHNICAL_OBSERVER_FAILURE'
    assert saved['phase']=='TECHNICAL_OBSERVER_FAILURE_SAVED'
    assert saved['observer_exception'] is not None and saved['fingerprint_exception'] is None
    assert not saved['shared_None_fallback']
    if attempts:
        assert saved['recursive_payload']==payload
        assert saved['actual_fingerprint']==ev.payload_digest(payload)
        assert state['hook'] is not None
    else:
        assert saved['recursive_payload'] is None and saved['actual_fingerprint'] is None
    evidence=Path(os.environ['E_R3_OBSERVER_REPAIR_OUTPUT'])/(mode+'.json')
    dump_json(evidence,{'scope':'off-domain mock ONLY','registered_input_used':False,'record':saved,
                        'technical_STOP_raised':True,'payload_calls':len(calls),'cleanup_attempts':state['sets'].count('restore')})


@pytest.mark.parametrize('error', [TypeError('body type error'),ValueError('body value error')])
def test_body_and_restoration_errors_are_both_retained(tmp_path,monkeypatch,error):
    from trotterlib import rte_compiled_cost as shared
    configure();mocked_profile(monkeypatch,'restore_error')
    def acquire(q):raise error
    monkeypatch.setattr(shared,'_circuit_payload',acquire)
    helper=ev.legacy_helpers(ROOT/'scripts/tracks/algorithm_codesign/audit_df_sparse_correction.py')
    path=tmp_path/'double-failure.json'
    def save(record):dump_json(path,record,replace_existing=path.exists())
    with pytest.raises(ev.ObserverFailure):ev.observe_path('legacy',frames(),'fixed',helper,save,{})
    saved=load_json(path)
    assert saved['observation']=='TECHNICAL_OBSERVER_FAILURE'
    assert saved['fingerprint_exception']['message']==str(error)
    assert saved['observer_lifecycle']['exceptions']['restoration']
    assert saved['shared_None_fallback'] and saved['definition_work']['recursive_payload_attempts']==1


def test_normal_observer_setup_restore_verified_and_hash_unchanged(tmp_path,monkeypatch):
    from trotterlib import rte_compiled_cost as shared
    configure();state=mocked_profile(monkeypatch,'normal')
    payload={'instructions':[],'global_phase':0,'fingerprint_policy':'unchanged-test-policy'}
    monkeypatch.setattr(shared,'_circuit_payload',lambda q:payload)
    helper=ev.legacy_helpers(ROOT/'scripts/tracks/algorithm_codesign/audit_df_sparse_correction.py')
    path=tmp_path/'normal.json'
    def save(record):dump_json(path,record,replace_existing=path.exists())
    expected=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    result=ev.observe_path('legacy',frames(),expected,helper,save,{})
    assert result['observation']=='MATCH' and result['actual_fingerprint']==expected
    assert result['observer_lifecycle']['installation_verified'] and result['observer_lifecycle']['restoration_verified']
    assert result['observer_lifecycle']['exceptions']=={} and result['observer_exception'] is None
    assert result['definition_work']['recursive_payload_attempts']==1 and state['hook'] is None


def test_worker_mock_does_not_start_legacy_after_current_observer_failure(tmp_path,monkeypatch):
    from trottertracks.algorithm_codesign.df_correction_comparison import worker as oldworker,native_evaluation
    configure();out=tmp_path/launch.RESULT;out.mkdir(parents=True)
    art=tmp_path/launch.ART;art.mkdir(parents=True)
    contract=json.loads((ROOT/launch.ART/'contract_v1.json').read_bytes())
    (art/'contract_v1.json').write_text(json.dumps(contract))
    monkeypatch.setattr(worker,'__file__',str(tmp_path/'src/trottertracks/algorithm_codesign/e_r3_identity_diagnostic/worker.py'))
    monkeypatch.setattr(worker,'claim_child',lambda root:{'off_domain':True})
    monkeypatch.setattr(worker,'saved_frame_rows',lambda *args:'mock-off-domain')
    monkeypatch.setattr(oldworker,'runtime_check',lambda *args:None)
    monkeypatch.setattr(native_evaluation,'deserialize_frames',lambda rows:frames())
    monkeypatch.setattr(ev,'legacy_helpers',lambda path:{})
    seen=[]
    def fail(label,frames,expected,helper,save,provenance):
        seen.append(label);save({'phase':'TECHNICAL_OBSERVER_FAILURE_SAVED','off_domain':True})
        raise ev.ObserverFailure('mock worker observer failure; technical STOP')
    monkeypatch.setattr(ev,'observe_path',fail);monkeypatch.setattr(sys,'argv',['off-domain-worker-mock'])
    with pytest.raises(ev.ObserverFailure):worker.main()
    assert seen==['current'] and not (out/'legacy.json').exists()
    assert not (out/'identity_result.json').exists()


def test_v2_namespace_cannot_reuse_old_pending_authorization_or_old_result():
    assert str(launch.ART).endswith('/v2') and str(launch.RESULT).endswith('/v2')
    assert launch.RECEIPT.name.endswith('_v2.md')
    assert not (ROOT/launch.RESULT).exists()
    with pytest.raises(PermissionError,match='Separate explicit'):launch.verify_launch(ROOT)
