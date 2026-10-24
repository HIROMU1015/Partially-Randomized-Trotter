"""Approval/IO and mocked end-to-end tests; every input is artificial."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock,patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_stage1 as s
from trottertracks import c1_stage1_execution as run


def artificial_packet(path):
    h=np.diag([-.3,-.1,.2,.4]); g=np.zeros((4,)*4)
    with path.open('xb') as stream:
        np.savez(stream,one_mo=h,eri_chemist=g,nuclear_repulsion=np.array(.5),
                 mo_coeff=np.eye(4),df_one_body=np.kron(h,np.eye(2)).astype(complex),
                 df_lambdas=np.ones(10),df_g_matrices=np.zeros((10,8,8),complex),
                 energy=np.array('DO_NOT_READ_SAVED_SCIENCE'))


def packet(tmp_path):
    input_path=tmp_path/'ARTIFICIAL.npz';artificial_packet(input_path)
    source=tmp_path/'source.json';run.save(source,dict(implementation_commit='mock',files={}))
    contract=tmp_path/'contract.json'
    run.save(contract,dict(scope=run.SCOPE,conditions_adopted=True,execution_authorized=False,
             grant_issued=False,input=dict(sha256=run.sha(input_path)),
             fit_settings=s.FIT_SETTINGS,allowance={'version':s.ALLOWANCE_VERSION},
             compatibility={'version':s.COMPAT_VERSION},
             source_manifest_sha256=run.sha(source),implementation_source_commit='mock',
             call_limits=dict(fit={},evaluation_max_R8=s.expected_evaluation_calls(8,36))))
    with patch.object(run,'P_H4_SHA256',run.sha(input_path)):
        return run.preflight(tmp_path,contract,source,input_path)


def test_preflight_does_not_decode_arrays(tmp_path):
    with patch.object(np,'load',side_effect=AssertionError('no numeric decode')):
        v=packet(tmp_path)
    assert v['contract']['execution_authorized'] is False


def test_hash_drift_stops(tmp_path):
    v=packet(tmp_path);v['input'].write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='INPUT_HASH'):
        run.preflight(v['root'],v['contract_path'],v['source_path'],v['input'])


def test_missing_or_wrong_approval_stops_before_numeric_load(tmp_path):
    v=packet(tmp_path);ap=tmp_path/'approval.json';run.save(ap,dict(approved=True))
    output=tmp_path/'out'
    with patch.object(run.subprocess,'check_output',return_value='mock'), patch.object(np,'load',side_effect=AssertionError('no load')):
        with pytest.raises(PermissionError):run.execute(v,ap,output,'unused')
    assert not output.exists()


def test_journal_quota_and_partial_attempts(tmp_path):
    trace=run.journal(tmp_path,'fit',dict(fit=1))
    trace('fit')
    with pytest.raises(ValueError,match='CALL_LIMIT'):trace('fit')
    assert len((tmp_path/'fit_calls.jsonl').read_text().splitlines())==1


def test_input_whitelist_and_dtypes(tmp_path):
    path=tmp_path/'ARTIFICIAL.npz';artificial_packet(path)
    a=run.load_input(path,evaluation=True)
    assert 'energy' not in a and set(a)=={'one_mo','eri_chemist','nuclear_repulsion','mo_coeff','df_one_body','df_lambdas','df_g_matrices'}


def test_mock_cross_environment_run_export_and_STOP(tmp_path):
    v=packet(tmp_path);output=tmp_path/'out';ap=tmp_path/'approval.json';run.save(ap,{})
    fit=Mock()
    def fake_process(command,**kwargs):
        fit(command)
        a=run.load_input(v['input']);J=np.diag([-.2,.3,.4,.6])[None];U=np.eye(4)[None]
        result=SimpleNamespace(success=True,status=0,message='mock',x=np.zeros(1),jac=np.zeros(1),fun=0.,nit=1,nfev=1,njev=1)
        calls=s.Calls();calls.mark('input_load')
        arrays,metadata=s.fit_once(a['one_mo'],a['eri_chemist'],a['nuclear_repulsion'],calls,fit_function=Mock(return_value=(J,U,result)))
        run.write_export(output,arrays,metadata,a,v,calls)
        return SimpleNamespace(wait=lambda:0)
    with patch.object(run,'verify_approval',return_value={}), patch.object(run,'runtime_identity',return_value={'type':'mock'}), patch.object(run.subprocess,'Popen',side_effect=fake_process):
        result=run.execute(v,ap,output,Path('mock-fit-python'))
    assert fit.call_count==1 and '--worker-fit' in fit.call_args.args[0]
    assert result['status']=='STAGE1_REPRESENTATION_PASS'
    assert run.read(output/'STOP.json')['mandatory_stop'] is True
    assert run.read(output/'fit_export.json')['calls']['fit']==1
    assert len(list((output/'operators').glob('*.npz')))==10
    (output/'fit_export.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='EXPORT_PROVENANCE'):
        run.load_export(output,v)


def test_worker_failure_leaves_partial_STOP_no_retry(tmp_path):
    v=packet(tmp_path);output=tmp_path/'out';ap=tmp_path/'approval.json';run.save(ap,{})
    child=Mock(return_value=SimpleNamespace(wait=lambda:2))
    with patch.object(run,'verify_approval',return_value={}),patch.object(run,'runtime_identity',return_value={}),patch.object(run.subprocess,'Popen',child):
        with pytest.raises(RuntimeError,match='FIT_WORKER_STOP'):
            run.execute(v,ap,output,Path('mock'))
    assert child.call_count==1
    assert run.read(output/'STOP.json')['automatic_retry'] is False


def test_interrupted_parent_stops_only_its_child_and_retains_STOP(tmp_path):
    v=packet(tmp_path);output=tmp_path/'out';ap=tmp_path/'approval.json';run.save(ap,{})
    process=SimpleNamespace(pid=12345,wait=Mock(side_effect=[KeyboardInterrupt(),0]))
    with patch.object(run,'verify_approval',return_value={}),patch.object(run,'runtime_identity',return_value={}),patch.object(run.subprocess,'Popen',return_value=process),patch.object(run.os,'killpg') as kill:
        with pytest.raises(KeyboardInterrupt):run.execute(v,ap,output,Path('mock'))
    kill.assert_called_once_with(12345,run.signal.SIGTERM)
    assert run.read(output/'STOP.json')['automatic_retry'] is False


def test_CLI_execution_requires_approval_before_any_file_read(tmp_path):
    process=subprocess.run([sys.executable,str(ROOT/'scripts/run_research_c_c1_stage1.py'),
         '--execute','--contract','missing','--source','missing','--input','missing'],capture_output=True,text=True)
    assert process.returncode==2 and 'new explicit approval' in process.stderr


def test_previous_approval_scope_cannot_authorize_compatibility_run(tmp_path):
    v=packet(tmp_path);ap=tmp_path/'approval.json'
    run.save(ap,dict(scope='C1_STAGE1_P_H4_ONE_FIT_V1',approved=True))
    with patch.object(run.subprocess,'check_output',return_value='mock'):
        with pytest.raises(PermissionError,match='NEW_EXPLICIT'):
            run.verify_approval(v,ap,tmp_path/'fresh')
