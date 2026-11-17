"""Finish byte-only checking of the already copied completed925 attempt.

SOURCE95 engine files stay frozen. Its first checker assumed7168 was the
maximum, while the first rejected field was7168 and the maximum is7747.
This verifier corrects only that metadata assertion and starts no work.
"""
from pathlib import Path
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,sys,time


def check(code_root,settings_path,copy_root,source_run,prior_evidence):
    code_root=Path(code_root).resolve();sys.path[:0]=[str(code_root/'src')]
    from trottertracks.rpe_pbs import Gateway
    from trottertracks import rpe_q3_worker  # Existing trusted exact-reader policy.
    spec=importlib.util.spec_from_file_location('fixed_checker',code_root/'scripts/check_rpe_pbs_exact_transfer.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    settings=json.loads(Path(settings_path).read_text());settings.update(remote_root=copy_root,namespace_by_run=False)
    assert settings['transport_source']=='95a74e4486131a97bd5235221bf2d8ab9f99f591'
    assert settings['gateway_python']=='/home/AbeHiromu/projects/rpe-pbs-20261118/runtime/venv311rc1/bin/python'
    job='98693b829295fc884330';attempt='18c2c7bc2d7b4a5581ef2543b1359e1b';token='exact-transfer-metadata'
    baseline=Path(source_run)/'distributed/transfers'/job/attempt
    original=(baseline/'OUTCOME.json').read_bytes();observed=json.loads(original)
    facts=list(checker.integer_facts(observed))
    assert {item['path']:item['digits']for item in facts}=={
        '$.MSE_lower.numerator':7168,'$.MSE_lower.denominator':7174,
        '$.MSE_upper.numerator':7741,'$.MSE_upper.denominator':7747}
    recorded=json.loads(Path(prior_evidence).read_text())['attempts'][0]['files']
    assert hashlib.sha256(original).hexdigest()==recorded['OUTCOME.json']['sha256']
    gateway=Gateway(settings);started=time.monotonic();hashes={}
    try:
        normal=gateway.call(dict(action='exchange'))
        requests=normal['requests'][token];assert len(requests)==1
        request=requests[0];payload=request['payload']
        assert request['endpoint']=='/v1/complete'and payload['job_id']==job and payload['attempt_id']==attempt
        raw=base64.b64decode(payload['files']['OUTCOME.json'])
        assert raw==original and json.loads(raw)==observed
        assert list(checker.integer_facts(json.loads(raw)))==facts
        initial_command=gateway.process.args
        gateway.call(dict(action='exchange',responses={token:{request['request_id']:{'ok':True,'result':{'ok':True}}}}))
        gateway.close()
        after=gateway.call(dict(action='harvest',token=token))
        assert len(after['attempts'])==1 and not after['incomplete']
        recovered=after['attempts'][0]
        assert recovered['assignment']['job_id']==job and recovered['assignment']['attempt_id']==attempt
        raw=base64.b64decode(recovered['files']['OUTCOME.json'])
        assert raw==original and json.loads(raw)==observed
        assert list(checker.integer_facts(json.loads(raw)))==facts
        for name,encoded in payload['files'].items():
            data=base64.b64decode(encoded)
            assert base64.b64decode(recovered['files'][name])==data
            assert (baseline/name).read_bytes()==data
            sha=hashlib.sha256(data).hexdigest();assert recorded[name]['sha256']==sha
            hashes[name]=sha
        for command in(initial_command,gateway.process.args):
            assert shlex.split(command[-1])[0]==settings['gateway_python']
        assert after['attempts'][0]['state']['state']=='TRANSFERRED'
    finally:gateway.close()
    return dict(status='PASS',scope='SAVED_COMPLETED_BYTES_AND_EXACT_INTEGER_TRANSFER_ONLY',
        engine_source='95a74e4486131a97bd5235221bf2d8ab9f99f591',
        original_controller_source='925985b4ea76a15f8a07c507179d4fa6c492eabd',
        scientific_source='28d976b99cdf8329adcc1b79327e329e947ef774',
        original_result_commit='840525ed6a2dc3e018dd568898b9bc9e06349578',
        original_outcome_sha256=hashlib.sha256(original).hexdigest(),
        job_id=job,attempt_id=attempt,copied_spool=copy_root,
        gateway_script=settings['gateway_script'],gateway_python=settings['gateway_python'],
        exact_integer_facts=facts,preserved_file_sha256=hashes,
        ordinary_completion_exchange_bytes_exact=True,post_exit_harvest_bytes_exact=True,
        decoded_outcome_dict_exact=True,reconnect_uses_same_dedicated_python=True,
        completed_saved_attempt_state='TRANSFERRED',
        verifier_correction=dict(original_assumption='Maximum integer digits==7168',
            observed='First failing lower numerator7168; maximum upper denominator7747',
            correction='Require all four exact integer fields and their original digit lengths/digests',
            engine_source_changed=False,scientific_guards_changed=False,
            copied_spool_reused=True,re_copy=False),
        original_run_changed=False,original_attempt_changed=False,original_settings_changed=False,
        float_conversion=False,integer_truncation=False,new_scientific_q3_calls=0,new_compile_calls=0,
        new_signal_calls=0,refits=0,new_queue_adoptions=0,new_pbs_submissions=0,
        same_controller_clock_seconds=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root',type=Path,required=True);parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--copy-root',required=True);parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--prior-evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=check(a.code_root,a.settings,a.copy_root,a.source_run,a.prior_evidence)
    result['corrected_verifier']=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        role='Technical metadata verifier; distinct from fixed95 engine')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',output=str(a.output))))
