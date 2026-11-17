"""Transfer an existing exact PBS outcome through an isolated metadata spool.

Only completed bytes are copied. No qsub, scientific worker, controller queue,
result adoption or original-run mutation is performed by this technical check.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from trottertracks.rpe_pbs import Gateway
from trottertracks import rpe_q3_worker  # Existing exact-integer reader policy.


STAGE = r'''
import base64,hashlib,json,pathlib,sys
value=json.load(sys.stdin)
source=pathlib.Path(value['source_attempt']).resolve()
allocation=pathlib.Path(value['source_allocation']).resolve()
destination=pathlib.Path(value['copy_root']).resolve()
if destination.exists() or destination==source or destination in source.parents or allocation==destination or allocation in destination.parents:
 raise ValueError('Use a new isolated metadata directory; original attempt is read-only')
names=['ASSIGNMENT.json','WORKER_PROFILE.json','WORKER_STATE.json','OUTCOME.json',
 'COLLECTED_OUTCOME.json','PROCESS_RETURN.json','WORKER_COLLECT_RETURN.json','SOURCE.json','ADAPTER_SOURCE.json']
raw={name:(source/name).read_bytes()for name in names}
before={name:hashlib.sha256(data).hexdigest()for name,data in raw.items()}
assignment=json.loads(raw['ASSIGNMENT.json'])
source_spec=json.loads((allocation/'SPEC.json').read_text())
assert source_spec['profile']['pbs']['worker_source']=='925985b4ea76a15f8a07c507179d4fa6c492eabd'
assert assignment['source_manifest']['commit']=='28d976b99cdf8329adcc1b79327e329e947ef774'
host=source_spec['profile']['host_id']
assert assignment.get('host_id',host)==host
token='exact-transfer-metadata'
target=destination/'allocations'/token
attempt=target/'workers'/host/assignment['job_id']/assignment['attempt_id']
attempt.mkdir(parents=True)
for name,data in raw.items():(attempt/name).write_bytes(data)
profile=json.loads((allocation/'PROFILE.json').read_text())
profile['work_dir']=str(attempt.parents[1])
profile['artifact_cache_directory']=str(target/'artifacts')
(target/'PROFILE.json').write_text(json.dumps(profile)+'\n')
(target/'SPEC.json').write_text(json.dumps({'token':token,'profile':{'host_id':host},'metadata_only':True})+'\n')
(target/'STATE.json').write_text(json.dumps({'state':'EXITED','metadata_only':True})+'\n')
sys.path[:0]=[value['code_root']+'/src']
from trottertracks.rpe_q3_worker import validate_completed,result_bundle
# The saved collection receipt prevents the adapter's collect/execute branches.
assert (attempt/'WORKER_COLLECT_RETURN.json').exists()
observed=validate_completed(json.loads(raw['WORKER_PROFILE.json']),assignment,attempt)
files=result_bundle(attempt)
payload={'host_id':host,'job_id':assignment['job_id'],
 'attempt_id':assignment['attempt_id'],'request_sha256':assignment['request_sha256'],'files':files}
request={'request_id':'saved-completion','endpoint':'/v1/complete','payload':payload}
(target/'requests').mkdir()
(target/'requests/saved-completion.json').write_text(json.dumps(request)+'\n')
assert all(hashlib.sha256((source/name).read_bytes()).hexdigest()==sha for name,sha in before.items())
print(json.dumps({'token':token,'request':request,'original_sha256':before,
 'outcome_status':observed['status'],'source_files':{name:base64.b64encode(data).decode()for name,data in raw.items()},
 'runtime':{'executable':sys.executable,'version':sys.version,'version_info':list(sys.version_info)},
 'qsub_markers':len(list(destination.rglob('QSUB_STARTED.json')))},separators=(',',':')))
'''


def integer_facts(value, path='$'):
    if type(value) is int:
        text=str(value)
        if len(text.lstrip('-'))>4300:
            yield dict(path=path,digits=len(text.lstrip('-')),decimal_sha256=hashlib.sha256(text.encode()).hexdigest())
    elif isinstance(value,dict):
        for key,item in sorted(value.items()):
            yield from integer_facts(item,path+'.'+key)
    elif isinstance(value,list):
        for index,item in enumerate(value):
            yield from integer_facts(item,path+'['+str(index)+']')


def check(settings, source_allocation, source_attempt, copy_root, *, code_root=ROOT):
    settings=dict(settings,remote_root=str(copy_root),namespace_by_run=False)
    runtime=settings.get('gateway_python')
    if not runtime or runtime=='/usr/bin/python3':
        raise ValueError('Explicit dedicated gateway_python is required for this check')
    payload=dict(source_attempt=str(source_attempt),source_allocation=str(source_allocation),
                 copy_root=str(copy_root),code_root=str(code_root))
    remote=[runtime,'-B','-c',STAGE]
    started=time.monotonic()
    process=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        settings['ssh_alias'],shlex.join(remote)],input=json.dumps(payload),text=True,
        capture_output=True,check=True,timeout=60)
    staged=json.loads(process.stdout)
    assert staged['qsub_markers']==0
    baseline={name:base64.b64decode(data)for name,data in staged.pop('source_files').items()}
    request=staged.pop('request')
    original=json.loads(baseline['OUTCOME.json'])
    facts=list(integer_facts(original))
    assert facts and max(item['digits']for item in facts)==7168
    gateway=Gateway(settings)
    try:
        normal=gateway.call(dict(action='exchange'))
        returned=normal['requests'][staged['token']]
        assert len(returned)==1 and returned[0]==request
        complete=returned[0]['payload']
        normal_raw=base64.b64decode(complete['files']['OUTCOME.json'])
        assert normal_raw==baseline['OUTCOME.json']
        assert json.loads(normal_raw)==original
        assert list(integer_facts(json.loads(normal_raw)))==facts
        initial_command=gateway.process.args
        gateway.call(dict(action='exchange',responses={staged['token']:{request['request_id']:{'ok':True,'result':{'ok':True}}}}))
        # A new login client must use the same dedicated runtime after exit.
        gateway.close()
        harvested=gateway.call(dict(action='harvest',token=staged['token']))
        recovered=next(item for item in harvested['attempts']if item['assignment']['attempt_id']==complete['attempt_id'])
        assert len(harvested['attempts'])==1 and not harvested['incomplete']
        harvested_raw=base64.b64decode(recovered['files']['OUTCOME.json'])
        assert harvested_raw==baseline['OUTCOME.json']
        assert json.loads(harvested_raw)==original
        assert list(integer_facts(json.loads(harvested_raw)))==facts
        final_command=gateway.process.args
        for command in(initial_command,final_command):
            assert shlex.split(command[-1])[0]==runtime
        hashes={}
        for name,data in baseline.items():
            if name in recovered['files']:
                assert base64.b64decode(recovered['files'][name])==data
                hashes[name]=hashlib.sha256(data).hexdigest()
    finally:
        gateway.close()
    return dict(status='PASS',scope='SAVED_OUTCOME_BYTE_AND_EXACT_INTEGER_TRANSFER_ONLY',
        code_root=str(code_root),gateway_script=settings['gateway_script'],gateway_python=runtime,
        remote_runtime=staged['runtime'],source_allocation=str(source_allocation),source_attempt=str(source_attempt),
        copied_spool=str(copy_root),job_id=complete['job_id'],attempt_id=complete['attempt_id'],
        source_controller_commit='925985b4ea76a15f8a07c507179d4fa6c492eabd',
        scientific_source='28d976b99cdf8329adcc1b79327e329e947ef774',
        same_source_integer_facts=facts,preserved_file_sha256=hashes,
        ordinary_completion_exchange_bytes_exact=True,post_exit_harvest_bytes_exact=True,
        reconnected_gateway_uses_same_dedicated_python=True,float_conversion=False,integer_truncation=False,
        original_run_changed=False,original_attempt_changed=False,original_settings_changed=False,
        scientific_evaluations=0,compile_calls=0,signal_calls=0,refits=0,new_queue_adoptions=0,new_pbs_submissions=0,
        same_controller_clock_seconds=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--source-allocation',required=True)
    parser.add_argument('--source-attempt',required=True)
    parser.add_argument('--copy-root',required=True)
    parser.add_argument('--remote-code-root',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=check(json.loads(args.settings.read_text()),args.source_allocation,args.source_attempt,args.copy_root,
                 code_root=args.remote_code_root)
    result['checker_source']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],output=str(args.output),scientific_evaluations=0)))


if __name__=='__main__':main()
