import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from trottertracks.rpe_q3_timing import compact_queue_reasons, read_lines
from trottertracks.rpe_worker_storage import compact_provider_source, restore_provider_source
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks.rpe_pbs import PbsManager
from rpe_romutail_retention import compact_readonly_pbs_replies, deduplicate_old_artifacts, archive_old_logs


def test_queue_delta_is_lossless_across_resume(tmp_path):
    initial = {f'job-{i}': 'WAIT' for i in range(1000)}
    changed = dict(initial, **{'job-1': 'READY'}); del changed['job-2']
    def row(inv, reasons, previous=None):
        return dict(event='QUEUE_WAIT_STATE', invocation=inv, **compact_queue_reasons(reasons, previous))
    events = [row('first', initial), row('first', changed, initial), row('first', changed, changed),
              row('second', initial)]
    log = tmp_path/'PROGRESS.jsonl'
    log.write_text(''.join(json.dumps(event)+'\n' for event in events))
    restored = read_lines(log)
    assert [x['reasons'] for x in restored] == [initial,changed,changed,initial]
    assert restored[1]['reason_counts'] == Counter(changed.values())
    assert len(json.dumps(events[2])) < len(json.dumps(events[0]))/20


def test_provider_retains_unique_source_and_evidence(tmp_path):
    root = tmp_path/'provider_workspace'; root.mkdir()
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.PIPE,text=True).strip()
    git('init'); git('config','user.name','test'); git('config','user.email','test@example.invalid')
    for path in ('src/base.py','scripts/changed.py','docs/archive/old.md','docs/research/raw.json'):
        target=root/path; target.parent.mkdir(parents=True,exist_ok=True);target.write_text(path)
    git('add','.'); git('commit','-m','base');source=git('rev-parse','HEAD')
    (root/'scripts/changed.py').write_text('local work')
    (root/'src/local.py').write_text('not tracked')
    assert compact_provider_source(root,source)>0
    assert not (root/'src/base.py').exists()
    assert not (root/'docs/archive/old.md').exists()
    assert (root/'scripts/changed.py').read_text()=='local work'
    assert (root/'src/local.py').read_text()=='not tracked'
    assert (root/'docs/research/raw.json').read_text()=='docs/research/raw.json'
    restore_provider_source(root)
    assert (root/'src/base.py').read_text()=='src/base.py'
    assert (root/'docs/archive/old.md').read_text()=='docs/archive/old.md'
    assert not (tmp_path/'RETAINED_PROVIDER_SOURCE.json').exists()


def test_prior_cache_shares_verified_content(tmp_path):
    old=ArtifactCache(tmp_path/'old'); raw=b'input data'; sha=hashlib.sha256(raw).hexdigest()
    old.put({sha:base64.b64encode(raw).decode()})
    current=ArtifactCache(tmp_path/'new',prior_roots=[old.root]); assert current.missing([sha])==[]
    assert current.path(sha).samefile(old.path(sha))
    assert current.get([sha])==old.get([sha])
    assert current.missing(['a'*64])==['a'*64]


def test_pbs_readonly_response_replayed_without_duplicate_body(tmp_path):
    manager=PbsManager.__new__(PbsManager);manager.directory=tmp_path
    token='token'; manager.state={'allocations':{token:{'spec':{'profile':{'host_id':'pbs'}}}}};manager.allocations=[]
    cache=ArtifactCache(tmp_path/'cas'); raw=b'abc'*10000; sha=hashlib.sha256(raw).hexdigest();cache.put({sha:base64.b64encode(raw).decode()})
    calls=[]
    def rpc(endpoint,payload):
        calls.append(endpoint);return dict(blobs=cache.get(payload['hashes']))
    manager.executor=SimpleNamespace(rpc=rpc,artifact_cache=cache)
    request=dict(endpoint='/v1/artifacts/get',request_id='one',payload=dict(host_id='pbs',hashes=[sha]))
    result=manager._dispatch(token,request)
    assert manager._dispatch(token,request)==result and len(calls)==1
    assert not (tmp_path/'responses'/token/'one.json').exists()
    assert (tmp_path/'responses-replay'/token/'one.json').stat().st_size<1000
    from pytest import raises
    with raises(ValueError,match='changed content'):
        manager._dispatch(token,dict(request,payload=dict(host_id='pbs',hashes=[])))


def stopped_run(state, name):
    run=state.parent/name;run.mkdir(); (run/'LAUNCH.json').write_text('{"invocation_id":"sample"}'); (run/'EXIT.json').write_text('{"invocation_id":"sample"}')
    index=state/'RUNS.json'; rows=json.loads(index.read_text()) if index.exists() else {}; rows[name]=dict(run_directory=str(run)); index.write_text(json.dumps(rows))
    (state/'REGISTRY.lock').touch(); return run


def test_old_retention_only_verified_readonly_replies(tmp_path):
    state=tmp_path/'state';state.mkdir(); old=stopped_run(state,'old'); other=stopped_run(state,'other')
    raw=b'known input'; sha=hashlib.sha256(raw).hexdigest(); encoded=base64.b64encode(raw).decode()
    a=ArtifactCache(old/'distributed/artifacts'); b=ArtifactCache(other/'distributed/artifacts')
    a.put({sha:encoded});b.put({sha:encoded}); assert not a.path(sha).samefile(b.path(sha))
    assert deduplicate_old_artifacts(state,minimum_bytes=1)['shared_files']==1
    assert a.path(sha).samefile(b.path(sha))
    replies=old/'distributed/pbs/responses/token';replies.mkdir(parents=True)
    record=dict(request_sha256='f'*64,response=dict(ok=True,result=dict(blobs={sha:encoded})))
    (replies/'good.json').write_text(json.dumps(record));record['response']['result']['blobs'][sha]='bad';(replies/'bad.json').write_text(json.dumps(record))
    (replies/'state.json').write_text(json.dumps(dict(request_sha256='e'*64,response=dict(ok=True,result=dict(accepted=True)))))
    outcome=compact_readonly_pbs_replies(state,minimum_bytes=1)
    assert outcome['compacted_responses']==1 and not (replies/'good.json').exists()
    assert (replies/'bad.json').is_file() and (replies/'state.json').is_file()
    assert json.loads((replies.parent.parent/'responses-replay/token/good.json').read_text())['hashes']==[sha]
    (old/'PROGRESS.jsonl').write_text('history\n');mount=tmp_path/'external';mount.mkdir()
    with patch('rpe_romutail_retention.mounted_separately',return_value=True):
        result=archive_old_logs(state,mount,minimum_bytes=1)
    assert len(result['archived_files'])==1 and (old/'PROGRESS.jsonl').is_symlink()
    assert (old/'PROGRESS.jsonl').read_text()=='history\n'
