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
    (old/'PROGRESS.jsonl').write_text('live append target\n'); archived=old/'invocations/sample/PROGRESS.jsonl'; archived.parent.mkdir(parents=True); archived.write_text('history\n');mount=tmp_path/'external';mount.mkdir()
    with patch('rpe_romutail_retention.mounted_separately',return_value=True):
        result=archive_old_logs(state,mount,minimum_bytes=1)
    assert len(result['archived_files'])==1 and archived.is_symlink()
    assert archived.read_text()=='history\n'
    assert not (old/'PROGRESS.jsonl').is_symlink()


def test_terminal_rpc_archive_retains_original_replay_path(tmp_path):
    from rpe_romutail_retention import archive_terminal_rpc_sessions
    state=tmp_path/'state';state.mkdir();run=stopped_run(state,'old');mount=tmp_path/'external';mount.mkdir()
    p=run/'distributed/pbs';closed='a'*32;owned='b'*32
    for token in (closed, owned):
        source=p/'responses'/token;source.mkdir(parents=True)
        (source/'one.json').write_bytes(b'{"request_sha256":"test","response":{"ok":true}}\n')
    (p/'STATE.json').write_text(json.dumps({'allocations':{closed:{'status':'F'},owned:{'status':'R'}}}))
    original=(p/'responses'/closed/'one.json').read_bytes()
    with patch('rpe_romutail_retention.mounted_separately',return_value=True):
        result=archive_terminal_rpc_sessions(state,mount,run_id='old')
    assert len(result['sessions'])==1
    assert (p/'responses'/closed).is_symlink() and (p/'responses'/closed/'one.json').read_bytes()==original
    assert not (p/'responses'/owned).is_symlink()
    manifest=Path(result['sessions'][0]['archive']).with_name(closed+'.archive.json')
    assert json.loads(manifest.read_text())['entries']['one.json']['sha256']==hashlib.sha256(original).hexdigest()


def test_share_only_exact_completed_event_triples(tmp_path):
    from rpe_romutail_retention import share_stopped_event_originals
    state=tmp_path/'state';state.mkdir();run=stopped_run(state,'old')
    for candidate, values in [('same',[b'abc']*3),('different',[b'abc',b'abc',b'xyz'])]:
        directory=run/'completion/candidates'/candidate
        for name,raw in zip(('events_PRIVATE.json','axes_X/events_PRIVATE.json','axes_Y/events_PRIVATE.json'),values):
            p=directory/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        (directory/'SHARED_EVENTS.json').write_text('{"digest":"saved"}')
    result=share_stopped_event_originals(state,'old')
    assert result['groups']==1 and result['shared_files']==2
    same=run/'completion/candidates/same';different=run/'completion/candidates/different'
    assert (same/'events_PRIVATE.json').samefile(same/'axes_X/events_PRIVATE.json')
    assert not (different/'events_PRIVATE.json').samefile(different/'axes_X/events_PRIVATE.json')


def test_install_files_keeps_shared_events_and_local_cas(tmp_path):
    from trottertracks.rpe_q3_coordinator import install_files
    from trottertracks.rpe_preparation_artifacts import worker_cache
    root=tmp_path/'transfer';raw=b'private events';encoded=base64.b64encode(raw).decode()
    names=['events_PRIVATE.json','axes_X/events_PRIVATE.json','axes_Y/events_PRIVATE.json']
    install_files(root,dict.fromkeys(names,encoded));install_files(root,dict.fromkeys(names,encoded))
    assert all((root/n).samefile(root/names[0]) for n in names)
    central=ArtifactCache(tmp_path/'central');sha=hashlib.sha256(raw).hexdigest();central.put({sha:encoded})
    worker=worker_cache(dict(work_dir=str(tmp_path/'worker'), central_artifact_cache_directory=str(central.root)))
    assert worker.missing([sha])==[] and worker.path(sha).samefile(central.path(sha))
    worker.path(sha).unlink();assert central.path(sha).read_bytes()==raw


def test_compact_hchain_report_is_json_identical(tmp_path):
    from trottertracks.rpe_local0_search import write
    data={'numbers':[1,2,3],'name':'日本語','nested':{'x':True}}
    full=tmp_path/'full.json';compact=tmp_path/'compact.json'
    write(full,data);write(compact,data,compact=True)
    assert json.loads(full.read_text())==json.loads(compact.read_text())==data
    assert compact.stat().st_size<full.stat().st_size


def test_share_only_accepted_request_with_matching_queue_digest(tmp_path):
    import sqlite3
    from rpe_romutail_retention import share_stopped_accepted_requests
    from trottertracks.rpe_q3_store import digest
    state=tmp_path/'state';state.mkdir();run=stopped_run(state,'old');d=run/'distributed';d.mkdir()
    request={'request_id':'original'}; raw=json.dumps(request).encode()
    with sqlite3.connect(d/'QUEUE.sqlite3') as db:
        db.execute('CREATE TABLE jobs(job_id,method,attempt_id,request_sha256,state)')
        for job, state_name, key in [('good','ACCEPTED',digest(request)),('unknown','UNKNOWN',digest(request)),('wrong','ACCEPTED','bad')]:
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)',(job,'M','a',key,state_name))
            for p in (run/'jobs/M'/job/'REQUEST.json',d/'transfers'/job/'a/REQUEST.json'):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    result=share_stopped_accepted_requests(state,'old')
    assert result['shared_files']==1
    for job, expected in [('good',True),('unknown',False),('wrong',False)]:
        assert (run/'jobs/M'/job/'REQUEST.json').samefile(d/'transfers'/job/'a/REQUEST.json')==expected


def test_share_provider_packs_only_when_both_files_match(tmp_path):
    from rpe_romutail_retention import share_stopped_provider_git_packs
    state=tmp_path/'state';state.mkdir();run=stopped_run(state,'old')
    name='pack-'+'a'*40; paths=[]
    for candidate,index in [('first',b'index'),('second',b'index'),('different',b'other')]:
        root=run/'pf_auto/candidates'/candidate/'pf_auto/pf_completion/provider_workspace/.git/objects/pack'
        root.mkdir(parents=True);(root/(name+'.pack')).write_bytes(b'packed');(root/(name+'.idx')).write_bytes(index);paths.append(root)
    result=share_stopped_provider_git_packs(state,'old')
    assert result['shared_files']==2
    assert (paths[0]/(name+'.pack')).samefile(paths[1]/(name+'.pack'))
    assert not (paths[0]/(name+'.pack')).samefile(paths[2]/(name+'.pack'))


def test_selective_rpc_archive_keeps_tiny_replies_and_reuses_verified_copy(tmp_path):
    from rpe_romutail_retention import archive_terminal_rpc_large_files
    state=tmp_path/'state';state.mkdir();run=stopped_run(state,'old');mount=tmp_path/'external';mount.mkdir()
    p=run/'distributed/pbs';token='a'*32;source=p/'responses'/token;source.mkdir(parents=True)
    (p/'STATE.json').write_text(json.dumps({'allocations':{token:{'status':'F'}}}))
    large=b'abcd'*10000;(source/'large.json').write_bytes(large);(source/'small.json').write_bytes(b'{}')
    saved=mount/'RomuTail-auto-coldstore/pbs-responses/old/sample'/token;saved.mkdir(parents=True)
    (saved/'large.json').write_bytes(large)
    with patch('rpe_romutail_retention.mounted_separately',return_value=True):
        result=archive_terminal_rpc_large_files(state,mount,run_id='old',minimum_bytes=100)
        again=archive_terminal_rpc_large_files(state,mount,run_id='old',minimum_bytes=100)
    assert result['sessions'][0]['files']==1 and not again['sessions']
    assert (source/'large.json').is_symlink() and (source/'large.json').read_bytes()==large
    assert not (source/'small.json').is_symlink() and (source/'small.json').read_bytes()==b'{}'


def test_vacuum_settled_queue_preserves_rows_and_excludes_unresolved(tmp_path):
    import sqlite3
    from contextlib import closing
    from rpe_romutail_retention import compact_stopped_queues
    state=tmp_path/'state';state.mkdir();old=stopped_run(state,'old');owned=stopped_run(state,'owned')
    for run,status in ((old,'ACCEPTED'),(owned,'UNKNOWN')):
        p=run/'distributed/QUEUE.sqlite3';p.parent.mkdir()
        with closing(sqlite3.connect(p)) as db, db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE jobs(job_id TEXT PRIMARY KEY,state TEXT,request TEXT)')
            db.execute('INSERT INTO jobs VALUES(?,?,?)',('keep',status,'exact'))
            db.execute('CREATE TABLE temporary_history(id INTEGER PRIMARY KEY,payload TEXT)')
            db.executemany('INSERT INTO temporary_history(payload) VALUES(?)',[('x'*4000,)]*400)
            db.execute('DELETE FROM temporary_history')
    before=(owned/'distributed/QUEUE.sqlite3').read_bytes()
    result=compact_stopped_queues(state,minimum_bytes=1)
    assert result['reclaimed_bytes']>0
    with closing(sqlite3.connect(old/'distributed/QUEUE.sqlite3')) as db:
        assert db.execute('SELECT * FROM jobs').fetchall()==[('keep','ACCEPTED','exact')]
        assert db.execute('PRAGMA journal_mode').fetchone()==('wal',)
    assert (owned/'distributed/QUEUE.sqlite3').read_bytes()==before


def test_storage_inventory_marks_release_and_alternate_and_never_authorizes_deletion(tmp_path):
    from rpe_romutail_retention import storage_dependency_inventory
    base=tmp_path/'repo/.rpe-central';state=base/'state';state.mkdir(parents=True)
    release=base/'releases/used';run=release/'search-evidence/old';run.mkdir(parents=True)
    planning=base/'planning/input';planning.mkdir(parents=True)
    extra=base/'deployment/unknown';extra.mkdir(parents=True)
    (state/'RUNS.json').write_text(json.dumps({'old':{'run_directory':str(run)}}))
    alternate=base.parent/'.git/objects/info/alternates';alternate.parent.mkdir(parents=True);alternate.write_text(str(planning/'objects')+'\n')
    data=storage_dependency_inventory(state);lookup={row['path']:row for row in data['entries']}
    assert lookup[str(release)]['status']=='REFERENCED'
    assert lookup[str(planning)]['status']=='REFERENCED'
    assert lookup[str(extra)]['status']=='UNCONFIRMED_NOT_A_DELETION_CANDIDATE'
    assert data['status']=='INVENTORY_ONLY' and extra.is_dir()


def test_stopped_provider_source_compaction_respects_pinned_restore(tmp_path):
    from rpe_romutail_retention import compact_stopped_provider_sources
    state=tmp_path/'state';state.mkdir();run=stopped_run(state,'old')
    root=run/'pf_auto/candidates/one/pf_auto/pf_completion/provider_workspace';root.mkdir(parents=True)
    def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True,stderr=subprocess.PIPE).strip()
    git('init');git('config','user.name','fixture');git('config','user.email','fixture@example.invalid')
    src=root/'src/file.py';src.parent.mkdir();src.write_text('fixture')
    git('add','.');git('commit','-m','fixture');source=git('rev-parse','HEAD')
    (root.parent/'PROVIDER_SOURCE.json').write_text(json.dumps({'source':source}))
    pinned=tmp_path/'code';code=pinned/'src/trottertracks/pf_distributed_preparation.py';code.parent.mkdir(parents=True);code.write_text('pass\n')
    (run/'CONTROLLER_CONFIG.json').write_text(json.dumps({'code_root':str(pinned)}))
    assert compact_stopped_provider_sources(state,'old')['compacted']==0 and src.exists()
    code.write_text('restore_provider_source(workspace)\n')
    assert compact_stopped_provider_sources(state,'old')['compacted']==1 and not src.exists()
    from rpe_romutail_retention import restore_stopped_provider_sources
    assert restore_stopped_provider_sources(state,'old')['restored']==1
    assert src.read_text()=='fixture'


def test_diagnostic_archive_preserves_and_verifies_original_bytes(tmp_path):
    from rpe_romutail_retention import pack_stopped_diagnostics,verify_diagnostic_archive,restore_diagnostic_archive
    import sqlite3
    state=tmp_path/'state';state.mkdir();run=stopped_run(state,'old')
    db=run/'distributed/QUEUE.sqlite3';db.parent.mkdir()
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE jobs(job_id TEXT,method TEXT,attempt_id TEXT,state TEXT)')
        conn.execute("INSERT INTO jobs VALUES('job','FULL','attempt','ACCEPTED')")
    diag=run/'jobs/FULL/job/TRANSPORT_TIMING.jsonl';diag.parent.mkdir(parents=True);diag.write_bytes(b'{"fixture":true}\n')
    archive=tmp_path/'diag.jsonl.xz';result=pack_stopped_diagnostics(state,'old',archive)
    assert result['files']==1 and result['originals_preserved'] and verify_diagnostic_archive(archive,'old')==1
    assert diag.read_bytes()==b'{"fixture":true}\n'
    diag.unlink()  # synthetic fixture only
    assert restore_diagnostic_archive(state,'old',archive)['restored']==1
    assert diag.read_bytes()==b'{"fixture":true}\n'

def test_retention_protects_transitive_live_ancestors_and_rechecks_lock(tmp_path):
    from rpe_romutail_retention import stopped_runs
    from recover_stopped_rpe_run import stopped_lock
    from pytest import raises
    state=tmp_path/'state';state.mkdir()
    grand=stopped_run(state,'grand'); parent=stopped_run(state,'parent')
    child=stopped_run(state,'child'); unrelated=stopped_run(state,'unrelated')
    (parent/'RUN_LINEAGE.json').write_text(json.dumps({'parent_directory':str(grand)}))
    (child/'ROUND.json').write_text(json.dumps({'executor':{'prior_run_dirs':[str(parent)]}}))
    (child/'EXIT.json').unlink()
    assert [run for run,_ in stopped_runs(state)]==[unrelated]
    with raises(BlockingIOError):
        with stopped_lock(state,grand,{'invocation_id':'sample'},protect_references=True):
            raise AssertionError('unexpected permission to compact ancestor')
    (child/'EXIT.json').write_text('{"invocation_id":"sample"}')
    assert {run for run,_ in stopped_runs(state)}=={grand,parent,child,unrelated}

def test_change_only_report_keeps_inode_and_mtime(tmp_path):
    from trottertracks.rpe_local0_search import write
    target=tmp_path/'ROUND_RESULT.json'
    value={'integer':2**100,'text':'α'*100000}
    write(target,value,compact=True,only_changed=True);before=target.stat()
    write(target,value,compact=True,only_changed=True);after=target.stat()
    assert (after.st_ino,after.st_mtime_ns)==(before.st_ino,before.st_mtime_ns)
    write(target,dict(value,new=True),compact=True,only_changed=True)
    assert json.loads(target.read_text())['new'] is True


def test_alternate_readiness_and_index_without_mutating_source(tmp_path):
    from rpe_romutail_retention import alternate_relocation_readiness
    from compact_rpe_git_objects import refresh_multi_pack_index
    source=tmp_path/'source';source.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-C',str(source),*args],text=True,stderr=subprocess.PIPE).strip()
    git('init');git('config','user.name','fixture');git('config','user.email','fixture@example.invalid')
    first=source/'first';first.write_text('one');git('add','.');git('commit','-m','first')
    git('repack','-d')
    second=source/'second';second.write_text('two');git('add','.');git('commit','-m','second')
    git('repack')
    objects=source/'.git/objects'
    result=refresh_multi_pack_index(objects)
    assert result['status']=='VERIFIED' and result['after']['multi_pack_index']
    assert result['before']['pack_count']==result['after']['pack_count']
    assert alternate_relocation_readiness(source,objects)['status']=='OBJECTS_AVAILABLE'
    empty=tmp_path/'empty';empty.mkdir()
    assert alternate_relocation_readiness(source,empty)['status']=='HOLD_MISSING_OBJECTS'
    (empty/'info').mkdir();(empty/'info/alternates').write_text(str(objects)+'\n')
    assert alternate_relocation_readiness(source,empty)['status']=='HOLD_TARGET_NOT_INDEPENDENT'
    assert not (objects/'info/alternates').exists()
