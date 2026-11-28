"""Storage-only regressions; no scientific adapter or PBS submission."""
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from trottertracks import rpe_worker_storage as storage
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor

GIB = 1024**3


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def independent_attempt(part, *, state='ACCEPTED', adopted=1, different=False, worker_state='TRANSFERRED'):
    spool = part/'worker'; job, attempt = 'job123','attempt123'; directory=spool/job/attempt
    write(directory/'WORKER_STATE.json',{'state':worker_state})
    write(directory/'ASSIGNMENT.json',dict(job_id=job, attempt_id=attempt, request_sha256='x'*64,
                                          request={'route':{'kind':'PR_DF'}}))
    write(directory/'OUTCOME.json',{});write(directory/'PROCESS_RETURN.json',{'returncode':0})
    (directory/'WORKER_PROCESS.lock').touch()
    raw = directory/job/'RESULT.json';raw.parent.mkdir();raw.write_bytes(b'abc'*6144)
    canonical=part/'jobs'/'PR_DF'/job/job/'RESULT.json';canonical.parent.mkdir(parents=True);canonical.write_bytes(b'xyz'*6144 if different else raw.read_bytes())
    db=part/'distributed/QUEUE.sqlite3';db.parent.mkdir()
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE jobs (job_id, attempt_id, request_sha256, method, state, adopted_epoch, completion_sha256)')
        conn.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?)',(job,attempt,'x'*64,'PR_DF',state,adopted,'y'*64))
    return spool,raw,canonical


def test_independent_copies_only_share_adopted_byte_identical_results(tmp_path):
    good, raw, original=independent_attempt(tmp_path/'good')
    unadopted, raw2, original2=independent_attempt(tmp_path/'unadopted',adopted=None)
    different, raw3, original3=independent_attempt(tmp_path/'different',different=True)
    unknown, raw4, original4=independent_attempt(tmp_path/'unknown',state='UNKNOWN')
    running, raw5, original5=independent_attempt(tmp_path/'running',worker_state='RUNNING')
    out=storage.share_independent_q3_copies([good,unadopted,different,unknown,running],[tmp_path],GIB)
    assert out['files']==1 and out['bytes']>0 and raw.samefile(original)
    assert not raw2.samefile(original2) and not raw3.samefile(original3)
    assert not raw4.samefile(original4) and not raw5.samefile(original5)
    assert raw.read_bytes()==b'abc'*6144
    assert storage.share_independent_q3_copies([good],[tmp_path],GIB)['files']==0


def test_independent_admission_sampling_never_reclaims_or_repeats_scan(tmp_path):
    policy=dict(limit_bytes=50*GIB, roots=[str(tmp_path)], sample_path=str(tmp_path/'STORAGE_USAGE_GATEWAY.json'))
    (tmp_path/'OFFLINE_ACTIVE').touch()
    with (patch.object(storage,'_used_bytes',return_value=48*GIB) as measure,
          patch.object(storage,'_reclaim') as cleanup,
          patch.object(storage.shutil,'disk_usage',return_value=type('Usage',(),{'free':40*GIB})())):
        first=storage.report(policy,spools=(str(tmp_path),),proactive=True,measurement_only=True)
        second=storage.report(policy,spools=(str(tmp_path),),proactive=True,measurement_only=True)
    assert measure.call_count==1 and cleanup.call_count==0 and first==second
    assert first['used_bytes']==48*GIB and first['admission_sample'] and first['reclaim_pending']
    assert storage.compute_storage_observation(policy,[policy['sample_path']])['report']==first
    child=dict(policy,parent_compute_sample=policy['sample_path'])
    assert storage.parent_compute_storage_report(child)==first


def test_maintenance_reclaims_without_using_estimate_for_admission(tmp_path):
    part=tmp_path/'run'/'independent-search'/'parts'/'part';spool,raw,original=independent_attempt(part)
    policy=dict(limit_bytes=50*GIB,roots=[str(tmp_path)],sample_path=str(tmp_path/'maintenance.json'))
    measured=dict(limit_bytes=50*GIB,used_bytes=48*GIB,reason=None,accepting_new_jobs=True,
                  reclaim_start_bytes=40*GIB,reclaim_pending=True)
    with patch.object(storage,'_used_bytes',side_effect=AssertionError('no full admission scan in maintenance')), \
         patch.object(storage,'_reclaim',side_effect=lambda a,b,used,target,**kwargs:(used,0)), \
         patch.object(storage,'_share_transferred_git_packs',return_value=0):
        result=storage.reclaim_from_measurement(policy,measured,pbs_runs=str(tmp_path))
    assert result['status']=='CHECKED' and result['shared_files']==1 and raw.samefile(original)
    assert measured['used_bytes']==48*GIB and not Path(policy['sample_path']).exists()


def test_storage_admission_accounts_for_existing_and_new_reservations():
    report={'storage':dict(used_bytes=48*GIB,limit_bytes=50*GIB,accepting_new_jobs=True,
                          filesystem_free_bytes=100*GIB,filesystem_min_free_bytes=5*GIB)}
    assert DistributedExecutor._fits_preparation_storage(report,{'storage_bytes':GIB},GIB)
    assert not DistributedExecutor._fits_preparation_storage(report,{'storage_bytes':GIB},2*GIB)
    assert not DistributedExecutor._fits_preparation_storage(report,{},3*GIB)
    assert not DistributedExecutor._fits_preparation_storage({'storage':{'used_bytes':None,'limit_bytes':50*GIB,'accepting_new_jobs':False}}, {})


def test_compact_evidence_keeps_all_original_locators(tmp_path):
    spool, raw, original = independent_attempt(tmp_path)
    raw.unlink(); original.unlink()
    for relative in ('OUTCOME.json','COLLECTED_OUTCOME.json','job123/FINITE_SUPPORT.json'):
        left=spool/'job123'/'attempt123'/relative; right=tmp_path/'jobs'/'PR_DF'/'job123'/relative
        data=b'compact saved evidence'*512
        left.write_bytes(data);right.write_bytes(data)
    result=storage.share_independent_q3_copies([spool],[tmp_path],GIB)
    assert result['files']==3
    for relative in ('OUTCOME.json','COLLECTED_OUTCOME.json','job123/FINITE_SUPPORT.json'):
        left=spool/'job123'/'attempt123'/relative;right=tmp_path/'jobs'/'PR_DF'/'job123'/relative
        assert left.samefile(right) and left.read_bytes()==right.read_bytes()


def test_storage_reservation_is_kept_until_owner_ends(tmp_path):
    from trottertracks.rpe_q3_store import JobStore
    store=JobStore(tmp_path/'queue.sqlite3')
    store.reserve('PR_DF','job',{'route':{'kind':'artificial'}},1,True)
    [row]=store.claim('local',1,assignment_detail={'storage_reservation_bytes':GIB})
    store.update('local','job',row['attempt_id'],'RUNNING',{'progress':'active'})
    assert store.row('job')['detail']['storage_reservation_bytes']==GIB


def test_cleanup_batches_measure_whole_policy_only_once(tmp_path):
    paths=[tmp_path/f'copy-{i:03}' for i in range(65)]
    def small_du(command, **kwargs):
        return ''.join(f'4096\t{name}\n' for name in command[command.index('--')+1:])
    with patch.object(storage,'_acknowledged_copies',return_value=[(1,i,path) for i,path in enumerate(paths)]), \
         patch.object(storage,'_remove_copy',return_value=True), \
         patch.object(storage.subprocess,'check_output',side_effect=small_du) as du, \
         patch.object(storage,'_used_bytes',return_value=0) as measure:
        used,removed=storage._reclaim([], [str(tmp_path)], 65*4096, 0)
    assert removed==65 and used==0 and du.call_count==3 and measure.call_count==1


def test_bundled_streaming_cache_installs_initialization_and_helpers(tmp_path, monkeypatch):
    import base64, hashlib, zipfile
    import trottertracks
    from trottertracks import rpe_preparation_artifacts as pinned
    from types import SimpleNamespace
    import run_rpe_pbs_storage_worker as launcher
    implementation = pinned.ArtifactCache
    class LegacyCache:
        def __init__(self, root): self.root = root
    monkeypatch.setattr(pinned, 'ArtifactCache', LegacyCache)
    monkeypatch.setattr(trottertracks, '_pipeline_preparation_artifacts', SimpleNamespace(ArtifactCache=implementation), raising=False)
    monkeypatch.setattr(trottertracks, '__path__', list(trottertracks.__path__))
    package=tmp_path/'worker.pyz'
    with zipfile.ZipFile(package,'w') as z:z.writestr('trottertracks/_pipeline_preparation_artifacts.py','')
    launcher.configure_artifact_transport(package)
    cache=LegacyCache(tmp_path/'cache')
    data=b'operational fixture';sha=hashlib.sha256(data).hexdigest()
    assert cache.missing([sha])==[sha]
    cache.put({sha:base64.b64encode(data).decode()})
    assert cache.missing([sha])==[] and cache.get([sha])[sha]==base64.b64encode(data).decode()


def test_cleanup_failure_does_not_invalidate_fresh_capacity_measurement(tmp_path):
    policy = dict(roots=[str(tmp_path)], limit_bytes=100, sample_path=str(tmp_path/'sample.json'))
    with patch.object(storage, '_used_bytes', return_value=81) as measure, \
         patch.object(storage, '_reclaim', side_effect=OSError('artificial cleanup failure')):
        result = storage.report(policy, spools=[str(tmp_path)], proactive=True)
    assert result['used_bytes'] == 81
    assert result['accepting_new_jobs'] is True
    assert 'cleanup_error' in result
    assert measure.call_count == 2


def test_background_storage_never_blocks_poll_or_offers_unmeasured_capacity(tmp_path):
    from threading import Event
    policy=dict(limit_bytes=100, roots=[str(tmp_path)], sample_path=str(tmp_path/'sample.json'))
    ready=Event(); started=Event()
    def measure(*args, **kwargs):
        started.set(); assert ready.wait(2)
        return dict(limit_bytes=100,used_bytes=10,accepting_new_jobs=True,reason=None,
                    filesystem_free_bytes=1000,filesystem_min_free_bytes=10)
    with patch.object(storage,'report',side_effect=measure):
        lane=storage.BackgroundStorage(policy)
        try:
            assert not lane.poll()['accepting_new_jobs']
            assert started.wait(1)
            assert not lane.poll()['accepting_new_jobs']
            ready.set(); lane.sample.result(timeout=2)
            assert lane.poll()['accepting_new_jobs']
        finally:
            ready.set(); lane.close()


def test_owned_sampler_child_is_reaped_when_stopping(tmp_path):
    import subprocess, sys, threading, time
    from trottertracks import rpe_worker_storage as storage
    stop = threading.Event()
    errors = []
    marker = tmp_path/'pid'
    def sample():
        storage.set_storage_stop_event(stop)
        try:
            storage._du_output([sys.executable,'-c',
                'import os,time,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)',str(marker)],
                timeout=60,text=True,stderr=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            errors.append('cancelled')
    thread = threading.Thread(target=sample)
    thread.start()
    try:
        deadline = time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert marker.exists()
        pid = int(marker.read_text())
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and errors == ['cancelled']
    assert not Path('/proc',str(pid)).exists()


def test_contended_storage_sampler_uses_only_recent_actual_sample(tmp_path):
    import fcntl,time,json
    policy=dict(limit_bytes=10*2**30,roots=[str(tmp_path)],sample_path=str(tmp_path/'sample.json'))
    sample=storage.report(policy,refresh=True,proactive=True,measurement_only=True)
    assert sample['used_bytes'] is not None
    with (tmp_path/'sample.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=storage.report(policy,refresh=True,proactive=True,measurement_only=True,nonblocking=True)
        assert result['measurement_in_progress'] is True
        assert result['measurement_age_seconds']<storage.SAMPLE_SECONDS
        assert result['used_bytes']==sample['used_bytes']
        saved=json.loads((tmp_path/'sample.json').read_text())
        saved['sampled_epoch']=time.time()-storage.SAMPLE_SECONDS-1
        (tmp_path/'sample.json').write_text(json.dumps(saved))
        result=storage.report(policy,refresh=True,proactive=True,measurement_only=True,nonblocking=True)
        assert result['reason']=='STORAGE_MEASUREMENT_PENDING' and not result['accepting_new_jobs']


def test_terminal_copy_ledger_reuses_estimates_and_invalidates_changed_paths(tmp_path):
    ledger=storage.TerminalCopyLedger()
    path=tmp_path/'terminal-copy';path.write_bytes(b'old')
    names=[str(path)]
    with patch.object(storage,'_du_output',return_value=f'4096\t{path}\n') as measure:
        assert ledger.sizes(names,10)=={str(path):4096}
        assert ledger.sizes(names,10)=={str(path):4096} and measure.call_count==1
        path.write_bytes(b'changed')
        ledger.sizes(names,10)
        assert measure.call_count==2
        ledger.retire(path);ledger.sizes(names,10)
        assert measure.call_count==3
        ledger.retain([])
        assert not ledger.entries


def test_incremental_storage_ledger_matches_du_with_hardlinks_sparse_mutation_and_delete(tmp_path):
    import os
    ledger=storage.IncrementalStorageLedger()
    left=tmp_path/'left';right=tmp_path/'right';left.mkdir();right.mkdir()
    file=left/'raw';file.write_bytes(b'a'*8192);os.link(file,right/'link')
    sparse=right/'sparse'
    with sparse.open('wb') as stream:stream.seek(1024*1024);stream.write(b'x')
    (left/'sym').symlink_to(sparse)
    roots=[str(left),str(right)]
    expected=lambda:storage._used_bytes(roots)
    assert ledger.scan(roots)==expected()
    file.write_bytes(b'b'*32768)
    assert ledger.scan(roots)==expected()
    (right/'link').unlink();(left/'new').mkdir();(left/'new'/'leaf').write_bytes(b'z'*4096)
    assert ledger.scan(roots)==expected()
    file.unlink();sparse.unlink()
    assert ledger.scan(roots)==expected()
    assert not ledger.disabled


def test_incremental_storage_ledger_disables_on_reconciliation_mismatch(tmp_path):
    ledger=storage.IncrementalStorageLedger()
    assert ledger.measure([str(tmp_path)],timeout=5,du=lambda:123)==123
    assert ledger.disabled
    assert ledger.measure([str(tmp_path)],timeout=5,du=lambda:456)==456
