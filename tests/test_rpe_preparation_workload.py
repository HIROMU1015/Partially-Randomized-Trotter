"""Operational-scale tests only; no scientific provider is invoked."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trottertracks import rpe_df_price_flanks as df
from trottertracks import rpe_q3_resources as R
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
from trottertracks.rpe_preparation_artifacts import ArtifactCache


def request(tmp_path, *, q=10, outer=20, paths=4, axes=('X', 'Y'), system='fixture', source='rev'):
    context = dict(physical_candidate=dict(system=system, method='PR', formula='S2', q=q, prefix=2),
                   binding=dict(rank=7, basis='fixture', compiler='c', control='directional'),
                   structure=dict(event_count=q * outer))
    doc = dict(cells=[dict(m=0, request=dict(physical_context=context, axes=list(axes)), indices=list(range(paths)))])
    name = f'workspace/{q}-{outer}-{paths}-{len(axes)}-{system}/df_prices/INPUTS.json'
    p = tmp_path / name.removeprefix('workspace/')
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(doc).encode(); p.write_bytes(raw)
    return dict(preparation=dict(controller_root=str(tmp_path), inputs={name: hashlib.sha256(raw).hexdigest()},
                                 stage='compile', provider='DF_CALIBRATION', source=dict(commit=source),
                                 parallel_df_price_axes=True))


def test_workload_scale_and_partial_observation(tmp_path):
    small = request(tmp_path); large = request(tmp_path, q=20, outer=40, paths=8)
    a, b = R.preparation_workload(small), R.preparation_workload(large)
    assert (a['path_units'], a['axis_jobs'], a['peak_units']) == (800, 8, 230)
    assert (b['path_units'], b['axis_jobs'], b['peak_units']) == (6400, 16, 860)
    sample = dict(workload=a, host='pbs', cpu_count=4, peak_rss_bytes=1000,
                  unique_event_bytes=6000, stored_path_units=600, elapsed_seconds=123, completed=False)
    estimate = R.preparation_forecast(large, [sample], cpu_count=8, host='pbs')
    assert estimate == dict(memory_bytes=11218, storage_bytes=64000)
    assert R.preparation_forecast(small, [sample], cpu_count=2)['memory_bytes'] == 750
    assert R.preparation_forecast(request(tmp_path, source='other'), [sample]) == {}
    assert R.preparation_forecast(request(tmp_path, system='other'), [sample]) == {}
    sample['completed'] = True
    assert R.preparation_forecast(large, [sample], cpu_count=4, host='pbs')['seconds'] > 123
    assert 'seconds' not in R.preparation_forecast(large, [sample], host='wsl')


def test_storage_budget_reserves_other_owners_without_blocking_independent_work():
    report = dict(storage=dict(used_bytes=40, limit_bytes=100))
    fits = DistributedExecutor._fits_preparation_storage
    assert fits(report, dict(storage_bytes=60))
    assert not fits(report, dict(storage_bytes=61))
    assert not fits(report, dict(storage_bytes=40), reserved=21)
    assert fits(report, {}, reserved=60)
    assert not fits(dict(storage=dict(limit_bytes=100)), dict(storage_bytes=1))
    assert not fits(dict(storage=dict(used_bytes=10, limit_bytes=100, filesystem_free_bytes=20, filesystem_min_free_bytes=5)), dict(storage_bytes=16))


def test_streaming_event_digest_and_lossless_compact_document(tmp_path):
    document = dict(events=[[dict(phase={'real': -0.0, 'imag': 0.0}, ids=('X', 'λ'), order=2)] * 15],
                    seeds=[2, 3], laws=[{'a': 0.1}], conditional=False)
    expected = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert df.streaming_digest(document) == expected
    assert df.streaming_digest(document['events']) == hashlib.sha256(json.dumps(document['events'], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    path = tmp_path / 'events_PRIVATE.json'
    def original(path, value):
        path.write_text(json.dumps(value, indent=2))
    df.compact_event_writer(original)(path, document)
    assert json.loads(path.read_text()) == json.loads(json.dumps(document))
    assert path.stat().st_size < len(json.dumps(document, indent=2).encode())
    with pytest.raises(FileExistsError):df.compact_event_writer(original)(path, document)


def test_artifact_capture_and_materialize_do_not_read_whole_files(tmp_path, monkeypatch):
    source = tmp_path / 'source'; source.mkdir(); data = source / 'evidence.json'
    data.write_bytes(b'raw evidence' * 100000)
    def fail(*args, **kwargs):raise AssertionError('whole file read')
    monkeypatch.setattr(Path, 'read_bytes', fail)
    cache = ArtifactCache(tmp_path / 'cache')
    manifest = cache.capture(source, ['evidence.json'])
    destination = tmp_path / 'destination'
    cache.materialize(manifest, destination)
    with (destination / 'evidence.json').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['evidence.json']
    with (destination / 'evidence.json').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError, match='Changed immutable input'):cache.materialize(manifest, destination)


def test_scale_classes_and_observation_do_not_double_count_axes(tmp_path):
    a = request(tmp_path/'controller', q=10, outer=20)
    b = request(tmp_path/'controller', q=30, outer=80)
    low = R.preparation_workload(a); high = R.preparation_workload(b)
    profile = dict(task_cpu_count=4)
    assert R.measured_resource_class(a, runtime=profile, preparation_detail=True, workload=low) != R.measured_resource_class(
        b, runtime=profile, preparation_detail=True, workload=high)
    assert R.measured_resource_class(a, runtime=profile, preparation_detail=True) == R.measured_resource_class(
        b, runtime=profile, preparation_detail=True)
    attempt = tmp_path/'attempt'; spec = a['preparation']; name = next(iter(spec['inputs']))
    fixed = attempt/name; fixed.parent.mkdir(parents=True)
    fixed.write_bytes((Path(spec['controller_root'])/name.removeprefix('workspace/')).read_bytes())
    path = fixed.parent/'calibration/acquisition/m0/p0';path.mkdir(parents=True)
    raw = path/'events_PRIVATE.json';raw.write_bytes(b'one shared file')
    for axis in ('X', 'Y'):
        axis_dir = path/('axes_'+axis);axis_dir.mkdir();(axis_dir/raw.name).hardlink_to(raw)
    (path/'SHARED_EVENTS.json').write_text('{}')
    obs = R.owned_preparation_observation(attempt, dict(request=a, job_id='job', attempt_id='attempt', request_sha256='hash'),
                                         profile, 12345)
    assert obs['unique_event_bytes'] == len(b'one shared file')
    assert obs['stored_path_units'] == 200
    assert obs['peak_rss_bytes'] == 12345


def test_paired_dictionary_uses_path_concurrency_and_matching_storage(tmp_path):
    legacy=request(tmp_path/'controller');paired=request(tmp_path/'controller')
    paired['preparation']['df_event_layout']='DICTIONARY_PAIRED_V1'
    old=R.preparation_workload(legacy);new=R.preparation_workload(paired)
    assert new['axis_jobs']==8 and new['path_workers']==4
    old_sample=dict(workload=old,cpu_count=8,peak_rss_bytes=10000,unique_event_bytes=10000,stored_path_units=200)
    new_sample=dict(workload=new,cpu_count=8,peak_rss_bytes=5000,unique_event_bytes=100,stored_path_units=200)
    assert R.preparation_forecast(paired,[old_sample],cpu_count=8)==dict(memory_bytes=7500,storage_bytes=40000)
    assert R.preparation_forecast(paired,[old_sample,new_sample],cpu_count=8)==dict(memory_bytes=7500,storage_bytes=400)
    assert R.preparation_forecast(legacy,[old_sample,new_sample],cpu_count=8)['storage_bytes']==40000
    assert R.preparation_forecast(legacy,[new_sample],cpu_count=8).get('storage_bytes') is None
    attempt=tmp_path/'attempt';spec=paired['preparation'];name=next(iter(spec['inputs']))
    fixed=attempt/name;fixed.parent.mkdir(parents=True);fixed.write_bytes((Path(spec['controller_root'])/name.removeprefix('workspace/')).read_bytes())
    path=fixed.parent/'calibration/acquisition/m0/p0/axes_XY';path.mkdir(parents=True)
    (path/'events_PRIVATE.json').write_bytes(b'paired codec')
    (path/'STRUCTURAL.json').write_text('{}')
    obs=R.owned_preparation_observation(attempt,dict(request=paired,job_id='j',attempt_id='a',request_sha256='sha'),dict(task_cpu_count=8),1000)
    assert obs['unique_event_bytes']==len(b'paired codec') and obs['stored_path_units']==200


def test_reversible_pbs_drain_keeps_deadline_and_process_stop():
    import threading
    import ast
    from trottertracks import rpe_pbs_worker
    # The bundled worker is imported under a zipapp-specific alias in production.
    bundled = Path(rpe_pbs_worker.__file__).with_name('_pipeline_pbs_worker.py')
    tree = ast.parse(bundled.read_text())
    function = next(x for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'AllocationWorker'
                    for x in c.body if isinstance(x, ast.FunctionDef) and x.name == '_sync_drain')
    namespace = {};exec(compile(ast.Module(body=[function], type_ignores=[]), str(bundled), 'exec'), namespace)
    for sync in (rpe_pbs_worker.AllocationWorker._sync_drain, namespace['_sync_drain']):
        worker = SimpleNamespace(stop=threading.Event(), remaining=lambda:100,
                                 draining=False, exit_reason=None, drain_seconds=20, minimum_task_seconds=10)
        sync(worker, {'stop':True}); assert worker.draining
        sync(worker, {}); assert worker.draining
        sync(worker, {'stop':False}); assert not worker.draining
        worker.remaining = lambda:10
        sync(worker, {'stop':False}); assert worker.draining
        assert worker.exit_reason == 'WALLTIME_ADMISSION_CUTOFF'
        worker.remaining = lambda:100
        sync(worker, {'stop':False}); assert worker.draining
        worker.stop.set();sync(worker, {'stop':False}); assert worker.draining


def test_probe_priority_matches_admitted_class_and_keeps_other_work():
    executor = DistributedExecutor.__new__(DistributedExecutor)
    executor.hosts = {'direct':{'resource_report':{'admitted_task_classes':['selected']}}}
    executor._fits_host_task_class = lambda host,row:row['kind']=='selected'
    rows = [dict(kind='other', job_id='a'),dict(kind='other',job_id='b'),dict(kind='selected',job_id='c')]
    assert [row['job_id'] for row in executor._prioritize_probe_classes('direct',rows,2,1)] == ['c','a']


def test_direct_inflight_offer_not_reused_by_other_or_late_poll():
    import time
    executor = DistributedExecutor.__new__(DistributedExecutor)
    current = time.monotonic()-5
    executor.hosts = {'direct':{'poll_started':current}, 'remote':{'poll_started':current}}
    executor.pbs_manager = None; executor.distributed = {'relay':{'workers':[{'host_id':'remote'}]}}
    assert executor._inflight_direct_offer('direct',current)
    assert not executor._inflight_direct_offer('direct',current-1)
    assert not executor._inflight_direct_offer('remote',current)
    executor.hosts['direct']['poll_started']=current-20
    assert not executor._inflight_direct_offer('direct',current-20)
