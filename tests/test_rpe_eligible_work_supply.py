from pathlib import Path
from threading import Condition
from types import SimpleNamespace
from unittest.mock import Mock, patch

from trottertracks import pf_saved_prices, rpe_live_q3_readiness
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor


def test_controller_scope_hold_skips_input_delivery_and_host_probes(tmp_path):
    request = dict(method='PR_DF', route={'kind': 'prdf_d2_common_entry'})
    executor = Mock()
    executor.distributed = dict(local_worker={'host_id': 'local'})
    executor.source_manifest = {}
    executor.run_dir = tmp_path
    executor._host_admission_policy.return_value = ({}, None)
    executor.state.closed = False
    executor.state.cv = Condition()
    hold = dict(reason='COUNT_ONLY_SCOPE_EXCEEDED', scientific_evaluation_calls=0)
    with patch.object(rpe_live_q3_readiness, 'controller_scope_hold', return_value=hold), patch.object(
            rpe_live_q3_readiness, 'controller_input_hashes', side_effect=AssertionError('unexpected input delivery')):
        decision = rpe_live_q3_readiness.probe(executor, request, 'job', dict(eligible_host_ids=[]))
    assert decision['controller_scope'] == hold
    assert not decision['eligible_host_ids']
    assert (tmp_path/'distributed/q3-readiness/job/CONTROLLER_SCOPE.json').is_file()


def test_ready_cpu_counts_only_work_that_can_be_sent_to_the_worker():
    executor = object.__new__(DistributedExecutor)
    executor.hosts = {'worker': {'resource_report': {'resources': {'allowed_cpu_count': 8}}}}
    executor.distributed = {}
    executor.allocation_config = {'priority_groups': []}
    executor.pbs_manager = None
    executor.ready_queue_waves = 2
    executor._offer_free = Mock(return_value=1)
    executor._offer_fresh = Mock(return_value=True)
    executor._host_admission = Mock(return_value={'accepting': True})
    executor._host_supports_stage = Mock(return_value=True)
    executor._fits_host_walltime = Mock(return_value=True)

    def row(job, cpu, eligible, stage=None):
        request = {'preparation': {'stage': stage}} if stage else {}
        profile = {'task_cpu_count': cpu}
        return dict(job_id=job, host_id='worker', request=request, request_sha256='fixed-'+job,
                    eligibility=dict(eligible=eligible, execution_profiles={'worker': profile}))

    rows = {value['job_id']: value for value in (
        row('running', 3, True), row('q3', 2, True), row('audit', 1, True, 'audit'),
        row('wrong-host', 20, False, 'compile'))}
    executor.store = SimpleNamespace(rows_for=lambda jobs: {job: rows[job] for job in jobs},
                                     eligible=lambda request, eligibility, *_, **__: eligibility['eligible'])
    ready = [SimpleNamespace(job=Path(job)) for job in ('q3', 'audit', 'wrong-host')]
    supply = executor._ready_cpu_supply(ready, [rows['running']], {}, None)['worker']
    assert supply['available_cpu'] == 5
    assert supply['ready_cpu_by_stage'] == {'q3': 2, 'audit': 1}
    assert supply['missing_cpu'] == 7


def test_pbs_keeps_replacement_work_that_fits_after_a_running_job_finishes():
    from trottertracks import rpe_pbs

    executor = object.__new__(DistributedExecutor)
    executor.hosts = {'pbs-0': {'resource_report': {'pbs': {'resource_pool': {
        'cpu_count': 8, 'memory_bytes': 13 * rpe_pbs.GIB}}}}}
    executor.pbs_manager = SimpleNamespace(configured={'pbs-0': {'profile': {}}}, settings={})
    executor.distributed = {}
    executor.allocation_config = {'priority_groups': []}
    executor.ready_queue_waves = 2
    executor.live_readiness_refill_seconds = 5
    executor._offer_free = Mock(return_value=1)
    executor._host_supports_stage = Mock(return_value=True)
    executor._fits_host_walltime = Mock(return_value=True)
    executor._offer_fresh = Mock(return_value=True)
    executor._host_admission = Mock(return_value={'accepting': True})
    jobs = {
        'running': dict(cpus=2, gib=8),
        'next': dict(cpus=4, gib=6, preparation={'stage': 'compile'}),
        'too_large': dict(cpus=1, gib=13),
    }
    rows = {name: dict(job_id=name, host_id='pbs-0', request=request, eligibility={}, request_sha256='fixed-'+name)
            for name, request in jobs.items()}
    executor.store = SimpleNamespace(rows_for=lambda jobs: {job: rows[job] for job in jobs},
                                     eligible=lambda *_, **__: True)
    with patch.object(rpe_pbs, 'resource_need',
                      side_effect=lambda request, *_: (request['cpus'], request['gib'] * rpe_pbs.GIB)), \
         patch.object(rpe_pbs, 'task_estimate', return_value=20):
        ready = [SimpleNamespace(job=Path(name)) for name in ('next', 'too_large')]
        supply = executor._ready_cpu_supply(ready, [rows['running']], {}, None)['pbs-0']
    assert supply['compatible_ready_jobs'] == 1
    assert supply['ready_jobs_fitting_free_resources'] == 0
    assert supply['ready_cpu_seconds_by_stage'] == {'compile': 80}


def test_partial_price_preflight_has_same_roles_and_leaves_blocked_pair_out():
    plan = dict(new_compile_cells=[dict(m=1, role='CALIBRATION'), dict(m=5, role='HOLDOUT')],
                maximum_new_compile=4, calibration_rounds=[1, 2], holdout_round=5)
    selected = pf_saved_prices.independent_plan(plan, {'1': [{'axis': 'X', 'action': 'HELD'}]})
    assert selected['new_compile_cells'] == [dict(m=5, role='HOLDOUT')]
    assert selected['maximum_new_compile'] == 2
    assert selected['calibration_rounds'] == [1, 2]
    assert selected['holdout_round'] == 5
    assert plan['maximum_new_compile'] == 4


def test_four_host_compile_admission_preserves_legacy_request_placement():
    executor = object.__new__(DistributedExecutor)
    executor.distributed = dict(pf_legacy_compile_hosts=['server2'],
        preparation_provider_hosts={'PF_COMPILE': ['local', 'server2', 'local0-wsl', 'pbs-0']})
    prep = {'stage': 'compile', 'provider': 'PF_COMPILE', 'inputs': {}}
    request = {'preparation': prep}
    assert executor._host_supports_stage('server2', request)
    assert not executor._host_supports_stage('local', request)
    prep['inputs']['private/pf-transport/scripts/run_pf_preparation_stage.py'] = 'fixed'
    assert all(executor._host_supports_stage(host, request)
               for host in ('local', 'server2', 'local0-wsl', 'pbs-0'))
    assert not executor._host_supports_stage('server1', request)


def test_portable_pf_request_ships_verified_wrapper_and_snapshot(tmp_path):
    import hashlib
    import json
    from trottertracks import rpe_preparation_bridge as bridge

    config = tmp_path/'config'
    snapshot = tmp_path/'snapshot'
    snapshot.write_bytes(b'saved snapshot')
    sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    config.write_text(json.dumps(dict(provider_root='/fixed/provider', systems={'H4': {
        'inputs': {'snapshot': dict(path='/fixed/snapshot', sha256=sha)}}})))
    source = dict(PF_NATIVE_ORIGIN_CONFIG='/fixed/config',
                  PF_NATIVE_ORIGIN_SHA256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  PF_NATIVE_H4_INPUTS=('snapshot',))
    producer = object.__new__(bridge.DistributedPreparationBridge)
    producer.settings = dict(controller_paths={'/fixed/config': {'path': str(config)},
                                               '/fixed/snapshot': {'path': str(snapshot)}})
    producer._blob = lambda data: hashlib.sha256(data).hexdigest()
    script = Path('/controller/scripts/run_pf_preparation_stage.py')
    inputs, bindings, checks = {}, {}, []
    with patch.object(bridge.runpy, 'run_path', return_value=source):
        runtime = producer._pf_native_runtime(script, inputs, bindings, checks)
    assert runtime['runtime'] == dict(input_config='/fixed/config', provider_root='/fixed/provider')
    assert inputs['private/pf-native/snapshot'] == sha
    assert bindings[str(script)] == '@WORK@/private/pf-transport/scripts/rpe_pf_shared_fingerprints.py'
    assert 'private/pf-transport/scripts/run_pf_preparation_stage.py' in inputs
    assert 'private/pf-transport/scripts/rpe_pf_shared_fingerprints.py' in inputs
    assert {row['path']:row['sha256'] for row in checks} == inputs
