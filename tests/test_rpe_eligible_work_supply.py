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
    executor.pbs_manager = None
    executor.ready_queue_waves = 2
    executor._offer_free = Mock(return_value=1)
    executor._host_supports_stage = Mock(return_value=True)
    executor._fits_host_walltime = Mock(return_value=True)

    def row(job, cpu, eligible, stage=None):
        request = {'preparation': {'stage': stage}} if stage else {}
        profile = {'task_cpu_count': cpu}
        return dict(job_id=job, host_id='worker', request=request,
                    eligibility=dict(eligible=eligible, execution_profiles={'worker': profile}))

    rows = {value['job_id']: value for value in (
        row('running', 3, True), row('q3', 2, True), row('audit', 1, True, 'audit'),
        row('wrong-host', 20, False, 'compile'))}
    executor.store = SimpleNamespace(row=rows.__getitem__, eligible=lambda request, eligibility, *_: eligibility['eligible'])
    ready = [SimpleNamespace(job=Path(job)) for job in ('q3', 'audit', 'wrong-host')]
    supply = executor._ready_cpu_supply(ready, [rows['running']], {}, None)['worker']
    assert supply['available_cpu'] == 5
    assert supply['ready_cpu_by_stage'] == {'q3': 2, 'audit': 1}
    assert supply['missing_cpu'] == 7


def test_partial_price_preflight_has_same_roles_and_leaves_blocked_pair_out():
    plan = dict(new_compile_cells=[dict(m=1, role='CALIBRATION'), dict(m=5, role='HOLDOUT')],
                maximum_new_compile=4, calibration_rounds=[1, 2], holdout_round=5)
    selected = pf_saved_prices.independent_plan(plan, {'1': [{'axis': 'X', 'action': 'HELD'}]})
    assert selected['new_compile_cells'] == [dict(m=5, role='HOLDOUT')]
    assert selected['maximum_new_compile'] == 2
    assert selected['calibration_rounds'] == [1, 2]
    assert selected['holdout_round'] == 5
    assert plan['maximum_new_compile'] == 4
