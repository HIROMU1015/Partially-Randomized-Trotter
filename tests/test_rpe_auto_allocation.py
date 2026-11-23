"""Automatic global capacity and host preference; artificial durable queues only."""
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock
from collections import Counter
from types import SimpleNamespace
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
from trottertracks import rpe_q3_coordinator as Q
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor, ConcurrentExecutor
import test_rpe_q3_host_eligibility_queue as Base

HIGH = ['server2', 'server1', 'local']
WSL = 'local0-wsl'


class CapacityIntegralTests(unittest.TestCase):
    def test_dynamic_capacity_integrates_each_interval_instead_of_final_capacity(self):
        # Pure event replay: no executor thread or job/adapter process starts.
        events = [(0,'EXECUTOR_START',dict(capacity=1)),
            (1,'ENQUEUE',dict(job='a')),(1,'ENQUEUE',dict(job='b')),(1,'ENQUEUE',dict(job='c')),
            (2,'AUTO_CAPACITY',dict(capacity=3)),(3,'LAUNCH',dict(job='a')),
            (4,'LAUNCH',dict(job='b')),(5,'LAUNCH',dict(job='c')),
            (6,'EXIT',dict(job='a')),(7,'EXIT',dict(job='b')),(8,'EXIT',dict(job='c')),
            (9,'AUTO_CAPACITY',dict(capacity=1)),(10,'EXECUTOR_STOP',{})]
        with tempfile.TemporaryDirectory() as directory:
            ex = ConcurrentExecutor.__new__(ConcurrentExecutor)
            ex.run_dir=Path(directory);ex.invocation='ARTIFICIAL_EVENT_REPLAY';ex.capacity=1;ex.dp_workers=None
            ex.lock=threading.RLock();ex.progress_start=0
            ex.state=SimpleNamespace(cv=threading.Condition(),tasks={},metrics=Counter(),by_method={},
                peak_active=3,peak_RSS_KiB=0,failure=None)
            (ex.run_dir/'PROGRESS.jsonl').write_text(''.join(json.dumps(dict(
                monotonic=t,event=event,invocation=ex.invocation,**fields))+'\n' for t,event,fields in events))
            result=ex.metrics()
            self.assertEqual(result['busy_job_seconds'],9)
            self.assertEqual(result['integrated_available_slot_seconds'],24)
            self.assertEqual(result['idle_slot_seconds_with_queued_candidate'],7)
            self.assertEqual(result['idle_slot_seconds_no_queued_candidate'],8)
            self.assertEqual(result['slot_utilization'],9/24)
            self.assertEqual(result['busy_job_seconds']+result['idle_slot_seconds_with_queued_candidate']+
                result['idle_slot_seconds_no_queued_candidate'],24)
            fixed=[(t,event,dict(fields,capacity=3) if event=='EXECUTOR_START' else fields)
                   for t,event,fields in events if event!='AUTO_CAPACITY']
            ex.capacity=3
            (ex.run_dir/'PROGRESS.jsonl').write_text(''.join(json.dumps(dict(
                monotonic=t,event=event,invocation=ex.invocation,**fields))+'\n' for t,event,fields in fixed))
            result=ex.metrics()
            self.assertEqual(result['integrated_available_slot_seconds'],30)
            self.assertEqual(result['slot_utilization'],9/(10*3))


class ReadySupplyTests(unittest.TestCase):
    def test_busy_workers_are_refilled_and_shared_jobs_are_counted_once(self):
        executor = DistributedExecutor.__new__(DistributedExecutor)
        executor.distributed = {}
        executor.pbs_manager = None
        executor.ready_queue_waves = 2
        executor.hosts = {
            host: dict(capacity=1, resource_report=dict(mode='AUTO', resources=dict(allowed_cpu_count=4)))
            for host in ('host-a', 'host-b')}
        executor._host_admission = Mock(return_value=dict(accepting=True))
        executor._offer_fresh = Mock(return_value=True)
        executor._offer_free = Mock(return_value=0)
        executor._host_supports_stage = Mock(return_value=True)
        executor._fits_host_walltime = Mock(return_value=True)
        executor._host_rank = Mock(return_value=0)

        def job(name, options):
            return dict(job_id=name, request={}, request_sha256=Q.digest({}), eligibility=dict(execution_profiles={
                host: dict(dp_workers=4) for host in options}))

        rows = {'shared': job('shared', executor.hosts),
                'a': job('a', ['host-a']), 'b': job('b', ['host-b'])}
        executor.store = SimpleNamespace(
            rows_for=lambda keys: {key: rows[key] for key in keys},
            eligible=lambda request, eligibility, host, *args, **kwargs: host in eligibility['execution_profiles'])
        rows['a']['host_id'] = 'host-a'
        rows['b']['host_id'] = 'host-b'
        queued = [SimpleNamespace(job=SimpleNamespace(name='shared'))]
        owned = [dict(job_id=name, host_id='host-'+name) for name in ('a', 'b')]
        offers = executor._ready_cpu_supply(queued, owned, {}, None)
        self.assertEqual([row['available_cpu'] for row in offers.values()], [0, 0])
        self.assertEqual([row['target_cpu'] for row in offers.values()], [8, 8])
        self.assertEqual(sum(row['ready_cpu'] for row in offers.values()), 4)
        self.assertEqual(sum(row['missing_cpu'] for row in offers.values()), 12)


class AutoAllocationTests(unittest.TestCase):
    setUp = Base.EligibilityQueue.setUp
    close = Base.EligibilityQueue.close
    close_all = Base.EligibilityQueue.close_all
    proposal = Base.EligibilityQueue.proposal
    policy = Base.EligibilityQueue.policy

    def executor(self, maximum='auto', **config):
        ex = DistributedExecutor(ROOT, self.run, self.routes, {}, self.problem,
            optimizer_source=self.source, max_concurrent=maximum, budgets={'METHOD': 12},
            distributed=dict(port=0, affinity_wait_seconds=0,
                auto_allocation=dict(discovery_grace_seconds=0, **config)))
        self.executors.append(ex)
        return ex

    def jobs(self, ex, count, allowed=None):
        view = ex.for_method('METHOD')
        proposals = [self.proposal('auto-'+str(n)) for n in range(count)]
        facts = self.policy(ex, [(p, allowed or HIGH+[WSL]) for p in proposals])
        for p in proposals: view._enqueue(p, demand=True)
        return proposals, facts

    def poll(self, ex, host, facts=None, slots=1, capacity=None, observe=False):
        return ex.rpc('/v1/poll', dict(host_id=host, source_sha=ex.optimizer_source,
            slots=slots, capacity=slots if capacity is None else capacity,
            cached_prefixes=[], host_fingerprints={job:d[host] for job,d in (facts or {}).items() if host in d},
            observe_only=observe))

    def test_two_preferred_free_slots_take_first_two_and_wsl_takes_surplus_now(self):
        ex = self.executor(); proposals, facts = self.jobs(ex, 3)
        self.poll(ex, 'server2', facts, observe=True)
        self.poll(ex, 'server1', facts, observe=True)
        self.poll(ex, 'local', facts, slots=0, observe=True)
        wsl = self.poll(ex, WSL, facts)['assignments']
        self.assertEqual(len(wsl), 1)
        first = self.poll(ex, 'server2', facts)['assignments']
        second = self.poll(ex, 'server1', facts)['assignments']
        self.assertEqual(len(first+second), 2)
        self.assertEqual(len({a['job_id'] for a in first+second+wsl}), 3)
        self.assertEqual(ex.state.active, 3)
        self.assertEqual(ex.for_method('METHOD').capacity, ex.capacity)
        self.assertEqual(ex.state.peak_active, 3)

    def test_small_queue_is_left_for_free_preferred_host(self):
        ex = self.executor(); _, facts = self.jobs(ex, 1)
        self.poll(ex, 'server1', facts, observe=True)
        self.assertEqual(self.poll(ex, WSL, facts)['assignments'], [])
        self.assertEqual(len(self.poll(ex, 'server1', facts)['assignments']), 1)

    def test_automatic_total_counts_only_request_compatible_free_slots(self):
        ex = self.executor();_, facts = self.jobs(ex, 3, ['server2'])
        self.poll(ex, 'server2', facts, slots=1, observe=True)
        incapable = self.poll(ex, 'server1', facts, slots=9, observe=True)
        self.assertEqual(incapable['capacity_feedback']['ready_jobs'], 0)
        self.assertEqual(ex.capacity, 1)
        self.assertEqual(len(self.poll(ex, 'server2', facts)['assignments']), 1)
        self.assertEqual(ex.capacity, 1)
        self.assertEqual(self.poll(ex, WSL, facts)['assignments'], [])

    def test_busy_host_counts_verified_queue_for_capacity_and_still_requires_current_facts(self):
        ex = self.executor()
        _, facts = self.jobs(ex, 3, ['server2'])
        busy = self.poll(ex, 'server2', slots=0, capacity=1)
        feedback = busy['capacity_feedback']
        self.assertEqual(feedback['ready_jobs'], 0)
        self.assertEqual(feedback['capacity_ready_jobs'], 3)
        self.assertEqual(self.poll(ex, 'server2', slots=2)['assignments'], [])
        stale = {job: {'server2': dict(row['server2'], runtime='CHANGED')}
                 for job, row in facts.items()}
        self.assertEqual(self.poll(ex, 'server2', stale, slots=0)['capacity_feedback']['capacity_ready_jobs'], 0)
        self.assertEqual(len(self.poll(ex, 'server2', facts, slots=2)['assignments']), 2)

    def test_busy_preferred_hosts_do_not_delay_only_free_wsl(self):
        ex = self.executor(); _, facts = self.jobs(ex, 1)
        for h in HIGH: self.poll(ex, h, facts, slots=0, capacity=1, observe=True)
        answer = self.poll(ex, WSL, facts)
        self.assertEqual(len(answer['assignments']), 1)
        self.assertNotIn('waiting_for_discovery', answer)

    def test_incompatible_or_changed_fingerprint_is_not_a_preferred_slot(self):
        ex = self.executor(); _, facts = self.jobs(ex, 2, [WSL, 'server1'])
        changed = {j:dict(d, server1=dict(d['server1'],runtime='CHANGED')) for j,d in facts.items()}
        self.assertEqual(self.poll(ex, 'server1', changed, slots=2, observe=True)['capacity_feedback']['ready_jobs'], 0)
        self.assertEqual(self.poll(ex, 'server2', facts, slots=9, observe=True)['capacity_feedback']['ready_jobs'], 0)
        self.assertEqual(len(self.poll(ex, WSL, facts)['assignments']), 1)

    def test_stale_preferred_offer_is_not_waited_for(self):
        ex = self.executor(offer_ttl_seconds=.01); _, facts = self.jobs(ex, 1)
        self.poll(ex, 'server1', facts, observe=True)
        ex.hosts['server1']['observed_monotonic'] -= 1
        self.assertEqual(len(self.poll(ex, WSL, facts)['assignments']), 1)

    def test_relay_offer_has_its_own_finite_lifetime(self):
        ex = self.executor(offer_ttl_seconds=3, remote_offer_ttl_seconds=15)
        ex.distributed['relay'] = {'workers': [{'host_id': WSL}]}
        _, facts = self.jobs(ex, 1)
        for host in (WSL, 'local'):
            self.poll(ex, host, facts, observe=True)
            ex.hosts[host]['observed_monotonic'] -= 6
        self.assertTrue(ex._offer_fresh(WSL))
        self.assertFalse(ex._offer_fresh('local'))
        ex.hosts[WSL]['observed_monotonic'] -= 10
        self.assertFalse(ex._offer_fresh(WSL))

    def test_first_wsl_poll_discovery_is_bounded_when_preferred_never_respond(self):
        ex = self.executor();ex.allocation_config['discovery_grace_seconds'] = .1
        _, facts = self.jobs(ex, 1)
        ex.state.allocation_start -= 100  # A slow plan must not expire discovery before the first worker poll.
        first = self.poll(ex, WSL, facts)
        self.assertEqual(first['assignments'], [])
        self.assertTrue(first['waiting_for_discovery'])
        ex.state.allocation_discovery[WSL] -= 1
        self.assertEqual(len(self.poll(ex, WSL, facts)['assignments']), 1)

    def test_known_preferred_surplus_does_not_wait_for_unresponsive_third_host(self):
        ex = self.executor();ex.allocation_config['discovery_grace_seconds'] = 1
        _, facts = self.jobs(ex, 3)
        self.poll(ex, 'server2', facts, observe=True)
        self.poll(ex, 'server1', facts, observe=True)
        answer = self.poll(ex, WSL, facts)
        self.assertEqual(len(answer['assignments']), 1)
        self.assertNotIn('waiting_for_discovery', answer)

    def test_paused_optional_host_is_not_timing_required_or_busy_wait(self):
        ex = self.executor();_, facts = self.jobs(ex, 1)
        ex.distributed.update(required_hosts=[WSL], host_admission=dict(hosts={
            WSL:dict(accepting=False, optional=True, max_slots=1)}))
        self.poll(ex, 'server1', facts, observe=True)
        events = [json.loads(line) for line in (self.run/'PROGRESS.jsonl').read_text().splitlines()]
        reasons = [event['reasons'] for event in events if event['event'] == 'QUEUE_WAIT_STATE'][-1]
        self.assertNotIn('REQUIRED_HOST_PARTICIPATION_WAIT', reasons.values())
        self.assertNotIn('ALL_ELIGIBLE_SLOTS_BUSY', reasons.values())
        self.assertEqual(len(self.poll(ex, 'server1', facts)['assignments']), 1)

    def test_explicit_global_cap_counts_virtual_preferred_slots(self):
        ex = self.executor(maximum=2); _, facts = self.jobs(ex, 3)
        self.poll(ex, 'server2', facts, observe=True)
        self.poll(ex, 'server1', facts, observe=True)
        self.assertEqual(self.poll(ex, WSL, facts)['assignments'], [])
        self.assertEqual(len(self.poll(ex, 'server2', facts)['assignments']), 1)
        self.assertEqual(len(self.poll(ex, 'server1', facts)['assignments']), 1)
        self.assertEqual(ex.state.active, 2)

    def test_auto_resume_keeps_ownership_and_saved_fixed_limit_cannot_increase(self):
        ex = self.executor(maximum=1); _, facts = self.jobs(ex, 2)
        assigned = self.poll(ex, 'server1', facts)['assignments'][0]
        self.close(ex)
        resumed = self.executor(maximum='auto')
        self.assertFalse(resumed.state.allocation_auto)
        self.assertEqual(resumed.capacity, 1)
        replay = self.poll(resumed, 'server1', capacity=1)['assignments']
        self.assertEqual(replay[0]['attempt_id'], assigned['attempt_id'])
        self.assertEqual(self.poll(resumed, WSL, facts)['assignments'], [])
        self.assertEqual(sum(e['event']=='ASSIGN' for e in resumed.store.events()), 1)

    def test_pause_and_capacity_reduction_drain_owned_attempts(self):
        ex = self.executor(); _, facts = self.jobs(ex, 3)
        ex.distributed['host_admission'] = dict(hosts={WSL:dict(accepting=True, optional=True)})
        first = self.poll(ex, WSL, facts, slots=2, capacity=2)['assignments']
        self.assertEqual(len(first), 2)
        for a in first:
            ex.rpc('/v1/state', dict(host_id=WSL, job_id=a['job_id'], attempt_id=a['attempt_id'], state='RUNNING'))
        ex.distributed['host_admission']['hosts'][WSL]['accepting'] = False
        self.assertEqual(self.poll(ex, WSL, facts, slots=1, capacity=1)['assignments'], [])
        self.assertEqual(sum(r['host_id']==WSL for r in ex.store.rows()), 2)
        self.assertEqual(len(self.poll(ex, 'server1', facts)['assignments']), 1)
        self.assertEqual(ex.state.active, 3)


if __name__ == '__main__': unittest.main()
