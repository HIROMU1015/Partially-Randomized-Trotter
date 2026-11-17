"""Synthetic resource bounds/slot trials and metadata-only relay discovery."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_q3_resources as R
from trottertracks import rpe_q3_worker as W
spec=importlib.util.spec_from_file_location('_resource_relay',ROOT/'scripts/run_rpe_q3_relay.py')
Relay=importlib.util.module_from_spec(spec);spec.loader.exec_module(Relay)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)


def profile(**policy):
    return dict(host_id='ARTIFICIAL',slots=1,dp_workers=2,
                slot_policy=dict(mode='auto',initial_slots=1,**policy))


def resources(cpu=8,memory=16000000,free=16000000):
    return dict(allowed_cpu_count=cpu,allowed_memory_bytes=memory,
                remaining_group_memory_bytes=free,machine_mem_available_bytes=free)


def feedback(count,seconds,wait=0,**extra):
    return dict(controller_invocation='ARTIFICIAL_CLOCK',usable_results=count,adopted_results=count,
                measurement_seconds=seconds,decision_wait_seconds=wait,ready_jobs=8,**extra)


class ResourceTests(unittest.TestCase):
    def test_process_cgroup_allocation_bounds_are_not_machine_idle_resources(self):
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary);proc=base/'proc';group=base/'cgroup'
            write(proc/'self/cgroup','0::/scope\n')
            write(proc/'meminfo','MemTotal: 1000000 kB\nMemAvailable: 900000 kB\n')
            write(group/'scope/cpu.max','200000 100000\n')
            write(group/'scope/cpuset.cpus.effective','0-7\n')
            write(group/'scope/memory.max','200000000\n')
            write(group/'scope/memory.current','50000000\n')
            write(group/'memory.max','500000000\n')
            with patch.object(R.os,'cpu_count',return_value=96):
                got=R.resource_snapshot(profile(),proc_root=proc,cgroup_root=group,
                    affinity={0,1,2,3},environ={'SLURM_CPUS_PER_TASK':'3'})
            self.assertEqual(got['machine_cpu_count'],96)
            self.assertEqual(got['allowed_cpu_count'],2)
            self.assertEqual(got['allowed_memory_bytes'],200000000)
            self.assertEqual(got['remaining_group_memory_bytes'],150000000)
            self.assertIn('NONEXCLUSIVE',got['cpu_scope'])
            self.assertTrue(got['observational_only'])

    def test_owned_tree_reads_only_explicit_supervisor_and_children(self):
        with tempfile.TemporaryDirectory()as temporary:
            base=Path(temporary)
            for pid,rss,child in ((11,20,'12'),(12,30,'')):
                write(base/str(pid)/'status','VmRSS: '+str(rss)+' kB\n')
                values=['R']+['0']*20;values[11]='10';values[12]='20'
                write(base/str(pid)/'stat',str(pid)+' (owned process) '+' '.join(values))
                write(base/str(pid)/'task'/str(pid)/'children',child)
            got=R.owned_tree_sample(11,proc_root=base)
            self.assertEqual(got['rss_bytes'],50*1024)
            self.assertEqual(got['sampled_pids'],2)
            self.assertGreater(got['cpu_seconds'],0)

    def test_cold_trial_measured_improvement_revert_and_drain(self):
        p=profile();jobs=[dict(state='TRANSFERRED',peak_rss_bytes=1000000,dp_workers=2)]
        state,offer=R.decide_slots(p,{},resources(),[],feedback(0,0),now=0)
        self.assertEqual(offer['capacity'],1)
        state,offer=R.decide_slots(p,state,resources(),jobs,feedback(1,10,10),now=10)
        self.assertEqual(offer['capacity'],2)
        self.assertEqual(offer['stage'],'RESOURCE_BOUNDED_EXPLORATORY_TRIAL')
        state,offer=R.decide_slots(p,state,resources(),jobs,feedback(2,15,14),now=15)
        self.assertEqual(offer['capacity'],2)
        self.assertEqual(offer['stage'],'MEASURED_THROUGHPUT_AND_WAIT')
        state,offer=R.decide_slots(p,state,resources(),jobs,feedback(3,20,18),now=20)
        self.assertEqual(offer['capacity'],3)
        state,offer=R.decide_slots(p,state,resources(),jobs,feedback(4,40,40),now=40)
        self.assertEqual(offer['capacity'],2)
        self.assertEqual(offer['stage'],'MEASURED_TRIAL_REVERT_DRAIN')
        running=[dict(state='RUNNING',peak_rss_bytes=1000000,dp_workers=2)]*2
        state,offer=R.decide_slots(p,state,resources(cpu=1),running,feedback(4,41,41),now=41)
        self.assertEqual(offer['capacity'],1)
        self.assertEqual(offer['slots'],0)
        self.assertTrue(offer['running_jobs_may_exceed_reduced_capacity'])

    def test_equal_trial_is_not_improvement_and_polling_does_not_inflate_rate(self):
        p=profile();jobs=[dict(state='TRANSFERRED',peak_rss_bytes=1000000)]
        state,_=R.decide_slots(p,{},resources(),jobs,feedback(0,0),now=0)
        for second in (1,2,30,99):
            state,_=R.decide_slots(p,state,resources(),jobs,feedback(0,second),now=second)
        state,offer=R.decide_slots(p,state,resources(),jobs,feedback(1,100,10),now=100)
        self.assertEqual(state['trial']['rate'],.01)
        state,offer=R.decide_slots(p,state,resources(),jobs,feedback(2,200,20),now=200)
        self.assertEqual(offer['capacity'],1)
        self.assertEqual(offer['stage'],'MEASURED_TRIAL_REVERT_DRAIN')

    def test_unknown_and_untransferred_completion_hold_slots_fixed_mode_unchanged(self):
        for status in ('UNKNOWN','COMPLETED_PENDING_TRANSFER'):
            _,offer=R.decide_slots(profile(),{},resources(),[dict(state=status)])
            self.assertEqual(offer['slots'],0)
        _,offer=R.decide_slots(dict(slots=3),{},resources(cpu=1),[dict(state='RUNNING')])
        self.assertEqual(offer['capacity'],3)
        self.assertEqual(offer['slots'],2)
        self.assertEqual(offer['stage'],'EXPLICIT_FIXED_PROFILE')

    def test_mailbox_feedback_persists_and_resume_keeps_slot_state(self):
        with tempfile.TemporaryDirectory()as temporary:
            p=dict(profile(),root=str(ROOT),interpreter=sys.executable,route_views={},work_dir=temporary)
            with patch.object(R,'resource_snapshot',return_value=resources()):
                first=W.mailbox(p,'resources',dict(capacity_feedback=feedback(0,0)))
                second=W.mailbox(p,'status')
            self.assertEqual(first['capacity'],1)
            self.assertEqual(second['capacity'],1)
            self.assertTrue((Path(temporary)/'SLOT_STATE.json').exists())
            self.assertEqual(json.loads((Path(temporary)/'CAPACITY_FEEDBACK.json').read_text())['usable_results'],0)
            self.assertFalse(list(Path(temporary).glob('*/*/ASSIGNMENT.json')))


class DiscoveryTests(unittest.TestCase):
    def relay(self,base,workers):
        config=base/'relay.json';write(config,json.dumps(dict(state_directory=str(base/'state'),
            controller={},workers=workers,external_tunnel=True,poll_seconds=.02)))
        return Relay.Relay(config)

    def test_all_ready_host_offers_precede_claims_even_when_wsl_listed_first(self):
        with tempfile.TemporaryDirectory()as temporary:
            workers=[dict(host_id='wsl',transport='wsl'),dict(host_id='a'),dict(host_id='b')]
            relay=self.relay(Path(temporary),workers);calls=[]
            def remote(worker,action,value=None):
                self.assertEqual(action,'status')
                return dict(host_id=worker['host_id'],slots=1,capacity=1,source_sha='ARTIFICIAL',attempts=[],cached_prefixes=[]),0
            def post(path,value):
                calls.append((value['host_id'],value.get('observe_only',False)))
                return dict(assignments=[])
            with patch.object(relay,'remote',side_effect=remote),patch.object(relay,'post',side_effect=post):
                relay.cycle()
            relay.discovery_pool.shutdown(wait=True)
            first_claim=next(i for i,(_,observe)in enumerate(calls)if not observe)
            self.assertEqual({h for h,observe in calls[:first_claim]if observe},{'a','b','wsl'})
            self.assertEqual([h for h,observe in calls if not observe],['a','b','wsl'])

    def test_unresponsive_preferred_host_does_not_block_wsl_or_duplicate_status(self):
        with tempfile.TemporaryDirectory()as temporary:
            workers=[dict(host_id='missing'),dict(host_id='wsl',transport='wsl')]
            relay=self.relay(Path(temporary),workers);gate=threading.Event();calls=[];statuses=[]
            def remote(worker,action,value=None):
                statuses.append(worker['host_id'])
                if worker['host_id']=='missing':gate.wait(2)
                return dict(host_id=worker['host_id'],slots=1,capacity=1,source_sha='ARTIFICIAL',attempts=[],cached_prefixes=[]),0
            def post(path,value):
                calls.append((value['host_id'],value.get('observe_only',False)))
                return dict(assignments=[])
            try:
                with patch.object(relay,'remote',side_effect=remote),patch.object(relay,'post',side_effect=post):
                    started=time.monotonic();relay.cycle()
                    self.assertLess(time.monotonic()-started,.3)
                    self.assertIn(('wsl',False),calls)
                    relay.cycle()
                    self.assertEqual(statuses.count('missing'),1)
                    gate.set();relay.discovery_pool.shutdown(wait=True)
            finally:
                gate.set()
                relay.discovery_pool.shutdown(wait=True)


if __name__=='__main__':unittest.main()
