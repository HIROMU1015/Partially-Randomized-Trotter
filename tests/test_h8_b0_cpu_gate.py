"""CPU diagnostics and immutable failed admissions, using no H8 scientific data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h8_b0_anchor_runtime as r
import h8_b0_anchor_contract as c
import launch_h8_b0_anchors as launch
import probe_h8_b0_anchor_cpus as probe


def window(busy_by_CPU):
    first={i:[0]*8 for i in busy_by_CPU}
    last={i:[busy,0,0,100-busy,0,0,0,0] for i,busy in busy_by_CPU.items()}
    return r.idle_receipt(first,last,busy_by_CPU)


def cores(n=8):
    return {i:dict(package_id='0',core_id=str(i),siblings=[i]) for i in range(n)}


class CPURecords(unittest.TestCase):
    def test_threshold_unchanged_equal_pass_above_stop(self):
        self.assertEqual(window({0:5})['status'],'PASS')
        self.assertEqual(window({0:6})['status'],'STOP')
        w=window({0:17,2:0,5:0})
        self.assertEqual(w['failing_CPUs'],[0]);self.assertEqual(len(w['CPUs']),3)
        self.assertEqual(w['CPUs'][0]['busy_ticks'],17)
        self.assertFalse(w['own_activity_subtracted'])
    def test_counter_unknown_reset_or_empty_not_zero_busy(self):
        for first,last in [({},{}),({0:[1]*8},{0:[0]*8}),({0:[0]*8},{0:[0]*8}),
                ({0:[0]*8},{0:[0]*7}),({0:[0]*8},{0:[0,0,0,True,0,0,0,0]})]:
            self.assertEqual(r.idle_receipt(first,last,[0])['CPUs'][0]['status'],'UNKNOWN_COUNTERS')
        with self.assertRaises(ValueError):r.idle_receipt({}, {}, [])
    def test_exception_has_complete_counters(self):
        try:r.check_idle({0:[0]*8},{0:[10,0,0,90,0,0,0,0]},[0])
        except r.CPUIdleError as error:
            self.assertEqual(str(error),'PHYSICAL_CORE_BUSY');self.assertEqual(error.receipt['failing_CPUs'],[0])
        else:self.fail('busy CPU passed')
    def test_measure_does_not_subtract_self_ticks(self):
        own=[dict(user_ticks=0,system_ticks=0,last_CPU=0,affinity=[0]),
             dict(user_ticks=20,system_ticks=0,last_CPU=0,affinity=[0])]
        with patch.object(r,'self_cpu',side_effect=own),patch.object(r,'cpu_counters',side_effect=[{0:[0]*8},{0:[20,0,0,80,0,0,0,0]}]),patch.object(r.time,'sleep'):
            result=r.measure_idle([0]);self.assertEqual(result['status'],'STOP')
            self.assertEqual(result['self_cpu_ticks_delta'],20)
        with self.assertRaises(ValueError):r.measure_idle([0],seconds=.25)
    def test_one_bad_window_is_not_hidden_by_latest_idle(self):
        windows=[window({i:(17 if i==0 else 0) for i in range(8)}),window({i:0 for i in range(8)}),window({i:0 for i in range(8)})]
        selected=probe.choose(windows,cores())
        self.assertEqual(selected['proposed_cpus'],[1,2,3,4,5,6])
        self.assertFalse(selected['execution_authorized']);self.assertFalse(selected['CPUs_reassigned'])
    def test_smt_sibling_load_and_duplicate_physical_core(self):
        inventory=cores(8);inventory[0]['siblings']=[0,8]
        windows=[window({**{i:0 for i in range(8)},8:20}) for _ in range(3)]
        self.assertNotIn(0,probe.choose(windows,inventory)['proposed_cpus'])
        inventory=cores(6);inventory[1]['core_id']='0'
        self.assertEqual(probe.choose([window({i:0 for i in range(6)})]*3,inventory)['status'],'STOP_NO_SIX_STABLE_PHYSICAL_CORES')
    def test_missing_window_or_sibling_fails_closed(self):
        with self.assertRaises(ValueError):probe.choose([window({i:0 for i in range(8)})]*2,cores())
        inventory=cores(6);inventory[0]['siblings']=[0,99]
        self.assertEqual(probe.choose([window({i:0 for i in range(6)})]*3,inventory)['proposed_cpus'],[])
    def test_probe_fixed_three_windows_no_affinity_change(self):
        receipt=window({i:0 for i in range(8)})
        with patch.object(probe,'inventory',return_value=cores()),patch.object(r,'measure_idle',return_value=receipt) as observed,\
             patch.object(probe.os,'sched_setaffinity') as affinity:
            self.assertEqual(probe.probe()['proposed_cpus'],[0,1,2,3,4,5]);self.assertEqual(observed.call_count,3)
            affinity.assert_not_called()


class FailedAdmission(unittest.TestCase):
    def test_busy_failure_saved_before_fork_and_attempt_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);p=c.plan()
            cfg=dict(SOURCE='a'*40,execution_sealed=True,plan_sha256=c.digest(p),resources=p['resources'],
                output=str(base/'output'),control=str(base/'control'),lock=str(base/'lock'))
            grant=dict(schema='H8_B0_ANCHOR_NEW_SCOPE_GRANT_V1',approved_by_user=True,runtime_authorization=True,
                approval_text='artificial only',scope=c.SCOPE,SOURCE=cfg['SOURCE'],plan_sha256=c.digest(p),
                resources=p['resources'],call_caps=p['call_caps'],one_shot=True,retry=False,resume=False,
                next_stage_authorized=False,driver_cpu=0,monitor_cpu=1,worker_cpus=[2,3,4,5],allowed_cpus=list(range(6)))
            config_path=base/'config.json';config_path.write_text(json.dumps(cfg))
            grant['config_sha256']=r.sha(config_path);grant_path=base/'grant.json';grant_path.write_text(json.dumps(grant))
            grant_sha=r.sha(grant_path);failure=window({0:17,1:0,2:0,3:0,4:0,5:0})
            with patch.object(r,'private_path',side_effect=Path),patch.object(r,'verify_files'),patch.object(r,'protected'),\
                 patch.object(r,'topology',return_value=set(range(6))),patch.object(r,'measure_idle',return_value=failure) as sample,\
                 patch.object(launch.os,'sched_getaffinity',return_value=set(range(6))),patch.object(launch.os,'sched_setaffinity') as affinity,\
                 patch.object(launch.observer,'supervise') as supervise:
                with self.assertRaises(r.CPUIdleError):launch.launch(config_path,grant_path,base/'fresh_gate.json')
                terminal=json.loads((base/'launcher_admission01/terminal.json').read_text())
                cpu=json.loads((base/'launcher_admission01/CPU_initial.json').read_text())
                self.assertEqual(cpu['failing_CPUs'],[0]);self.assertEqual(terminal['stage'],'INITIAL_FRESH_CPU_RESOURCE_GATE')
                self.assertEqual(terminal['reported_exit_code'],1);self.assertFalse(terminal['driver_started'])
                self.assertEqual(terminal['path_state'],dict(control=False,output=False,lock=False))
                self.assertEqual(r.sha(grant_path),grant_sha);supervise.assert_not_called()
                self.assertEqual(affinity.call_args_list[-1].args[1],set(range(6)))
                with self.assertRaisesRegex(ValueError,'ATTEMPT_ALREADY_SPENT'):launch.launch(config_path,grant_path,base/'fresh_gate.json')
                self.assertEqual(sample.call_count,1)
                self.assertFalse((base/'fresh_gate.json').exists())
    def test_refresh_failure_is_recorded_before_exception(self):
        failure=window({0:8})
        with patch.object(r,'verify_authorization',return_value=[0]),patch.object(r,'sha',return_value='config'),patch.object(r,'protected'),\
             patch.object(r,'topology',return_value={0}),patch.object(r,'measure_idle',return_value=failure):
            saved=[]
            with self.assertRaises(r.CPUIdleError):launch.fresh({}, {},Path('/unused'),cpu_record=saved.append)
            self.assertEqual(saved,[failure])
    def test_resource_contract_and_science_caps_unchanged(self):
        original=c.original.proposal(c.plan()['binding'])
        p=c.plan();self.assertEqual(p['call_caps'],original['call_caps'])
        for k,v in original['resources'].items():self.assertEqual(p['resources'][k],v)
        self.assertEqual(p['cases'],original['cases']);self.assertEqual(p['waves'],original['waves'])
        self.assertFalse(p['CPU_admission']['automatic_reassignment'])
        self.assertEqual(p['CPU_admission']['maximum_busy_fraction'],'1/20')


if __name__=='__main__':unittest.main()
