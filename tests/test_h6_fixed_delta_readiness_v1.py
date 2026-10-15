"""Stdlib synthetic/mock checks; no molecular inputs or science imports."""
import cmath
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h6_fixed_delta_plan_v1 as p
import h6_fixed_delta_runner_v1 as r


def config():
    return dict(kind=p.KIND,SOURCE='a'*40,plan=p.preparation(),
        resources=dict(worker_cpus=list(range(20,28)),driver_cpu=28))


def grant(c):
    return dict(kind=p.KIND,approved=True,runtime_authorization=True,user_approval='synthetic only',
        config_digest=p.digest(c),SOURCE=c['SOURCE'],one_shot=True,retry=False,next_stage_authorized=False,
        allowed_cpus=list(range(20,29)),stage_maximum=c['plan']['demand'])


def signal(q=4):
    return dict(cell=p.cells(q)[0],signals=dict(corrected=dict(real=.6,imag=-.2),
        exact_truncated=dict(real=.61,imag=-.21)),reference=dict(real=.62,imag=-.22),
        allowance=dict(real=1e-10,imag=2e-10,log_B_margin=0.),log_B=0.,
        independent_forward_signals={'B0_only':True},precision_rows={})


class PlanTests(unittest.TestCase):
    def test_exact_panel(self):
        x=p.preparation();self.assertEqual(len(x['cells']),15)
        self.assertEqual([(x['T'],x['q']) for x in x['times']],list(p.TIMES))
    def test_fixed_local_step(self):
        for T,q in p.TIMES:self.assertEqual(T/q,.2);self.assertEqual(T/q/4,.05)
    def test_fixed_methods(self):
        for q in (2,4,8):self.assertEqual([x['method'] for x in p.cells(q)],['B0','B0I','B1','B1','B2'])
    def test_exact_demand(self):
        self.assertEqual(p.demand(p.registrations()),dict(groups=72,compile=144,trajectory=64,
            sampling_calls=320,rte_events=1280,control_probe=40))
    def test_no_T08_dispatch(self):self.assertNotIn(.8,[x['task']['T'] for x in p.registrations()])
    def test_unique_seeds(self):
        seeds=[x['task']['seed'] for x in p.registrations() if x['task']['seed'] is not None]
        self.assertEqual(len(seeds),len(set(seeds)))
    def test_seed_reproducible(self):self.assertEqual(p.registrations(),p.registrations())
    def test_no_old_namespace(self):self.assertNotIn('H6_MATCHED_V1_FRESH_COST',p.NAMESPACE)
    def test_seed_scope_reject(self):
        with self.assertRaises(ValueError):p.seed(.8,p.cells(4)[-1],0)
    def test_seed_replica_reject(self):
        with self.assertRaises(ValueError):p.seed(.4,p.cells(2)[-1],32)
    def test_noninteger_q_reject(self):
        with self.assertRaises(ValueError):p.cells(True)
    def test_heavy_partition(self):
        tasks=p.registrations();self.assertEqual(sum(x['task']['heavy'] for x in tasks),2)
    def test_unapproved_defaults(self):
        c=p.preparation();self.assertFalse(c['approved']);self.assertFalse(c['runtime_authorization'])
        self.assertEqual(c['allowed_cpus'],[]);self.assertIsNone(c['absolute_launch_command'])
    def test_stop_defaults(self):
        c=p.preparation();self.assertTrue(c['mandatory_stop']);self.assertFalse(c['retry']);self.assertFalse(c['resume'])
    def test_normalization_accumulation(self):
        # Same illustrative local finite paired Taylor distribution, different R.
        tau=.01;b=sum(tau**k/math.factorial(k)*math.hypot(1,tau/(k+1)) for k in (0,2))
        logs=[q*4*math.log(b) for _,q in p.TIMES]
        self.assertEqual(logs[1],2*logs[0]);self.assertEqual(logs[2],2*logs[1])


class PhaseTests(unittest.TestCase):
    def test_all_T_phases(self):
        for T,q in p.TIMES:
            before=signal(q);out=p.rotate_record(before,T,.173)
            z=out['signals']['corrected'];self.assertAlmostEqual(complex(**dict(real=z['real'],imag=z['imag'])),
                cmath.exp(-1j*T*.173)*complex(.6,-.2))
    def test_reference_unchanged(self):
        for T,q in p.TIMES:self.assertEqual(p.rotate_record(signal(q),T,.7)['reference'],signal(q)['reference'])
    def test_no_input_mutation(self):
        a=signal();b=deepcopy(a);p.rotate_record(a,.8,.7);self.assertEqual(a,b)
    def test_no_B0_oracle_mislabel(self):
        x=p.rotate_record(signal(),.8,.1);self.assertNotIn('independent_forward_signals',x)
    def test_complex_error_closure(self):
        x=p.rotate_record(signal(),.8,.3)['error_decomposition'];self.assertLess(x['complex_closure_discrepancy'],1e-14)
    def test_not_global_phase(self):
        # Control branch 0 is identity; branch 1 has exp(-i*T*C).
        phase=cmath.exp(-1j*.8*.7);self.assertNotAlmostEqual(phase.real,1.)
        self.assertAlmostEqual((1+phase).real/2,(1+math.cos(.8*.7))/2)
    def test_zero_scalar(self):self.assertEqual(p.rotate_record(signal(),.8,0.)['signals'],signal()['signals'])
    def test_nonfinite_scalar(self):
        for c in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):p.rotate_record(signal(),.8,c)
    def test_negative_error(self):
        with self.assertRaises(ValueError):p.rotate_record(signal(),.8,0.,-1.)
    def test_wrong_T_q(self):
        with self.assertRaises(ValueError):p.rotate_record(signal(),1.6,0.)
    def test_allowance_not_decreased(self):
        x=p.rotate_record(signal(),.8,.3);self.assertGreater(x['allowance']['real'],signal()['allowance']['real'])


class GateTests(unittest.TestCase):
    def test_bound_mock_grant(self):
        c=config()
        with patch.object(r.os,'sched_getaffinity',return_value=set(range(64))):r.grant_gate(c,grant(c))
    def test_old_grant_rejected(self):
        with self.assertRaises(ValueError):r.grant_gate(config(),dict(approved_by_user=True))
    def test_unapproved_rejected(self):
        c=config();g=grant(c);g['approved']=False
        with self.assertRaises(ValueError):r.grant_gate(c,g)
    def test_binding_mutation_rejected(self):
        c=config();g=grant(c);c['SOURCE']='b'*40
        with self.assertRaises(ValueError):r.grant_gate(c,g)
    def test_extra_cpu_rejected(self):
        c=config();g=grant(c);g['allowed_cpus'].append(29)
        with self.assertRaises(ValueError):r.grant_gate(c,g)
    def test_budget_mutation_rejected(self):
        c=config();g=grant(c);g['stage_maximum']=dict(compile=143)
        with self.assertRaises(ValueError):r.grant_gate(c,g)
    def test_no_retry(self):
        c=config();g=grant(c);g['retry']=True
        with self.assertRaises(ValueError):r.grant_gate(c,g)
    def test_concurrency_classification(self):
        self.assertEqual(r.resources(config(),'normal')['cost_workers'],8)
        self.assertEqual(r.resources(config(),'heavy')['cost_workers'],1)
        self.assertEqual(r.resources(config(),'signal')['cost_workers'],2)
    def test_as_parent_worker_ceiling(self):
        for mode in ('normal','heavy','signal'):
            x=r.resources(config(),mode);self.assertGreaterEqual(x['coordinator_hard_address_space_bytes'],x['cost_address_space_bytes'])
    def test_max_slots_fit_rss(self):
        for mode in ('normal','heavy','signal'):
            x=r.resources(config(),mode);self.assertLessEqual(x['cost_workers']*x['worker_rss_stop_bytes']+
                x['coordinator_rss_reserve_bytes']+x['aggregate_margin_bytes'],x['aggregate_rss_bytes'])
    def test_external_observer_freshness(self):self.assertEqual(r.observation_policy()['maximum_observation_seconds'],5.)
    def test_forbidden_tmp(self):
        with self.assertRaises(ValueError):r.private('/tmp/no')
    def test_science_import_is_lazy(self):
        spec=importlib.util.spec_from_file_location('delta_science',r.ROOT/'scripts/h6_fixed_delta_science_v1.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertNotIn('qiskit',sys.modules);self.assertNotIn('numpy',sys.modules)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=os.environ['H6_TEST_HOME'])
        self.path=Path(self.temp.name);self.writer=r.Writer(self.path)
        self.task=dict(T=.4)
    def tearDown(self):self.temp.cleanup()
    def test_exclusive_output(self):
        self.writer.write('one.json',{})
        with self.assertRaises(FileExistsError):self.writer.write('one.json',{})
    def test_byte_cap(self):
        with self.assertRaises(ValueError):r.Writer(self.path,1).write('one.json',{})
    def test_missing_terminal_not_success(self):
        with self.assertRaises(FileNotFoundError):r.admit(self.path,self.task)
    def test_stop_terminal_rejected(self):
        self.writer.write('worker_terminal.json',dict(status='STOP',task=self.task))
        with self.assertRaises(ValueError):r.admit(self.path,self.task)
    def test_incomplete_pair_rejected(self):
        self.writer.write('worker_terminal.json',dict(status='COMPLETE',task=self.task))
        self.writer.write('cost_result.json',dict(task=self.task,rows=[dict(axis='cosine')]))
        with self.assertRaises(ValueError):r.admit(self.path,self.task)


if __name__=='__main__':unittest.main(verbosity=2)
