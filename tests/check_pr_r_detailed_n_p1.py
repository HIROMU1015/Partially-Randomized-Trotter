"""Artificial/mock numerical leaves, real frozen B grouping and public metadata."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pr_r_detailed_n_p1 as m
from check_pf_pr_rpe_pipeline import fake_shot


def pair():
    return dict(reference=dict(real=1.,imag=0.),independent_reference=dict(real=1.,imag=0.),signal_difference=0.,certified=False)


class P1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.static=m.preflight(ROOT)

    def fixture(self):
        pkt=deepcopy(self.static)
        for item in pkt['saved_refs'].values():item['evidence']=pair()
        for old in pkt['baselines'].values():
            r=old['record'];s=r['identity']['structure']
            r['diagnostics']=dict(signal=dict(reference=dict(real=1.,imag=0.)),
                signed_weights=s['signed_weights'],allocation=s['allocation'],event_count=s['event_count'],
                reference_solver_energy_ha=0.,state_residual_ha=0.)
        return pkt

    def harness(self,pkt,*,failed_R=None,scalar_fail=False):
        bridge=m.b_bridge(ROOT,pkt['rows'][0]['N_request']['law_binding']['definition'])
        loads=[];actions=[];scalar=[]
        def loader(root,system):
            def load(task):
                loads.append(system);return SimpleNamespace(scientific_binding=task['binding'],half_steps={}),{}
            return load
        bridge.bound_context_factory=loader
        class FakeBatch:
            def __init__(self,task,base,source,context,reference):
                self.task=task;self.context=context;self.reference=reference;self.last_run=None
            def __call__(self,task,rows):
                ctx,_=self.context(task)
                for T in sorted({r['candidate']['T'] for r in rows}):self.reference(ctx,T)
                result=[]
                for row in rows:
                    c=row['candidate'];actions.append(c['id']);n=fake_shot(task,row,passed=c['R_outer']!=failed_R)
                    n.update(origin='EXPLICIT_PR_R_VARIANT_N',diagnostics=dict(synthetic_only=True))
                    result.append(n)
                self.last_run=dict(finite_mean_attempts=len(rows),PR_exact_tail_controls_completed=len(rows))
                return result
        def analyze(task,row,*args):
            scalar.append(row['candidate']['id']);n=fake_shot(task,row,passed=not scalar_fail)
            n.update(origin='EXPLICIT_PR_R_VARIANT_N',diagnostics=dict(synthetic_only=True));return n
        return bridge,FakeBatch,analyze,loads,actions,scalar

    def execute(self,pkt,out,**kwargs):
        bridge,batch,analyze,loads,actions,scalar=self.harness(pkt,**kwargs)
        with patch.object(m,'b_bridge',return_value=bridge),patch.object(m.nmodel,'RBatch',batch), \
             patch.object(m.nmodel,'analyze',side_effect=analyze),patch.object(m,'new_reference',return_value=pair()) as ref:
            got=m.execute(pkt,'ARTIFICIAL_SOURCE',out)
        return got,loads,actions,scalar,ref.call_count

    def test_passive_preflight_fixed12_and_no_numpy(self):
        code="from pathlib import Path;from trottertracks import pr_r_detailed_n_p1 as m;m.preflight(Path.cwd());import sys;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit'))"
        env=__import__('os').environ.copy();env['PYTHONPATH']='src:scripts'
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)
        self.assertEqual(len(self.static['rows']),12)
        self.assertFalse(any(r['m']==12 for r in self.static['rows']))
        bridge=m.b_bridge(ROOT,self.static['rows'][0]['N_request']['law_binding']['definition'])
        self.assertNotIn('FixedSavedPrices',bridge.RMenuBatch.__init__.__globals__)
        self.assertNotIn('run_evaluation',bridge.RMenuBatch.__init__.__globals__)

    def test_mock_full12_scope_baselines_first_shared_context_and_reference(self):
        pkt=self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new';got,loads,actions,scalar,refs=self.execute(pkt,out)
            self.assertEqual(got['status'],'P1_COMPLETE_CONDITIONAL_N')
            self.assertEqual((len(actions),len(set(actions)),len(scalar),refs),(9,9,3,1))
            self.assertEqual(sorted(loads),['H4','H6'])
            self.assertEqual(actions[0],'2d2af6ca830dff14a695')
            self.assertEqual(len(got['records']),12);self.assertEqual(got['scope_observed']['scalar_candidates_completed'],12)
            saved=json.loads((out/'N_HANDOFF.json').read_text())
            self.assertEqual(saved['shots'],got['records']);self.assertEqual(saved['prices'],[])
            self.assertTrue(got['source_preserved']);self.assertTrue(got['input_preserved'])
            self.assertIsNone(got['total_resources'])
            for n in got['records']:self.assertEqual(n['N_comparison_policy'],m.POLICY)
            with self.assertRaises(FileExistsError):self.execute(pkt,out)

    def test_saved_baseline_fail_stops_before_context_or_new_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            got,loads,actions,scalar,refs=self.execute(self.fixture(),Path(tmp)/'new',scalar_fail=True)
        self.assertEqual(got['status'],'P1_STOP');self.assertEqual((loads,actions,refs),([],[],0))
        self.assertEqual(len(scalar),1);self.assertEqual(len(got['records']),12)
        self.assertEqual(sum(n['plan']['status']=='NOT_EXECUTED_AFTER_P1_STOP' for n in got['records']),11)

    def test_B_group_failure_preserves_unexecuted_no_R_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            got,loads,actions,scalar,refs=self.execute(self.fixture(),Path(tmp)/'new',failed_R=11)
        self.assertEqual(got['status'],'P1_STOP_UNQUALIFIED_N')
        self.assertEqual(len(actions),len(set(actions)));self.assertEqual(refs,1)
        statuses={n['identity']['candidate']['id']:n['plan']['status'] for n in got['records']}
        self.assertEqual(statuses['5e2581e9c4f25bdf8529'],'UNQUALIFIED_ROUND')
        self.assertEqual(statuses['4269eafc8c0ef8922c23'],'NOT_EXECUTED_AFTER_N_FAILURE')

    def test_missing_or_foreign_reference_never_falls_back(self):
        for kind in ('missing','foreign'):
            pkt=self.fixture();key=('H4',.1)
            if kind=='missing':del pkt['saved_refs'][key]
            else:pkt['saved_refs'][key]['binding']={'state':'FOREIGN'}
            with tempfile.TemporaryDirectory() as tmp:
                got,loads,actions,scalar,refs=self.execute(pkt,Path(tmp)/'new')
            self.assertEqual(got['status'],'P1_STOP');self.assertEqual((loads,actions,refs),([],[],0))

    def test_two_path_gate_and_input_source_loss_are_never_qualified(self):
        r=dict(binding={'state':'TOY'},T=.1,source={'path':'TOY'},evidence=pair())
        r['evidence']['independent_reference']['imag']=.2
        with self.assertRaises(ValueError):m.checked_reference(r,{'state':'TOY'},.1)
        pkt=self.fixture();pkt['sources'][m.NEW_FILES[0]]='WRONG'
        with tempfile.TemporaryDirectory() as tmp:got,*_=self.execute(pkt,Path(tmp)/'new')
        self.assertEqual(got['status'],'P1_STOP_SOURCE_OR_INPUT_CHANGED');self.assertFalse(got['source_preserved'])


if __name__=='__main__':unittest.main(verbosity=2)
