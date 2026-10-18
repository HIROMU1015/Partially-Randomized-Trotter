"""Artificial schedules and toy matrices only; no stored molecular inputs."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('m5_test',ROOT/'src/trotterlib/h46_m5_plan.py')
k=importlib.util.module_from_spec(spec);spec.loader.exec_module(k)


class M5Tests(unittest.TestCase):
    def test_exact_source_literals_and_operation_order(self):
        data=json.loads((ROOT/'references/h46_m5_best_formula_v1.json').read_text())
        source=data['function_source'].replace(' -> List[float]','')
        env={};exec(source,env);compact=env[data['function_name']]()
        expected=list(reversed(compact[1:]))+[compact[0]]+compact[1:]
        self.assertEqual([v.hex() for v in k.weights('4th(m5_best)')],[v.hex() for v in expected])

    def test_symmetry_negative_piece_and_normalization(self):
        w=k.weights('4th(m5_best)');self.assertEqual(len(w),11);self.assertEqual(w,w[::-1])
        self.assertEqual(sum(v<0 for v in w),1);self.assertLess(abs(math.fsum(w)-1),1e-14)
        # Numerical residual is reported with scale; exact zero is not demanded.
        self.assertLess(abs(math.fsum(v**3 for v in w)),1e-12)

    def test_allocation_calls_frozen_pure_function(self):
        w=k.weights('4th(m5_best)')
        for n in (11,16,32,44):
            self.assertEqual(k.allocation(w,n),k.frozen_allocation()(w,n))
            self.assertEqual(sum(k.allocation(w,n)),n);self.assertGreaterEqual(min(k.allocation(w,n)),1)
        with self.assertRaises(ValueError):k.allocation(w,10)

    def test_signed_time_order_and_rte_budgets(self):
        for label in k.FORMULAS:
            s=k.schedule(label,2,'PER_PIECE',4)
            self.assertEqual(len(s),2*len(k.weights(label)))
            self.assertEqual(sum(x['rte_steps'] for x in s),8*len(k.weights(label)))
            self.assertAlmostEqual(math.fsum(x['signed_time'] for x in s),.8)
            self.assertEqual([x['piece'] for x in s[:len(k.weights(label))]],list(range(len(k.weights(label)))))
            absolute=k.schedule(label,2,'ABS_TIME_PER_OUTER',16)
            self.assertEqual(sum(x['rte_steps'] for x in absolute),32)

    def test_b0_keeps_base_scalar_b0i_restores_tail(self):
        for mode in ('FULL','B0','B0I','PR'):
            seen=[]
            out=k.lowered_schedule('4th(m5_best)',2,'PER_PIECE',2,mode=mode,scalar=.3,tail_identity=.7,T=.8,lower_s2=lambda **kw:seen.append(kw) or kw)
            self.assertEqual(len(out),22)
            expected=.8*(1. if mode in ('B0I','PR') else .3)
            self.assertAlmostEqual(math.fsum(x['scalar_time'] for x in seen),expected)
            self.assertTrue(all(x['control_policy']=='symmetric_directional' for x in seen))
            self.assertEqual(sum(x['rte_steps'] for x in seen),44 if mode=='PR' else 0)

    def test_pilot_fixed_call_counts(self):
        for system in k.SYSTEMS:
            p=k.pilot(system);self.assertEqual(len(p['deterministic']),9);self.assertEqual(len(p['PR']),3)
            self.assertEqual(p['compile_axes'],2*(9+3*2));self.assertEqual(p['native_evolutions'],15)
            self.assertFalse(p['ranking_authorized']);self.assertFalse(p['independent_confirmation'])

    def test_sparse_screen_and_baseline_domains_retained(self):
        for system,detcount in [('H4',52),('H6',92)]:
            s=k.screen(system);self.assertEqual(len(s['PR']),54);self.assertEqual(len(s['new_M5_deterministic']),detcount)
            ids=[x['id'] for x in s['PR']+s['new_M5_deterministic']];self.assertEqual(len(ids),len(set(ids)))
            self.assertTrue(s['existing_S2_Y4_deterministic_domains_retained']);self.assertFalse(s['automatic_q16_extension'])
            self.assertEqual({x['K'] for x in s['PR']},{2,4})

    def test_four_independent_pilot_lanes_and_stop_isolation(self):
        nodes=k.graph();p2=[n['id'] for n in nodes if n['phase']==2]
        self.assertEqual(len(p2),4);self.assertEqual(set(k.ready(nodes,[],p2)),set(p2))
        self.assertEqual(set(k.ready(nodes,[],p2,['H4_P2_DET'])),set(p2)-{'H4_P2_DET'})
        self.assertTrue(all(n['end_STOP'] for n in nodes))

    def test_no_unapproved_cross_phase_dispatch(self):
        nodes=k.graph();allids=[n['id'] for n in nodes];completed=[n['id'] for n in nodes if n['phase']==2]
        self.assertEqual(k.ready(nodes,completed,allids),[])
        self.assertEqual(set(k.ready(nodes,completed,allids,active_phase=3)),{'H4_P3_SIGNAL','H6_P3_SIGNAL'})
        self.assertEqual(k.ready(nodes,completed,[],active_phase=3),[])

    def test_cli_refuses_execute_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'not_created.json'
            r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/plan_h46_m5_best.py'),'--execute','--output',str(p)],capture_output=True,text=True)
            self.assertEqual(r.returncode,2);self.assertFalse(p.exists());self.assertIn('SEPARATE_SCIENTIFIC_APPROVAL',r.stdout)

    def test_invalid_scopes(self):
        for args in [('m5',1,'PER_PIECE',2),('2nd',0,'PER_PIECE',2),('2nd',True,'PER_PIECE',2),('2nd',1,'unknown',2),('2nd',1,'PER_PIECE',0)]:
            with self.assertRaises(ValueError):k.schedule(*args)

    def test_artificial_noncommuting_fourth_order(self):
        import numpy as np
        A=np.array([[0,1],[1,0]],complex)*.73;B=np.diag([-.42,.42]).astype(complex)
        def exp(H,t):
            value,vectors=np.linalg.eigh(H);return (vectors*np.exp(-1j*t*value))@vectors.conj().T
        target=exp(A+B,.8);errors=[]
        for q in (2,4,8,16):
            U=np.eye(2,dtype=complex)
            for row in k.schedule('4th(m5_best)',q,'PER_PIECE',1):
                t=row['signed_time'];U=exp(A,t/2)@exp(B,t)@exp(A,t/2)@U
            errors.append(np.linalg.norm(U-target))
        for a,b in zip(errors,errors[1:]):self.assertLess(abs(math.log2(a/b)-4),.2)


if __name__=='__main__':unittest.main()
