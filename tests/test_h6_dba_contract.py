"""Artificial contracts: thresholds, scopes, seeds and scoped refusal."""
import importlib.util
from pathlib import Path
import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('dba',Path(__file__).resolve().parents[1]/'scripts/h6_dba_contract.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class Contract(unittest.TestCase):
    def test_threshold_independent_exact(self):
        self.assertFalse(c.axis_check('102',100)['PASS'])
        self.assertFalse(c.axis_check('98',100)['PASS'])
        self.assertTrue(c.axis_check('10199/100',100)['PASS'])
        self.assertFalse(c.axis_check('103',100)['PASS'])

    def test_no_nonfinite_or_zero(self):
        for value in (0,-1,1.0,True,float('nan')):
            with self.assertRaises(ValueError):c.axis_check('100',value)

    def test_minimal_scope(self):
        p=c.plan();self.assertEqual(p['D']['native_compile'],2)
        self.assertEqual(p['B']['native_compile'],32)
        self.assertEqual(p['A']['native_compile'],0)
        self.assertEqual(len({x['id'] for x in p['A']['cells']}),16)
        for x in p['A']['cells']:self.assertEqual(x['R'],x['q']*x['r'])

    def test_seed_order_independent_and_distinct(self):
        tasks=[(g['cell']['id'],g['stratum'],i) for g in c.plan()['B']['groups'] for i in range(8)]
        values={t:c.seed('B',*t,'fake-input-hash') for t in tasks}
        self.assertEqual(len(set(values.values())),16)
        self.assertEqual(values,{t:c.seed('B',*t,'fake-input-hash') for t in reversed(tasks)})

    def test_draft_rejected_before_inputs(self):
        with self.assertRaisesRegex(ValueError,'EXPLICIT_SCOPED'):
            c.verify({},dict(stage='D',approved_by_user=False),'D')

    def test_stage_grant_not_transferable(self):
        with self.assertRaisesRegex(ValueError,'STAGE'):
            c.verify({},dict(stage='D'),'B')

    def test_no_automatic_extension(self):
        p=c.plan()
        self.assertFalse(p['retry']);self.assertFalse(p['cost_reaccounting'])
        self.assertTrue(p['mandatory_STOP_per_stage'])
        self.assertEqual(p['B']['K4_s2_multi'],'UNRESOLVED_NOT_MEASURED')

    def test_cosine_failure_prevents_sine_compile(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import h6_dba_native as native
        built=[];written={}
        def wrapper(*args,**kw):
            built.append(kw['axis']);return SimpleNamespace(data=[1])
        result=SimpleNamespace(compiler_settings_hash='compiler',actual_circuit_fingerprint='fp',rz_count=100,cx_count=20)
        m=SimpleNamespace(build_native_hadamard_wrapper=wrapper,canonical_qiskit_circuit_fingerprint=lambda x:'fp',
                          transpile_and_measure_cost=lambda *a,**kw:result)
        writer=SimpleNamespace(write=lambda name,value:written.update({name:value}))
        with self.assertRaisesRegex(ValueError,'AXIS_MODEL_ERROR_GE_2_PERCENT'):
            native.compile_pair(None,m,None,{'binding':{'compiler':'compiler'}},writer,
                                predictions={'cosine':'102','sine':'100'})
        self.assertEqual(built,['cosine'])
        self.assertFalse(written['measured_cosine.json']['model_check']['PASS'])


if __name__=='__main__':unittest.main()
