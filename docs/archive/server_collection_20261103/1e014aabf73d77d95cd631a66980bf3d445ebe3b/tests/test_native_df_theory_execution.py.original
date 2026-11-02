"""Artificial resource-gate and saved-accounting checks; no science imports."""
import copy
from fractions import Fraction
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

ROOT=Path(__file__).resolve().parents[1]
def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
launcher=module('theory_launcher','launch_native_df_theory_evaluation.py')
join=module('theory_join','connect_native_df_theory_saved_costs.py')


class ResourceChecks(unittest.TestCase):
    def setUp(self):
        zero=dict(avg10=0.,avg60=0.,avg300=0.,total=0.)
        self.base=dict(memory={'MemAvailable':1000},pressure={'host':{'some':zero.copy()},'cg':{'some':zero.copy()}},events={'cg':{'oom':0,'oom_kill':0}})
    def test_idle(self):self.assertIsNone(launcher.reason(self.base,self.base,100))
    def test_headroom(self):self.assertEqual(launcher.reason(self.base,self.base,1001),'MEMORY_HEADROOM')
    def test_host_threshold(self):
        current=copy.deepcopy(self.base);current['pressure']['host']['some']['avg10']=1
        self.assertEqual(launcher.reason(current,self.base,100),'MEMORY_PRESSURE')
    def test_cgroup_pressure(self):
        current=copy.deepcopy(self.base);current['pressure']['cg']['some']['avg10']=.001
        self.assertEqual(launcher.reason(current,self.base,100),'MEMORY_PRESSURE')
    def test_oom(self):
        current=copy.deepcopy(self.base);current['events']['cg']['oom_kill']=1
        self.assertEqual(launcher.reason(current,self.base,100),'OOM_INCREMENT')


class SavedJoin(unittest.TestCase):
    def test_artificial_join_and_reject_inconsistent_old_total(self):
        # Only temp within the user's home. No default /tmp use.
        with tempfile.TemporaryDirectory(prefix='theory-test-',dir=Path.home()) as td:
            path=Path(td);m={'inputs':{}};theory=[]
            for system in ('H6','H8'):
                axes={'real':{'shots':2},'imag':{'shots':3}}
                row=dict(cell_id='fixture',epsilon=.01,factor=1,accounting={'axes':axes},
                    metrics={key:{'axis_means':{'cosine':4,'sine':4},'G_exact':{'numerator':'20','denominator':'1'}} for key in ('RZ','CX')})
                d=path/(system+'.json');d.write_text(json.dumps({system+'_EXPLORATION92':[row]}))
                f=path/(system+'family.json');f.write_text(json.dumps({'cases':[]}))
                m['inputs'][system+'_d0']={'path':str(d)};m['inputs'][system+'_B0I_results']={'path':str(f)}
                for model in ('A','B'):
                    for axis in ('cosine','sine'):
                        cost={k:join.encode(Fraction(6)) for k in ('rz','cx')}
                        theory.append(dict(system=system,candidate_id='fixture',model=model,axis=axis,lower=cost,upper=cost))
            result=join.connect(m,theory)
            self.assertEqual(len(result),8);self.assertTrue(all(join.rational(x['theory_lower'])==30 for x in result))
            row['metrics']['RZ']['G_exact']['numerator']='21';d.write_text(json.dumps({'H8_EXPLORATION92':[row]}))
            with self.assertRaisesRegex(ValueError,'SAVED_COMPILED_ACCOUNTING'):join.connect(m,theory)


if __name__=='__main__':unittest.main()
