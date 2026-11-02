"""Current counter versus pinned native schedule, artificial inputs only."""
import ast
import importlib.util
import math
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value
a=load('compat_prior_audit',ROOT/'scripts/audit_h46_s4_count_mismatch.py')
m=load('compat_current_counter',ROOT/'src/trotterlib/native_df_gate_model.py')


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.f=a.functions()

    def test_coefficient_hex(self):
        native=self.f.weights.yoshida_4th_list()
        times=m.pf_schedule(1,2.,1,'4th','symmetric_directional')
        self.assertEqual([times[i].time.hex() for i in (0,2,4)],
                         [native[i].hex() for i in (1,0,1)])

    def test_all_stage_hex_block_order_and_modes(self):
        for q in (1,2,3,8):
            for t in (.8,-.8,0.,-0.):
                for formula in ('2nd','4th'):
                    actual=m.pf_schedule(3,t,q,formula,'symmetric_directional')
                    expected=self.f.schedule.deterministic_control_plan(3,t,q,formula)
                    self.assertEqual([(s.block,s.time.hex(),s.mode) for s in actual],
                                     [(s.term,s.time.hex(),s.mode.lower()) for s in expected])

    def test_registered_negative_time_hex(self):
        for q,hexvalue in ((1,'-0x1.5ca78b1289ec1p-1'),(8,'-0x1.5ca78b1289ec1p-4')):
            s=m.pf_schedule(1,.8,q,'4th','symmetric_directional')
            self.assertEqual(next(x.time.hex() for x in s if x.time<0),hexvalue)

    def test_s2_schedule_unchanged(self):
        old=self.f.m
        for q in (1,3,8):
            for policy in ('ordinary','symmetric_directional'):
                fields=lambda xs:[(s.block,s.mode,s.time.hex()) for s in xs]
                self.assertEqual(fields(m.pf_schedule(4,.8,q,'2nd',policy)),
                                 fields(old.pf_schedule(4,.8,q,'2nd',policy)))

    def test_only_schedule_function_changed(self):
        def rest(source):
            tree=ast.parse(source)
            tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='pf_schedule']
            return ast.dump(tree,include_attributes=False)
        self.assertEqual(rest((ROOT/'src/trotterlib/native_df_gate_model.py').read_bytes()),rest(self.f.source['counter']))

    def test_artificial_zero_selection_count_differences_resolved(self):
        f=a.functions();f.m=m
        for eta in ((1.,2.,-3.),(1.,5.,-6.)):
            for q in (1,8):
                r=a.artificial_pair(f,eta,q=q)
                self.assertEqual(set(r['delta_native_minus_model'].values()),{0})
                for s in r['stages']:
                    self.assertEqual(s['model_time_hex'],s['native_time_hex'])
                    self.assertEqual(s['model_rz_indices'],s['native_rz_indices'])

    def test_support_phase_and_explicit_zero_rules(self):
        pp=self.f.ops.DiagonalEvolutionPrimitives(.3,((0,0.),(1,1e-30)),((0,1,0.),(1,2,.2)))
        f=a.functions();f.m=m
        for mode in ('UNCONTROLLED','DIRECTIONAL'):
            self.assertEqual(a.append_native(f,pp,mode,3),m.diagonal_from_primitives(pp,mode.lower(),3))

    def test_wrapper_and_scalar_unchanged(self):
        for axis in ('X','Y'):
            circuit=m.Circuit((m.Gate('p',(2,),.4),),-.2)
            full=m.count(m.hadamard_wrapper(circuit,2,axis))
            self.assertEqual(full,m.count(circuit)+m.count(m.hadamard_wrapper(m.Circuit(),2,axis)))
            self.assertEqual(m.hadamard_wrapper(circuit,2,axis).global_phase,-.2)

    def test_invalid_inputs_fail_closed(self):
        for q in (0,-1,True,1.5):
            with self.assertRaises(ValueError):m.pf_schedule(1,.8,q,'4th')
        for time in (float('nan'),float('inf'),-float('inf')):
            with self.assertRaises(ValueError):m.pf_schedule(1,time,1,'4th')
        with self.assertRaises(ValueError):m.pf_schedule(1,.8,1,'invalid')


if __name__=='__main__':unittest.main(verbosity=2)
