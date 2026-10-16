"""Off-domain scalar engineering tests. No saved Hamiltonian/signal inputs."""
import ast
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch
import mpmath as mp
from jsonschema import Draft202012Validator

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v2 import bounds, boundary, control, spec
from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v2.supervisor import supervise
mp.mp.dps=150


def prohibit_science_open(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).absolute()
        artifact=ROOT/'artifacts'
        if artifact in p.parents and not str(p).startswith(str(ROOT/spec.ART)+'/'):
            raise AssertionError('Saved scientific artifact open prohibited in technical tests')
sys.addaudithook(prohibit_science_open)


def m(x):return mp.mpf(x.numerator)/x.denominator


class ScalarEnclosures(unittest.TestCase):
    def setUp(self):self.B=bounds.Bounds(64,spec.RESOURCES)
    def encloses(self,pair,x):self.assertLessEqual(m(pair[0]),x);self.assertGreaterEqual(m(pair[1]),x)
    def test_pi(self):self.encloses(self.B.pi(),mp.pi)
    def test_sin(self):
        for x in [F(0),F(1,17),F(1,2),F(3,2),F(1,2**500)]:
            with self.subTest(x=str(x)):self.encloses(self.B.sin_positive(x),mp.sin(m(x)))
    def test_cos(self):
        for x in [F(0),F(1,19),F(1),F(3,2)]:
            with self.subTest(x=str(x)):self.encloses(self.B.cos_positive(x),mp.cos(m(x)))
    def test_atan_range_reductions(self):
        pi=self.B.pi()
        for x in [F(0),F(1,13),F(1,2),F(3,4),F(1),F(19),F(2**80)]:
            with self.subTest(x=str(x)):self.encloses(self.B.atan_interval((x,x),pi),mp.atan(m(x)))
    def test_log_range_reduction(self):
        for x in [F(1)+F(1,97),F(2),F(17),F(2**40)+F(1,19)]:
            with self.subTest(x=str(x)):self.encloses(self.B.log_above_one(x),mp.log(m(x)))
    def test_outward_pack(self):
        lo,hi=F(-2,7),F(3,11);q=self.B.pack((lo,hi));self.assertLessEqual(q[0],lo);self.assertGreaterEqual(q[1],hi)
    def test_finer_precision_narrows_pi(self):
        a=self.B.pi();b=bounds.Bounds(128,spec.RESOURCES).pi();self.assertLess(b[1]-b[0],a[1]-a[0])
    def test_caps(self):
        r=dict(spec.RESOURCES,max_bound_steps=0)
        with self.assertRaises(bounds.BoundLimit):bounds.Bounds(64,r).pi()
        r=dict(spec.RESOURCES,max_series_terms_per_call=0)
        with self.assertRaises(bounds.BoundLimit):bounds.Bounds(64,r).sin_positive(F(1))
        r=dict(spec.RESOURCES,max_rational_bits=4)
        with self.assertRaises(bounds.BoundLimit):bounds.Bounds(64,r).check(F(1,2**20))
    def test_invalid_domains(self):
        for fn,x in [(self.B.sin_positive,F(-1)),(self.B.cos_positive,F(2)),(self.B.log_above_one,F(1))]:
            with self.assertRaises(ValueError):fn(x)


class ComparisonCases(unittest.TestCase):
    def setUp(self):self.B=bounds.Bounds(64,spec.RESOURCES);self.pi=self.B.pi()
    def compare(self,*args):return boundary.compare(*args,self.B,self.pi)
    def test_equal(self):self.assertEqual(self.compare(F(4,5),F(1,3),F(4,5),F(1,3))['status'],'ALL_COMMON_GAMMA_EQUAL')
    def test_equal_bias(self):self.assertEqual(self.compare(F(9,10),F(1,3),F(4,5),F(1,3))['status'],'ALL_COMMON_GAMMA_SHARED_HIGHER')
    def test_shared_dominance(self):self.assertEqual(self.compare(F(9,10),F(1,5),F(4,5),F(2,5))['status'],'ALL_COMMON_GAMMA_SHARED_HIGHER')
    def test_other_dominance(self):self.assertEqual(self.compare(F(4,5),F(2,5),F(9,10),F(1,5))['status'],'ALL_COMMON_GAMMA_SHARED_LOWER')
    def test_no_domain(self):self.assertEqual(self.compare(F(1),F(2),F(1),F(0))['status'],'NO_COMMON_PHASE_DOMAIN')
    def test_unresolved_domain(self):
        x=sum(self.pi)/4
        self.assertEqual(self.compare(F(1),x,F(1),F(0))['status'],'ARITHMETIC_UNRESOLVED')
    def test_tradeoff_x_nonpositive(self):
        r=self.compare(F(9,10),F(1,2),F(4,5),F(1,10))
        self.assertEqual(r['status'],'ALL_COMMON_GAMMA_SHARED_LOWER')
        self.assertEqual(r['boundary_location_status'],'NO_INTERIOR_ROOT_X_NONPOSITIVE')
    def test_direct_tradeoff_enclosure(self):
        a,u,c,v=F(9,10),F(2,5),F(4,5),F(1,5)
        r=self.compare(a,u,c,v);self.assertEqual(r['status'],'UNIQUE_TRADEOFF_BOUNDARY')
        lo,hi=map(F,r['boundary_interval_rad'])
        x=m(a)*mp.cos(m(u))-m(c)*mp.cos(m(v));y=m(a)*mp.sin(m(u))-m(c)*mp.sin(m(v))
        value=mp.atan(y/x);self.assertLessEqual(m(lo),value);self.assertGreaterEqual(m(hi),value)
        self.assertEqual(r['continuous_relation_after'],'SHARED_HIGHER')
        self.assertEqual(r['strict_integer_saving'],'NOT_INFERRED_FROM_CONTINUOUS_ORDER')
    def test_swapped_tradeoff(self):
        a,u,c,v=F(9,10),F(2,5),F(4,5),F(1,5)
        direct=self.compare(a,u,c,v);reverse=self.compare(c,v,a,u)
        self.assertEqual(direct['boundary_interval_rad'],reverse['boundary_interval_rad'])
        self.assertEqual(reverse['formula_applicability'],'REVIEW_TRADEOFF_SWAPPED')
        self.assertEqual(reverse['continuous_relation_after'],'SHARED_LOWER')
    def test_coefficient_unresolved(self):
        c=F(str(mp.cos(mp.mpf('0.5'))/mp.cos(mp.mpf('0.1'))))
        r=self.compare(F(1),F(1,2),c,F(1,10))
        self.assertEqual(r['status'],'ARITHMETIC_UNRESOLVED')
    def test_invalid_visibility(self):
        with self.assertRaises(ValueError):self.compare(F(0),F(0),F(1),F(0))


class MetadataAndGuard(unittest.TestCase):
    def setUp(self):self.c=json.loads((ROOT/spec.ART/'execution_contract_candidate.json').read_text())
    def test_candidate_and_null_fields(self):
        control.validate_contract(self.c)
        self.assertTrue(all(v is None for v in self.c['science'].values()))
        self.assertIsNone(self.c['resources']['wall_time_seconds'])
        self.assertIsNone(self.c['resources']['cpu_time_seconds'])
        self.assertNotIn('gamma_rad',self.c['science'])
    def test_candidate_closed_before_git_and_science(self):
        with patch.object(control,'git',side_effect=AssertionError('git reached')):
            with self.assertRaises(control.ExecutionClosed):control.authorize(ROOT,spec.ART+'/execution_contract_candidate.json',None,None)
    def test_flags_alone_cannot_authorize(self):
        c=copy.deepcopy(self.c);c.update(adopted=True,coverage_model_adopted=True,status='ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL')
        with self.assertRaises(control.ExecutionClosed):control.validate_contract(c,True)
    def test_time_caps_required_even_when_science_filled(self):
        c=copy.deepcopy(self.c)
        c.update(adopted=True,coverage_model_adopted=True,status='ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL')
        c['science'].update({k:v[0] for k,v in spec.SELECTIONS.items()})
        c['science'].update(alpha_total='1/20',arithmetic_bits=64)
        with self.assertRaisesRegex(control.ExecutionClosed,'time caps not adopted'):
            control.validate_contract(c,True)
    def test_permit_required(self):
        with self.assertRaises(control.ExecutionClosed):control.run(ROOT,None)
    def test_no_scientific_entry_imported(self):
        self.assertNotIn('trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v2.analysis',sys.modules)
    def test_AST_without_bytecode_compile(self):
        for p in spec.SOURCE_FILES:
            if p.endswith('.py'):ast.parse((ROOT/p).read_text())
    def test_schema_metaschema(self):Draft202012Validator.check_schema(json.loads((ROOT/spec.ART/'result_schema.json').read_text()))
    def test_fixed_identifiers_only(self):
        self.assertEqual(len(spec.keys()),108);self.assertEqual(len(spec.contrast_keys()),48)
    def test_immutable_input_and_scope(self):
        for name in ['input','model','direction','grid']:
            c=copy.deepcopy(self.c);c[name]=None
            with self.assertRaises(control.ExecutionClosed):control.validate_contract(c)
    def test_resource_and_precision_types(self):
        for field in ['wall_time_seconds','cpu_time_seconds']:
            c=copy.deepcopy(self.c);c['resources'][field]=True
            with self.assertRaises(control.ExecutionClosed):control.validate_contract(c)
        c=copy.deepcopy(self.c);c['science']['arithmetic_bits']=True
        with self.assertRaises(control.ExecutionClosed):control.validate_contract(c)
        for alpha in ['0','1','2']:
            c=copy.deepcopy(self.c);c['science']['alpha_total']=alpha
            with self.assertRaises(control.ExecutionClosed):control.validate_contract(c)


class SupervisorTechnicalOnly(unittest.TestCase):
    def test_successful_non_numeric_worker(self):
        r=supervise(lambda:'GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT',1)
        self.assertEqual(r['status'],'COMPLETE_WORKER_EXIT')
    def test_hard_wall_stop_preserves_failure_classification(self):
        r=supervise(lambda:time.sleep(1),0.03)
        self.assertTrue(r['wall_cap_reached']);self.assertEqual(r['status'],'TECHNICAL_STOP')
        self.assertFalse(r['retry'])
    def test_worker_failure(self):
        r=supervise(lambda:'TECHNICAL_STOP',1)
        self.assertEqual(r['status'],'TECHNICAL_STOP')


class RuntimeCompatibility(unittest.TestCase):
    def test_optional_api_present(self):
        from types import SimpleNamespace
        from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v2 import runtime_compatibility as rt
        calls=[]
        with patch.object(rt,'sys',SimpleNamespace(set_int_max_str_digits=calls.append)):
            rt.configure_integer_serialization()
        self.assertEqual(calls,[25000])

    def test_optional_api_absent(self):
        from types import SimpleNamespace
        from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v2 import runtime_compatibility as rt
        with patch.object(rt,'sys',SimpleNamespace()):
            rt.configure_integer_serialization()

    def test_actual_interpreter_max_rational_serialization(self):
        from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v2.runtime_compatibility import configure_integer_serialization
        configure_integer_serialization()
        value=(1<<spec.RESOURCES['max_rational_bits'])-1
        text=str(value)
        self.assertLess(len(text),25000)
        self.assertEqual(int(text),value)

    def test_actual_worker_startup_and_checkpoint_without_science(self):
        # Exercise the real resource/startup/save path on an isolated child.
        # Mock only the scientific evaluator, with metadata-only artificial JSON.
        import tempfile
        from types import ModuleType
        from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v2.supervisor import supervise
        c=spec.template()
        c['resources']=dict(c['resources'],cpu_time_seconds=160,wall_time_seconds=320)
        with tempfile.TemporaryDirectory(prefix='m1s-gamma-startup-test-') as directory:
            root=Path(directory)
            raw=b'{}\n';(root/'input.json').write_bytes(raw)
            c['input']=dict(c['input'],path='input.json',bytes=len(raw),sha256=control.digest(raw))
            outpath=spec.OUTPUT_PREFIX+'technical_startup_only'
            output=root/outpath;output.mkdir(parents=True)
            permit=control._Permit(control._TOKEN,c,{'output_path':outpath})
            module=ModuleType('artificial_metadata_evaluator')
            def evaluate(raw,c,permit,result,checkpoint):
                self.assertEqual(raw,{})
                result['status']='GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT'
                checkpoint(result)
            module.evaluate=evaluate
            with patch.dict(sys.modules,{'trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v2.analysis':module}):
                summary=supervise(lambda:control.worker_run(root,permit,256*1024*1024),5)
            self.assertEqual(summary['worker_exitcode'],0)
            saved=json.loads((output/'result.json').read_text())
            self.assertIsNone(saved['failure'])
            self.assertEqual(len(saved['rows']),108)
            self.assertTrue(all(x['status']=='NOT_EVALUATED' for x in saved['rows']))
            self.assertFalse((output/'result.partial.tmp').exists())
            self.assertFalse((output/'exclusive_launch_marker.json').exists())


if __name__=='__main__':unittest.main()
