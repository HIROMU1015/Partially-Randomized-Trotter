"""Only artificial arrays/scalars, except fixed-file bytes/header inspection."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import io
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mpmath
import numpy as np
import precision_adapter as a
import run_precision_isolation as runner

I = np.eye(2, dtype=complex)


def payload(value):
    value=complex(value)
    return {'real':value.real,'imag':value.imag}


def pair(law,reference):
    return {name:{'mean':payload(value)} for name,value in [('law',law),('reference',reference)]}


def stage(law,reference,converged=True):
    return {'best_pair':pair(law,reference),'convergence':{'converged':converged}}


def synthetic_factors():
    return {name:{'forward':I.copy(),'reverse':I.copy(),'tail':I.copy()} for name in ('law','reference')}


class PrecisionTests(unittest.TestCase):
    def test_precision_ladder_is_derived_and_bounded(self):
        self.assertEqual(a.precision_ladder(70,131072),(96,160,224))
        with self.assertRaises(ValueError):a.precision_ladder(0,131072)

    def test_binary64_conversion_preserves_exact_dyadics_not_decimal_substitution(self):
        ctx=mpmath.mp.clone();ctx.prec=160
        for value in (.1,-.1,np.nextafter(1.,2.),np.nextafter(0.,1.),float(2**80)):
            numerator,denominator=float(value).as_integer_ratio()
            exact=a.exact_real(ctx,value)
            self.assertEqual(exact*denominator,ctx.mpf(numerator))
        self.assertNotEqual(a.exact_real(ctx,.1),ctx.mpf('0.1'))
        ctx.prec=32
        with self.assertRaises(ValueError):a.exact_real(ctx,.1)

    def test_shape_dtype_and_nonfinite_inputs_are_rejected(self):
        for value in (np.eye(2,dtype=np.complex64),np.eye(2),np.ones((2,3),dtype=complex),
                      np.array([[complex(float('nan'),0)]],dtype=complex)):
            with self.assertRaises(ValueError):a.validated_array(value)
        with self.assertRaises(ValueError):a.validated_array(I,vector=True)
        with self.assertRaises(ValueError):a.exact_real(a.scalar_context(),float('inf'))

    def test_known_large_q_nilpotent_power(self):
        step=np.array([[1.,2.**-20],[0.,1.]],dtype=complex)
        job={'path':'law','mode':'fixed_step','bits':96,'q':131072,'step':step,
             'psi':np.array([1.,.5],dtype=complex)}
        result=a.evaluate_job(job)
        ctx=a.scalar_context()
        self.assertEqual(a.number(ctx,result['mean']),ctx.mpf('1.3125'))

    def test_binary_power_matches_closed_form_and_backend_on_toy(self):
        ctx=a.scalar_context();matrix=ctx.matrix([[1,ctx.mpc(0,1)/8],[0,1]])
        expected=ctx.matrix([[1,ctx.mpc(0,1)*7/8],[0,1]])
        actual=a.binary_power(matrix,7)
        self.assertEqual(actual,expected)
        self.assertEqual(actual,matrix**7)
        with self.assertRaises(ValueError):a.binary_power(matrix,-1)

    def test_noncommuting_factor_order_is_not_reversed(self):
        job={'path':'law','mode':'fixed_factors','bits':96,'q':1,
             'forward':np.array([[1,1],[0,1]],dtype=complex),
             'reverse':np.diag([1j,1.]).astype(complex),'tail':np.diag([.5,.25]).astype(complex),
             'psi':np.array([1,1j],dtype=complex)}
        actual=a.evaluate_job(job)
        self.assertEqual(a.number(a.scalar_context(),actual['mean']),-.25+.5j)

    def test_saved_attenuation_is_not_applied_twice_or_rebuilt(self):
        job={'path':'law','mode':'fixed_factors','bits':96,'q':3,'forward':I,'reverse':I,
             'tail':I*.5,'psi':np.array([1.,0.],dtype=complex)}
        result=a.evaluate_job(job)
        self.assertEqual(a.number(a.scalar_context(),result['mean']),.125)
        self.assertFalse(result['interval_certified'])

    def test_modes_use_distinct_fixed_step_and_factor_inputs(self):
        base={'path':'law','bits':96,'q':1,'forward':I,'reverse':I,'tail':I*.5,
              'step':I*.25,'psi':np.array([1.,0.],dtype=complex)}
        step=a.evaluate_job({**base,'mode':'fixed_step'})
        factors=a.evaluate_job({**base,'mode':'fixed_factors'})
        ctx=a.scalar_context()
        self.assertEqual(a.number(ctx,step['mean']),.25)
        self.assertEqual(a.number(ctx,factors['mean']),.5)

    def test_clone_context_does_not_change_global_precision(self):
        before=mpmath.mp.prec
        a.evaluate_job({'path':'law','mode':'fixed_step','bits':160,'q':2,'step':I,
                        'psi':np.array([1.,0.],dtype=complex)})
        self.assertEqual(mpmath.mp.prec,before)

    def test_convergence_and_candidate_labels_do_not_certify_rpe(self):
        baseline={name:payload(v) for name,v in [('law',2e-10),('reference',0)]}
        previous=pair(0,0);current=pair(1e-15,0)
        self.assertTrue(a.convergence(previous,current)['converged'])
        result=a.interpret(True,baseline,stage(0,0))
        self.assertEqual(result['candidate_labels'],['POWER_ROUNDOFF_PLAUSIBLE'])
        self.assertIsNone(result['u_num_bound'])
        self.assertFalse(result['RPE_tolerance_certified'])
        self.assertEqual(result['original_numeric_diagnostic_status'],'INCONSISTENT')
        assembly=a.interpret(True,baseline,stage(2e-10,0),stage(0,0))
        self.assertEqual(assembly['candidate_labels'],['CONSTRUCTION_ROUNDOFF_PLAUSIBLE'])
        retained=a.interpret(True,baseline,stage(2e-10,0),stage(2e-10,0))
        self.assertIn('IMPLEMENTATION_OR_DEFINITION_DIFFERENCE_POSSIBLE',retained['candidate_labels'])

    def test_unconverged_or_threshold_edge_is_insufficient(self):
        baseline={'law':payload(2e-10),'reference':payload(0)}
        self.assertEqual(a.interpret(True,baseline,stage(0,0,False))['candidate_labels'],['INSUFFICIENT_EVIDENCE'])
        self.assertEqual(a.interpret(False,baseline,stage(0,0))['candidate_labels'],['INSUFFICIENT_EVIDENCE'])
        self.assertEqual(a.gap_relation(pair(1e-10,0))['relation_to_original_atol'],'NEAR_THRESHOLD')
        self.assertEqual(a.interpret(True,{'law':payload(0),'reference':payload(0)},stage(0,0))['candidate_labels'],['INSUFFICIENT_EVIDENCE'])

    def test_ladder_third_tier_is_only_used_after_failed_convergence(self):
        factors=synthetic_factors();psi=np.array([1.,0.],dtype=complex);steps={'law':I,'reference':I}
        def batch(jobs,pool,record,output):
            return [{'path':j['path'],'bits':j['bits'],'mean':payload(2e-12 if j['bits']==96 else 0)} for j in jobs]
        with patch.object(runner,'evaluate_batch',side_effect=batch) as call:
            result=runner.run_ladder('fixed_step',factors,psi,steps,(96,160,224),None,{},None)
        self.assertEqual(call.call_count,2)
        self.assertEqual(result['precision_bits_used'],[96,160,224])
        self.assertTrue(result['convergence']['converged'])

    def test_actual_metadata_uses_canonical_archive_names_without_array_decode(self):
        with patch.object(runner.np,'load',side_effect=AssertionError('array decode')):
            metadata=runner.fixed_metadata()
        self.assertEqual(set(metadata['operator_headers']),runner.OPERATOR_KEYS)
        self.assertEqual(metadata['operator_headers']['law_tail_mean']['shape'],(70,70))
        self.assertEqual(metadata['old_result']['evaluation']['numeric_diagnostic_status'],'INCONSISTENT')

    def test_default_cli_does_not_enter_science(self):
        with patch.object(runner.np,'load',side_effect=AssertionError('array decode')), \
             patch.object(runner,'load_arrays',side_effect=AssertionError('H4 loader')), \
             patch.object(a,'float_baseline',side_effect=AssertionError('power')), \
             patch.object(a,'evaluate_job',side_effect=AssertionError('high precision')), \
             redirect_stdout(io.StringIO()) as stream:
            runner.main([])
        self.assertEqual(json.loads(stream.getvalue())['status'],'PREPARED_NOT_SCIENTIFICALLY_EXECUTED')
        self.assertEqual(json.loads(stream.getvalue())['original_numeric_diagnostic_status'],'INCONSISTENT')

    def test_interruption_preserves_stop_record_and_rejects_overwrite(self):
        metadata={'inputs':[],'precision_bits':(96,160,224)}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'synthetic-only'
            with patch.object(runner,'source_metadata',return_value={'commit':'synthetic'}), \
                 patch.object(runner,'fixed_metadata',return_value=metadata), \
                 patch.object(runner,'load_arrays',side_effect=RuntimeError(str(runner.ROOT)+'/synthetic')):
                with self.assertRaises(RuntimeError):runner.execute(output)
                record=json.loads((output/'RESULT.json').read_text())
                self.assertEqual(record['execution_status'],'INTERRUPTED_OR_INVALID_MANDATORY_STOP')
                self.assertIn('<repository>',record['error']['message'])
                original=(output/'RESULT.json').read_bytes()
                with self.assertRaises(FileExistsError):runner.execute(output)
                self.assertEqual((output/'RESULT.json').read_bytes(),original)

    def test_spawned_workers_and_synthetic_driver_preserve_stage_scope(self):
        metadata={'inputs':[],'precision_bits':(96,160,224),'old_result':{'evaluation':{'means':{
            'law':payload(1.),'reference':payload(1.)}}}}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'synthetic-only'
            with patch.object(runner,'source_metadata',return_value={'commit':'synthetic'}), \
                 patch.object(runner,'fixed_metadata',return_value=metadata), \
                 patch.object(runner,'load_arrays',return_value=(synthetic_factors(),np.array([1.,0.],dtype=complex))):
                runner.execute(output)
            record=json.loads((output/'RESULT.json').read_text())
            self.assertEqual(record['execution_status'],'COMPLETED_MANDATORY_STOP')
            self.assertEqual(len(record['high_precision_jobs']),4)
            self.assertNotIn('fixed_factors',record)
            self.assertIsNone(record['interpretation']['u_num_bound'])
            with np.load(output/'frozen_steps.npz',allow_pickle=False) as saved:self.assertEqual(set(saved.files),{'law','reference'})

    def test_factor_stage_is_conditional_on_converged_retained_gap(self):
        baseline={'law':payload(2e-10),'reference':payload(0)}
        metadata={'inputs':[],'precision_bits':(96,160,224),'old_result':{'evaluation':{'means':baseline}}}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'synthetic-only'
            with patch.object(runner,'source_metadata',return_value={'commit':'synthetic'}), \
                 patch.object(runner,'fixed_metadata',return_value=metadata), \
                 patch.object(runner,'load_arrays',return_value=(synthetic_factors(),np.array([1.,0.],dtype=complex))), \
                 patch.object(a,'float_baseline',return_value=(baseline,{'law':I,'reference':I})), \
                 patch.object(runner,'run_ladder',side_effect=[stage(2e-10,0),stage(0,0)]) as ladder:
                runner.execute(output)
            self.assertEqual([c.args[0] for c in ladder.call_args_list],['fixed_step','fixed_factors'])
            record=json.loads((output/'RESULT.json').read_text())
            self.assertEqual(record['interpretation']['candidate_labels'],['CONSTRUCTION_ROUNDOFF_PLAUSIBLE'])

    def test_baseline_failure_stops_before_high_precision_pool(self):
        baseline={'law':payload(0),'reference':payload(0)}
        metadata={'inputs':[],'precision_bits':(96,160,224),'old_result':{'evaluation':{'means':{
            'law':payload(1.),'reference':payload(1.)}}}}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'synthetic-only'
            with patch.object(runner,'source_metadata',return_value={'commit':'synthetic'}), \
                 patch.object(runner,'fixed_metadata',return_value=metadata), \
                 patch.object(runner,'load_arrays',return_value=(synthetic_factors(),np.array([1.,0.],dtype=complex))), \
                 patch.object(a,'float_baseline',return_value=(baseline,{'law':I,'reference':I})), \
                 patch.object(runner,'ProcessPoolExecutor',side_effect=AssertionError('No high precision')):
                runner.execute(output)
            record=json.loads((output/'RESULT.json').read_text())
            self.assertFalse(record['baseline_reproduced'])
            self.assertEqual(record['interpretation']['candidate_labels'],['INSUFFICIENT_EVIDENCE'])
            self.assertEqual(record['high_precision_jobs'],[])


if __name__=='__main__':
    unittest.main()
