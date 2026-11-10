"""Synthetic accounting and read-only CDF8 checks. No H4 numerical rerun."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import track_d_resource_interface as api


def artificial(method):
    inputs = api.ScientificInputs({'id':'ARTIFICIAL'}, {'id':'ARTIFICIAL_TARGET'},
        ((0,1),(1,2)), 'ARTIFICIAL_IDEAL', 'ARTIFICIAL_DECODER')
    profile = api.ControlProfile({'id':'ARTIFICIAL_COMPILER'}, 'ARTIFICIAL_CUT')
    rows = []
    for m, q in inputs.rounds:
        for axis, base in [('X',1),('Y',2)]:
            v = base+m+(.5 if method=='reference' else 0.)
            rows.append(dict(m=m,q=q,axis=axis,method=method,binding=deepcopy(inputs.binding),
                compiler=deepcopy(profile.compiler),scope=api.SCOPE,
                estimate_unit='whole_path_one_interrogation',source={'kind':'ARTIFICIAL'},
                expectation_basis='qualified_expectation',attenuation_price_multiplier=False,
                mean={k:v*(i+1) for i,k in enumerate(api.bc.METRICS)},
                peak_qubits=5 if method=='reference' else 3))
    prices = api.MethodPrices(inputs,profile,method,tuple(rows))
    accuracy = dict(same_accuracy=True,binding=inputs.binding,target=inputs.target)
    plans = [dict(m=0,shots=2,target_id='ARTIFICIAL'),dict(m=1,shots=3,target_id='ARTIFICIAL')]
    return inputs,accuracy,plans,prices


class Synthetic(unittest.TestCase):
    def test_method_switch_uses_existing_accounting_and_peak_width(self):
        for method,total,width in [('symmetric_directional',21.,3),('reference',26.,5)]:
            inputs,accuracy,plans,prices = artificial(method)
            before = deepcopy(prices)
            original = api.accounting.aggregate_totals
            with patch.object(api.accounting,'aggregate_totals',wraps=original) as delegated:
                result = api.aggregate_supplied(inputs,accuracy,plans,prices)
            delegated.assert_called_once()
            self.assertEqual(result['totals'],{k:total*(i+1) for i,k in enumerate(api.bc.METRICS)})
            self.assertEqual(result['shots'],10)
            self.assertEqual(result['peak_qubits'],width)
            self.assertFalse(result['actual_success_guaranteed'])
            self.assertIsNone(result['intervals'])
            self.assertEqual(prices,before)

    def test_missing_duplicate_or_kernel_price_rejected(self):
        inputs,a,p,c = artificial('reference')
        for rows in (c.rows[:-1],c.rows+(c.rows[0],)):
            with self.assertRaises(ValueError):api.aggregate_supplied(inputs,a,p,replace(c,rows=rows))
        bad = deepcopy(c);bad.rows[0]['scope']='compiled_time_evolution_subcircuit'
        with self.assertRaises(ValueError):api.aggregate_supplied(inputs,a,p,bad)

    def test_scientific_identity_and_unqualified_accuracy(self):
        inputs,a,p,c = artificial('reference')
        wrong = replace(inputs,law_model='OTHER_LAW')
        self.assertIsNone(api.aggregate_supplied(wrong,a,p,c)['totals'])
        a['same_accuracy']=False
        self.assertIsNone(api.aggregate_supplied(inputs,a,p,c)['totals'])

    def test_invalid_width_price_units_and_finite_sample(self):
        inputs,a,p,c = artificial('reference')
        mutations=[('peak_qubits',None),('method','UNKNOWN'),('compiler',{}),
                   ('expectation_basis','finite_sample'),('source',None),
                   ('attenuation_price_multiplier',True)]
        for k,v in mutations:
            bad=deepcopy(c);bad.rows[0][k]=v
            with self.assertRaises(ValueError):api.aggregate_supplied(inputs,a,p,bad)
        for value in (float('nan'),float('inf'),-1.,True):
            bad=deepcopy(c);bad.rows[0]['mean']['cx_count']=value
            with self.assertRaises(ValueError):api.aggregate_supplied(inputs,a,p,bad)
        bad=deepcopy(c);bad.rows[0]['mean']['qubits']=5
        with self.assertRaises(ValueError):api.aggregate_supplied(inputs,a,p,bad)

    def test_conditional_price_assumptions_and_target_binding(self):
        inputs,a,p,c = artificial('reference')
        bad=deepcopy(c);bad.rows[0]['expectation_basis']='explicit_conditional_model'
        with self.assertRaises(ValueError):api.aggregate_supplied(inputs,a,p,bad)
        bad.rows[0]['model_assumptions']=['ARTIFICIAL_MODEL']
        self.assertIsNotNone(api.aggregate_supplied(inputs,a,p,bad)['totals'])
        bad.rows[0]['target_id']='OTHER_TARGET'
        with self.assertRaises(ValueError):api.aggregate_supplied(inputs,a,p,bad)

    def test_bad_shots_and_horizon_not_silently_generalized(self):
        inputs,a,p,c = artificial('reference')
        for value in (None,0,True,1.5):
            bad=deepcopy(p);bad[0]['shots']=value
            with self.assertRaises(ValueError):api.aggregate_supplied(inputs,a,bad,c)
        with self.assertRaises(ValueError):api.aggregate_supplied(inputs,a,p[:-1],c)


@contextmanager
def no_numerical_evaluation():
    # If the saved selector accidentally starts any old accounting or shot
    # path, fail before that function can operate on real H4 records.
    with (patch.object(api.accounting,'aggregate_totals',side_effect=AssertionError('H4 reaggregation forbidden')),
          patch.object(api.accounting,'_legacy',side_effect=AssertionError('shot/provider import forbidden')),
          patch.object(api.bc,'aggregate_registered',side_effect=AssertionError('H4 accounting forbidden')),
          patch.object(api.bc,'read_b4',side_effect=AssertionError('use frozen saved metadata'))):
        yield


class SavedReadOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with no_numerical_evaluation():cls.study=api.SavedCDF8Study(ROOT)

    def test_all_methods_scenarios_return_stored_values_only(self):
        with no_numerical_evaluation():
            for scenario in api.SCENARIOS:
                raw=json.loads(subprocess.check_output(['git','show',api.TOTAL+':'+api.DIRECTORY+scenario+'/result.json'],cwd=ROOT))
                for method in api.METHODS:
                    result=self.study.select(self.study.inputs,method,scenario=scenario)
                    column='symmetric' if method=='symmetric_directional' else 'reference'
                    self.assertEqual(result['total_resources'],{k:raw['totals'][k][column] for k in api.bc.METRICS})
                    self.assertEqual(result['uncertainty'],raw['uncertainty'])
                    self.assertEqual(result['total_origin'],'SAVED_RESULT_NOT_REAGGREGATED')
                    self.assertEqual(result['shot_evidence']['stored_total_shots'],14738)
                    self.assertEqual(result['peak_qubits'],17 if method=='reference' else 9)
                    self.assertEqual(len(result['prices'].rows),26)
                    self.assertFalse(result['actual_success_guaranteed'])
                    self.assertIsNone(result['uncertainty']['total_confidence_interval'])

    def test_q1_direct_cohorts_and_long_q_proxy_keep_metadata(self):
        with no_numerical_evaluation():r=self.study.select(self.study.inputs,'reference')
        for row in r['prices'].rows:
            if row['q']==1:self.assertEqual(row['cut'],dict(kind='complete_event_within_step',after=16))
            label='DIRECT_FINITE_SAMPLE_PRICE_INPUT' if row['q']<=32 else 'EXPLORATORY_LONG_Q_PROXY'
            self.assertEqual(row['price_classification'],label)
            self.assertFalse(row['price_uncertainty_qualified'])

    def test_other_PR_target_state_law_or_decoder_not_qualified(self):
        inputs=self.study.inputs
        variants=[]
        for key,value in [('PR','ordered S2/s1/r16/K0'),('frame','original LD3'),
                          ('h',.05),('state','own_DF_ground'),('cdf','OTHER_CDF'),
                          ('law_sha256','OTHER_LAW')]:
            b=deepcopy(inputs.binding);b[key]=value;variants.append(replace(inputs,binding=b))
        t=deepcopy(inputs.target);t['energy_ha']=.000159;variants.append(replace(inputs,target=t))
        variants.extend([replace(inputs,law_model='actual'),replace(inputs,decoder='OTHER'),replace(inputs,rounds=inputs.rounds[:-1])])
        with no_numerical_evaluation():
            for other in variants:
                r=self.study.select(other,'reference')
                self.assertIsNone(r['prices']);self.assertIsNone(r['total_resources'])
                self.assertIsNone(r['shot_evidence']['rows']);self.assertFalse(r['shot_evidence']['common_N'])

    def test_other_control_profile_no_price_or_shot_fallback(self):
        profile=self.study.profile
        compiler=deepcopy(profile.compiler);compiler['optimization_level']=2
        bool_compiler=deepcopy(profile.compiler);bool_compiler['optimization_level']=True
        profiles=[replace(profile,compiler=compiler),replace(profile,compiler=bool_compiler),
                  replace(profile,cut_policy='OTHER_CUT'),replace(profile,head_tail_C1=True),
                  replace(profile,phase_policy='DROP_PHASE'),replace(profile,topology='ROUTED')]
        with no_numerical_evaluation():
            for other in profiles:
                r=self.study.select(self.study.inputs,'reference',profile=other)
                self.assertFalse(r['fallback_used']);self.assertIsNone(r['total_resources'])
                self.assertFalse(r['shot_evidence']['common_N'])

    def test_actual_unknown_method_and_scenario_do_not_gain_qualification(self):
        with no_numerical_evaluation():
            r=self.study.select(self.study.inputs,'reference',lane='actual')
            self.assertIsNone(r['total_resources']);self.assertIsNone(r['shot_evidence']['rows'])
            with self.assertRaises(ValueError):self.study.select(self.study.inputs,'other')
            with self.assertRaises(ValueError):self.study.select(self.study.inputs,'reference',scenario='best')

    def test_returned_objects_cannot_modify_saved_study(self):
        with no_numerical_evaluation():
            r=self.study.select(self.study.inputs,'reference');r['prices'].rows[0]['mean']['cx_count']=-1
            r['shot_evidence']['rows'][0]['N_X']=-1
            again=self.study.select(self.study.inputs,'reference')
            self.assertNotEqual(again['prices'].rows[0]['mean']['cx_count'],-1)
            self.assertNotEqual(again['shot_evidence']['rows'][0]['N_X'],-1)


if __name__=='__main__':unittest.main(verbosity=2)
