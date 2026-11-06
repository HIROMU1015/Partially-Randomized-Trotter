"""Artificial contracts/algebra/stats only; no molecular sampling or compile."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import check_track_d_c5 as old_tests
import run_track_d_c6b as runner
from trottertracks import track_d_c6b_holdout as pilot
import numpy as np
from qiskit.quantum_info import Operator


class C6bTests(unittest.TestCase):
    def request(self, sample=0):
        row=pilot.binding(sample)
        return pilot.HoldoutRequest(sample, tuple(row['seeds']), 'a'*64, 'b'*40)

    def record(self):
        return dict(**pilot.binding(0), cohort=pilot.COHORT, execution_source='b'*40,
            endpoint=pilot.api.c3.paired.ENDPOINT, law_sha256=pilot.api.LAW,
            input_packet_sha256=pilot.api.INPUT_PACKET, ordered_events=[[{}]*32 for _ in range(8)])

    def test_seeds_complete_disjoint_deterministic_and_legacy_guard_preserved(self):
        plan=pilot.seed_plan();rows=plan['generation_order']
        self.assertEqual(plan,pilot.seed_plan())
        self.assertEqual([r['sample_id'] for r in rows],[f'q8_sample{i:02d}' for i in range(32)])
        self.assertEqual(len({s for r in rows for s in r['seeds']}),256)
        self.assertEqual(plan['calibration_seed_overlap'],0)
        self.assertEqual(plan['master_tuple'],[2026110606,pilot.calibration.RESERVED_HOLDOUT_DOMAIN,pilot.calibration.BASE])
        for bad in [-1,32,True]:
            with self.assertRaises(ValueError):pilot.binding(bad)
        with self.assertRaises(ValueError):pilot.calibration.binding(8,0)
        with self.assertRaises(ValueError):pilot.calibration.binding(4,16)

    def test_request_event_binding_cut_source_and_record_shape(self):
        request=self.request()
        self.assertEqual(len(request.ordered_samples),8)
        self.assertNotEqual(request.fingerprint,self.request(1).fingerprint)
        for changes in [dict(q=4),dict(cut_after_complete_step=3),dict(seeds=(1,)*8),
                        dict(endpoint='0'*64),dict(execution_source='bad')]:
            with self.assertRaises(ValueError):replace(request,**changes)
        record=self.record();self.assertEqual(pilot.request_from_record(record).sample,0)
        for changes in [dict(q=4),dict(seed_domain='calibration'),dict(ordered_events=[[{}]*32]*7),
                        dict(ordered_events=[[{}]*31]*8),dict(law_sha256='0'*64)]:
            with self.assertRaises(ValueError):pilot.request_from_record({**record,**changes})

    def test_canonical_sampler_eight_calls_no_method_axis_resampling(self):
        event=S(to_dict=lambda:dict(synthetic='event'),taylor_order=0,product_component_ids=(),phase=1)
        context=S(arm=S(),config=S(rte_steps=32),tail=S())
        b=pilot.api.c3.base.original
        with patch.object(b,'configure',return_value=(context.tail,context.config,'distribution')) as conf, \
             patch.object(b.connection,'sample_occurrence',return_value=(event,)*32) as sampler:
            record=pilot.sample_record(context,31,'b'*40)
        self.assertEqual(conf.call_count,1);self.assertEqual(sampler.call_count,8)
        self.assertEqual([call.args[-1] for call in sampler.call_args_list],pilot.binding(31)['seeds'])
        self.assertEqual(pilot.request_from_record(record).sample,31)

    def test_q8_cut4_shared_assembly_algebra_with_signed_synthetic_steps(self):
        with old_tests.SyntheticFixture() as f:
            saved=pilot.api.c3.prepare_ordered(f.context,f.root,tuple(range(8)),4,fixture_selection='toy')
            steps=[]
            for i in range(8):
                with np.load(f.root/f'references/sample{i:02d}.npz') as z:target=z['system_reference']
                steps.append(dict(sample=i,events=f.steps[i],reference=target,
                    raw=pilot.api.c3.base.load_qpy(f.root/f'circuits/sample{i:02d}_raw.qpy')))
            fresh=pilot.api.c3.prepare_steps(f.context,steps,4,ordered_samples=tuple(range(8)),
                fixture_selection='toy',cohort=pilot.api.c3.paired.COHORT)
            for method in pilot.api.METHODS:
                for axis in pilot.api.AXES:
                    np.testing.assert_allclose(Operator(saved.wrappers[method,axis].remove_final_measurements(inplace=False)).data,
                        Operator(fresh.wrappers[method,axis].remove_final_measurements(inplace=False)).data,atol=1e-13,rtol=0)
            a,b,adj=[Operator(c).data for c in [fresh.later,fresh.earlier,fresh.later_adjoint]]
            np.testing.assert_allclose(a@b,fresh.reference,atol=1e-13,rtol=0)
            np.testing.assert_allclose(adj,a.conj().T,atol=1e-13,rtol=0)
            self.assertEqual(fresh.identity['strong_symmetric']['step_boundaries_fused'],7)
            self.assertEqual(fresh.identity['reference_optimization']['internal_step_boundaries_fused'],6)

    def records(self):
        return [dict(q=8,sample=i,method=m,axis=a,
            metrics={g:1000+i*(2 if m=='reference' else 1) for g in pilot.api.METRICS},
            peak_qubits=17 if m=='reference' else 9)
            for i in range(32) for m in pilot.api.METHODS for a in pilot.api.AXES]

    def test_paired_units_covariance_missing_duplicate_rejected(self):
        records=self.records();summary=pilot.paired_statistics(records);row=summary['rows'][0]
        self.assertEqual(summary['n_whole_paths'],32)
        self.assertEqual(row['paired_deltas_by_sample'],list(range(32)))
        self.assertEqual(row['paired_delta']['mean'],15.5)
        self.assertAlmostEqual(row['paired_delta']['sample_SD'],np.std(np.arange(32),ddof=1))
        self.assertAlmostEqual(row['method_sample_covariance'],2*np.var(np.arange(32),ddof=1))
        self.assertEqual(np.shape(summary['sample_covariance']),(28,28))
        self.assertEqual(np.shape(summary['paired_sample_covariance']),(12,12))
        self.assertTrue(summary['XY_are_not_independent_samples'])
        for bad in [records[:-1],records+[records[0]], [{**records[0],'q':4}]+records[1:]]:
            with self.assertRaises(ValueError):pilot.paired_statistics(bad)

    def test_diagnostic_conditioning_secondary_covariance_and_width_are_separate(self):
        summary=pilot.paired_statistics(self.records())
        model=dict(primary=[dict(axis=a,metric=g,predicted_paired_delta=15.5,tau=20.,
            calibration_variance_conditional=10000.) for a in pilot.api.AXES for g in pilot.api.METRICS],
            nominal_two_sided_normal_z=3.)
        result=pilot.diagnose(summary,model)
        self.assertEqual(result['status'],'EXPLORATORY_WITHIN_WIDTH')
        self.assertTrue(all(r['population_line_diagnostic']=='EXPLORATORY_OUTSIDE_WIDTH' for r in result['rows']))
        self.assertEqual(result['statistical_qualification'],'STATISTICALLY_UNQUALIFIED')
        self.assertFalse(result['confirmatory_PASS']);self.assertFalse(result['qualified_95_percent_coverage'])
        other=copy.deepcopy(model)
        for row in other['primary']:row['predicted_paired_delta']=1000.
        self.assertEqual(pilot.diagnose(summary,other)['status'],'EXPLORATORY_OUTSIDE_WIDTH')

    def test_frozen_model_reads_only_old_coefficients_family_12_scale_and_covariance(self):
        model=pilot.frozen_model(ROOT)
        saved=json.loads((ROOT/pilot.CALDIR/'exploratory_affine.json').read_text())
        self.assertEqual(model['saved_models'],saved['models'])
        self.assertEqual(len(model['primary']),12)
        for row in model['primary']:
            self.assertEqual(row['tau'],1+.05*row['symmetric_pre_holdout_scale'])
            self.assertEqual(row['predicted_paired_delta'],row['method_predictions']['reference']-row['method_predictions']['symmetric_directional'])
        self.assertFalse(model['model_refit']);self.assertFalse(model['confirmatory_PASS'])

    def test_scheduler_distributes_remainder_memory_and_path_compile_priority(self):
        tasks=[dict(kind='raw',method=None,memory_bytes=10,sample_id=str(i)) for i in range(3)]
        allocations=runner.task_allocation(8,30,tasks,8,0)
        self.assertEqual([t for _,t in allocations],[3,3,2])
        self.assertEqual(len(runner.task_allocation(8,20,tasks,8,0)),2)
        self.assertEqual(runner.task_allocation(8,9,tasks,8,0),[])
        direct=dict(kind='compiled',method='symmetric_directional',memory_bytes=1,sample_id='d')
        self.assertEqual(runner.task_allocation(8,11,[direct,tasks[0]],8,0)[1][1],7)

    def test_price_reuses_full_wrapper_P0_and_preserves_old_fixed_entry(self):
        request=self.request();api=pilot.api
        compiler=dict(qiskit='1.3.0',optimization_level=1,seed_transpiler=17,basis_gates=['rz','sx','x','cx'],
                      approximation_degree=1.,qubits_initially_zero=False)
        post=dict(status='PASS',arm='reference',axis='X',global_phase_fit=False,
                  metrics={**{g:10 for g in api.METRICS},'qubit_count':17})
        kwargs=dict(method='reference',axis='X',postcheck=post,raw_status='PASS',paired_status='PASS',
                    execution_source='b'*40,compiler=compiler)
        row=pilot.price_record(request,**kwargs)
        self.assertEqual(row['q'],8);self.assertFalse(row['qualification']['all_law'])
        for changes in [dict(raw_status='FAIL'),dict(compiler={}),dict(execution_source='c'*40),dict(cost_scope='kernel')]:
            with self.assertRaises(ValueError):pilot.price_record(request,**{**kwargs,**changes})
        with self.assertRaises(ValueError):api.price_record(api.OrderedRequest(8,tuple(range(8)),4),**kwargs)


if __name__=='__main__':unittest.main()
