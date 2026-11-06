"""Synthetic/data-contract tests only; no molecular sampling/compile/Aer."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import check_track_d_c5 as old_tests
from trottertracks import track_d_c6a_pilot as pilot
from trottertracks import track_d_whole_path_price as api
import numpy as np
from qiskit.quantum_info import Operator


class C6aTests(unittest.TestCase):
    def request(self,q=2,sample=0):
        b=pilot.binding(q,sample)
        return pilot.FreshRequest(q,sample,tuple(b['seeds']),'a'*64,'b'*40,b['cut_after_complete_step'])

    def test_seed_domains_order_unique_holdout_unused(self):
        plan=pilot.seed_plan();rows=plan['generation_order']
        self.assertEqual([(r['q'],r['sample']) for r in rows],[(q,i) for q in [2,4] for i in range(16)])
        self.assertEqual(len(set(s for r in rows for s in r['seeds'])),96)
        self.assertNotIn(plan['holdout_domain_reserved'],[r['seed_domain'] for r in rows])
        self.assertFalse(plan['holdout_generated']);self.assertEqual(plan,pilot.seed_plan())

    def test_request_rejects_wrong_seed_cut_domain_and_input(self):
        r=self.request()
        for changes in [dict(q=8),dict(sample=16),dict(seeds=(1,2)),dict(cut_after_complete_step=0),
                        dict(endpoint='0'*64),dict(law_sha256='0'*64),dict(execution_source='invalid')]:
            with self.assertRaises(ValueError):replace(r,**changes)
        self.assertNotEqual(r.fingerprint,self.request(sample=1).fingerprint)

    def test_canonical_connection_calls_once_per_occurrence_shared_methods_axes(self):
        event=S(to_dict=lambda:dict(synthetic='event'),taylor_order=0,product_component_ids=(),phase=1)
        context=S(arm=S(),config=S(rte_steps=32),tail=S())
        b=api.c3.base.original
        with patch.object(b,'configure',return_value=(context.tail,context.config,'distribution')) as configure, \
             patch.object(b.connection,'sample_occurrence',return_value=(event,)*32) as sample:
            record=pilot.sample_record(context,4,3,'b'*40)
        self.assertEqual(configure.call_count,1);self.assertEqual(sample.call_count,4)
        self.assertEqual([call.args[-1] for call in sample.call_args_list],pilot.binding(4,3)['seeds'])
        self.assertEqual(len(record['ordered_events']),4)
        request=pilot.request_from_record(record)
        self.assertEqual(request.seeds,tuple(record['seeds']))
        bad=dict(record,seed_domain='holdout')
        with self.assertRaises(ValueError):pilot.request_from_record(bad)

    def test_common_bound_steps_preserve_c5_entry_operators_order_phase_cut(self):
        # Two modes, signed synthetic tails, q2/q4 only; independent old adapter
        # and explicit-step ingestion must yield identical assembled circuits.
        with old_tests.SyntheticFixture() as f:
            for q in [2,4]:
                saved=api.c3.prepare_ordered(f.context,f.root,tuple(range(q)),q//2,fixture_selection='toy')
                steps=[]
                for i in range(q):
                    with np.load(f.root/f'references/sample{i:02d}.npz') as z:target=z['system_reference']
                    steps.append(dict(sample=i,events=f.steps[i],reference=target,
                        raw=api.c3.base.load_qpy(f.root/f'circuits/sample{i:02d}_raw.qpy')))
                fresh=api.c3.prepare_steps(f.context,steps,q//2,ordered_samples=tuple(range(q)),
                    fixture_selection='toy',cohort=api.c3.paired.COHORT)
                self.assertEqual(saved.identity,fresh.identity)
                self.assertEqual(saved.phase,fresh.phase)
                self.assertTrue(np.array_equal(saved.reference,fresh.reference))
                for method in api.METHODS:
                    for axis in api.AXES:
                        s=saved.wrappers[method,axis].remove_final_measurements(inplace=False)
                        r=fresh.wrappers[method,axis].remove_final_measurements(inplace=False)
                        np.testing.assert_allclose(Operator(s).data,Operator(r).data,atol=1e-13,rtol=0)
                np.testing.assert_allclose(Operator(fresh.later).data@Operator(fresh.earlier).data,
                                           fresh.reference,atol=1e-13,rtol=0)

    def records(self):
        rows=[]
        for q in [2,4]:
            for i in range(16):
                for method in api.METHODS:
                    for axis in api.AXES:
                        value=q*100+i*(2 if method=='reference' else 1)
                        rows.append(dict(q=q,sample=i,method=method,axis=axis,
                            metrics={g:value for g in api.METRICS},peak_qubits=17 if method=='reference' else 9))
        return rows

    def test_paired_covariance_ddof1_sample_unit_and_missing_rejected(self):
        records=self.records();s=pilot.paired_statistics(records);q2=s['by_q']['2']
        self.assertEqual(q2['n_independent_path_units'],16)
        r=q2['rows'][0]
        self.assertEqual(r['paired_delta']['mean'],7.5)
        expected=np.var(np.arange(16),ddof=1)
        self.assertAlmostEqual(r['method_sample_covariance'],2*expected)
        self.assertAlmostEqual(r['paired_delta']['sample_SD'],np.sqrt(expected))
        self.assertEqual(s['confidence_interval'],None)
        self.assertEqual(r['paired_deltas_by_sample'],list(range(16)))
        self.assertEqual(q2['covariance_rank_limit'],15)
        for bad in [records[:-1],records+[records[0]]]:
            with self.assertRaises(ValueError):pilot.paired_statistics(bad)

    def test_affine_reuses_generic_fit_and_has_no_holdout_qualification(self):
        fit=pilot.exploratory_affine(pilot.paired_statistics(self.records()))
        self.assertFalse(fit['long_q_qualified']);self.assertFalse(fit['holdout_generated'])
        self.assertEqual(fit['residual_degrees_of_freedom'],0)
        self.assertEqual(fit['models']['reference'][0]['slope'],100.)

    def test_fresh_prices_keep_full_scope_p0_and_old_fixed_guard(self):
        request=self.request()
        compiler=dict(qiskit='1.3.0',optimization_level=1,seed_transpiler=17,basis_gates=['rz','sx','x','cx'],
                      approximation_degree=1.,qubits_initially_zero=False)
        post=dict(status='PASS',arm='reference',axis='X',global_phase_fit=False,
                  metrics={**{g:10 for g in api.METRICS},'qubit_count':17})
        kwargs=dict(method='reference',axis='X',postcheck=post,raw_status='PASS',paired_status='PASS',
                    execution_source='b'*40,compiler=compiler)
        row=pilot.price_record(request,**kwargs)
        self.assertEqual(row['estimate_kind'],'paired_calibration_pilot_compiled')
        self.assertIsNone(row['population_expectation']);self.assertFalse(row['qualification']['long_q'])
        for changes in [dict(cost_scope='serial_kernel'),dict(raw_status='FAIL'),dict(compiler={}),
                        dict(execution_source='c'*40)]:
            with self.assertRaises(ValueError):pilot.price_record(request,**{**kwargs,**changes})
        with self.assertRaises(ValueError):api.price_record(api.OrderedRequest(4,(0,1,2,3),2),**kwargs)


if __name__=='__main__':unittest.main()
