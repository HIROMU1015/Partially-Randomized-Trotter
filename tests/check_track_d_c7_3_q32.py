"""Synthetic q32/domain/statistics/pipeline checks; no molecular scientific calls."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import check_track_d_c5 as legacy
import run_track_d_c7_3_q32 as runner
from trottertracks import track_d_c7_3_q32 as pilot
from trottertracks import track_d_c6b_holdout as old_q8
import numpy as np
from qiskit.quantum_info import Operator


class C73Tests(unittest.TestCase):
    def request(self,i=0):
        return pilot.Q32Request(i,tuple(pilot.binding(i)['seeds']),'a'*64,'b'*40)

    def records(self):
        return [dict(q=32,sample=i,method=m,axis=a,
            metrics={g:(10000+i+(pilot.CX_CANDIDATE if g=='cx_count' else i) if m=='reference' else 10000+i)
                     for g in pilot.api.METRICS},peak_qubits=17 if m=='reference' else 9)
            for i in range(8) for m in pilot.api.METHODS for a in pilot.api.AXES]

    def test_new_seeds_fixed_domain_disjoint_and_old_guards_preserved(self):
        p=pilot.seed_plan()
        self.assertEqual(len(p['generation_order']),8)
        self.assertEqual(len({v for r in p['generation_order'] for v in r['seeds']}),256)
        self.assertEqual(p['C6a_C6b_C7_1_seed_overlap'],0)
        self.assertNotEqual(pilot.DOMAIN,old_q8.DOMAIN)
        from trottertracks import track_d_c7_1_q16 as old_q16
        self.assertEqual((old_q16.Q,old_q16.SAMPLES,old_q16.CUT),(16,16,8))
        self.assertNotEqual(pilot.DOMAIN,old_q16.DOMAIN)
        self.assertEqual(p,pilot.seed_plan())
        with self.assertRaises(ValueError):pilot.calibration.binding(16,0)
        with self.assertRaises(ValueError):replace(old_q8.HoldoutRequest(0,tuple(old_q8.binding(0)['seeds']),'a'*64,'b'*40),q=16)
        for bad in [-1,8,True]:
            with self.assertRaises(ValueError):pilot.binding(bad)
        for change in [dict(q=8),dict(cut_after_complete_step=15),dict(cohort=old_q8.COHORT),dict(seeds=(0,)*32)]:
            with self.assertRaises(ValueError):replace(self.request(),**change)

    def test_sampler_reuses_canonical_calls_and_request_rejects_changed_binding(self):
        event=S(to_dict=lambda:dict(synthetic='event'),taylor_order=0,product_component_ids=(),phase=1)
        context=S(arm=S(),config=S(rte_steps=32),tail=S())
        b=pilot.api.c3.base.original
        with patch.object(b,'configure',return_value=(context.tail,context.config,'distribution')), \
             patch.object(b.connection,'sample_occurrence',return_value=(event,)*32) as sample:
            record=pilot.sample_record(context,0,'b'*40)
        self.assertEqual(sample.call_count,32)
        self.assertEqual(pilot.request_from_record(record).q,32)
        self.assertEqual([c.args[-1] for c in sample.call_args_list],pilot.binding(0)['seeds'])
        for change in [dict(q=8),dict(seed_domain=old_q8.DOMAIN),dict(ordered_events=record['ordered_events'][:-1])]:
            with self.assertRaises(ValueError):pilot.request_from_record({**record,**change})

    def test_q32_cut16_shared_builder_actual_inverse_phase_and_XY(self):
        # Repeated two-mode TOY inputs test control flow, not a molecular cohort.
        c3=pilot.api.c3
        with legacy.SyntheticFixture() as f:
            steps=[]
            for i in range(32):
                j=i%8
                with np.load(f.root/f'references/sample{j:02d}.npz') as z:target=z['system_reference']
                steps.append(dict(sample=i,events=f.steps[j],reference=target,
                    raw=c3.base.load_qpy(f.root/f'circuits/sample{j:02d}_raw.qpy')))
            bundle=c3.prepare_steps(f.context,steps,16,ordered_samples=tuple(range(32)),
                fixture_selection='two-mode synthetic only',cohort='synthetic')
            a,b,adj=[Operator(c).data for c in [bundle.later,bundle.earlier,bundle.later_adjoint]]
            np.testing.assert_allclose(a@b,bundle.reference,atol=1e-12,rtol=0)
            np.testing.assert_allclose(adj,a.conj().T,atol=1e-12,rtol=0)
            vacuum=np.zeros(4,complex);vacuum[0]=np.exp(1j*bundle.phase)
            np.testing.assert_allclose(bundle.reference[:,0],vacuum,atol=1e-12,rtol=0)
            self.assertEqual(bundle.identity['strong_symmetric']['step_boundaries_fused'],31)
            self.assertEqual(bundle.identity['reference_optimization']['internal_step_boundaries_fused'],30)
            self.assertEqual(bundle.identity['scalar_identity_occurrences'],32)
            record=dict(ordered_events=[[dict(rotation_angle=e.rotation_angle,application_sequence=[dict(
                basis_id=e.application_sequence[0].basis_id,
                diagonal_pauli_support=e.application_sequence[0].diagonal_pauli_support,
                coefficient_sign=e.application_sequence[0].coefficient_sign)]) for e in f.steps[i%8]] for i in range(32)])
            features=pilot.raw_features(bundle,record)
            self.assertEqual(features['events'],64)
            self.assertEqual(features['logical_CSWAP'],4)
            self.assertTrue(features['aggregate_only_no_ordered_events_published'])
            observations={}
            for m in pilot.api.METHODS:
                for axis in pilot.api.AXES:
                    obs=c3.base.artificial.induced_observable(bundle.wrappers[m,axis])
                    expected=c3.base.artificial.expected_observable(bundle.reference,axis)
                    np.testing.assert_allclose(obs,expected,atol=1e-12,rtol=0)
                    observations[m,axis]=obs
            np.testing.assert_allclose(observations['reference','X']+1j*observations['reference','Y'],
                bundle.reference,atol=1e-12,rtol=0)

    def test_complete_statistics_covariance_and_integer_CX_residual_separate_from_width(self):
        records=self.records();summary=pilot.paired_statistics(records)
        self.assertEqual(summary['n_whole_paths'],8)
        self.assertEqual(np.shape(summary['sample_covariance']),(28,28))
        self.assertEqual(np.shape(summary['paired_sample_covariance']),(12,12))
        self.assertEqual(summary['rows'][0]['paired_delta']['mean'],3.5)
        self.assertTrue(pilot.cx_candidate_residuals(summary)['rows'][0]['all_exact_integer_zero'])
        records[0]['metrics']['cx_count']+=1
        residual=pilot.cx_candidate_residuals(pilot.paired_statistics(records))
        self.assertEqual(residual['status'],'CX_CANDIDATE_VIOLATED')
        self.assertFalse(residual['all_path_theorem'])
        for bad in [records[:-1],records+[records[0]],[{**r,'q':8} for r in records]]:
            with self.assertRaises(ValueError):pilot.paired_statistics(bad)

    def test_fixed_prediction_vs_population_line_diagnosis_different_estimands(self):
        summary=pilot.paired_statistics(self.records())
        model=dict(primary=[dict(axis='X',metric='rz_count',predicted_paired_delta=3.5,
            calibration_variance_conditional=100,tau=5)],nominal_two_sided_normal_z=2)
        result=pilot.diagnose(summary,model);row=result['rows'][0]
        self.assertEqual(row['primary_status'],'EXPLORATORY_WITHIN_WIDTH')
        self.assertEqual(row['population_line_diagnostic'],'EXPLORATORY_OUTSIDE_WIDTH')
        self.assertEqual(row['prediction_minus_observed'],-row['prediction_error'])
        self.assertFalse(result['confirmatory_PASS'])
        self.assertEqual(row['primary_margin'],2*np.std(range(8),ddof=1)/np.sqrt(8))

    def test_frozen_model_uses_saved_covariance_q32_weights_without_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/pilot.CALDIR;folder.mkdir(parents=True)
            labels=[(m,a,g) for m in pilot.api.METHODS for a in pilot.api.AXES for g in pilot.api.METRICS]
            cov=[[9 if x==y and x[0]=='symmetric_directional' else
                  16 if x==y else 10 if x[1:]==y[1:] else 0 for y in labels] for x in labels]
            stats={'by_q':{str(q):dict(sample_covariance_labels=labels,sample_covariance=cov,n_independent_path_units=16) for q in [2,4]}}
            fit=dict(models={m:[dict(axis={'X':'cosine','Y':'sine'}[a],metric=g,slope=3 if m=='reference' else 4,intercept=2)
                               for a in pilot.api.AXES for g in pilot.api.METRICS] for m in pilot.api.METHODS})
            for leaf,value in [('paired_statistics.json',stats),('exploratory_affine.json',fit),
                ('C6b_planning_proposal.json',dict(conditional_parameter_covariance=dict(synthetic=True)))]:
                (folder/leaf).write_text(json.dumps(value))
            model=pilot.frozen_model(Path(temporary));row=model['primary'][0]
            self.assertEqual(row['calibration_variance_conditional'],(196+225)*(16+9-20)/16)
            self.assertEqual(row['predicted_paired_delta'],-32)
            self.assertEqual(row['tau'],1+.05*130)
            self.assertFalse(model['model_refit']);self.assertFalse(model['C6b_q8_used_for_fit'])
            self.assertEqual(len(model['primary']),12)

    def test_pipeline_only_queues_four_after_raw_pass_and_caps_complete_run_at32(self):
        from concurrent.futures import Future
        class Pool:
            calls=[]
            def __init__(self,**kwargs):self.kwargs=kwargs
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def submit(self,fn,*args):
                self.calls.append((fn,args));future=Future()
                if fn==runner.raw_worker:future.set_result((args[0]['sample_id'],dict(status='PASS')))
                else:
                    sid,task,_=args;future.set_result((sid,task[2]+'_'+task[3],dict(status='PASS')))
                return future
        stop=S(is_set=lambda:False,set=lambda:None)
        resources=dict(numerical_threads=7,memory={'MemAvailable':1000},
            raw_task_memory_estimate_bytes=1,reference_task_memory_estimate_bytes=1,direct_task_memory_estimate_bytes=1)
        events=pilot.seed_plan()['generation_order'];raw={};compiled={}
        with tempfile.TemporaryDirectory() as temp,patch.object(runner.historical_pipeline,'ProcessPoolExecutor',Pool), \
             patch.object(runner.historical_pipeline,'memory_available',return_value=1000):
            runner.pipeline(events,Path(temp),Path(temp),resources,stop,Path(temp),raw,compiled)
        self.assertEqual(len(raw),8);self.assertEqual(sum(map(len,compiled.values())),32)
        started=set()
        for fn,args in Pool.calls:
            if fn==runner.raw_worker:started.add(args[0]['sample_id'])
            else:self.assertIn(args[0],started)

    def test_reused_feature_recurrence_and_wire_roles_without_compile(self):
        from qiskit import QuantumCircuit
        q=QuantumCircuit(5,1);q.h(4);q.cx(4,0);q.rz(.23,2);q.cx(0,2);q.h(4);q.measure(4,0)
        result=pilot.wrapper_features(q,'reference')
        self.assertEqual(result['wire_roles'],['system','system','reference','reference','ancilla'])
        for key,gate in [('rz_depth','rz'),('cx_depth','cx'),('total_depth',None)]:
            expected=q.depth() if gate is None else q.depth(lambda x:x.operation.name==gate)
            self.assertEqual(result['weighted_wire_finish'][key]['depth'],expected)
        self.assertFalse(result['prediction_without_compile'])

    def test_sampling_context_reuses_law_without_any_head_circuit(self):
        base=pilot.api.c3.base
        arm=S(model=S(modes=8),plan=S(atoms=(),extracted_identity=.04))
        tail=S(lambda_r=2.);config=S()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            np.savez(root/'input.npz',h=np.zeros((1,1)),g=np.zeros((1,1,1,1)),constant=.17)
            np.savez(root/'candidate.npz',J=np.zeros((8,4,4)),U=np.zeros((8,4,4)))
            np.savez(root/'sector.npz',J=np.zeros((8,4,4)),U=np.zeros((8,4,4)))
            (root/'atoms.json').write_text(json.dumps(dict(atoms=[])))
            law=root/'artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json'
            law.parent.mkdir(parents=True);law.write_text(json.dumps(dict(identity=.04,lambda_R=2.)))
            contract=dict(input='input.npz',candidate='candidate.npz',sector='sector.npz',atoms='atoms.json',
                          saved_inputs={'input.npz':pilot.api.INPUT_PACKET,'candidate.npz':pilot.api.c3.paired.ENDPOINT})
            with patch.object(base.original,'prepare_arm',return_value=arm) as prepare, \
                 patch.object(base.original,'configure',return_value=(tail,config,None)), \
                 patch.object(base,'StrongBuilder',side_effect=AssertionError('NO_HEAD_BEFORE_EVENTS')):
                context=pilot.sampling_context(root,contract)
            self.assertIs(context.arm,arm);self.assertIs(context.tail,tail)
            self.assertFalse(hasattr(context,'forward'));self.assertEqual(prepare.call_count,1)


if __name__=='__main__':unittest.main()
