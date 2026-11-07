"""q64 request, synthetic builder, statistics and first-path gate tests."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_track_d_h4_reference as boot
boot.bootstrap(os.environ['TRACK_D_ORIGIN'])
import check_track_d_c5 as legacy
from trottertracks import track_d_c8_1_q64 as pilot
from trottertracks import track_d_c7_3_q32 as old
from qiskit.quantum_info import Operator
import numpy as np


class C8Tests(unittest.TestCase):
    def request(self):return pilot.Q64Request(0,tuple(pilot.binding(0)['seeds']),'a'*64,'b'*40)

    def test_domain_disjoint_and_old_request_guards_unchanged(self):
        plan=pilot.seed_plan();self.assertEqual(len(plan['generation_order']),8)
        self.assertEqual(len({s for r in plan['generation_order'] for s in r['seeds']}),512)
        self.assertEqual(plan['previous_seed_overlap'],0)
        self.assertEqual((old.Q,old.CUT),(32,16));self.assertEqual(self.request().q,64)
        self.assertTrue(all(x.startswith('q64_sample00_step') for x in self.request().ordered_samples))
        for change in [dict(q=32),dict(cut_after_complete_step=31),dict(seeds=(0,)*64),dict(cohort=old.COHORT)]:
            with self.assertRaises(ValueError):replace(self.request(),**change)
        with self.assertRaises(ValueError):replace(old.Q32Request(0,tuple(old.binding(0)['seeds']),'a'*64,'b'*40),q=64)

    def test_sampler_is_exactly_64_canonical_calls_no_new_sampler(self):
        event=S(to_dict=lambda:dict(synthetic='event'),taylor_order=0,product_component_ids=(),phase=1)
        context=S(arm=S(),config=S(rte_steps=32),tail=S());base=pilot.api.c3.base.original
        with patch.object(base,'configure',return_value=(context.tail,context.config,'distribution')),patch.object(base.connection,'sample_occurrence',return_value=(event,)*32) as sample:
            record=pilot.sample_record(context,0,'b'*40)
        self.assertEqual(sample.call_count,64);self.assertEqual(pilot.request_from_record(record).q,64)
        self.assertEqual([c.args[-1] for c in sample.call_args_list],pilot.binding(0)['seeds'])
        with self.assertRaises(ValueError):pilot.request_from_record({**record,'ordered_events':record['ordered_events'][:-1]})

    def test_64_toy_steps_strong63_reference62_and_actual_adjoint(self):
        c3=pilot.api.c3
        with legacy.SyntheticFixture() as f:
            steps=[]
            for i in range(64):
                j=i%8
                with np.load(f.root/f'references/sample{j:02d}.npz') as z:target=z['system_reference']
                steps.append(dict(sample=i,events=f.steps[j],reference=target,raw=c3.base.load_qpy(f.root/f'circuits/sample{j:02d}_raw.qpy')))
            bundle=c3.prepare_steps(f.context,steps,32,ordered_samples=tuple(range(64)),fixture_selection='2mode ARTIFICIAL only',cohort='synthetic')
            a,b,adj=[Operator(c).data for c in (bundle.later,bundle.earlier,bundle.later_adjoint)]
            np.testing.assert_allclose(a@b,bundle.reference,atol=1e-12,rtol=0)
            np.testing.assert_allclose(adj,a.conj().T,atol=1e-12,rtol=0)
            self.assertEqual(bundle.identity['strong_symmetric']['step_boundaries_fused'],63)
            self.assertEqual(bundle.identity['reference_optimization']['internal_step_boundaries_fused'],62)
            self.assertEqual(bundle.identity['scalar_identity_occurrences'],64)
            for method in pilot.api.METHODS:
                for axis in pilot.api.AXES:
                    np.testing.assert_allclose(c3.base.artificial.induced_observable(bundle.wrappers[method,axis]),
                        c3.base.artificial.expected_observable(bundle.reference,axis),atol=1e-12,rtol=0)

    def records(self):
        return [dict(q=64,sample=i,method=m,axis=a,metrics={g:100000+i+(pilot.CX_CANDIDATE if g=='cx_count' else i) if m=='reference' else 100000+i for g in pilot.api.METRICS},peak_qubits=17 if m=='reference' else 9) for i in range(8) for m in pilot.api.METHODS for a in pilot.api.AXES]

    def test_n8_complete_cohort_and_integer_CX_counterexample(self):
        records=self.records();s=pilot.paired_statistics(records);self.assertEqual(s['n_whole_paths'],8)
        result=pilot.cx_candidate_residuals(s);self.assertTrue(all(r['all_exact_integer_zero'] for r in result['rows']))
        records[0]['metrics']['cx_count']+=1
        self.assertEqual(pilot.cx_candidate_residuals(pilot.paired_statistics(records))['status'],'CX_CANDIDATE_VIOLATED')
        with self.assertRaises(ValueError):pilot.paired_statistics(records[:-1])

    def synthetic_pipeline(self,fail=False):
        from concurrent.futures import Future
        from threading import Event
        import run_track_d_c8_1_q64 as runner
        starts=[];stop=Event()
        class Pool:
            def __init__(self,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def submit(self,func,*args):
                f=Future()
                if func==runner.raw_worker:
                    sid=args[0]['sample_id'];starts.append((sid,'raw'))
                    f.set_result((sid,dict(status='PASS',peak_RSS_KiB=1)))
                else:
                    sid,task,stage,_=args;name=task[2]+'_'+task[3];starts.append((sid,stage))
                    f.set_result((sid,name,stage,dict(status='FAIL' if fail and sid=='q64_sample00' and stage=='post' else 'PASS')))
                return f
        events=[dict(sample_id=f'q64_sample{i:02d}') for i in range(8)]
        resources=dict(numerical_threads=8,memory=dict(MemAvailable=10**12),raw_memory_estimate=10000,post_memory_estimate=10000)
        with patch.object(runner,'ProcessPoolExecutor',Pool),patch.object(runner,'write'),patch.object(runner.old.frozen,'memory_available',return_value=10**12),patch.object(runner.previous,'paired_postcheck'):
            if fail:
                with self.assertRaises(ValueError):runner.pipeline(events,Path('.'),Path('.'),resources,stop,Path('.'),{}, {})
            else:runner.pipeline(events,Path('.'),Path('.'),resources,stop,Path('.'),{}, {})
        return starts

    def test_sample00_all4_postchecks_precede_any_remaining_raw(self):
        starts=self.synthetic_pipeline();first_other=next(i for i,v in enumerate(starts) if v[0]!='q64_sample00')
        self.assertEqual(starts[:first_other].count(('q64_sample00','post')),4)
        self.assertEqual(starts[:first_other].count(('q64_sample00','compile')),4)

    def test_sample00_failure_never_starts_remaining_paths(self):
        starts=self.synthetic_pipeline(fail=True)
        self.assertTrue(all(sid=='q64_sample00' for sid,stage in starts))


if __name__=='__main__':unittest.main()
