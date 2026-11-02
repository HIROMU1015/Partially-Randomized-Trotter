"""Scheduling/cache correctness on artificial objects; scientific calls are 0."""
from concurrent.futures import Future
import copy
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import execute as x
import performance as p


class Event:
    def __init__(self, angle=0.1):
        self.angle=angle

    def to_dict(self):
        return dict(rotation_angle=self.angle, phase=dict(real=1.,imag=0.))


def inputs():
    prep=NS(preparation_hash='ARTIFICIAL_PREPARATION')
    occurrence=NS(component_specs=('ARTIFICIAL_SPEC',),events=(Event(),),controlled=True,
        ancilla_qubit=2,preserve_event_order=True,cancel_adjacent_equal_bases=True,
        control_diagonal_only=True,identity_as_relative_ancilla_phase=True,
        tail_id='ARTIFICIAL_TAIL',tail_hash='ARTIFICIAL_HASH',occurrence_rte_steps=1)
    request=NS(preparation=prep,step_time=.8,rte_occurrence=occurrence,rte_basis_plan=None,seed=1)
    return dict(id='ARTIFICIAL',T=.8),prep,[request],.2,lambda _: 'ARTIFICIAL_FP'


class Performance(unittest.TestCase):
    def test_heavy_first_is_only_a_permutation_of_fixed_draws(self):
        plan=x.read_plan()
        jobs=[(v['candidate'],cohort,i) for cohort in plan['cohorts'] for v in plan['candidates']
              for i in range(v['n_exploration'] if cohort==plan['cohorts'][0] else v['n_confirmation'])]
        ordered=p.order_jobs(jobs,x.native.cost_proxy)
        identity=lambda j:(j[0]['id'],j[1],j[2])
        self.assertEqual(len(ordered),56689)
        self.assertEqual({identity(j) for j in ordered},{identity(j) for j in jobs})
        work=lambda j:x.native.cost_proxy(j[0])*(8 if j[0]['system']=='H4' else 12)**2
        self.assertEqual([work(j) for j in ordered],sorted((work(j) for j in jobs),reverse=True))

    def test_admission_distinguishes_own_work_from_other_load(self):
        observation=dict(idle_cpu=0.,available_ram=100000)
        self.assertEqual(p.admission(observation,8,16,own_cpu=8),8)
        self.assertEqual(p.admission(observation,8,16,own_cpu=4),4)
        self.assertEqual(p.admission(dict(observation,idle_cpu=8),8,16,own_cpu=8),16)
        self.assertEqual(p.admission(observation,0,16),0)

    def test_admission_respects_memory_and_fractional_quota(self):
        self.assertEqual(p.admission(dict(idle_cpu=16,available_ram=400),8,16,footprint=100),10)
        self.assertEqual(p.admission(dict(idle_cpu=.5,available_ram=10000),0,1),1)
        with self.assertRaises(RuntimeError):
            p.admission(dict(idle_cpu=16,available_ram=99),8,16,footprint=100)
        monitor=p.AdaptiveAdmission(None,'ARTIFICIAL_UNUSED_PATH',100)
        monitor.note_result(dict(peak_RSS_bytes=400));monitor.note_result(dict(peak_RSS_bytes=200))
        self.assertEqual(monitor.footprint,400)

    def test_scheduler_grows_and_drains_without_reordering_draw_identity(self):
        row=x.read_plan()['candidates'][0]['candidate']
        jobs=[(row,'ARTIFICIAL',i) for i in range(7)]
        submitted=[];observed=[]
        pool=MagicMock()
        def submit(function,job):
            submitted.append(job[2]);f=Future();f.set_result(dict(index=job[2]));return f
        pool.submit.side_effect=submit
        def wait(pending,**kwargs):
            observed.append(len(pending));return {next(iter(pending))},set(pending)-{next(iter(pending))}
        # First completion opens three slots, then new submissions drain to one.
        targets=iter([3,1,1,1,1,1,1])
        with patch.object(x,'wait',side_effect=wait),patch.object(x.time,'monotonic',side_effect=iter(range(0,2000,31))):
            result=x.process_jobs(jobs,1,monitor=lambda active,capacity:next(targets),executor=pool,capacity=3)
        self.assertEqual(sorted(submitted),list(range(7)))
        self.assertEqual(sorted(v['index'] for v in result),list(range(7)))
        self.assertEqual(observed[:3],[1,3,2])
        self.assertEqual(max(observed),3)

    def test_persistent_pool_is_constructed_once(self):
        with patch.object(x,'ProcessPoolExecutor') as factory:
            with x.PoolOwner() as owner:
                first=owner.get(8);second=owner.get(8)
                self.assertIs(first,second)
            self.assertEqual(factory.call_count,1)
            first.shutdown.assert_called_once_with(wait=True,cancel_futures=True)

    def test_contexts_are_reused_across_cohorts_and_refreshed_for_changed_input(self):
        plan=x.read_plan()
        with patch.dict(x._CONTEXT_SIGNATURES,{},clear=True),patch.dict(x.native.CONTEXT,{},clear=True),patch.object(x.native,'native_context',return_value=NS()) as build:
            self.assertEqual(x.ensure_contexts(plan),2)
            self.assertEqual(x.ensure_contexts(plan),0)
            changed=copy.deepcopy(plan);changed['systems']['H4']['binding']['state']='DIFFERENT'
            self.assertEqual(x.ensure_contexts(changed),1)
            self.assertEqual(build.call_count,3)

    def test_schedule_reuse_returns_unmodified_fresh_descriptors(self):
        original=MagicMock(return_value=[dict(signed_time=.8,rte_steps=4)])
        memo=p.ScheduleMemo(original)
        a=memo('2nd',1,'ABS_TIME_PER_OUTER',4,T=.8);a[0]['signed_time']=99
        b=memo('2nd',1,'ABS_TIME_PER_OUTER',4,T=.8)
        self.assertEqual(b[0]['signed_time'],.8)
        self.assertEqual((memo.hits,memo.misses,original.call_count),(1,1,1))

    def test_identical_numeric_events_reuse_build_but_not_sampling(self):
        args=inputs();original=MagicMock(return_value=NS(circuit='ARTIFICIAL_CIRCUIT'))
        memo=p.ComposeMemo(original)
        first=memo(*args)
        args[2][0].seed=999  # A new draw can yield the identical numerical event.
        second=memo(*args)
        self.assertIs(first,second);self.assertEqual(original.call_count,1)
        self.assertEqual((memo.hits,memo.misses),(1,1))

    def test_event_scalar_spec_and_policy_changes_never_hit(self):
        for change in ('angle','scalar','spec','control','basis','time','row'):
            with self.subTest(change=change):
                args=list(inputs());original=MagicMock(return_value=NS())
                memo=p.ComposeMemo(original);memo(*args)
                occurrence=args[2][0].rte_occurrence
                if change=='angle':occurrence.events[0].angle=.2
                if change=='scalar':args[3]=.3
                if change=='spec':occurrence.component_specs=tuple(['DIFFERENT_SPEC'])
                if change=='control':occurrence.control_diagonal_only=False
                if change=='basis':args[2][0].rte_basis_plan=NS(to_dict=lambda:dict(policy='DIFFERENT'))
                if change=='time':args[2][0].step_time=1.6
                if change=='row':args[0]['T']=1.6
                memo(*args);self.assertEqual(original.call_count,2)

    def test_signed_zero_and_different_preparation_do_not_alias(self):
        args=list(inputs());original=MagicMock(return_value=NS());memo=p.ComposeMemo(original)
        args[3]=0.;memo(*args);args[3]=-0.;memo(*args)
        args[2][0].preparation=NS(preparation_hash='OTHER');memo(*args)
        self.assertEqual(original.call_count,3)

    def test_worker_IPC_receipt_does_not_send_raw_trial_payload(self):
        row=x.read_plan()['candidates'][0]['candidate']
        raw=dict(candidate=row,cohort='ARTIFICIAL',index=1,peak_RSS_bytes=123,wall_seconds=.5,
                 prices={'cosine':list(range(1000))},checks=[dict(value='long artificial record')]*100)
        with patch.object(x,'_CONTROLLER_PID',-1),patch.object(x,'_WORKER_MEMOS',(NS(hits=0,misses=0),NS(hits=0))),patch.object(x.native,'cost_trial',return_value=raw):
            slim=x.worker_trial((row,'ARTIFICIAL',1))
        self.assertNotIn('prices',slim);self.assertNotIn('checks',slim)
        self.assertEqual(slim['index'],raw['index'])
        self.assertLess(len(json.dumps(slim)),len(json.dumps(raw))/5)


if __name__=='__main__':
    unittest.main()
