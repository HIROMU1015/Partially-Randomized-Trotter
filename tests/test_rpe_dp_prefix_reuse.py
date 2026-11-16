"""Artificial tables only. Saved kernel/accelerator SOURCE, no H4 evaluation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from copy import deepcopy
from fractions import Fraction
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
VIEW=Path(os.environ.get('RPE_FIXED_VIEW','/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-route-view-b-bd42dd1e9'))
# The source branch intentionally leaves historical kernel files in their fixed view.
sys.path[:0]=[str(VIEW/'src'),str(VIEW/'tests')]
from trottertracks import rpe_total_dp_parallel as P
spec=importlib.util.spec_from_file_location('_d2_prefix',ROOT/'src/trottertracks/rpe_dp_prefix_reuse.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
from check_rpe_total_dp_parallel import tables
from check_rpe_total_decoder_risk import make,CASES
R=P.R


def example(n=13,shots=2,states=32,pairs=200000):
    args=make(vector=[shots]*n,raws=[(.25,-.5)]*n)
    scope=dict(source={'fixture':'ARTIFICIAL'},maximum_frontier_states=states,maximum_transition_pairs=pairs)
    tt=tables(args)
    return args,tt,scope


def changed_tail(tt,shots=3):
    row=dict(m=len(tt)-1,raw=dict(real=.25,imag=-.5),N_X=shots,N_Y=shots)
    t=R.D.round_distribution(row,h=10.,target_energy=0.,radius=1.,probability_rule=R.D.EXACT_AFFINE,maximum_outcomes=10000)
    return deepcopy(tt[:-1])+[t]


def artificial_child(tt,scope,out):
    value=P.Accelerator(workers=1)(tt,h=10.,target=0.,scope=scope)
    (out/'RESULT.json').write_text(json.dumps(value))


class PrefixTests(unittest.TestCase):
    def engine(self,**kwargs):
        return C.accelerator_class(P)(cache_namespace={'fixture':'same_configuration_signal_contract'},workers=1,**kwargs)

    def test_m12_only_change_exact_all_fields_and_12_rounds_reused(self):
        _,tt,scope=example();store=C.PrefixStore();engine=self.engine(store=store)
        for trial in (tt,changed_tail(tt)):
            expected=P.Accelerator(workers=1)(trial,h=10.,target=0.,scope=scope)
            self.assertEqual(engine(trial,h=10.,target=0.,scope=scope),expected)
        self.assertEqual([e['reused_rounds']for e in engine.cache_events if e['event']=='HIT'],[12])
        self.assertEqual(len(engine.observations),13)

    def test_ordinary_full_q3_evaluate_and_intake_judgment_match(self):
        for limit in (0,100):  # both sides of the model filter, no adopted real tolerance
            args=make('zero_tie_rich');args['options']['q3_limit']=dict(source={'fixture':'ARTIFICIAL'},scope='MODEL_MSE_FILTER_ONLY',MSE_upper=limit)
            other=deepcopy(args);other['allocations'][0]['shots'][-1].update(N_X=3,N_Y=3)
            expected=[R.evaluate(**x)for x in (args,other)]
            with C.installed(P,directory=None,cache_namespace={'fixture':'evaluate'}):
                engine=P.Accelerator(workers=1)
                with patch.object(R,'total_state_dp',engine):
                    for trial,want in zip((args,other),expected):self.assertEqual(R.evaluate(**trial),want)
                self.assertTrue(any(e['event']=='HIT'for e in engine.cache_events))
            self.assertEqual(expected[1][0]['diagnostic']['deterministic_risk']['q3']['within_limit'],limit==100)

    def test_residual_and_scope_boundaries_and_no_mass_frontier(self):
        for states,pairs in ((0,0),(0,10000),(1,7),(2,30),(8,2000)):
            _,tt,scope=example(n=4,states=states,pairs=pairs);engine=self.engine()
            engine(tt,h=10.,target=0.,scope=scope)
            other=changed_tail(tt)
            self.assertEqual(engine(other,h=10.,target=0.,scope=scope),R.total_state_dp(other,h=10.,target=0.,scope=scope),(states,pairs))
            self.assertTrue(any(e['event']=='HIT'for e in engine.cache_events))

    def test_prefix_change_scope_target_h_length_and_namespace_miss(self):
        _,tt,scope=example(n=4);store=C.PrefixStore();engine=self.engine(store=store)
        engine(tt,h=10.,target=0.,scope=scope)
        variants=[(changed_tail(tt[:-1])+tt[-1:],10.,0.,scope),(tt,11.,0.,scope),(tt,10.,.01,scope),
                  (tt,10.,0.,dict(scope,maximum_frontier_states=3)),(tt[:-1],10.,0.,scope)]
        for trial,h,target,sc in variants:
            self.assertEqual(engine(trial,h=h,target=target,scope=sc),P.Accelerator(workers=1)(trial,h=h,target=target,scope=sc))
            self.assertFalse(any(e['event']=='HIT'for e in engine.cache_events))
        engine=C.accelerator_class(P)(cache_namespace={'fixture':'OTHER'},store=store,workers=1)
        engine(tt,h=10.,target=0.,scope=scope);self.assertFalse(any(e['event']=='HIT'for e in engine.cache_events))

    def test_earlier_checkpoint_can_reuse_only_unchanged_prefix(self):
        _,tt,scope=example(n=5);engine=self.engine(checkpoint_rounds=[1,3])
        engine(tt,h=10.,target=0.,scope=scope)
        trial=tt[:3]+changed_tail(tt[:4])[-1:]+tt[4:]
        self.assertEqual(engine(trial,h=10.,target=0.,scope=scope),P.Accelerator(workers=1)(trial,h=10.,target=0.,scope=scope))
        self.assertEqual([e['reused_rounds']for e in engine.cache_events if e['event']=='HIT'],[2])

    def test_disk_reopen_signed_zero_large_exact_masses_and_provenance_only_scope(self):
        _,tt,scope=example(n=4)
        for t in tt:
            factor=2**3000+7;t['denominator']*=factor
            for o in t['outcomes']:o['weight']*=factor
        with tempfile.TemporaryDirectory()as d:
            first=self.engine(store=C.PrefixStore(d));first(tt,h=10.,target=0.,scope=scope)
            second=self.engine(store=C.PrefixStore(d));trial=changed_tail(tt)
            sc=dict(scope,source={'fixture':'different_invocation_same_limits'})
            self.assertEqual(second(trial,h=10.,target=0.,scope=sc),P.Accelerator(workers=1)(trial,h=10.,target=0.,scope=sc))
            self.assertTrue(any(e['event']=='HIT'for e in second.cache_events))
            value=dict(signedzero=-0.,mass=P.Mass(2**10000,3),fraction=Fraction(-7,13))
            self.assertEqual(C._decode(C._encode(value),P.Mass),value)
            self.assertEqual(C._decode(C._encode(value),P.Mass)['signedzero'].hex(),'-0x0.0p+0')

    def test_invalid_suffix_and_corrupt_checkpoint_never_yield_pass(self):
        _,tt,scope=example(n=4)
        with tempfile.TemporaryDirectory()as d:
            engine=self.engine(store=C.PrefixStore(d));engine(tt,h=10.,target=0.,scope=scope)
            bad=changed_tail(tt);bad[-1]['denominator']+=1
            self.assertEqual(engine(bad,h=10.,target=0.,scope=scope),P.Accelerator(workers=1)(bad,h=10.,target=0.,scope=scope))
            p=next(Path(d).glob('*.json'));p.write_text('{"broken":true}')
            trial=changed_tail(tt)
            self.assertEqual(engine(trial,h=10.,target=0.,scope=scope),P.Accelerator(workers=1)(trial,h=10.,target=0.,scope=scope))
            self.assertTrue(any(e['event']=='READ_REJECTED'for e in engine.cache_events))

    def test_existing_observed_subclass_constructor_and_process_inheritance(self):
        _,tt,scope=example(n=4);events=[]
        class Observed(P.Accelerator):pass
        with tempfile.TemporaryDirectory()as d:
            with C.installed(P,directory=d,cache_namespace={'fixture':'existing_worker'},event_sink=events.append):
                engine=Observed(workers=2,outcome_batch=2);first=engine(tt,h=10.,target=0.,scope=scope)
                engine=Observed(workers=2,outcome_batch=2);second=engine(changed_tail(tt),h=10.,target=0.,scope=scope)
            self.assertEqual(first,P.Accelerator(workers=1)(tt,h=10.,target=0.,scope=scope))
            self.assertEqual(second,P.Accelerator(workers=1)(changed_tail(tt),h=10.,target=0.,scope=scope))
            self.assertTrue(any(e['event']=='HIT'for e in events))

    def test_adapter_context_fork_worker_disk_reuse(self):
        import multiprocessing as mp
        adapter_spec=importlib.util.spec_from_file_location('_prefix_adapter',ROOT/'scripts/run_rpe_optimizer_q3_job.py')
        A=importlib.util.module_from_spec(adapter_spec);adapter_spec.loader.exec_module(A)
        _,tt,scope=example(n=4)
        with tempfile.TemporaryDirectory()as d:
            request=dict(method='PR_DF',configuration_id='ARTIFICIAL',expected={'configuration_id':'ARTIFICIAL'},
                route=dict(view_commit='ARTIFICIAL_FIXED'),execution_profile=dict(prefix_cache_directory=str(Path(d)/'cache')))
            inputs={k:{'fixture':'ARTIFICIAL'}for k in A.CONTRACT}
            case=dict(configuration={'fixture':'ARTIFICIAL'},signal_packet={'fixture':'ARTIFICIAL'})
            for i,trial in enumerate((tt,changed_tail(tt))):
                out=Path(d)/str(i);out.mkdir()
                with A.prefix_cache_context(request,case,inputs,out):
                    child=mp.get_context('fork').Process(target=artificial_child,args=(trial,scope,out))
                    child.start();child.join(20);self.assertEqual(child.exitcode,0)
                result=json.loads((out/'RESULT.json').read_text())
                self.assertEqual(result,P.Accelerator(workers=1)(trial,h=10.,target=0.,scope=scope))
            events=[json.loads(l)for l in(Path(d)/'1'/'DP_PREFIX_EVENTS.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['event']=='HIT'for e in events))
            self.assertTrue(all(e['pid']!=os.getpid()for e in events))

    def test_original_guards_stay_active(self):
        with patch.object(P,'REFERENCE_DP_AST_SHA256','WRONG'):
            with self.assertRaisesRegex(ValueError,'Reference total DP SOURCE changed'):self.engine()
        with patch.object(C,'ACCELERATOR_SOURCE_SHA256','WRONG'):
            with self.assertRaisesRegex(ValueError,'Unsupported accelerator SOURCE'):self.engine()


if __name__=='__main__':unittest.main(verbosity=2)
