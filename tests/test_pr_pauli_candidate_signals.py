"""Wrapper fixtures only; no provider, mean, compile or Q3 executes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_pauli_signal_candidate_wrapper',ROOT/'scripts/run_pr_pauli_candidate_signals.py')
W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)


class SignalsWrapperTests(unittest.TestCase):
    def setup_fixture(self,tmp):
        workspace=Path(tmp)/'owned';workspace.mkdir()
        for name in W.ENTRY_FILES:
            p=workspace/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'FIXED')
        config={'id':'ARTIFICIAL','spec':{'L_D':4,'formula':'4th(m5_best)','kappa':{'value':2}},'rounds':[]}
        request=dict(workspace=str(workspace),directory='results/candidate',configuration=config,
                     tail_descriptor={'prefix':4,'fixture':'ARTIFICIAL'},saved_packets=[])
        state={'head':'INITIAL','counter':0};invocations=[]
        def git(w,*args):
            if args[:2]==('rev-parse','HEAD'):return state['head']
            raise AssertionError(args)
        def commit(w,paths,message):
            state['counter']+=1;state['head']='COMMIT'+str(state['counter']);return state['head']
        def entry(w,interpreter,args,log,marker=None):
            invocations.append(list(args));log.write_text('ARTIFICIAL')
            output=Path(args[args.index('--out')+1]);output.mkdir(parents=True,exist_ok=True)
            if '--plan'in args:
                W.write(output/'PLAN.json',dict(configuration_id=config['id'],request_rounds=[1,2],reuse_rounds=[0],rebind_rounds=[],counts={'REQUEST':2}))
            elif '--execute'in args:
                self.assertIsNotNone(marker)
                W.write(marker,dict(status='RUNNING',pid=-1))
                W.write(output/'result.json',dict(status=W.SUCCESS,scientific_source=args[args.index('--source')+1],scope={'new_mean_calls':2}))
                W.write(marker.with_name('SIGNALS_RETURN.json'),dict(returncode=0))
            elif '--assemble'in args:
                W.write(output/'MEAN_HANDOFF.json',dict(packets=[dict(configuration=config,signal_coverage_complete=True)]))
            return 0
        patches=[patch.object(W,'owned_checkout_lock',lambda w:__import__('contextlib').nullcontext()),
                 patch.object(W.subprocess,'check_output',return_value=b'FIXED'),patch.object(W,'git',side_effect=git),
                 patch.object(W,'commit',side_effect=commit),patch.object(W,'publish',return_value='OWNED_BRANCH'),
                 patch.object(W,'entry',side_effect=entry)]
        return request,state,invocations,patches

    def test_all_stages_descriptor_and_exact_candidate_passed_without_reselection(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory()as d:
            req,state,calls,patches=self.setup_fixture(d);original=deepcopy(req['configuration'])
            with ExitStack()as stack:
                for p in patches:stack.enter_context(p)
                first=W.run(req,execute=True);self.assertEqual(first['status'],'SIGNALS_READY')
                second=W.run(req,execute=True);self.assertEqual(second['status'],'SIGNALS_READY')
            self.assertEqual(len(calls),3)
            self.assertTrue(all('--tail-descriptor'in c for c in calls))
            self.assertEqual(req['configuration'],original)
            self.assertNotEqual(first['execution_source'],first['packet_commit'])
            self.assertEqual(second['new_signal_calls_this_invocation'],0)
            self.assertEqual(first['packet_source'],second['packet_source'])

    def test_plan_only_does_not_execute_or_commit_science(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory()as d:
            req,state,calls,patches=self.setup_fixture(d)
            with ExitStack()as stack:
                for p in patches:stack.enter_context(p)
                result=W.run(req)
            self.assertEqual(result['status'],'PLANNED_NOT_EXECUTED')
            self.assertEqual(len(calls),1);self.assertEqual(state['counter'],0)

    def test_ambiguous_prior_science_not_retried(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory()as d:
            req,state,calls,patches=self.setup_fixture(d)
            with ExitStack()as stack:
                for p in patches:stack.enter_context(p)
                W.run(req)
                base=Path(req['workspace'])/req['directory']
                W.write(base/'SIGNAL_STATE.json',dict(status='INPUTS_FIXED',execution_source='INITIAL'))
                W.write(base/'SIGNALS_STARTED.json',dict(status='STARTING'))
                result=W.run(req,execute=True)
            self.assertEqual(result['status'],'HELD_INCOMPLETE_PRIOR_EXECUTION')
            self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main(verbosity=2)
