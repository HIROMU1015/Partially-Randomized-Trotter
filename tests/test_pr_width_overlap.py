"""Explicit multi-width preparation/Q3 overlap; artificial providers/workers only."""
from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import check_pr_width_search as F
import run_rpe_schedule_search as E


def classes(overlap,fatal=False):
    observations=dict(threads=[],waited=[],installed=[])
    class Completion(F.ArtificialCompletion):
        def complete(self,request):
            observations['threads'].append(threading.get_ident())
            step=F.KS._frac(request['design']['delta_pf'])
            if step!=Fraction(.1):
                if overlap:
                    deadline=time.monotonic()+8
                    while not list((self.root/'jobs').glob('*/*/OUTCOME.json')):
                        if time.monotonic()>deadline:raise AssertionError('Initial READY Q3 never advanced during preparation')
                        time.sleep(.005)
                    observations['waited'].append(request['method'])
                self.state['price_received']=True
            return super().complete(request)

        def install(self,context,problem,records):
            routes=super().install(context,problem,records)
            observations['installed'].append([r['configuration_id']for r in records])
            if fatal:
                for rules in routes.values():
                    for rule in rules:rule['artificial']['force_unknown']=True
            return routes

    class Driver(F.ArtificialDriver):
        def __init__(self,args):
            super().__init__(args)
            if self.state.get('overlap_fixture_seeded'):return
            self.plan()
            comp=self.completion_api()
            # Saved READY records are fixture inputs before the normal loop;
            # the changed-width records remain selected preparation requests.
            for point in self.state['points']:
                if F.KS._frac(point['requested_delta_pf'])!=Fraction(.1):continue
                record=comp.complete(point['completion_request'])
                point.update(status='REGISTERED',configuration_id=record['configuration_id'],
                    completion=dict(key=record['key'],status=record['status']))
            self.state['overlap_fixture_seeded']=True;self.save()

        def completion_api(self):
            if self.completion is None:self.completion=Completion(self.run)
            return self.completion
    return Driver,observations


def setup(root,overlap,fatal=False):
    root=Path(root);args=F.WidthSearchTests().files(root)
    spec=json.loads((root/'ROUND.json').read_text())
    spec['search_overrides']['requested_widths']={'PR_PAULI':[.1,.2],'PR_DF':[.1,.05]}
    spec['search_overrides']['lookahead']=0
    spec['executor']=dict(scheduler='concurrent',max_concurrent_jobs=2,dp_workers=1)
    spec['max_new_q3_per_method']={'PR_PAULI':2,'PR_DF':2}
    if overlap:spec['overlap_ready_batch']=True
    F.write(root/'ROUND.json',spec)
    Driver,observations=classes(overlap,fatal)
    return args,Driver,observations


def call(args,Driver):
    with redirect_stdout(io.StringIO()):return E.main(args,driver_factory=Driver)


def records(path):
    return [json.loads(line)for line in Path(path).read_text().splitlines()]


class WidthOverlap(unittest.TestCase):
    def test_explicit_overlap_advances_ready_queue_and_keeps_late_work_same_invocation_and_budget(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-overlap-')as tmp:
            args,Driver,observations=setup(tmp,True);d=call(args,Driver)
            self.assertCountEqual(observations['waited'],['PR_PAULI','PR_DF'])
            self.assertTrue(any(t!=threading.get_ident()for t in observations['threads']))
            queue=records(d.run/'PROGRESS.jsonl')
            self.assertEqual(sum(e['event']=='EXECUTOR_START'for e in queue),1)
            launched=[e for e in queue if e['event']=='LAUNCH']
            self.assertEqual(len(launched),4)
            self.assertEqual(len({e['invocation']for e in launched}),1)
            for method in ('PR_PAULI','PR_DF'):
                jobs=list((d.run/'jobs'/method).glob('*/REQUEST.json'))
                self.assertEqual(len(jobs),2)
                self.assertEqual(len({json.loads(p.read_text())['configuration_id']for p in jobs}),2)
            phases=records(d.run/'ROUND_STAGES.jsonl')
            self.assertTrue(any(e['phase']=='WIDTH_READY_DURING_PREPARATION'for e in phases))
            self.assertTrue(any(e['phase']=='WIDTH_SEARCH_AND_Q3'for e in phases))
            baseline={str(p):p.read_bytes()for p in d.run.glob('jobs/*/*/LAUNCH.json')}
            calls=len(observations['threads']);call(args,Driver)
            self.assertEqual(baseline,{str(p):p.read_bytes()for p in d.run.glob('jobs/*/*/LAUNCH.json')})
            self.assertEqual(len(observations['threads']),calls)

    def test_default_off_uses_synchronous_preparation_and_no_interim_phase(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-no-overlap-')as tmp:
            args,Driver,observations=setup(tmp,False);d=call(args,Driver)
            self.assertEqual(observations['waited'],[])
            self.assertTrue(all(t==threading.get_ident()for t in observations['threads']))
            self.assertFalse(any(e['phase']=='WIDTH_READY_DURING_PREPARATION'for e in records(d.run/'ROUND_STAGES.jsonl')))
            self.assertEqual(len(list(d.run.glob('jobs/*/*/LAUNCH.json'))),4)

    def test_fatal_ready_result_prevents_late_q3_and_preserves_stop_on_resume(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-overlap-fatal-')as tmp:
            args,Driver,observations=setup(tmp,True,True)
            from trottertracks import pr_candidate_completion as C
            original=E.K.CLI.make_executor
            def executor(*args,**kwargs):
                ex,optimize=original(*args,**kwargs)
                def result(*a,**k):
                    value=optimize(*a,**k)
                    # V2 can retain missing-input requests even when a READY
                    # candidate returns UNKNOWN. They must not reopen work.
                    for row in value.values():
                        if row['stop_reason']=='UNKNOWN_RESULT':
                            row['input_requests']=[dict(kind='CONFIGURATION_INPUTS_MISSING',configuration_id='ARTIFICIAL_MISSING')]
                    return value
                return ex,result
            with patch.object(E.K.CLI,'make_executor',side_effect=executor),patch.object(C,'continue_selected_preparation',
                    side_effect=AssertionError('Fatal ready result must not continue preparation')):
                d=call(args,Driver)
            self.assertEqual(d.state['stop']['reason'],'V2_RESULT_REQUIRES_REVIEW')
            self.assertEqual(len(list(d.run.glob('jobs/*/*/LAUNCH.json'))),2)
            self.assertFalse(any(e['phase']=='WIDTH_SEARCH_AND_Q3'for e in records(d.run/'ROUND_STAGES.jsonl')))
            before=len(observations['threads']);call(args,Driver)
            self.assertEqual(before,len(observations['threads']))
            self.assertEqual(len(list(d.run.glob('jobs/*/*/LAUNCH.json'))),2)


if __name__=='__main__':unittest.main(verbosity=2)


def test_width_plan_reconstructs_base_context_in_background_without_second_selection():
    from types import SimpleNamespace
    from threading import Event
    started=Event();planned=Event()
    fixture=SimpleNamespace(batch_problem=lambda:{'fixed':'problem'},saved_catalog=lambda p:None)
    def context(problem):
        started.set();assert planned.wait(2);return {'context':problem}
    def plan():
        assert started.wait(2);planned.set();return ['one'],{'report':1}
    fixture.context=context;fixture.plan=plan
    assert E.WidthDriver.plan_with_base_context(fixture)==(['one'],{'report':1})
    assert fixture._prepared_base_context==({'fixed':'problem'},{'context':{'fixed':'problem'}})
