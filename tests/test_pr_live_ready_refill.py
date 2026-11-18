"""Normal width entry can dispatch a prepared candidate while Q3 is waiting."""
from fractions import Fraction
from pathlib import Path
import tempfile
import subprocess
import threading
import time
import unittest
from unittest.mock import patch

import test_pr_width_overlap as overlap


class LiveReadyRefillTests(unittest.TestCase):
    def run_fixture(self, directory, enabled, idle_method=None, standby=False):
        directory = Path(directory)
        args, _, _ = overlap.setup(directory, True)
        spec = overlap.E.O.load_json(directory / 'ROUND.json')
        spec['live_ready_refill'] = enabled
        spec['search_overrides'].update(lookahead=4, lookahead_mode='HYBRID', work_conserving_search=standby)
        spec['executor']['max_concurrent_jobs'] = 4
        overlap.F.write(directory / 'ROUND.json', spec)
        base, _ = overlap.classes(True)
        both_preparing = threading.Barrier(2)
        prepared = set()
        first_selector_done = threading.Event()

        class Completion(overlap.F.ArtificialCompletion):
            def __init__(self, root):
                super().__init__(root)
                self.lock = threading.RLock()

            def complete(self, request):
                design = request['design']
                if overlap.F.KS._frac(design['delta_pf']) != Fraction(.1):
                    # Observe genuine worker launch before making new routes ready.
                    deadline = time.monotonic() + 10
                    while len(list(self.root.glob('jobs/*/*/LAUNCH.json'))) < 2:
                        if time.monotonic() > deadline:
                            raise AssertionError('Initial Q3 jobs never started')
                        time.sleep(.01)
                    if enabled:
                        both_preparing.wait(10)
                    if request['method'] == idle_method:
                        if not first_selector_done.wait(5):
                            raise AssertionError('The first selector did not become idle')
                    with self.lock:
                        prepared.add(request['method'])
                        self.state['price_received'] = True
                        return super().complete(request)
                with self.lock:
                    return super().complete(request)

            def install(self, context, problem, records):
                routes = super().install(context, problem, records)
                for rec in records:
                    design = rec['request']['design']
                    if overlap.F.KS._frac(design['delta_pf']) == Fraction(.1):
                        for rule in routes[rec['request']['method']]:
                            if rec['configuration_id'] in rule['configuration_ids']:
                                method = rec['request']['method']
                                if enabled and method != idle_method:
                                    rule['artificial']['release_on_second_launch'] = list(dict.fromkeys(
                                        [method, idle_method] if idle_method else [method]))
                return routes

        class Driver(base):
            def completion_api(self):
                if self.completion is None:
                    self.completion = Completion(self.run)
                return self.completion

        original = overlap.E.O.optimize

        def observe_idle(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get('ready_source') is not None and kwargs.get('methods') == [idle_method]:
                first_selector_done.set()
            return result

        run_adapter = overlap.E.O.Executor.run_adapter

        def gated_adapter(view, job, rule, mode):
            methods = rule['artificial'].get('release_on_second_launch')
            if not methods or mode != 'execute':
                return run_adapter(view, job, rule, mode)
            # The artificial child waits for another launch before calling the
            # existing adapter. This exercises ordering without a sleep race.
            gate = """from pathlib import Path
import os, sys, time
root = Path(sys.argv[1]); methods = sys.argv[2].split(','); deadline = time.monotonic() + 12
while any(len(list((root / m).glob('*/LAUNCH.json'))) < 2 for m in methods):
    if time.monotonic() > deadline:
        sys.exit('No second launch while the first worker held its slot')
    time.sleep(.01)
os.execv(sys.executable, [sys.executable, '-B', *sys.argv[3:]])
"""
            cmd = [rule['interpreter'], '-B', '-c', gate, str(view.run_dir / 'jobs'), ','.join(methods),
                   str(view.adapter), '--request', str(job / 'REQUEST.json'), '--output', str(job), '--mode', mode]
            with (job / 'EXECUTE.log').open('a') as log:
                return subprocess.Popen(cmd, cwd=view.root, stdout=log, stderr=subprocess.STDOUT)

        with (patch.object(overlap.E.O, 'optimize', side_effect=observe_idle),
              patch.object(overlap.E.O.Executor, 'run_adapter', gated_adapter)):
            result = overlap.call(args, Driver)
        self.assertEqual(prepared, {'PR_DF', 'PR_PAULI'})
        self.assertFalse((result.run / 'RUNTIME_ERRORS.jsonl').exists())
        events = overlap.records(result.run / 'PROGRESS.jsonl')
        exits = [r for r in events if r['event'] == 'EXIT']
        self.assertEqual(len(exits), 4)
        self.assertTrue(all(r['returncode'] == 0 for r in exits))
        self.assertLessEqual(max(r['active'] for r in events if r['event'] == 'LAUNCH'), 4)
        both = []
        for method in ('PR_DF', 'PR_PAULI'):
            launched = [r for r in events if r['event'] == 'LAUNCH' and r['method'] == method]
            self.assertEqual(len(launched), 2)
            first, second = launched
            exit_first = next(r for r in events if r['event'] == 'EXIT' and r['job'] == first['job'])
            both.append(event_duration(second, first, exit_first))
        markers = {p: p.read_bytes() for p in result.run.glob('jobs/*/*/LAUNCH.json')}
        before_resume = overlap.E.O.load_json(result.run / 'CONFIRMED_MINIMA.json')['methods']
        overlap.call(args, Driver)
        self.assertEqual(markers, {p: p.read_bytes() for p in result.run.glob('jobs/*/*/LAUNCH.json')})
        after_resume = overlap.E.O.load_json(result.run / 'CONFIRMED_MINIMA.json')['methods']
        return both, events, [(m, before_resume[m]['verified_all_min'], after_resume[m]['verified_all_min'])
                              for m in sorted(before_resume)]

    def test_late_ready_refills_slots_before_inflight_answers_and_resumes(self):
        with tempfile.TemporaryDirectory(prefix='pr-live-ready-') as tmp:
            baseline, base_events, base_totals = self.run_fixture(Path(tmp) / 'old', False)
            live, live_events, live_totals = self.run_fixture(Path(tmp) / 'live', True)
        self.assertTrue(all(delta >= 0 for delta in baseline), baseline)
        self.assertTrue(all(delta < 0 for delta in live), live)
        self.assertFalse(any(r['event'] == 'Q3_WAIT_YIELD_FOR_READY' for r in base_events))
        self.assertEqual({r['method'] for r in live_events if r['event'] == 'Q3_WAIT_YIELD_FOR_READY'},
                         {'PR_DF', 'PR_PAULI'})
        self.assertEqual(live_totals, base_totals)
        self.assertTrue(all(first == resumed and first is not None for _, first, resumed in live_totals))

    def test_work_conserving_mode_through_normal_width_entry(self):
        with tempfile.TemporaryDirectory(prefix='pr-standby-ready-') as tmp:
            offsets, events, totals = self.run_fixture(tmp, True, standby=True)
            stages = overlap.records(Path(tmp) / 'run' / 'DRIVER_PROGRESS.jsonl')
        self.assertTrue(all(x < 0 for x in offsets))
        self.assertTrue(all(before == after for _, before, after in totals))
        self.assertTrue(any(e.get('dispatch_reason') == 'PER_DECISION_PER_METHOD_WIDTH_INTERLEAVED_FAMILY_ROUND_ROBIN'
                            for e in stages))

    def test_idle_method_refills_slots_while_peer_is_still_running(self):
        with tempfile.TemporaryDirectory(prefix='pr-idle-ready-') as tmp:
            _, events, _ = self.run_fixture(tmp, True, idle_method='PR_DF')
        df_first, df_next = [r for r in events if r['event'] == 'LAUNCH' and r['method'] == 'PR_DF']
        pauli_first = next(r for r in events if r['event'] == 'LAUNCH' and r['method'] == 'PR_PAULI')
        ended = {r['job']: r['monotonic'] for r in events if r['event'] == 'EXIT'}
        self.assertGreater(df_next['monotonic'], ended[df_first['job']])
        self.assertLess(df_next['monotonic'], ended[pauli_first['job']])


def event_duration(second, first, exit_first):
    assert second['invocation'] == first['invocation']
    return second['monotonic'] - exit_first['monotonic']
