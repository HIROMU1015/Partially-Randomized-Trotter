"""Plan-output projection and normal CLI hooks; artificial metadata only, no scientific leaf."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('_compact_plan_cli', ROOT / 'scripts/run_rpe_method_optimizer.py')
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)


def fixture():
    method = 'ARTIFICIAL'
    bounds = Fraction(10**300 + 1, 10**306 + 7)
    evaluated = [dict(configuration_id='ARTIFICIAL_CFG', N_X=[2], N_Y=[2],
        cost=Fraction(2), status=CLI.O.PASS, MSE_lower=bounds, MSE_upper=bounds, source='PRIOR_ROUND_JOB', job='/artificial/job1'),
        dict(configuration_id='ARTIFICIAL_CFG', N_X=[1], N_Y=[1], cost=Fraction(1), status=CLI.O.FAIL,
             MSE_lower=bounds, MSE_upper=bounds, source='PRIOR_ROUND_JOB', job='/artificial/job2')]
    best = dict(candidate='ARTIFICIAL_CFG', configuration_id='ARTIFICIAL_CFG', configuration={'L_D': 1},
        N_X=[2], N_Y=[2], status=CLI.O.PASS, MSE_lower=Fraction(1, 4), MSE_upper=Fraction(1, 3),
        cost=Fraction(2), cost_display='2')
    pending = dict(candidate='ARTIFICIAL_CFG', configuration_id='ARTIFICIAL_CFG', vector=[1, 2], cost=Fraction(3, 2),
                   cls='UPDATE', origin={'kind': 'ARTIFICIAL'})
    result = dict(method=method, best=best, pending_request=pending, stop_reason='NEW_SCIENCE_NOT_APPROVED',
        new_calls=0, evaluated_count=2, evaluated=evaluated, resume_records=evaluated,
        trace=[dict(MSE_upper=bounds)], held=[], ready_candidates=['ARTIFICIAL_CFG'], not_ready_candidates=[],
        budget=dict(max_new_q3=0, science_approved=False, spent_before=0, new_calls=0),
        prior_rounds=dict(run_dirs=['/artificial/prior'], loaded=2, already_known=0, refused=0),
        input_requests=[], search_range={'rule': 'ARTIFICIAL'})
    round_spec = dict(round_id='ARTIFICIAL_COMPACT_PLAN', methods=[method], initial='all', science_approved=False,
                      max_new_q3_per_method={method: 0})
    payload = dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1', round=round_spec, methods={method: result},
                   pf_preparation=None, optimizer_source='ARTIFICIAL_SOURCE')
    return method, result, round_spec, payload


class CompactPlanTests(unittest.TestCase):
    def test_projection_preserves_selection_and_counts_infeasible_without_adopting_it(self):
        method, result, _, payload = fixture()
        original = deepcopy(payload)
        projected = CLI.compact_plan_result(payload, {'evidence': {'saved_catalog_inputs': 'ARTIFICIAL_C'}})
        actual = projected['methods'][method]
        for key in ('best', 'pending_request', 'stop_reason', 'budget', 'prior_rounds', 'evaluated_count'):
            self.assertEqual(actual[key], CLI.slim(result[key]))
        self.assertEqual(actual['best']['cost'], 2)
        self.assertEqual(actual['evaluation_summary']['by_status'], {CLI.O.PASS: 1, CLI.O.FAIL: 1})
        self.assertEqual(actual['evaluation_summary']['resume_records'], 2)
        self.assertNotIn('evaluated', actual)
        self.assertNotIn('resume_records', actual)
        self.assertNotIn('trace', actual)
        self.assertFalse(projected['resume_records_exported'])
        self.assertEqual(payload, original)

    def test_omitted_huge_bounds_are_never_decimalized(self):
        _, _, _, payload = fixture()
        with patch.object(CLI, 'outward', side_effect=AssertionError('omitted bounds must not be walked')):
            output = CLI.compact_plan_result(payload, {})
            json.dumps(CLI.O.jsonable(output))

    def test_only_explicit_zero_budget_unapproved_scope_is_accepted(self):
        _, _, round_spec, _ = fixture()
        for changes in (dict(science_approved=True), dict(science_approved=None),
                        dict(max_new_q3_per_method={'ARTIFICIAL': 1}),
                        dict(max_new_q3_per_method={'ARTIFICIAL': 'UNLIMITED'}),
                        dict(max_new_q3_per_method={})):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'explicit zero budgets'):
                CLI.compact_plan_scope(dict(round_spec, **changes))

    def test_compact_output_is_not_silently_used_as_resume_records(self):
        _, _, _, payload = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'compact.json'
            CLI.write(path, CLI.compact_plan_result(payload, {}))
            with self.assertRaisesRegex(ValueError, 'not a resume-record export'):
                CLI.resume_input([path])

    def test_normal_round_hooks_default_and_compact(self):
        from trottertracks import pf_candidate_selection as PFSelect
        method, result, round_spec, _ = fixture()
        context = dict(expected={}, candidates=[], tau=Fraction(1), saved=None)
        problem = dict(methods={method: {}}, execution_routes={}, search={'max_steps_per_method': 1})
        for compact in (False, True):
            with self.subTest(compact=compact), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp);run = root / 'run';round_path = root / 'round.json'
                config = dict(round_spec, **({'plan_output': 'compact'} if compact else {}))
                round_path.write_text(json.dumps(config))
                events = []
                executor = SimpleNamespace(finish=lambda: events.append('finish'), launched=lambda _: 0)
                def setup(ex, ctx):
                    self.assertIs(ex, executor);self.assertIs(ctx, context);events.append('setup')
                def optimized(*args, **kwargs):
                    self.assertEqual(events, ['setup']);events.append('search');return {method: result}
                def callback(actual, ctx, routes):
                    self.assertIs(actual[method], result);self.assertIs(ctx, context)
                    self.assertFalse((run / 'ROUND_RESULT.json').exists());events.append('callback')
                args = SimpleNamespace(round=round_path, run_dir=run)
                with patch.object(CLI.O, 'prepare', side_effect=AssertionError('prepared context must be reused')), \
                        patch.object(CLI.O, 'apply_published'), patch.object(CLI, 'head', return_value='ARTIFICIAL_SOURCE'), \
                        patch.object(PFSelect, 'prepare', return_value=(context, {}, None)), \
                        patch.object(CLI, 'make_executor', return_value=(executor, optimized)), redirect_stdout(io.StringIO()):
                    returned = CLI.round_mode(args, problem, [], prepared_context=context,
                                              executor_setup=setup, result_callback=callback)
                self.assertIs(returned[method], result)
                self.assertEqual(events, ['setup', 'search', 'finish', 'callback'])
                data = json.loads((run / 'ROUND_RESULT.json').read_text())
                needs = json.loads((run / 'NEEDS.json').read_text())
                self.assertEqual(data['schema'], 'RPE_OPTIMIZER_COMPACT_PLAN_V1' if compact else 'RPE_OPTIMIZER_ROUND_RESULT_V1')
                self.assertEqual(data['methods'][method]['best']['cost']['numerator'], 2)
                self.assertEqual('evaluated' in data['methods'][method], not compact)
                if compact:
                    self.assertEqual(needs['round_result'], 'ROUND_RESULT.json')
                    self.assertNotIn('pending_Q3', needs)
                else:
                    self.assertEqual(set(needs), {'candidates', 'pending_Q3'})
                    self.assertEqual(data['methods'][method], CLI.O.jsonable(CLI.slim(result)))

    def test_invalid_compact_scope_is_rejected_before_preparation(self):
        _, _, round_spec, _ = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'round.json'
            path.write_text(json.dumps(dict(round_spec, science_approved=True, plan_output='compact')))
            args = SimpleNamespace(round=path, run_dir=Path(tmp) / 'run')
            with patch.object(CLI.O, 'prepare', side_effect=AssertionError('must reject before preparation')):
                with self.assertRaisesRegex(ValueError, 'explicit zero budgets'):
                    CLI.round_mode(args, {}, [])


if __name__ == '__main__':
    unittest.main()
