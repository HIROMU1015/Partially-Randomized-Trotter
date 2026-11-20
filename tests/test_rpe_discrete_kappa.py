"""Exact finite PR domains on artificial metadata; no scientific providers."""
from fractions import Fraction as F
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_schedule_search as W
from trottertracks import rpe_method_optimizer as O
import run_rpe_schedule_search as E


class DiscreteGrid(unittest.TestCase):
    def test_grid_is_exact_finite_and_exhausts_without_refinement(self):
        grid = [1, 2, 4, 8, 16, 32, 64]
        families = {key: W.Family(key, [dict(m=0, c=F(1, 1000), q=1, J=1)])
                    for key in ('a', 'b')}
        points = [dict(family='a', kappa=F(3), status='FEASIBLE', best=F(0))]
        seen = []
        for _ in range(10):
            proposed, _, report = W.plan_wide(families, points, dict(kappa_values=grid, per_decision=3))
            self.assertLessEqual(len(proposed), 3)
            self.assertTrue(all(p['kappa'] in grid for p in proposed))
            seen.extend((p['family'], p['kappa']) for p in proposed)
            points.extend(dict(p, status='NO_FEASIBLE_FOUND', best=None) for p in proposed)
            if not proposed:
                break
        self.assertEqual(set(seen), {(f, F(k)) for f in families for k in grid})
        self.assertEqual(len(seen), 14)
        self.assertEqual(report['stop'], 'RULE_EXHAUSTED')
        self.assertIsNone(report['global_best'])  # off-grid saved point cannot set an incumbent

    def test_requested_values_survive_physical_schedule_aliasing(self):
        f = W.Family('f', [dict(m=0, c=F(1, 1000), q=1, J=1)])
        proposals, _, _ = W.plan_wide({'f': f}, [], dict(kappa_values=['1', '2', '4']))
        self.assertEqual([p['kappa'] for p in proposals], [F(1), F(2), F(4)])
        self.assertEqual(len({f.R(p['kappa']) for p in proposals}), 1)

    def test_pending_grid_point_is_neither_retried_nor_ignored_at_stop(self):
        point = dict(family='f', kappa=F(2), status='IN_PROGRESS', best=None)
        proposals, _, report = W.plan_wide({'f': object()}, [point], dict(kappa_values=['2']))
        self.assertEqual(proposals, [])
        self.assertEqual(report['stop'], 'WAIT_FOR_PENDING')

    def test_distributed_driver_supplies_the_discrete_backlog_while_a_split_is_pending(self):
        grid = ['1', '2', '4']
        driver = E.WidthDriver.__new__(E.WidthDriver)
        driver.a = SimpleNamespace(plan_only=False)
        driver.wide = dict(kappa_values=grid, per_decision=1)
        driver.spec = dict(science_approved=True, executor=dict(distributed=dict(preparation={'enabled': True})))
        driver.work_conserving_search = driver.live_ready_refill = True
        driver.families = lambda: {'a': object(), 'b': object()}
        driver.state = dict(cycle=1, history=[], families={
            key: dict(method='PR_DF', L_D=ld, formula='2nd', K=2, requested_delta_pf='1/10')
            for key, ld in [('a', 5), ('b', 6)]}, points=[
                dict(family='a', method='PR_DF', kappa='1', status='COMPLETION'),
                dict(family='b', method='PR_DF', kappa='2', status='HELD', reason='ORIGINAL_STOP')])

        proposals, _ = driver.plan()
        self.assertEqual([(p['family'], p['kappa']) for p in proposals],
                         [('b', F(1)), ('a', F(2)), ('a', F(4)), ('b', F(4))])
        self.assertEqual(driver.state['points'][1]['reason'], 'ORIGINAL_STOP')
        self.assertEqual(driver.plan()[0], [])
        self.assertEqual(driver.wide['per_decision'], 1)

    def test_exact_fraction_membership_and_domain_validation(self):
        self.assertEqual(W.kappa_values(dict(kappa_values=['2', '1/3', '2'])), [F(1, 3), F(2)])
        self.assertTrue(W.kappa_allowed(dict(kappa_values=['1/3']), F(1, 3)))
        self.assertFalse(W.kappa_allowed(dict(kappa_values=['1/3']), F(float(1 / 3))))
        for setting in [dict(kappa_values=[]), dict(kappa_values=['0']),
                        dict(kappa_values=['-1']), dict(kappa_values=['4'], kappa_bounds=['1', '2'])]:
            with self.assertRaises(ValueError):
                W.kappa_values(setting)

    def test_all_candidate_sources_share_the_design_filter(self):
        setting = dict(family='PR', L_D_values=[5], absolute_kappa_values=['1', '2', '4'],
                       generation=dict(formulas=['2nd']), requested_delta_pf_values=['1/10'])
        problem = dict(methods=dict(PR_DF=setting, FULL_PF=dict(family='PF')))
        rows = [dict(configuration_id='keep', kappa='2', L_D=5, formula='2nd', delta_pf='1/10'),
                dict(configuration_id='offgrid', kappa='3', L_D=5, formula='2nd', delta_pf='1/10'),
                dict(configuration_id='wrongformula', kappa='2', L_D=5, formula='4th', delta_pf='1/10'),
                dict(configuration_id='wrongwidth', kappa='2', L_D=5, formula='2nd', delta_pf='1/5')]
        context = dict(catalog=dict(candidates=rows), candidates=[
            dict(method='PR_DF', configuration_id=r['configuration_id'], key=dict(L_D=r['L_D'])) for r in rows])
        context['candidates'].extend([
            dict(method='PR_DF', configuration_id='installed', key=dict(L_D=5), kappa_value='4',
                 generated_configuration=dict(spec=dict(L_D=5, formula='2nd', delta_pf='1/10'))),
            dict(method='PR_DF', configuration_id='badinstalled', key=dict(L_D=5), kappa_value='3',
                 generated_configuration=dict(spec=dict(L_D=5, formula='2nd', delta_pf='1/10'))),
            dict(method='PR_DF', configuration_id=None, key=dict(L_D=6)),
            dict(method='FULL_PF', configuration_id='pf', key={})])
        O.scope_absolute_kappa(context, problem)
        self.assertEqual([c['configuration_id'] for c in context['candidates']], ['keep', 'installed', 'pf'])
        self.assertEqual(len(context['scope_excluded']), 5)

    def test_plan_only_grid_does_not_enter_completion_or_executor(self):
        driver = E.WidthDriver.__new__(E.WidthDriver)
        driver.a = SimpleNamespace(plan_only=True)
        driver.spec = dict(search_overrides={})
        driver.wide = dict(kappa_values=['1', '2', '4'])
        driver.widths = dict(PR_DF=[dict(raw='1/10')])
        driver.methods = ['PR_DF']
        driver.problem = dict(methods=dict(PR_DF=dict(L_D_values=[5])))
        driver.state = dict(cycle=0, families=dict(f=dict(L_D=5, status='READY_METADATA')))
        driver.plan = Mock(return_value=([], {}))
        driver.save = Mock()
        driver.complete_points = Mock(side_effect=AssertionError('PlanOnly must not complete inputs'))
        driver.batch = Mock(side_effect=AssertionError('PlanOnly must not start an executor'))
        driver.loop()
        self.assertEqual(driver.state['scope_plan']['kappa_values_by_method'], dict(PR_DF=['1', '2', '4']))
        self.assertEqual(driver.state['stop']['reason'], 'DISCRETE_SCHEDULE_PLAN_ONLY')
        driver.complete_points.assert_not_called()
        driver.batch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
