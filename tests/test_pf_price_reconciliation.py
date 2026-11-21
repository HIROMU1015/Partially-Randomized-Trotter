"""PF price decisions reopen for new evidence while preserving prior attempts."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trottertracks import pf_candidate_selection as selection
from trottertracks import pf_saved_prices as prices


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PFPriceReconciliationTests(unittest.TestCase):
    def test_axis_recovery_uses_only_clear_unmeasured_wrappers(self):
        original = dict(selected_axis_compile=True, maximum_new_compile=4, new_compile_cells=[
            dict(m=2, axes=['X', 'Y'], role='CALIBRATION'), dict(m=4, axes=['X', 'Y'], role='HOLDOUT')])
        overlap = {'2': [dict(axis='X', action='HELD'), dict(axis='Y', action='NEW_COMPILE_MISSING_CONFIRMED')],
                   '4': [dict(axis='X', action='HELD'), dict(axis='Y', action='HELD')]}
        reduced = prices.independent_plan(original, overlap)
        self.assertEqual(reduced['new_compile_cells'], [dict(m=2, axes=['Y'], role='CALIBRATION')])
        self.assertEqual(reduced['maximum_new_compile'], 1)
        self.assertEqual(original['new_compile_cells'][0]['axes'], ['X', 'Y'])

    def test_received_stage_reopens_the_exact_pf_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            base = run/'pf_auto/price-recovery'
            records = {
                'finished': dict(status='PREPARATION_INCOMPLETE', price_receipt_recheck=True,
                                 reason='DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:PF_COMPILE:received'),
                'failed': dict(status='PREPARATION_INCOMPLETE', reason='STAGE_FAILED:PF_AUDIT:NO_AUTOMATIC_RETRY'),
            }
            runtime = SimpleNamespace(received_after_reservation_stop=lambda path, reason: (
                path == base/'candidates/finished/pf_auto/pf_completion/provider_workspace/docs/research/distributed_selected_pf/finished'
                and reason.endswith(':received')))
            selection.reconcile_received_preparation(run, records, runtime)
            self.assertEqual(records['finished']['status'], 'PREPARATION_WAITING')
            self.assertEqual(records['failed']['status'], 'PREPARATION_INCOMPLETE')

    def test_restored_candidates_are_announced_before_price_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            entry = {'saved': {'id': 'saved'}}
            write(run/'pf_auto/WAVES.json', {'saved': {'status': 'READY_RETURNED_TO_SAME_OPTIMIZER',
                                                       'configurations': entry}})
            context = dict(pf_evidence_registry={'configurations': {}}, candidates=[])
            spec = dict(methods=['TRUNCATED_PF'], pf_completion={'mode': 'complete'})
            events = []

            def refresh(root, problem, old, methods, registry, artificial):
                return dict(old, pf_evidence_registry=registry)

            def announce(updated, routes):
                self.assertEqual(updated['pf_evidence_registry']['configurations'], entry)
                events.append('READY')

            with patch.object(selection, '_refresh', side_effect=refresh), \
                 patch.object(selection.PF, 'registry_routes', return_value={}), \
                 patch.object(selection, 'reconsider_prices', side_effect=lambda *args: events.append('RECONCILE')), \
                 patch.object(selection, 'select', return_value={'selected': None}):
                selection.prepare_next(run, {}, context, spec, run, {}, object(), announce=announce)
            self.assertEqual(events, ['READY', 'RECONCILE'])

    def test_pf_wave_announces_first_ready_while_other_candidate_prepares(self):
        from copy import deepcopy
        first = dict(configuration_id='first', key={'formula': '2nd', 'p': 6, 'delta_over_h': '1'},
                     method='TRUNCATED_PF', ready=False, needs=['PRICE'])
        second = dict(first, configuration_id='second', key=dict(first['key'], p=7))
        context = dict(pf_evidence_registry={'configurations': {}}, candidates=[first, second],
                       catalog={'candidates': []})
        spec = dict(methods=['TRUNCATED_PF'], pf_completion={'mode': 'complete', 'max_candidates': 2})
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            announced = []
            ready = threading.Event()

            def prepare(*args, selection, **kwargs):
                cid = selection['selected']['configuration_id']
                if cid == 'second':
                    self.assertTrue(ready.wait(timeout=3))
                return {'pf_evidence_registry': {'configurations': {cid: {'id': cid}}}}, {}, {
                    'status': 'READY_RETURNED_TO_SAME_OPTIMIZER'}

            def announce(updated, routes):
                saved = json.loads((run/'pf_auto/WAVES.json').read_bytes())
                announced.append(updated['pf_evidence_registry']['configurations'])
                if list(announced[-1]) == ['first']:
                    self.assertEqual(saved['first']['status'], 'READY_RETURNED_TO_SAME_OPTIMIZER')
                    self.assertEqual(saved['second']['status'], 'PREPARATION_SCHEDULED')
                    ready.set()

            def refresh(root, problem, old, methods, registry, artificial):
                return dict(old, pf_evidence_registry=deepcopy(registry))

            with patch.object(selection, 'prepare', side_effect=prepare), \
                 patch.object(selection, '_refresh', side_effect=refresh), \
                 patch.object(selection.PF, 'registry_routes', return_value={}):
                selection.prepare_next(run, {}, context, spec, run, {}, object(), announce=announce)
            self.assertEqual([list(snapshot) for snapshot in announced], [['first'], ['first', 'second']])

    def test_completed_subset_requires_original_stage_end_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            leaf = Path(tmp)
            provider = leaf/'provider_workspace/docs/research/distributed_selected_pf/cfg'
            axes = [{'physical_id': 'circuit', 'axis': axis, 'gate': 'C_IMPLEMENTATION_GUARD_PASS'}
                    for axis in ('X', 'Y')]
            write(leaf/'RESPONSE.json', {'items': [{
                'configuration_id': 'cfg', 'status': 'PRICE_EVIDENCE_REQUIRED',
                'audited_independent_prices': axes}]})
            write(provider/'PF_AUDIT.RESULT.json', dict(status='AUDIT_PASS', prices=axes))
            for stage in ('PF_COMPILE', 'PF_AUDIT'):
                write(provider/(stage+'.DISTRIBUTED.json'), dict(job_id=stage))
                write(provider/(stage+'.RETURN.json'), dict(job_id=stage, returncode=0))
            self.assertEqual(prices.completed_partial({}, leaf, 'cfg'), [['circuit', 'X'], ['circuit', 'Y']])
            write(provider/'PF_COMPILE.RETURN.json', dict(job_id='other', returncode=0))
            self.assertIsNone(prices.completed_partial({}, leaf, 'cfg'))
            write(provider/'PF_COMPILE.RETURN.json', dict(job_id='PF_COMPILE', returncode=0))
            write(provider/'STOP.json', dict(reason='mandatory'))
            self.assertIsNone(prices.completed_partial({}, leaf, 'cfg'))

    def test_rechecks_when_original_owner_publishes_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            original = run/'original'
            row = dict(status='PRIOR_OWNER_HOLD',
                       price_reconciliation=dict(original_owner_directories=[str(original)]))
            for status, generation in [('PRICE_RECHECK_AVAILABLE', 1), ('PRICE_ACTION_UNAVAILABLE', 1)]:
                selection.reconsider_prices(ROOT, run, {}, {}, {'cfg': row})
                self.assertEqual(row['status'], status)
                self.assertEqual(row['price_recheck_generation'], generation)
                row['status'] = 'PRICE_ACTION_UNAVAILABLE'
            write(original/'COMPLETION.json', {'configuration_id': 'cfg'})
            selection.reconsider_prices(ROOT, run, {}, {}, {'cfg': row})
            self.assertEqual(row['status'], 'PRICE_RECHECK_AVAILABLE')
            self.assertEqual(row['price_recheck_generation'], 2)

    def test_incompatible_receipt_does_not_hold_an_otherwise_clear_round(self):
        physical = {'id': 'physical'}
        profile = {'compiler': 'fixed'}
        requests = [dict(m=3, axis=axis, physical_context=dict(physical_candidate=physical, profile=profile),
                         measurement_axis=axis, observable=axis, outcome_convention='fixed',
                         quantity='wrapper_cost', units='RZ', P=0) for axis in ('X', 'Y')]
        hit = dict(source={'commit':'archived'}, value=dict(requests[0],
                   price_profile=dict(compiler='different', lowering_id='H46_LONG_TIME_NATIVE_COMPOSE'),
                   classification='DIRECT_COMPILED_DETERMINISTIC', expectation='DETERMINISTIC'))
        plan = dict(configuration={'id':'cfg'}, new_compile_cells=[{'m':3}],
                    prepared=dict(pending=dict(requests=[dict(m=3, physical_candidate=physical)])))

        def duplicate_check(_, *, selected):
            return dict(rows=[dict(row, action='NEW_COMPILE_MISSING_CONFIRMED', findings=[])
                              for row in selected])

        native = SimpleNamespace(duplicate_check=duplicate_check)
        evidence = SimpleNamespace(physical_key=lambda candidate: candidate['id'],
                                   I=SimpleNamespace(price_requests=lambda *_: requests))
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {
                'run_pf_m0_prices': native, 'trottertracks.pf_evidence_preparation': evidence}):
            out = Path(tmp)/'AUDIT.json'
            self.assertEqual(prices.unresolved_rounds(ROOT, plan, out, {('physical','X'):[hit]}), {})
            self.assertTrue(all(row['action']=='NEW_COMPILE_MISSING_CONFIRMED'
                                for row in json.loads(out.read_bytes())['rows']))
