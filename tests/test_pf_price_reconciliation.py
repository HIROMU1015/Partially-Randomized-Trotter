"""PF price decisions reopen for new evidence while preserving prior attempts."""
import json
from pathlib import Path
import tempfile
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
