"""Artificial controls for new glue; no molecular inputs are loaded."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location('hybrid_execution', ROOT / 'scripts/run_pauli_hybrid_rte_comparison.py')
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class SyntheticExecutionChecks(unittest.TestCase):
    def test_weighted_paired_prices_and_saved_N(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            dest = Path(tmp)
            (dest / 'SOURCE.json').write_text('{"commit":"synthetic"}')
            folder = dest / 'synthetic/confirmation01/cases/test/prices/confirmation'
            for i, (c, s) in enumerate(((2, 3), (4, 5))):
                p = folder / f'{i:04d}'
                p.mkdir(parents=True)
                (p / 'result.json').write_text(json.dumps(dict(index=i, orders=[0],
                    axes=dict(cosine=dict(rz_count=c), sine=dict(rz_count=s)))))
            nominee = dict(molecule='synthetic', candidate_id='test', saved_rows=[dict(
                N_cos='7', N_sin='11', eligibility='CONDITIONAL_GUARD_ELIGIBLE', B_exact='9/8',
                bias_cos_exact='1/100', bias_sin_exact='-1/200', u_cos_exact='1/1000000')])
            with patch.object(D, 'OUT', dest):
                rows, costs = D.aggregate(nominee)
            self.assertEqual(rows[0]['G_exact'], '65')
            self.assertEqual(float(rows[0]['G_SE']), 18.0)
            self.assertEqual((rows[0]['N_cos'], rows[0]['N_sin'], rows[0]['B_exact']), ('7', '11', '9/8'))
            self.assertEqual(costs['axes']['cosine']['mean_exact'], '3')
            self.assertEqual(rows[0]['bias_sin_exact'], '-1/200')

    def test_fixed_extension_rule_and_existing_uncertainty_floor(self):
        new = dict(G_exact='100', G_SE='5', price_n='48')
        other = dict(G_exact='101', G_SE='1')
        self.assertTrue(D.extension_decision(new, other)['extend'])
        self.assertFalse(D.extension_decision(new, dict(G_exact='1000', G_SE='1'))['extend'])
        result = D.extension_decision(new, dict(G_exact='101', G_SE='10'))
        self.assertFalse(result['extend'])
        self.assertEqual(result['reason'], 'OLD_COMPARATOR_UNCERTAINTY_DOMINATES')

    def test_existing_H4_audited_builder_on_two_qubits(self):
        from hx_study.pauli_signal_adapter import PauliHamiltonian
        ham = PauliHamiltonian(2, .13, (('ZZ', .7), ('XI', -.2), ('IY', .1)), '{}')
        capture = D.Capture()
        cfg = D.R.S.config(1, 2, 2, 2)
        body, wrappers, sidecar = D.h4_trajectory(ham, cfg, 123, capture)
        reference_body, reference_wrappers, _ = D.R.S.central_trajectory(ham, cfg, seed=123)
        self.assertEqual(body, reference_body)
        self.assertEqual(wrappers, reference_wrappers)
        self.assertEqual(len(capture.events), 4)
        self.assertTrue(all(e['taylor_order'] in (0, 2) for e in capture.events))
        self.assertEqual(sidecar['ancilla'], 2)
        self.assertEqual(set(wrappers), {'cosine', 'sine'})
        self.assertEqual(wrappers['sine'].num_qubits, 3)
        json.dumps(dict(events=capture.events, control=sidecar))


if __name__ == '__main__':
    unittest.main()
