"""Split stage-2 checks: saved JSON metadata and stub evaluators only.

Never loads saved arrays, evaluates a real mean, builds circuits or prices.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
package = ModuleType('trotterlib'); package.__path__ = [str(ROOT/'src/trotterlib')]
sys.modules.setdefault('trotterlib', package)
from trottertracks import df_head_pauli_tail_split_signals as F  # noqa: E402
import run_df_head_pauli_tail_split_signals as R  # noqa: E402

S3, L, p1 = F.S3, F.L, F.p1
LD3_RAW = '26bac4f320abb345b90ba5cb56a461ff22e1229d'
LD3_PACKET = dict(commit='cdbce383f9944b7f9da1cbbd7851406dbbaae88a',
                  path='docs/research/df_head_pauli_tail_ld3_signals_20261116/MEAN_HANDOFF.json')
SOURCE, RAW = '2'*40, '3'*40


def stub_record(job, ok=True):
    """A record with the request's identities; numbers are artificial."""
    c = deepcopy(job['physical_context']['physical_candidate']); st = job['physical_context']['structure']
    return dict(status=F.LOCAL if ok else 'NUMERICAL_FAILURE', physical_context=deepcopy(job['physical_context']),
                candidate=c, B_identity=deepcopy(job['B_identity']), B_law_binding=deepcopy(job['B_law_binding']),
                B_configuration_id=job['B_configuration_id'], B_candidate_id=job['B_candidate_id'], m=job['m'],
                signed_weights=deepcopy(st['signed_weights']), allocation=deepcopy(st['allocation']), event_count=st['event_count'],
                signal={k: dict(real=.5, imag=.25) for k in ('raw', 'corrected', 'exact_tail', 'reference', 'total_signed_bias',
                        'outer_PF_or_discard_bias', 'finite_RTE_bias')} | dict(log_B=.01, B=1.01, attenuation=.99),
                allowance=dict(u=1e-9, log_B_margin=1e-9), numerical=dict(gate_pass=ok, closure=1e-12, power_unitarity_error=1e-12),
                numerical_failure_reasons=[] if ok else ['ARTIFICIAL'], source=deepcopy(job['physical_context']['numerical_recipe']['source']),
                input_source=dict(artificial=True), reference_source=dict(artificial=True), scalar_once=True,
                N_X=None, N_Y=None, C=None, G=None, global_RMSE_bound=None, scientific_PASS_FAIL=None)


class SplitSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ld5 = F.load_fixed(ROOT, F.SPECS[5])
        cls.ld3 = F.load_fixed(ROOT, F.SPECS[3])

    def setUp(self):
        guard = patch.object(np, 'load', side_effect=AssertionError('No saved arrays'))
        guard.start(); self.addCleanup(guard.stop)

    def test_LD3_spec_reproduces_published_packet_and_fixed_inputs(self):
        saved, item, config, b = self.ld3
        s_saved, s_item, s_config, _ = S3.load_fixed(ROOT)
        self.assertEqual((item, config), (s_item, s_config))
        old = dict(commit=R.LD3_SIGNALS, path=p1.provider.MODULE, adapter=S3.MODULE, kernel_source=S3.P1_KERNEL,
                   recipe=p1.provider.RECIPE)
        jobs = {m: p1.job(config, m, old, b) for m in F.ROUNDS}
        records = {m: L.read(ROOT, LD3_RAW, F.SPECS[3]['directory']+'run01/LD3_m%d.json' % m)['record'] for m in F.ROUNDS}
        value = F.packet(config, records, jobs, raw_commit=LD3_RAW, science_source=R.LD3_SIGNALS, spec=F.SPECS[3],
                         helper_source=R.HELPER)
        published = L.read(ROOT, **LD3_PACKET); published.pop('raw_status')
        self.assertEqual(value, published)

    def test_LD5_fixed_inputs_and_B_helper(self):
        saved, item, config, b = self.ld5
        self.assertEqual(config['id'], F.SPECS[5]['config_id'])
        self.assertEqual([r['R_outer'] for r in config['rounds']], [1]*11+[2, 4])
        b_item = {k: item[k] for k in ('payload', 'dictionary_sha256', 'source')}
        self.assertEqual(S3.b_intake_dictionary_identity(b_item), config['spec']['split']['tail_dictionary'])
        self.assertEqual(S3.operators_locator(item)['path'], F.SPECS[5]['p0']['directory']+'LD5_OPERATORS.npz')
        bad = deepcopy(F.SPECS[5]); bad['config_id'] = 'WRONG'
        with self.assertRaises(ValueError):
            F.load_fixed(ROOT, bad)

    def test_run_order_stop_and_packet_for_LD5(self):
        saved, item, config, b = self.ld5
        jobs = {m: p1.job(config, m, R.numerical(SOURCE), b) for m in F.ROUNDS}
        result = dict(records=[]); names = []
        self.assertTrue(F.run(jobs, lambda m, j: stub_record(j), lambda n, v: names.append(n), result, F.SPECS[5]))
        self.assertEqual(result['status'], F.success(F.SPECS[5]))
        self.assertIn('LD5_m12.json', names)
        records = {r['m']: r['record'] for r in result['records']}
        value = F.packet(config, records, jobs, raw_commit=RAW, science_source=SOURCE, spec=F.SPECS[5], helper_source=R.HELPER)
        pkt = value['packets'][0]
        self.assertEqual((pkt['prefix'], pkt['evaluated_rounds'], value['status']), (5, list(F.ROUNDS), F.success(F.SPECS[5])))
        self.assertEqual(pkt['rounds'][12]['mean_source']['path'], F.SPECS[5]['directory']+'run01/LD5_m12.json')
        result = dict(records=[])
        self.assertFalse(F.run(jobs, lambda m, j: stub_record(j, ok=m != 2), lambda *a: None, result, F.SPECS[5]))
        self.assertEqual((result['status'], len(result['records'])), ('STOP_NUMERICAL_FAILURE', 3))

    def test_runner_execute_and_assemble_entries_with_stubs(self):
        saved, item, config, b = self.ld5
        ctx = SimpleNamespace(tails={5: dict(dictionary_identity=config['spec']['split']['tail_dictionary'])})
        real_read = L.read
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'signals'; run01 = out/'run01'
            def read(root, commit, path):
                if commit == RAW:
                    return json.loads((out/Path(path).relative_to(F.SPECS[5]['directory'])).read_text())
                return real_read(root, commit, path)
            fixed = (dict(saved, inputs={}, context_source=dict(path=__file__, sha256=F.p0.sha(__file__))), item, config, b)
            with patch.object(R.R3, 'head', return_value=SOURCE), patch.object(R, 'sources', return_value=[]), \
                    patch.object(F, 'load_fixed', return_value=fixed), \
                    patch.object(R.R3, 'ld2_context_control', return_value=dict(equal=True)), \
                    patch.object(S3, 'load_context', return_value=ctx), \
                    patch.object(p1.provider, 'saved_h4_reference', return_value=dict(artificial=True)), \
                    patch.object(p1.provider, 'evaluate', side_effect=lambda c, job, ref, root: stub_record(job)):
                result = R.execute(run01, SOURCE, F.SPECS[5])
            self.assertEqual(result['status'], F.success(F.SPECS[5]), result.get('error'))
            self.assertEqual(result['scope']['new_mean_calls'], 13)
            with patch.object(R.R3, 'head', return_value=RAW), patch.object(R.R3, 'is_ancestor', return_value=True), \
                    patch.object(F.p0, 'blob', side_effect=lambda root, commit, path: (ROOT/path).read_bytes()), \
                    patch.object(F, 'load_fixed', return_value=fixed), patch.object(L, 'read', side_effect=read):
                value = R.assemble(out, SOURCE, RAW, F.SPECS[5])
            self.assertEqual(value['status'], F.success(F.SPECS[5]))
            self.assertEqual(value['B_intake_dictionary_identity'], R.HELPER)
            self.assertEqual(len((out/'ROUND_SUMMARY.csv').read_text().splitlines()), 14)


if __name__ == '__main__':
    unittest.main()
