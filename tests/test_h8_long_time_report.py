"""Artificial completed/partial cohort evidence; no H8 action or compiler."""
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h8_long_time_adapter as a
import report_h8_long_time_execution as report


class SavedEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        private = root / 'private'
        rows = []
        batches = [dict(status='PASS', SOURCE='native-fixture', signal_SOURCE='signal-fixture',
                        source_files={}, binding={}, trials=[], summaries={}) for _ in range(2)]
        for i in range(19):
            pr = i < 4
            c = dict(id=str(i), method='PR' if pr else 'FULL', formula='2nd',
                     prefix=7 if pr else 28, q=32, R_outer=32 if pr else None, K=4 if pr else None)
            ep = dict(candidate=c, status='ELIGIBLE', N_cos=2, N_sin=3, log_B=0., log_B_margin=1e-12)
            rows.append(dict(endpoint=ep))
            samples = [dict(candidate_id=c['id'], status='PASS', index=j, cohort='EXPLORATION', conditional=False,
                endpoint_digest=a.digest(ep), log_B=0., prices=dict(cosine=dict(rz_count=10 + i + j), sine=dict(rz_count=20 + i)),
                checks=[dict(phase_alignment=False, error=0., tolerance=1e-8) for _ in range(7)] if j == 0 else [])
                for j in range(9 if pr else 1)]
            batch = batches[i % 2]
            batch['trials'].extend(samples)
            batch['summaries'][c['id']] = a.paired_G(samples, ep, 'EXPLORATION')
        signal = dict(status='PASS', SOURCE='signal-fixture', binding={}, results=rows)
        a.save(private / 'stage_A_combined.json', signal)
        for name in ('stage_A01', 'stage_A02'):
            a.save(private / 'runs' / name / 'result.json', dict(fixture=True))
        for name, batch in zip(('stage_B01', 'stage_B02'), batches):
            a.save(private / 'runs' / name / 'result.json', batch)
        (root / 'docs/research/h8_long_time_execution_20261103').mkdir(parents=True)
        return private

    def test_complete_fixed_cohorts_preserve_exact_weighted_mean(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private = self.fixture(root)
            with patch.object(a, 'ROOT', root), redirect_stdout(io.StringIO()):
                value = report.collect(private, root / 'output')
            first = next(v for v in value['rows'] if v['candidate']['id'] == '0')
            self.assertEqual(first['mean_G_exact'], '88')
            self.assertEqual(first['n'], 9)
            self.assertEqual(len(value['trials']), 51)
            saved = json.loads(gzip.decompress((root / 'output/exploration.json.gz').read_bytes()))
            self.assertEqual(saved['confirmation']['status'], 'NOT_RUN')
            self.assertFalse(saved['global_grid_optimum_claim'])

    def test_partial_representative_or_wrong_signal_source_is_rejected(self):
        for change in ('missing_representative', 'wrong_signal_source'):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                private = self.fixture(root)
                path = private / 'runs/stage_B01/result.json'
                batch = a.read(path)
                if change == 'missing_representative':
                    batch['trials'][0]['checks'] = []
                else:
                    batch['signal_SOURCE'] = 'different'
                path.write_text(json.dumps(batch))
                with patch.object(a, 'ROOT', root), self.assertRaises(ValueError):
                    report.collect(private, root / 'output')
                self.assertFalse((root / 'output/exploration.json.gz').exists())


if __name__ == '__main__':
    unittest.main()
