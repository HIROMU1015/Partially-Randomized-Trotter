"""Saved public controller metadata only; no Q3/scientific numerical work."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from trottertracks.saved_rpe_cost_catalog import Reader
from trottertracks.h4_search_run_evidence import discover_runs

ROOT = Path(__file__).resolve().parents[1]
P03 = 'd7a976a629a6db68197b24f8f86734e109c8bf5d'
P04 = 'e3b4c1114139258a2c6e5d3e14de8b73fc4f7cda'


class SavedRunMetadata(unittest.TestCase):
    def test_saved_different_rounds_keep_new_and_reused_distinct(self):
        runs, issues = discover_runs(Reader(ROOT), {'P03': P03, 'P04': P04})
        selected = {r['label']: r for r in runs if '/kappa_abs_round03/' in r['directory'] or '/kappa_abs_round04/' in r['directory']}
        self.assertEqual(len(selected), 2)
        for run in selected.values():
            self.assertTrue(run['targets'])
            self.assertTrue(run['evaluations'])
            fresh = [e for e in run['evaluations'] if e['new_evaluation'] is True]
            self.assertTrue(fresh)
            self.assertTrue(all(e['origin'] == 'NEW_Q3_JOB' for e in fresh))
            reused = [e for e in run['evaluations'] if e['origin'] == 'PRIOR_ROUND_JOB']
            self.assertTrue(reused)
            self.assertTrue(all(e['new_evaluation'] is False for e in reused))
            self.assertTrue(all(e['adopted_evaluation'] is None for e in run['evaluations']))
        self.assertFalse(issues)
        self.assertTrue(all('/checks/' not in r['source']['path'] for r in runs))
        known_id = 'd7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3'
        known = [e for r in selected.values() for e in r['evaluations'] if e['configuration_id'] == known_id]
        self.assertTrue(known)
        self.assertTrue(all(e['h'] is not None and e['formula'] == '2nd' and
                            len(e['schedule']) == 13 and e['configuration_source'] for e in known))
        self.assertTrue(all('identity' not in s for e in known for s in e['schedule']))

    def test_checkpoint_copy_is_not_a_second_controller_run(self):
        runs,issues=discover_runs(Reader(ROOT),{'D':'36ad751d4ebf0ec160532615da6425b268a0c18a'})
        copies=[r for r in runs if r['label']=='D_FULL_PF_S2_2H_Q3_ONE_V1']
        self.assertEqual(len(copies),1)
        self.assertGreaterEqual(len(copies[0]['snapshot_history']),2)
        self.assertTrue(any('/first_snapshot/'in s['source']['path'] for s in copies[0]['snapshot_history']))

    def test_publication_hash_host_and_worker_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(['git', 'init', '-q', tmp], check=True)
            p = root / 'docs/research/demo/run01/ROUND_RESULT.json'
            p.parent.mkdir(parents=True)
            result = dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1', round=dict(round_id='DEMO', methods=['PR_DF']),
                          methods={'PR_DF': dict(stop_reason='STEP_LIMIT', evaluated=[dict(configuration_id='C',
                              N_X=[1]*13, N_Y=[1]*13, status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',
                              source='NEW_Q3_JOB', job='docs/research/demo/run01/jobs/PR_DF/J', cost={'numerator': 2, 'denominator': 1})])})
            p.write_text(json.dumps(result))
            publication = root / 'docs/research/demo/SEARCH_PUBLICATION.json'
            doc = dict(schema='H4_SEARCH_PUBLICATION_V1', run_id='DEMO', role='CONTROLLER',
                       run_directory='docs/research/demo/run01', execution_hosts=['TEST-HOST'],
                       files={'result': dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())})
            publication.write_text(json.dumps(doc))
            def commit():
                subprocess.run(['git', 'add', '.'], cwd=root, check=True)
                subprocess.run(['git', '-c', 'user.name=T', '-c', 'user.email=t@example.test', 'commit', '-qm', 'fixture'], cwd=root, check=True)
                return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            sha = commit()
            runs, issues = discover_runs(Reader(root), {'any-new-branch': sha})
            published = [r for r in runs if r.get('publication_source')]
            self.assertEqual(len(published), 1)
            self.assertEqual(published[0]['hosts'], ['TEST-HOST'])
            self.assertTrue(published[0]['evaluations'][0]['new_evaluation'])
            self.assertFalse(issues)
            doc['role'] = 'WORKER'; publication.write_text(json.dumps(doc)); sha = commit()
            runs, issues = discover_runs(Reader(root), {'worker': sha})
            self.assertTrue(any('controller-adopted' in i['reason'] for i in issues))
            self.assertFalse(runs)
            doc['role'] = 'CONTROLLER'; doc['files']['result']['sha256'] = '0'*64
            publication.write_text(json.dumps(doc)); sha = commit()
            runs, issues = discover_runs(Reader(root), {'bad': sha})
            self.assertTrue(any('SHA256' in i['reason'] for i in issues))
            self.assertFalse(runs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
