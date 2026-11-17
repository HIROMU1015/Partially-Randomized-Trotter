"""Normal sync integration on an isolated public Git fixture; scientific calls 0."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import sync_saved_rpe_publications as S

ROOT = Path(__file__).resolve().parents[1]


class NormalMapSync(unittest.TestCase):
    def test_other_environment_publication_dedup_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(['git', 'init', '-q', '-b', 'catalog', tmp], check=True)
            def put(path, value):
                p = root / path; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(value) if not isinstance(value, str) else value)
                return p
            def commit():
                subprocess.run(['git', 'add', '.'], cwd=root, check=True)
                subprocess.run(['git', '-c', 'user.name=T', '-c', 'user.email=t@example.test', 'commit', '-qm', 'saved fixture'], cwd=root, check=True)
                return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            for p in ('scripts/sync_saved_rpe_publications.py', 'src/trottertracks/saved_rpe_cost_catalog.py',
                      'src/trottertracks/pf_optimizer_connection.py', 'src/trottertracks/h4_search_map.py',
                      'src/trottertracks/h4_search_run_evidence.py', 'src/trottertracks/h4_search_map_html.py'):
                put(p, (ROOT / p).read_text())
            put('docs/research/catalog/INPUTS.json', dict(sources=[], comparison=dict(accuracy={})))
            put('docs/research/ROADMAP.md', '<!-- C_AUTO_PROGRESS_START -->\nold\n<!-- C_AUTO_PROGRESS_END -->\n')
            for label, origin in [('HOST_A', 'NEW_Q3_JOB'), ('HOST_B', 'PRIOR_ROUND_JOB')]:
                directory = 'docs/research/' + label + '/run01'
                result = dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1', round=dict(round_id=label, methods=['PR_PAULI']),
                    methods={'PR_PAULI': dict(stop_reason='STEP_LIMIT', evaluated=[dict(configuration_id='SAME_PHYSICAL',
                        N_X=[2]*13, N_Y=[2]*13, status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', source=origin,
                        job=directory + '/jobs/PR_PAULI/J', formula='2nd', L_D=3, requested_delta_pf='1/5',
                        cost=dict(numerator=100, denominator=1))])})
                p = put(directory + '/ROUND_RESULT.json', result)
                validation = put(directory + '/PRICE_MODEL_VALIDATIONS.json', dict(
                    schema='RPE_PRICE_MODEL_VALIDATIONS_V1', models=[dict(model_id='frozen-shared',
                    identity=dict(method='PR_PAULI', L_D=3, frozen_source=dict(commit='f'*40,path='MODEL.json')),
                    status='PENDING', label='検証待ち', validation_points=[])], applications=[dict(
                    configuration_id='SAME_PHYSICAL', model_id='frozen-shared', points=[dict(m=0,q=1,R=2)])]))
                put('docs/research/' + label + '/SEARCH_PUBLICATION.json', dict(schema='H4_SEARCH_PUBLICATION_V1',
                    run_id=label, role='CONTROLLER', run_directory=directory, execution_hosts=[label],
                    files={'result': dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()),
                           'PRICE_MODEL_VALIDATIONS.json': dict(path=str(validation.relative_to(root)),
                               sha256=hashlib.sha256(validation.read_bytes()).hexdigest())}))
            sha = commit()
            # New environment's name is deliberately outside C's legacy branch patterns.
            for name in ('known-old-branch', 'independent-environment/publication'):
                subprocess.run(['git', 'update-ref', 'refs/remotes/origin/' + name, sha], cwd=root, check=True)
            config = dict(remote='origin', branch='catalog', branches=['known-old-branch'],
                          directory='docs/research/catalog', inputs='docs/research/catalog/INPUTS.json',
                          roadmap='docs/research/ROADMAP.md', search_map='docs/research/map')
            catalog = dict(comparison=dict(accuracy={}), candidates=[], method_minima={}, duplicates_removed=0)
            real_git = S.git
            def transport(r, *args):
                if args[0] == 'ls-remote':
                    return '\n'.join(sha + '\trefs/heads/' + n for n in ('known-old-branch', 'independent-environment/publication'))
                if args[0] == 'fetch':
                    return ''
                return real_git(r, *args)
            # Only external transport and saved catalog leaf are artificial.
            # Public discovery, immutable readers, normal sync, map and HTML are real.
            with patch.object(S, 'git', side_effect=transport), \
                 patch.object(S, 'reconcile', side_effect=lambda *a: (deepcopy(catalog), {}, [])), \
                 patch.object(S, 'report', return_value='artificial saved catalog fixture\n'):
                first = S.synchronize(root, config, fetch=True, publish=False)
                self.assertEqual(first['status'], 'UPDATED')
                index_path = root / 'docs/research/map/INDEX.json'
                index = json.loads(index_path.read_text())
                self.assertEqual(len(index['runs']), 2)
                self.assertEqual(len(index['points']), 1)
                self.assertIn('independent-environment/publication', index['included_public_heads'])
                point = index['points'][0]
                self.assertEqual(len(point['occurrences']), 2)
                self.assertEqual(len(index['price_validation_models']), 1)
                self.assertEqual(point['price_validations'][0]['label'], '検証待ち')
                self.assertIn('検証待ち', (index_path.parent/'index.html').read_text())
                self.assertEqual({o['new_evaluation'] for o in point['occurrences']}, {True, False})
                self.assertEqual({h for o in point['occurrences'] for h in o['hosts']}, {'HOST_A', 'HOST_B'})
                before = {p.name: p.read_bytes() for p in index_path.parent.iterdir() if p.is_file()}
                second = S.synchronize(root, config, fetch=True, publish=False)
                self.assertEqual(second['status'], 'UNCHANGED')
                self.assertEqual(before, {p.name: p.read_bytes() for p in index_path.parent.iterdir() if p.is_file()})
                self.assertEqual(index['new_scientific_calls'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
