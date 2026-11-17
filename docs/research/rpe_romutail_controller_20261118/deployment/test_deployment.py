"""Synthetic operational-overlay preservation checks; no scientific entry."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('controller_deployment', ROOT / 'scripts/prepare_romutail_controller_deployment.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class DeploymentTests(unittest.TestCase):
    def test_scientific_settings_and_archived_files_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original = base / 'original'
            original.mkdir()
            profile = dict(host_id='local', root=str(ROOT), interpreter='/usr/bin/python3',
                           route_views={}, slots=1, preparation_profiles=[])
            M.put(base / 'PROFILE.json', profile)
            problem = dict(methods={'PR_PAULI': dict(L_D_values=[1, 2])}, accuracy={'fixture': 'SYNTHETIC'})
            round_value = dict(science_approved=False, max_new_q3_per_method={'PR_PAULI': 0},
                search_overrides=dict(schedule_wide=dict(kappa_bounds=['1/2', '8'])),
                executor=dict(max_concurrent_jobs='auto', prior_run_dirs=[], distributed=dict(
                    preparation=dict(budgets=dict(signal=0, compile=0, audit=0)))))
            service = dict(code_root=str(ROOT), code_source='original', computation_source='SYNTHETIC',
                           absolute_kappa_route={'entry': 'schedule'})
            for name, value in [('SERVICE.json', service), ('PROBLEM.json', problem),
                                ('ROUND.default.json', round_value), ('ROUND.absolute-kappa.json', round_value)]:
                M.put(original / name, value)
            before = {p.name: p.read_bytes() for p in original.iterdir()}
            M.put(base / 'MANIFEST.json', dict(roots=[]))
            layout = dict(code_root=str(ROOT), controller_python='/usr/bin/python3',
                state_directory=str(base / 'state'), controller_path_map={}, controller_paths={},
                publication=dict(repository=str(base / 'publication'), branch='synthetic', push=False),
                hosts={'local':dict(base_profile=str(base / 'PROFILE.json'), work_dir=str(base / 'spool'),
                                   artifact_cache_directory=str(base / 'cache'))})
            source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            M.prepare(original, base / 'MANIFEST.json', layout, source, base / 'deployment')
            updated = M.read(base / 'deployment/ROUND.default.json')
            self.assertEqual(updated['search_overrides'], round_value['search_overrides'])
            self.assertEqual(updated['max_new_q3_per_method'], round_value['max_new_q3_per_method'])
            self.assertFalse(updated['science_approved'])
            self.assertEqual(updated['executor']['distributed']['preparation']['budgets'], dict(signal=0,compile=0,audit=0))
            self.assertEqual(updated['executor']['distributed']['affinity_wait_seconds'], 0)
            self.assertNotIn('max_attempted_per_method', updated['candidate_completion'])
            self.assertEqual(M.read(base / 'deployment/PROBLEM.json'), problem)
            self.assertEqual(M.read(base / 'deployment/SERVICE.json')['entry'], 'schedule')
            self.assertEqual(updated['candidate_completion']['source'], source)
            self.assertEqual(updated['executor']['distributed']['worker_script'],
                             str(ROOT / 'scripts/run_rpe_q3_worker.py'))
            # A transport/controller fix can retain the prior wrapper and
            # worker versions without changing scientific settings or inputs.
            legacy_source = subprocess.check_output(['git', 'rev-parse', 'HEAD^'], cwd=ROOT, text=True).strip()
            legacy_worker = base / 'fixed-worker.py'
            legacy_worker.write_text('# Synthetic fixed worker path; never executed.\n')
            release_manifest=base/'RELEASE_COMPONENTS.json';host_layout=base/'HOST.local.json'
            M.put(release_manifest,dict(schema='RPE_RELEASE_COMPONENTS_V1',release_id='SYNTHETIC'))
            M.put(host_layout,dict(host_id='local',component_roots={}))
            layout.update(candidate_completion_source=legacy_source, worker_script=str(legacy_worker),
                release_manifest=str(release_manifest),host_layouts={'local':str(host_layout)})
            M.prepare(original, base / 'MANIFEST.json', layout, source, base / 'control-update')
            controlled = M.read(base / 'control-update/ROUND.default.json')
            self.assertEqual(controlled['candidate_completion']['source'], legacy_source)
            self.assertEqual(controlled['executor']['distributed']['worker_script'], str(legacy_worker))
            self.assertEqual(controlled['max_new_q3_per_method'], round_value['max_new_q3_per_method'])
            self.assertEqual(controlled['search_overrides'], round_value['search_overrides'])
            self.assertEqual(controlled['executor']['distributed']['preparation']['budgets'],
                             round_value['executor']['distributed']['preparation']['budgets'])
            self.assertEqual(M.read(base / 'control-update/SERVICE.json')['code_source'], source)
            deployed_service=M.read(base/'control-update/SERVICE.json')
            self.assertEqual(deployed_service['release_manifest'],str(release_manifest))
            self.assertEqual(deployed_service['host_layouts'],{'local':str(host_layout)})
            for path in (release_manifest,host_layout):
                self.assertEqual(deployed_service['fixed_settings'][str(path)],M.checksum(path))
            self.assertEqual(controlled['price_validation']['store'],str(base/'price-validation/models.sqlite3'))
            self.assertTrue(controlled['price_validation']['publish_shared_models'])
            self.assertEqual(M.read(base / 'control-update/ROUND.absolute-kappa.json')['candidate_completion']['source'],
                             legacy_source)
            self.assertEqual({p.name:p.read_bytes() for p in original.iterdir()}, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
