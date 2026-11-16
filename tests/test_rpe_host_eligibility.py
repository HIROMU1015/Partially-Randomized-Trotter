"""Saved-evidence policy regressions only; no remote connection or scientific job."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trottertracks import rpe_host_eligibility as H
from trottertracks.rpe_q3_coordinator import digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class HostEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rpe-host-eligibility-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.host = 'arbitrary-host'
        self.profile = dict(host_id=self.host, root='/owned/fixed-source', interpreter='/owned/venv/bin/python',
            route_views={'b' * 40: '/owned/fixed-route'}, dp_workers=3, prefix_cache_directory='/owned/cache')
        request = dict(request_id='arbitrary-job', method='ARBITRARY_METHOD', configuration_id='arbitrary-configuration',
            N_X=[3, 2], N_Y=[4, 2], cost=dict(numerator=9, denominator=1), origin=dict(kind='EXISTING_RULE'),
            route=dict(kind='saved_route_kind', view_commit='b' * 40, view='/controller/view',
                       interpreter='/controller/python', templates=[]), expected=dict(contract='fixed'),
            optimizer_source='a' * 40, execution_profile=dict(dp_workers=3, prefix_checkpoint_rounds=[1]))
        self.assignment = dict(job_id=request['request_id'], method=request['method'], request=request,
            request_sha256=digest(request), source_manifest=dict(commit='a' * 40, files={'src/kernel.py': 'd' * 64}),
            input_hashes=[dict(commit='c' * 40, path='saved/input.json', sha256='e' * 64)])
        self.resolved = H._resolved_request(self.assignment, self.profile)
        self.root = self.base / 'evidence' / self.host
        self.cell = self.root / 'dry' / 'arbitrary-candidate'
        save(self.cell / 'RESOLVED_REQUEST.json', self.resolved)
        self.adapter = dict(status='DRY_CHECK_PASS', mode='dry-check', request_id=request['request_id'], route='saved_route_kind')
        save(self.cell / 'adapter/DRY_CHECK.json', self.adapter)
        self.readiness = dict(status='REAL_HOST_DRY_CHECK_PASS', host_id=self.host, job_id=self.assignment['job_id'],
            source=self.assignment['source_manifest']['commit'], request_sha256=self.assignment['request_sha256'],
            input_hashes=self.assignment['input_hashes'], route=self.resolved['route'],
            scientific_evaluation_calls=0, queue_operations=0,
            resolved_request_sha256=H._sha(self.cell / 'RESOLVED_REQUEST.json'),
            command=[self.profile['interpreter'], '-B', 'adapter.py', '--mode', 'dry-check'],
            adapter_result=self.adapter, python_version='3.12.saved-observation')
        save(self.cell / 'READINESS.json', self.readiness)
        self.facts = dict(python=self.profile['interpreter'], python_version=self.readiness['python_version'],
            source_head=dict(returncode=0, stdout='a' * 40 + '\n'),
            source_files={'src/kernel.py': dict(expected='d' * 64, actual='d' * 64, match=True)},
            route_views={'b' * 40: dict(path='/owned/fixed-route', realpath='/owned/fixed-route',
                head=dict(returncode=0, stdout='b' * 40 + '\n'), tracked_source_diff=dict(returncode=0, stdout=''))})
        save(self.root / 'HOST_FACTS.json', self.facts)
        self.descriptor = dict(kind='received', root=str(self.root),
            manifest=str(self.base / 'TRANSFER_MANIFEST.json'), receipt=str(self.base / 'RETURN_RECEIPT.json'),
            manifest_root=str(self.base / 'evidence'))
        self.receive()

    def receive(self):
        manifest = {path.relative_to(self.base / 'evidence').as_posix(): H._sha(path)
                    for path in self.root.rglob('*') if path.is_file()}
        save(Path(self.descriptor['manifest']), manifest)
        save(Path(self.descriptor['receipt']), dict(transfer_manifest_sha256=H._sha(self.descriptor['manifest']),
            verified_transferred_files=len(manifest)))

    def derive(self, assignment=None, profile=None, descriptor=None):
        policy = H.derive_host_eligibility([assignment or self.assignment], [profile or self.profile],
            {self.host: descriptor or self.descriptor})
        return policy['jobs'][self.assignment['job_id']]

    def test_arbitrary_method_and_candidate_are_eligible_from_bound_evidence(self):
        row = self.derive()
        self.assertEqual(row['eligible_host_ids'], [self.host])
        self.assertEqual(row['hosts'][self.host]['status'], 'PASS')
        self.assertTrue(all(v == 'MATCH' for v in row['hosts'][self.host]['dimensions'].values()))

    def test_same_configuration_with_new_xy_distribution_is_unchecked(self):
        changed = deepcopy(self.assignment)
        changed['request']['N_Y'][0] += 1
        changed['request_sha256'] = digest(changed['request'])
        self.assertEqual(self.derive(changed)['hosts'][self.host]['status'], 'UNCHECKED')

    def test_changed_source_or_input_hash_is_unchecked(self):
        for part in ('source', 'input'):
            changed = deepcopy(self.assignment)
            if part == 'source':
                changed['source_manifest']['files']['src/kernel.py'] = 'f' * 64
            else:
                changed['input_hashes'][0]['sha256'] = 'f' * 64
            self.assertEqual(self.derive(changed)['hosts'][self.host]['status'], 'UNCHECKED')

    def test_changed_route_or_runtime_requirement_is_unchecked(self):
        changed = deepcopy(self.profile)
        changed['route_views']['b' * 40] = '/owned/new-route'
        self.assertEqual(self.derive(profile=changed)['hosts'][self.host]['status'], 'UNCHECKED')
        changed = deepcopy(self.profile)
        changed['runtime'] = dict(version='3.12.changed-in-place')
        self.assertEqual(self.derive(profile=changed)['hosts'][self.host]['status'], 'UNCHECKED')

    def test_bound_failure_remains_fail(self):
        failed = deepcopy(self.readiness)
        failed.update(status='REAL_HOST_DRY_CHECK_MISSING_OR_INVALID_INPUT', error='fixed runtime contract failed')
        save(self.cell / 'READINESS.json', failed)
        self.receive()
        row = self.derive()
        self.assertEqual(row['eligible_host_ids'], [])
        self.assertEqual(row['hosts'][self.host]['status'], 'FAIL')
        self.assertIn('fixed runtime contract failed', row['hosts'][self.host]['reasons'])

    def test_missing_receipt_or_resolved_request_is_unchecked(self):
        Path(self.descriptor['receipt']).unlink()
        self.assertEqual(self.derive()['hosts'][self.host]['status'], 'UNCHECKED')
        self.receive()
        (self.cell / 'RESOLVED_REQUEST.json').unlink()
        self.assertEqual(self.derive()['hosts'][self.host]['status'], 'UNCHECKED')

    def test_tampered_received_readiness_is_unchecked(self):
        changed = deepcopy(self.readiness)
        changed['source'] = 'f' * 40
        save(self.cell / 'READINESS.json', changed)
        self.assertEqual(self.derive()['hosts'][self.host]['status'], 'UNCHECKED')

    def test_local_checker_version_does_not_replace_adapter_runtime(self):
        runtime = dict(executable=self.profile['interpreter'], version='3.11.adapter-actual')
        raw = dict(profile=self.profile, runtime=runtime, source=self.assignment['source_manifest'],
            source_head='a' * 40, input_hashes=self.assignment['input_hashes'],
            request_sha256=self.assignment['request_sha256'],
            views=[dict(path='/owned/fixed-route', head='b' * 40, tracked_changes={})])
        cache = self.base / 'DRY_CACHE.json'
        save(cache, dict(methods={'anything': dict(result_sha256=H._sha(self.cell / 'READINESS.json'), fingerprint=raw)}))
        descriptor = dict(kind='local', root=str(self.root), runtime_cache=str(cache))
        row = self.derive(descriptor=descriptor)
        self.assertEqual(row['hosts'][self.host]['status'], 'PASS')
        self.assertEqual(row['hosts'][self.host]['current_host_fingerprint']['runtime']['version'], '3.11.adapter-actual')
        operational = dict(self.profile, work_dir='/new/owned/spool', slots=12)
        relocated = self.derive(profile=operational, descriptor=descriptor)
        self.assertEqual(relocated['hosts'][self.host]['status'], 'PASS')
        self.assertEqual(row['hosts'][self.host]['current_host_fingerprint_sha256'],
                         relocated['hosts'][self.host]['current_host_fingerprint_sha256'])

    def test_actual_runtime_version_change_at_same_path_changes_poll_fingerprint(self):
        raw = dict(profile=self.profile, runtime=dict(executable=self.profile['interpreter'], version='3.11.first'),
            source=self.assignment['source_manifest'], source_head='a' * 40,
            input_hashes=self.assignment['input_hashes'], request_sha256=self.assignment['request_sha256'],
            views=[dict(path='/owned/fixed-route', head='b' * 40, tracked_changes={})])
        with patch('trottertracks.rpe_plan_readiness.dry_check_fingerprint', return_value=raw):
            first = H.current_host_fingerprint(self.assignment, self.profile)
        changed = deepcopy(raw)
        changed['runtime']['version'] = '3.11.changed'
        with patch('trottertracks.rpe_plan_readiness.dry_check_fingerprint', return_value=changed):
            second = H.current_host_fingerprint(self.assignment, self.profile)
        self.assertNotEqual(digest(first), digest(second))

    def test_missing_or_malformed_evidence_is_unchecked(self):
        row = H.derive_host_eligibility([self.assignment], [self.profile], {})['jobs'][self.assignment['job_id']]
        self.assertEqual(row['hosts'][self.host]['status'], 'UNCHECKED')
        save(self.cell / 'READINESS.json', {'malformed': True})
        self.receive()
        self.assertEqual(self.derive()['hosts'][self.host]['status'], 'UNCHECKED')


if __name__ == '__main__':
    unittest.main()
