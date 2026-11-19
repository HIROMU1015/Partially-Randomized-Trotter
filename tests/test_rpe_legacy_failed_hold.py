"""Imported terminal local failures stay excluded without inventing outcomes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.rpe_q3_coordinator import digest
from trottertracks.rpe_search_continuation import verify_prior_q3_holds


class LegacyFailedHold(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root/'files'/'prior'
        self.run = self.root/'new'
        self.job_id = 'a'*20
        self.method = 'PR_DF'
        self.job = self.parent/'jobs'/self.method/self.job_id
        self.original = '/synthetic/source-host/original-run'
        self.problem = dict(problem=dict(system='ARTIFICIAL', hamiltonian_data=dict(
            snapshot=dict(path='/new/location/input.npz', sha256='bound-input'))),
            rpe=dict(M=0), target=dict(energy_ha=0), accuracy=dict(threshold='1/100'))
        old_problem = deepcopy(self.problem)
        old_problem['problem']['hamiltonian_data']['snapshot']['path'] = '/old/location/input.npz'
        self.request = dict(request_id=self.job_id, method=self.method, configuration_id='fixed', N_X=[2], N_Y=[3])
        self.write(self.parent/'FIXED_INPUTS.json', dict(problem=old_problem))
        self.write(self.job/'REQUEST.json', self.request)
        self.write(self.job/'LAUNCH.json', dict(pid=42, kind='prdf_d2_split'))
        self.write(self.job/'STARTING.json', dict(parent_pid=24, automatic_retry=False))
        self.write(self.job/'PROCESS_RETURN.json', dict(returncode=1, signal=None, automatic_retry=False))
        names = ['REQUEST.json', 'LAUNCH.json', 'STARTING.json', 'PROCESS_RETURN.json', 'FIXED_INPUTS.json']
        hashes = {name: self.sha((self.parent if name == 'FIXED_INPUTS.json' else self.job)/name) for name in names}
        files = []
        for name, sha in hashes.items():
            relative = Path(name) if name == 'FIXED_INPUTS.json' else Path('jobs')/self.method/self.job_id/name
            files.append(dict(path=str(Path('files/prior')/relative), original=str(Path(self.original)/relative), sha256=sha))
        manifest = self.root/'MANIFEST.json'
        self.write(manifest, dict(roots=[dict(snapshot_root='files/prior', original_root=self.original)], files=files))
        check = self.root/'SOURCE_HOST_CHECK.json'
        self.write(check, dict(host_id='source-host', original_parent_directory=self.original, active_owner_count=0,
            checked_job_ids=[self.job_id], job_process_ids={self.job_id: dict(pid=42, parent_pid=24)}))
        self.hold = dict(kind='LEGACY_FAILED_RETURN', parent_directory=str(self.parent), method=self.method,
            job_id=self.job_id, host_id='source-host', request_sha256=digest(self.request), files_sha256=hashes,
            provenance_manifest=dict(path=str(manifest), sha256=self.sha(manifest)),
            liveness=dict(path=str(check), sha256=self.sha(check)))
        self.spec = dict(methods=[self.method], executor=dict(prior_run_dirs=[str(self.parent)], prior_q3_holds=[self.hold]))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def verify(self):
        return verify_prior_q3_holds(self.root, self.run, self.problem, self.spec)

    def test_terminal_receipt_and_relocated_inputs_keep_exact_allocation_held(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*.json')}
        result = self.verify()
        self.assertEqual(result[str(self.job)]['N_X'], [2])
        self.assertEqual(result[str(self.job)]['N_Y'], [3])
        self.assertEqual(result[str(self.job)]['reason'], 'LEGACY_FAILED_RETURN_NO_RETRY')
        self.assertFalse((self.job/'OUTCOME.json').exists())
        self.assertFalse((self.parent/'EXIT.json').exists())
        self.assertEqual({p: p.read_bytes() for p in self.root.rglob('*.json')}, before)

    def test_hold_never_allows_replay(self):
        self.hold['replay_exact'] = True
        with self.assertRaisesRegex(ValueError, 'cannot be replayed'):
            self.verify()

    def test_modified_receipt_or_completed_outcome_is_rejected(self):
        receipt = self.job/'PROCESS_RETURN.json'
        self.write(receipt, dict(returncode=0, automatic_retry=False))
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            self.verify()
        self.hold['files_sha256']['PROCESS_RETURN.json'] = self.sha(receipt)
        # Updating a supplied hash cannot rewrite the imported file inventory.
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            self.verify()

    def test_live_source_owner_is_rejected_even_with_matching_check_hash(self):
        p = Path(self.hold['liveness']['path'])
        value = json.loads(p.read_text())
        value['active_owner_count'] = 1
        self.write(p, value)
        self.hold['liveness']['sha256'] = self.sha(p)
        with self.assertRaisesRegex(ValueError, 'still owns an attempt'):
            self.verify()

    def test_different_model_or_fixed_input_hash_is_rejected(self):
        self.problem['problem']['hamiltonian_data']['snapshot']['sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'terminal identity differs'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
