"""Remote cache proofs reuse ended same-evaluation CAS without deleting data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('reclaim_storage', ROOT/'scripts/reclaim_rpe_worker_storage.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class RemoteCacheProof(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state = Path(self.temporary.name)
        self.current = self.state/'current'
        put(self.current/'CONTROLLER_CONFIG.json', dict(state_directory=str(self.state)))
        (self.current/'distributed').mkdir()
        with sqlite3.connect(self.current/'distributed/QUEUE.sqlite3') as db:
            db.execute('CREATE TABLE jobs (host_id TEXT, state TEXT, request TEXT)')
        self.index = {'current': dict(run_directory=str(self.current), evaluation_identity='E1')}
        put(self.state/'RUNS.json', self.index)

    def owner(self, name, *, evaluation='E1', finished=True, exit_invocation='I1'):
        owner = self.state/name
        put(owner/'LAUNCH.json', dict(invocation_id='I1'))
        if finished:
            put(owner/'EXIT.json', dict(invocation_id=exit_invocation))
        self.index[name] = dict(run_directory=str(owner), evaluation_identity=evaluation)
        put(self.state/'RUNS.json', self.index)
        return owner

    def blob(self, owner, content):
        sha = hashlib.sha256(content).hexdigest()
        path = owner/'distributed/artifacts'/sha[:2]/sha
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return sha, len(content), path

    def job(self, state, payload, host='server2'):
        with sqlite3.connect(self.current/'distributed/QUEUE.sqlite3') as db:
            db.execute('INSERT INTO jobs VALUES (?,?,?)', (host, state, json.dumps(payload)))

    def test_current_and_ended_same_evaluation_are_byte_verified(self):
        prior = self.owner('prior')
        a, na, pa = self.blob(self.current, b'current')
        b, nb, pb = self.blob(prior, b'previous completed input')
        bad, nbad, pbad = self.blob(prior, b'corrupted')
        pbad.write_bytes(b'incorrect')  # Same byte count, wrong SHA.
        wrongsize, ns, ps = self.blob(prior, b'wrong inventory size')
        linksha, nl, link = self.blob(prior, b'linked')
        target = prior/'external'; target.write_bytes(link.read_bytes())
        link.unlink(); link.symlink_to(target)
        inventory = {a: na, b: nb, bad: nbad, wrongsize: ns+1, linksha: nl}
        self.assertEqual(R.verified_remote_duplicates(self.current, 'server2', inventory, 1000), {a: na, b: nb})
        self.assertEqual(pa.read_bytes(), b'current')
        self.assertEqual(pb.read_bytes(), b'previous completed input')

    def test_live_foreign_and_unmatched_exit_owners_are_excluded(self):
        inventory = {}
        for name, options in [('live', dict(finished=False)),
                              ('foreign', dict(evaluation='E2')),
                              ('resumed', dict(exit_invocation='OLD'))]:
            sha, n, _ = self.blob(self.owner(name, **options), name.encode())
            inventory[sha] = n
        self.assertEqual(R.verified_remote_duplicates(self.current, 'server2', inventory, 1000), {})

    def test_queued_assigned_and_pending_references_are_preserved(self):
        prior = self.owner('prior')
        inventory = {}
        for state in ('QUEUED', 'ASSIGNED'):
            sha, n, _ = self.blob(prior, state.encode())
            inventory[sha] = n
            self.job(state, {'nested': [dict(blob=sha)]}, host='other-worker')
        sha, n, _ = self.blob(prior, b'pending')
        inventory[sha] = n
        put(self.current/'distributed/PREPARATION_PENDING.json', {'inputs': [sha]})
        self.assertEqual(R.verified_remote_duplicates(self.current, 'server2', inventory, 1000), {})

    def test_active_remote_execution_disables_proofs(self):
        sha, n, _ = self.blob(self.owner('prior'), b'input')
        self.job('RUNNING', {}, host='server2')
        self.assertEqual(R.verified_remote_duplicates(self.current, 'server2', {sha: n}, 1000), {})


if __name__ == '__main__':
    unittest.main()
