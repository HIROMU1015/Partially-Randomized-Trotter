"""Exercise the live-worker and interrupted metadata compaction paths."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from trottertracks import rpe_local_pack_maintenance as maintenance
from trottertracks.rpe_q3_store import JobStore


class LocalPackMaintenance(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.run = root/'run'
        self.run.mkdir()
        self.distributed = self.run/'distributed'
        self.distributed.mkdir()
        JobStore(self.distributed/'QUEUE.sqlite3')
        (root/'REGISTRY.lock').touch()
        self.registry=root/'RUNS.json'
        self.write(self.registry, {'run':dict(run_directory=str(self.run))})
        self.write(root/'CURRENT_TARGET.json', dict(run_directory=str(root/'unrelated-selected-run')))
        self.write(self.run/'CONTROLLER_CONFIG.json', dict(state_directory=str(root)))
        self.spool = root/'spool'
        self.spool.mkdir()
        self.worker_lock = self.spool/'WORKER.lock'
        self.worker_lock.touch()
        self.sample = root/'STORAGE_USAGE.json'
        self.sample.with_suffix('.lock').touch()
        cache = root/'preparation-metadata-cache'
        self.repo = cache/'bundle'/'objects.git'
        self.repo.parent.mkdir(parents=True)
        (self.repo.parent/'IMPORT.lock').touch()
        self.git('init', '--bare', '-q')
        self.pack = self.repo/'objects/pack'
        self.object_data = b'FIXED METADATA INPUT\n' * 1000
        self.oid = self.git('hash-object', '-w', '--stdin', input=self.object_data).strip()
        self.git('update-ref', 'refs/tags/immutable-input', self.oid.decode())
        self.git('-c', 'repack.writeBitmaps=false', 'repack', '-adq')
        for path in list(self.pack.iterdir()):
            shutil.copy2(path, path.with_name('pack-' + 'd'*40 + path.suffix))
        partial = self.pack/'tmp_pack_retained'
        partial.write_bytes(b'unindexed pack marker')
        partial.chmod(0o444)
        self.policy = dict(limit_bytes=50*1024**3, roots=[str(cache), str(self.spool)],
                           sample_path=str(self.sample))
        self.profile = dict(host_id='local', work_dir=str(self.spool), storage_limit=self.policy)
        self.write(self.distributed/'LOCAL_WORKER.json', self.profile)
        self.write(self.distributed/'LOCAL_DAEMON.json',
                   dict(profile=str(self.distributed/'LOCAL_WORKER.json'), pid=30_000))
        self.pending = root/'metadata-repack/pending'
        self.objects = maintenance._packed_objects(self.pack)
        self.soft = dict(reclaim_pending=True)
        self.idle = dict(reclaim_pending=False)

    def git(self, *args, input=None):
        return subprocess.check_output(['git', '-C', str(self.repo.parent), '--git-dir',
                                        str(self.repo), *args], input=input, stderr=subprocess.PIPE)

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data) + '\n')

    def hard_hold(self):
        report = dict(accepting_new_jobs=False, reason='STORAGE_LIMIT_REACHED')
        identity = {key: self.policy[key] for key in ('limit_bytes', 'roots')}
        self.write(self.sample, dict(policy=identity, sampled_epoch=time.time(), report=report))
        self.write(self.run/'PROGRESS.jsonl', dict(event='QUEUE_WAIT_STATE',
            hosts=dict(local=dict(fresh_free_slots=0, storage=report))))

    def compact(self, observation):
        # Keep the fixture small while exercising the real Git pack code and locks.
        read_bytes = Path.read_bytes

        def read_process_arguments(path):
            if path == Path('/proc/30000/cmdline'):
                return b'/fixed/run_rpe_q3_worker.py\0' + str(self.distributed/'LOCAL_WORKER.json').encode() + b'\0'
            return read_bytes(path)

        with patch.object(maintenance, '_choose_candidate', return_value=[self.pack]), \
             patch.object(maintenance, '_used_bytes', return_value=self.policy['limit_bytes']), \
             patch.object(Path, 'read_bytes', read_process_arguments):
            return maintenance.compact_local_metadata(self.run, self.profile, observation)

    def assert_objects_preserved(self):
        self.assertEqual(maintenance._packed_objects(self.pack), self.objects)
        self.assertEqual(self.git('cat-file', 'blob', 'refs/tags/immutable-input'), self.object_data)
        self.assertEqual((self.pack/'tmp_pack_retained').read_bytes(), b'unindexed pack marker')
        self.assertFalse(self.pending.exists())

    def test_stage_while_polling_and_apply_during_quota_hold(self):
        self.assertEqual(self.compact(self.idle), {})
        with self.worker_lock.open('rb') as worker:
            fcntl.flock(worker, fcntl.LOCK_EX)
            self.hard_hold()
            (self.run/'PROGRESS.jsonl').write_text('{"event":"QUEUE_WAIT_STATE"}\n')
            result = self.compact(self.soft)
            self.assertEqual(result['metadata_compaction'], 'PACK_SWITCH_WAIT_LOCAL_IDLE_OR_STORAGE_HOLD')
            self.assertTrue(self.pending.is_dir())
            attempt = self.spool/'job/owned'
            attempt.mkdir(parents=True)
            self.write(attempt/'ASSIGNMENT.json', {})
            self.write(attempt/'WORKER_STATE.json', dict(state='RUNNING'))
            self.hard_hold()
            self.assertEqual(self.compact(self.soft)['metadata_compaction'],
                             'PACK_SWITCH_WAIT_OWNED_ATTEMPTS')
            self.write(attempt/'WORKER_STATE.json', dict(state='UNKNOWN'))
            self.write(attempt/'PROCESS_RETURN.json', dict(returncode=1))
            self.hard_hold()
            result = self.compact(self.soft)
            self.assertGreater(result['compacted_metadata_bytes'], 0)
            self.assertTrue((attempt/'PROCESS_RETURN.json').is_file())
            with self.worker_lock.open('rb') as other:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assert_objects_preserved()

    def test_recovers_interrupted_switch_below_proactive_threshold(self):
        with self.worker_lock.open('rb') as worker:
            fcntl.flock(worker, fcntl.LOCK_EX)
            self.hard_hold()
            (self.run/'PROGRESS.jsonl').write_text('{"event":"QUEUE_WAIT_STATE"}\n')
            self.compact(self.soft)
        staged = self.pending/'objects.git/objects/pack'
        self.assertTrue(maintenance._exchange_pack_directories(self.pack, staged))
        result = self.compact(self.idle)
        self.assertEqual(result['metadata_compaction'], 'PACK_COMPACTED')
        self.assert_objects_preserved()

    def test_other_active_run_with_shared_spool_keeps_staged_pack_uninstalled(self):
        other=self.run.parent/'another-run'
        (other/'distributed').mkdir(parents=True)
        self.write(other/'distributed/LOCAL_WORKER.json', self.profile)
        self.write(other/'LAUNCH.json', dict(invocation_id='live'))
        self.write(self.registry, {'run':dict(run_directory=str(self.run)),
                                   'another-run':dict(run_directory=str(other))})
        self.assertEqual(self.compact(self.soft)['metadata_compaction'], 'PACK_SWITCH_WAIT_LOCAL_OWNER')
        self.assertTrue(self.pending.is_dir())
        self.write(other/'EXIT.json', dict(invocation_id='live'))
        self.assertEqual(self.compact(self.soft)['metadata_compaction'], 'PACK_COMPACTED')
        self.assert_objects_preserved()
