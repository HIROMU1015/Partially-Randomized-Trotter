"""Artificial admission/storage only. No G7/F3 evaluation or sampling."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
M = importlib.import_module('trottertracks.algorithm_codesign.g6_w2c_b3_admission')
D = M.D
BASE = json.loads((ROOT/M.BASE_PATH).read_text())


def fixture(root):
    pins = dict(BASE['source_pins'])
    for name in (M.DRIVER, M.GUARD, M.ADMISSION, D.FASTPATH_SOURCE):
        pins[name] = M.digest((ROOT/name).read_bytes())
    paths = set(pins) | {M.BASE_PATH, BASE['saved_evidence']['path']}
    for name in paths:
        p = root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, p)
    record = root/'adoption.json'
    record.write_text('{"artificial_only":true,"package":"R1"}\n')
    ledger_dir = root/'ledger'
    ledger_dir.mkdir(mode=0o700)
    ledger = M.ClaimLedger.initialize(ledger_dir)
    release = '1'*40
    candidate = deepcopy(BASE)
    candidate.update(adopted=True, status='ARTIFICIAL_ONLY', source_pins=pins,
                     output_relative='artificial-output')
    candidate['decisions'] = {'angle': M.A6, 'seed': BASE['seed_proposal'],
        'schedule_and_output': BASE['schedule_proposal'], 'source_release': release,
        'external_admission': 'DURABLE_ATOMIC_R1_V1'}
    c = {'schema': 'b3_formal_admission_v1', 'content_adopted': True,
         'contract_grants_execution': False, 'retry': 0, 'resume': False, 'mandatory_STOP': True,
         'release': {'commit': release, 'f01_status': 'F01_INTEGRATED_REGRESSION_PASS',
                     'failure_provenance_regression_sha256': '2'*64},
         'adoption': {'adopted_package': 'R1', 'instruction_sha256': '3'*64,
                      'record_path': 'adoption.json', 'record_sha256': M.digest(record.read_bytes())},
         'candidate': candidate, 'source_pins': pins,
         'source_roles': {'science': sorted(set(pins)-{M.DRIVER,M.GUARD,M.ADMISSION}),
                          'driver': [M.DRIVER], 'guard': [M.GUARD], 'admission': [M.ADMISSION]},
         'input_pins': {'adoption.json': M.digest(record.read_bytes()),
                       BASE['saved_evidence']['path']: BASE['saved_evidence']['sha256']},
         'scope': 'ARTIFICIAL_SCOPE_ONE', 'claim_id': 'ARTIFICIAL_CLAIM_ONE'}
    auth = authorize(c, root, ledger)
    return c, ledger, auth


def authorize(c, root, ledger):
    binding = M.validate_r1(c, root, True)
    statement = root/'artificial-statement.txt'
    statement.write_bytes(b'ARTIFICIAL fixture. Not a user science approval.\n')
    auth_path = root/'artificial-authorization.json'
    raw = M.encoded({'schema': 'b3_independent_approval_v1', 'kind': 'ARTIFICIAL_NO_SCIENCE',
                     'binding': binding, 'statement_reference': str(statement),
                     'statement_sha256': M.digest(statement.read_bytes())})
    auth_path.write_bytes(raw)
    anchor = M.TrustAnchor('ARTIFICIAL_NO_SCIENCE', D.identity(c), c['release']['commit'],
                          str(auth_path), M.digest(raw), str(statement), M.digest(statement.read_bytes()),
                          str(ledger.directory), ledger.store_id)
    return M.PinnedApproval(anchor)


def attempt(args):
    root, c, store_id, anchor = args
    ledger = M.ClaimLedger(Path(root)/'ledger', store_id)
    port = M.ArtificialAdmission(c, root, ledger, M.PinnedApproval(anchor))
    try:
        port.admit_artificial(D.identity(c['candidate']))
        port.stop('ARTIFICIAL_RACE_SUCCESS')
        return True
    except (M.Denied, sqlite3.Error, OSError):
        return False


def die_at(root, c, store_id, anchor, point):
    ledger = M.ClaimLedger(Path(root)/'ledger', store_id)
    def fault(p):
        if p == point:
            os._exit(88)
    port = M.ArtificialAdmission(c, root, ledger, M.PinnedApproval(anchor), fault=fault)
    port.admit_artificial(D.identity(c['candidate']))


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.c, self.ledger, self.auth = fixture(self.root)

    def port(self, c=None, auth='default', fault=lambda p: None):
        return M.ArtificialAdmission(c or self.c, self.root, self.ledger,
                                    self.auth if auth == 'default' else auth, fault=fault)

    def admit(self, port=None, c=None):
        return (port or self.port(c)).admit_artificial(D.identity((c or self.c)['candidate']))

    def assert_spent(self):
        self.assertIsNotNone(self.ledger.row(self.c['scope']))
        with self.assertRaises((M.Denied, sqlite3.Error, OSError)):
            self.admit()

    def test_happy_path_and_stop(self):
        p = self.port()
        with p:
            receipt = self.admit(p)
            self.assertTrue(receipt['artificial_only'])
            self.assertEqual(self.ledger.row(self.c['scope'])[4], 'ADMITTED')
            self.assertFalse((self.root/'artificial-output').exists())
        self.assertEqual(self.ledger.row(self.c['scope'])[4], 'STOPPED')
        self.assert_spent()

    def test_artificial_cannot_be_boundary_port(self):
        p = self.port()
        self.assertFalse(hasattr(p, 'admit_once'))
        with self.assertRaises(D.Stop):
            D.Boundary().enter(self.c['candidate'], p)
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_production_defaults_to_no_authority(self):
        p = M.ProductionAdmission(self.c, self.root, self.ledger)
        with self.assertRaises(M.Denied):
            p.admit_once(D.identity(self.c['candidate']))
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_artificial_authority_cannot_authorize_production(self):
        p = M.ProductionAdmission(self.c, self.root, self.ledger, self.auth)
        with self.assertRaises(M.Denied):
            p.admit_once(D.identity(self.c['candidate']))
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_independent_anchor_kind_rejects_real_label_in_artificial(self):
        auth = M.PinnedApproval(replace(self.auth.anchor, kind='REAL_USER_SCIENCE_APPROVAL'))
        with self.assertRaises(M.Denied):
            self.admit(self.port(auth=auth))

    def test_missing_authorization(self):
        Path(self.auth.anchor.authorization_reference).unlink()
        with self.assertRaises(OSError):
            self.admit()

    def test_authorization_content_hash(self):
        Path(self.auth.anchor.authorization_reference).write_text('{"approved":true}')
        with self.assertRaisesRegex(M.Denied, 'AUTHORIZATION_CONTENT_HASH'):
            self.admit()

    def test_actual_statement_content_hash(self):
        Path(self.auth.anchor.statement_reference).write_text('changed artificial statement')
        with self.assertRaisesRegex(M.Denied, 'USER_STATEMENT_CONTENT_HASH'):
            self.admit()

    def test_approval_binding_even_with_matching_document_hash(self):
        p = Path(self.auth.anchor.authorization_reference)
        authdoc = json.loads(p.read_bytes());authdoc['binding']['scope'] = 'ARTIFICIAL_OTHER'
        p.write_bytes(M.encoded(authdoc))
        auth = M.PinnedApproval(replace(self.auth.anchor, authorization_sha256=M.digest(p.read_bytes())))
        with self.assertRaisesRegex(M.Denied, 'AUTHORIZATION_BINDING_MISMATCH'):
            self.admit(self.port(auth=auth))

    def test_driver_digest(self):
        with self.assertRaisesRegex(M.Denied, 'DRIVER_CONTRACT_DIGEST'):
            self.port().admit_artificial('f'*64)

    def test_ledger_identity(self):
        auth = M.PinnedApproval(replace(self.auth.anchor, ledger_store_id='unknown'))
        with self.assertRaisesRegex(M.Denied, 'TRUSTED_LEDGER_IDENTITY'):
            self.admit(self.port(auth=auth))

    def test_missing_ledger_not_recreated(self):
        self.ledger.path.unlink()
        with self.assertRaises(OSError):
            self.admit()
        self.assertFalse(self.ledger.path.exists())

    def test_corrupt_ledger_fails_closed(self):
        self.ledger.path.write_bytes(b'PARTIAL ARTIFICIAL DATABASE')
        with self.assertRaises(sqlite3.Error):
            self.admit()
        self.assertEqual(self.ledger.path.read_bytes(), b'PARTIAL ARTIFICIAL DATABASE')

    def test_no_reinitialize(self):
        with self.assertRaises(FileExistsError):
            M.ClaimLedger.initialize(self.ledger.directory)

    def test_scope_unique_thread_race(self):
        barrier = threading.Barrier(32)
        def run(i):
            barrier.wait()
            return attempt((str(self.root), self.c, self.ledger.store_id, self.auth.anchor))
        with ThreadPoolExecutor(max_workers=32) as pool:
            outcomes = list(pool.map(run, range(32)))
        self.assertEqual(sum(outcomes), 1)
        self.assert_spent()

    def test_scope_unique_owned_process_race(self):
        args = (str(self.root), self.c, self.ledger.store_id, self.auth.anchor)
        with mp.get_context('fork').Pool(32) as pool:
            outcomes = pool.map(attempt, [args]*32, chunksize=1)
        self.assertEqual(sum(outcomes), 1)
        self.assert_spent()

    def test_changed_digest_and_other_claim_cannot_reuse_scope(self):
        self.admit()
        c = deepcopy(self.c);c['claim_id'] = 'ARTIFICIAL_SECOND_CLAIM'
        auth = authorize(c, self.root, self.ledger)
        with self.assertRaises(sqlite3.IntegrityError):
            self.admit(self.port(c, auth), c)

    def test_same_claim_cannot_be_used_in_other_scope(self):
        self.admit()
        c = deepcopy(self.c);c['scope'] = 'ARTIFICIAL_OTHER_SCOPE'
        auth = authorize(c, self.root, self.ledger)
        with self.assertRaises(sqlite3.IntegrityError):
            self.admit(self.port(c, auth), c)

    def test_failure_before_reservation(self):
        def fault(point):
            if point == 'before_reserve':raise OSError('artificial before reserve')
        p = self.port(fault=fault)
        with self.assertRaises(OSError):self.admit(p)
        self.assertIsNone(self.ledger.row(self.c['scope']))
        with self.assertRaises(M.Denied):self.admit(p)

    def test_uncommitted_transaction_rolls_back_no_receipt(self):
        def fault(point):
            if point == 'insert_before_commit':raise OSError('artificial partial transaction')
        p = self.port(fault=fault)
        with self.assertRaises(OSError):self.admit(p)
        self.assertIsNone(self.ledger.row(self.c['scope']))
        with self.assertRaises(M.Denied):self.admit(p)

    def test_after_reservation_commit(self):
        def fault(point):
            if point == 'after_reserve_commit':raise OSError('artificial acknowledgement lost')
        with self.assertRaises(OSError):self.admit(self.port(fault=fault))
        self.assertEqual(self.ledger.row(self.c['scope'])[4], 'RESERVED')
        self.assert_spent()

    def test_partial_claim_detail_after_reservation(self):
        def fault(point):
            if point == 'partial_claim_detail_write':raise OSError('artificial partial detail')
        with self.assertRaises(OSError):self.admit(self.port(fault=fault))
        self.assert_spent()

    def test_before_output_creation_after_reservation(self):
        def fault(point):
            if point == 'before_output_creation':raise OSError('artificial output failure')
        with self.assertRaises(OSError):self.admit(self.port(fault=fault))
        self.assertFalse((self.root/'artificial-output').exists())
        self.assert_spent()

    def test_output_collision_before_reserve(self):
        (self.root/'artificial-output').mkdir()
        with self.assertRaisesRegex(M.Denied, 'OUTPUT_COLLISION'):self.admit()
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_output_collision_after_reserve(self):
        def fault(point):
            if point == 'before_binding_recheck':(self.root/'artificial-output').mkdir()
        with self.assertRaisesRegex(M.Denied, 'OUTPUT_COLLISION'):self.admit(self.port(fault=fault))
        self.assert_spent()

    def test_changed_binding_after_reserve(self):
        def fault(point):
            if point == 'before_binding_recheck':
                Path(self.auth.anchor.statement_reference).write_bytes(b'changed')
        with self.assertRaises(M.Denied):self.admit(self.port(fault=fault))
        self.assert_spent()

    def test_failure_and_stop_save_failures(self):
        def fault(point):
            if point in ('failure_record_save', 'stop_record_save'):raise OSError('artificial no disk')
        p = self.port(fault=fault);self.admit(p)
        result = p.stop('ARTIFICIAL_FAILURE')
        self.assertEqual(len(result['record_errors']), 2)
        self.assertEqual(self.ledger.row(self.c['scope'])[4], 'ADMITTED')
        self.assert_spent()

    def test_partial_stop_file_write_preserved(self):
        p = self.port();self.admit(p)
        real_write = os.write
        calls = []
        def write(fd, data):
            calls.append(1)
            if len(calls) > 1:raise OSError('artificial short control write')
            return real_write(fd, data[:7])
        with patch.object(M.os, 'write', write):
            result = p.stop('ARTIFICIAL_PARTIAL_WRITE')
        self.assertEqual(len(result['record_errors']), 1)
        self.assertEqual((self.ledger.directory/(p.nonce+'.STOP.json')).stat().st_size, 7)
        self.assert_spent()

    def test_owned_process_exit_after_durable_reservation(self):
        proc = mp.get_context('fork').Process(target=die_at, args=(str(self.root), self.c,
                  self.ledger.store_id, self.auth.anchor, 'after_reserve_commit'))
        proc.start();proc.join()
        self.assertEqual(proc.exitcode, 88)
        self.assertEqual(self.ledger.row(self.c['scope'])[4], 'RESERVED')
        self.assert_spent()

    def test_owned_process_exit_inside_uncommitted_transaction(self):
        proc = mp.get_context('fork').Process(target=die_at, args=(str(self.root), self.c,
                  self.ledger.store_id, self.auth.anchor, 'insert_before_commit'))
        proc.start();proc.join()
        self.assertEqual(proc.exitcode, 88)
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_stop_same_port_reentry(self):
        p = self.port();self.admit(p);p.stop()
        with self.assertRaises(M.Denied):self.admit(p)
        self.assert_spent()

    def test_context_failure_keeps_reservation(self):
        with self.assertRaises(OSError):
            with self.port() as p:
                self.admit(p)
                raise OSError('artificial transition before science')
        self.assertEqual(self.ledger.row(self.c['scope'])[4], 'STOPPED')
        self.assert_spent()

    def test_symlink_output_rejected(self):
        (self.root/'artificial-output').symlink_to(self.root/'absent')
        with self.assertRaisesRegex(M.Denied, 'PATH_SYMLINK'):self.admit()


    def test_partial_sql_detail_update_rolls_back_but_scope_spent(self):
        observed = []
        def fault(point):
            if point == 'detail_update_before_commit':
                observed.append(self.ledger.row(self.c['scope'])[4])
                raise OSError('artificial failure after SQL UPDATE before COMMIT')
        with self.assertRaises(OSError):self.admit(self.port(fault=fault))
        self.assertEqual(observed, ['RESERVED'])
        self.assert_spent()

    def test_owned_process_exit_during_detail_update(self):
        proc = mp.get_context('fork').Process(target=die_at, args=(str(self.root), self.c,
                  self.ledger.store_id, self.auth.anchor, 'detail_update_before_commit'))
        proc.start();proc.join()
        self.assertEqual(proc.exitcode, 88)
        self.assertEqual(self.ledger.row(self.c['scope'])[4], 'RESERVED')
        self.assert_spent()

    def test_no_science_modules_loaded(self):
        self.admit()
        for suffix in ('g7_generator', 'return_aggregation', 'g7_provider', 'g6_f3_contract',
                       'g6_f3_precision', 'g6_f3_generator'):
            self.assertNotIn('trottertracks.algorithm_codesign.'+suffix, sys.modules)


    def test_reservation_binds_approval_hash_and_store(self):
        self.admit()
        binding = json.loads(self.ledger.row(self.c['scope'])[2])
        self.assertEqual(binding['authorization']['sha256'], self.auth.anchor.authorization_sha256)
        self.assertEqual(binding['ledger_store_id'], self.ledger.store_id)

    def test_stop_save_is_not_retried(self):
        p = self.port();self.admit(p)
        with patch.object(M.os, 'write', side_effect=OSError('artificial save failure')) as write:
            first = p.stop();second = p.stop()
        self.assertEqual(write.call_count, 1)
        self.assertEqual(first, second)
        self.assert_spent()

    def test_directory_sync_failure_after_commit_burns_scope(self):
        with patch.object(M, 'sync_directory', side_effect=OSError('artificial directory sync failure')):
            with self.assertRaises(OSError):self.admit()
        self.assert_spent()

    def test_same_adapter_thread_race(self):
        p = self.port()
        def run(i):
            try:self.admit(p);return True
            except M.Denied:return False
        with ThreadPoolExecutor(max_workers=32) as pool:
            self.assertEqual(sum(pool.map(run, range(32))), 1)
        p.stop();self.assert_spent()

    def test_owned_process_exit_after_admitted_before_output(self):
        proc = mp.get_context('fork').Process(target=die_at, args=(str(self.root), self.c,
                  self.ledger.store_id, self.auth.anchor, 'before_output_creation'))
        proc.start();proc.join()
        self.assertEqual(proc.exitcode, 88)
        self.assertEqual(self.ledger.row(self.c['scope'])[4], 'ADMITTED')
        self.assert_spent()

    def test_execute_no_authority_cannot_import_science(self):
        port = M.ProductionAdmission(self.c, self.root, self.ledger)
        with patch.object(D, 'load_modules') as load, patch.object(D, 'load_saved') as saved:
            with self.assertRaises(M.Denied):
                D.execute(self.c['candidate'], root=self.root, output=self.root/'artificial-output', approval_port=port)
        load.assert_not_called();saved.assert_not_called()
        self.assertIsNone(self.ledger.row(self.c['scope']))

    def test_source_modified_between_reservation_and_entry(self):
        def fault(point):
            if point == 'before_binding_recheck':
                (self.root/M.DRIVER).write_bytes(b'ARTIFICIAL changed driver copy')
        with self.assertRaisesRegex(M.Denied, 'SOURCE_PINS_MISMATCH'):
            self.admit(self.port(fault=fault))
        self.assert_spent()


# Each invalid binding is an independent test/fixture; authority stays separately pinned.
def invalid_test(edit):
    def test(self):
        c = deepcopy(self.c);edit(c)
        with self.assertRaises((M.Denied, ValueError, KeyError)):
            self.admit(self.port(c), c)
        self.assertIsNone(self.ledger.row(self.c['scope']))
    return test


def set_nested(*path, value):
    def edit(c):
        target = c
        for key in path[:-1]:target = target[key]
        target[path[-1]] = value
    return edit


INVALID = {
    'formal_digest': set_nested('candidate', 'status', value='ARTIFICIAL_EDITED'),
    'unadopted': set_nested('content_adopted', value=False),
    'contract_self_approves': set_nested('contract_grants_execution', value=True),
    'no_final_release': set_nested('release', 'commit', value=None),
    'f01_pending': set_nested('release', 'f01_status', value='F01_INTEGRATION_PENDING'),
    'no_f01_regression': set_nested('release', 'failure_provenance_regression_sha256', value=None),
    'source_hash': set_nested('source_pins', next(iter(BASE['source_pins'])), value='0'*64),
    'input_hash': set_nested('input_pins', BASE['saved_evidence']['path'], value='0'*64),
    'driver_hash': set_nested('source_pins', M.DRIVER, value='0'*64),
    'guard_hash': set_nested('source_pins', M.GUARD, value='0'*64),
    'admission_hash': set_nested('source_pins', M.ADMISSION, value='0'*64),
    'input_value': set_nested('candidate', 'input', 'x', value='1/2'),
    'policy': set_nested('candidate', 'policy', 'root_bits', value=128),
    'angle': set_nested('candidate', 'decisions', 'angle', 'K_phi', value=64),
    'seed': set_nested('candidate', 'decisions', 'seed', 'namespace', value='OLD_NAMESPACE'),
    'cell_n': set_nested('candidate', 'cells', 0, 'N', value=1),
    'call_budget': set_nested('candidate', 'call_budget', 'total_sampling_calls', value=31711104),
    'schedule': set_nested('candidate', 'decisions', 'schedule_and_output', 'flush_rows', value=1),
    'output_identity': set_nested('candidate', 'output_relative', value='different-output'),
    'output_escape': set_nested('candidate', 'output_relative', value='../outside'),
    'claim_identity': set_nested('claim_id', value='ARTIFICIAL_OTHER'),
    'scope_identity': set_nested('scope', value='ARTIFICIAL_OTHER'),
    'retry': set_nested('retry', value=1),
    'resume': set_nested('resume', value=True),
    'no_stop': set_nested('mandatory_STOP', value=False),
    'adoption_record': set_nested('adoption', 'record_sha256', value='0'*64),
}
for name, edit in INVALID.items():
    setattr(Tests, 'test_reject_'+name, invalid_test(edit))

if __name__ == '__main__':
    unittest.main()
