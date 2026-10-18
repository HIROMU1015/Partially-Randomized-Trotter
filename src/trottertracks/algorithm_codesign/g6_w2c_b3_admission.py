"""R1 admission implementation; no launcher, authority provisioning, or science calls.

The trust anchor must come from independent verification of an actual future user
science approval, never from the execution candidate. This release supplies none.
ArtificialAdmission exercises the same storage/binding engine but cannot produce
an old Boundary-compatible science receipt.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
import uuid

from . import g6_w2c_b3_driver as D

BASE_PATH = 'docs/research/track_b_g6_w2c_b3_driver_preparation_20261019/execution_contract_candidate.json'
BASE_SHA = 'dba9ed5b85fbb0a6599baf248a392aa8d3be30f8e816f1956a48563a49fd577c'
OLD_DRIVER_SHA = 'd7947af16baccbd3e0707fa9b4ffcd26650a6569eb6606ee3c08ec443d1dd7ee'
DRIVER = 'src/trottertracks/algorithm_codesign/g6_w2c_b3_driver.py'
GUARD = 'src/trottertracks/algorithm_codesign/g6_w2c_b3_readiness.py'
ADMISSION = 'src/trottertracks/algorithm_codesign/g6_w2c_b3_admission.py'
A6 = {'tau': '1/1000000', 'K_phi': 32, 'J': 32,
      'mode': 'MATHEMATICAL_ENCLOSURE_ONLY_NO_PHYSICAL_CHANGE'}
SCENARIO = 'B3_CORRECTNESS_AND_FIXED_FULL_N_TWO_PASS'
HEX = re.compile(r'[0-9a-f]{64}\Z')


class Denied(RuntimeError):
    """Terminal admission failure; no automatic retry or resume."""


def require(condition, message):
    if not condition:
        raise Denied(message)


def digest(data):
    return sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def private_directory(path):
    p = Path(path)
    require(p.is_absolute() and p == p.resolve(), 'NONCANONICAL_OR_SYMLINK_DIRECTORY')
    s = p.stat()
    require(stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid() and
            stat.S_IMODE(s.st_mode) == 0o700, 'PRIVATE_OWNED_DIRECTORY_REQUIRED')
    return p


def exact_path(root, relative):
    p = Path(relative)
    require(type(relative) is str and relative and not p.is_absolute() and
            '..' not in p.parts and str(p) == relative, 'INVALID_RELATIVE_PATH')
    result = root / p
    require(result == result.resolve() and result.is_relative_to(root), 'PATH_SYMLINK_OR_ESCAPE')
    return result


def read_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1, 'REGULAR_UNLINKED_ALIAS_REQUIRED')
        with os.fdopen(fd, 'rb', closefd=False) as f:
            return f.read()
    finally:
        os.close(fd)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ClaimLedger:
    """One local SQLite DB; no implicit creation/reinitialization on admission.

    RESERVED is already irrevocably spent. EXTRA + DELETE rollback journal, with
    SQLite's documented functioning local locking/fsync/storage assumptions.
    Missing/corrupt/unknown store fails closed. Never delete rows or roll back a
    successfully committed reservation. Initialization is a separate admin step.
    """
    @classmethod
    def initialize(cls, directory):
        directory = private_directory(directory)
        p = directory / 'claims.sqlite3'
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        store_id = str(uuid.uuid4())
        with sqlite3.connect(p) as db:
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('PRAGMA synchronous=EXTRA')
            db.execute('CREATE TABLE metadata (store_id TEXT PRIMARY KEY, version INTEGER NOT NULL)')
            db.execute('INSERT INTO metadata VALUES (?,1)', (store_id,))
            db.execute('''CREATE TABLE claims (
                scope TEXT PRIMARY KEY, claim_id TEXT UNIQUE NOT NULL,
                contract_digest TEXT NOT NULL, binding TEXT NOT NULL,
                nonce TEXT UNIQUE NOT NULL, status TEXT NOT NULL
                    CHECK(status IN ('RESERVED','ADMITTED','STOPPED')),
                owner_pid INTEGER NOT NULL, stop_record TEXT)''')
            db.commit()
        sync_directory(directory)
        return cls(directory, store_id)

    def __init__(self, directory, store_id):
        self.directory = private_directory(directory)
        self.path, self.store_id = self.directory / 'claims.sqlite3', store_id

    def connect(self):
        private_directory(self.directory)
        s = self.path.lstat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == os.getuid() and
                stat.S_IMODE(s.st_mode) == 0o600, 'UNSAFE_LEDGER_FILE')
        db = sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, timeout=30,
                             isolation_level=None)
        try:
            require(db.execute('PRAGMA journal_mode').fetchone()[0] == 'delete', 'JOURNAL_POLICY')
            db.execute('PRAGMA synchronous=EXTRA')
            require(db.execute('PRAGMA synchronous').fetchone()[0] == 3, 'SYNC_POLICY')
            require(db.execute('SELECT store_id,version FROM metadata').fetchall() ==
                    [(self.store_id, 1)], 'STORE_IDENTITY_MISMATCH')
            require(db.execute('PRAGMA quick_check').fetchall() == [('ok',)], 'LEDGER_INTEGRITY')
            return db
        except BaseException:
            db.close()
            raise

    def reserve(self, binding, fault=lambda point: None):
        fault('before_reserve')
        db = self.connect()
        nonce = str(uuid.uuid4())
        try:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO claims VALUES (?,?,?,?,?,\'RESERVED\',?,NULL)',
                       (binding['scope'], binding['claim_id'], binding['formal_digest'],
                        encoded(binding).decode(), nonce, os.getpid()))
            fault('insert_before_commit')
            db.commit()  # EXTRA syncs the rollback-journal deletion directory too.
            sync_directory(self.directory)  # No receipt if this additional sync fails.
            fault('after_reserve_commit')
            return nonce
        except BaseException:
            if db.in_transaction:
                db.rollback()  # Only uncommitted changes; no permission has been returned.
            raise
        finally:
            db.close()

    def transition(self, scope, nonce, old, new, record=None, fault=lambda point: None):
        db = self.connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            cursor = db.execute('UPDATE claims SET status=?,stop_record=? WHERE scope=? AND nonce=? AND status=?',
                                (new, None if record is None else encoded(record).decode(), scope, nonce, old))
            require(cursor.rowcount == 1, 'RESERVATION_STATE_MISMATCH')
            fault('detail_update_before_commit')
            db.commit()
        finally:
            if db.in_transaction:
                db.rollback()
            db.close()

    def row(self, scope):
        db = self.connect()
        try:
            return db.execute('SELECT claim_id,contract_digest,binding,nonce,status,stop_record FROM claims WHERE scope=?',
                              (scope,)).fetchone()
        finally:
            db.close()


@dataclass(frozen=True)
class TrustAnchor:
    """Trusted host configuration, outside the editable contract/CLI/environment.

    Creating a REAL_USER_SCIENCE_APPROVAL anchor requires independently verifying
    the actual user statement first. This module neither issues nor discovers it.
    The release, formal contract, store location/ID and authorization bytes are all
    externally pinned. Python callers able to replace code/anchors are trusted;
    this is not a security sandbox against arbitrary code running as the owner.
    """
    kind: str
    formal_digest: str
    release_commit: str
    authorization_reference: str
    authorization_sha256: str
    statement_reference: str
    statement_sha256: str
    ledger_directory: str
    ledger_store_id: str


class PinnedApproval:
    def __init__(self, anchor):
        require(type(anchor) is TrustAnchor, 'INDEPENDENT_TRUST_ANCHOR_REQUIRED')
        self.anchor = anchor

    def verify(self, binding, artificial):
        a = self.anchor
        require(a.kind == ('ARTIFICIAL_NO_SCIENCE' if artificial else 'REAL_USER_SCIENCE_APPROVAL'),
                'INDEPENDENT_SCIENCE_APPROVAL_REQUIRED')
        require(a.formal_digest == binding['formal_digest'] and
                a.release_commit == binding['release_commit'], 'APPROVAL_RELEASE_OR_CONTRACT_MISMATCH')
        raw = read_regular(Path(a.authorization_reference))
        require(digest(raw) == a.authorization_sha256, 'AUTHORIZATION_CONTENT_HASH')
        statement = read_regular(Path(a.statement_reference))
        require(digest(statement) == a.statement_sha256 and statement, 'USER_STATEMENT_CONTENT_HASH')
        auth = json.loads(raw)
        require(auth == {'schema': 'b3_independent_approval_v1', 'kind': a.kind,
                         'binding': binding, 'statement_reference': a.statement_reference,
                         'statement_sha256': a.statement_sha256}, 'AUTHORIZATION_BINDING_MISMATCH')
        return {'reference': a.authorization_reference, 'sha256': a.authorization_sha256,
                'statement_reference': a.statement_reference, 'statement_sha256': a.statement_sha256}


def validate_r1(contract, root, artificial):
    require(contract.get('schema') == 'b3_formal_admission_v1', 'FORMAL_CONTRACT_REQUIRED')
    require(contract.get('content_adopted') is True and contract.get('contract_grants_execution') is False,
            'R1_ADOPTION_REQUIRED_CONTRACT_IS_NOT_AUTHORITY')
    require(contract.get('retry') == 0 and type(contract['retry']) is int and
            contract.get('resume') is False and contract.get('mandatory_STOP') is True, 'ONE_SHOT_REQUIRED')
    release = contract['release']
    require(re.fullmatch('[0-9a-f]{40}', release.get('commit') or '') and
            release.get('f01_status') == 'F01_INTEGRATED_REGRESSION_PASS' and
            HEX.fullmatch(release.get('failure_provenance_regression_sha256') or ''), 'F01_INTEGRATION_PENDING')
    require(contract['adoption']['adopted_package'] == 'R1' and
            HEX.fullmatch(contract['adoption']['instruction_sha256']), 'ADOPTION_IDENTITY_REQUIRED')
    for group in ('source_pins', 'input_pins'):
        require(type(contract[group]) is dict and contract[group], 'PINS_REQUIRED')
        for name, pin in contract[group].items():
            require(type(pin) is str and HEX.fullmatch(pin), 'FULL_SHA256_REQUIRED')
            require(digest(read_regular(exact_path(root, name))) == pin, group.upper()+'_MISMATCH:'+name)
    ap = contract['adoption']['record_path']
    require(contract['input_pins'].get(ap) == contract['adoption']['record_sha256'], 'ADOPTION_RECORD_PIN')
    base_data = read_regular(exact_path(root, BASE_PATH))
    require(digest(base_data) == BASE_SHA, 'FIXED_BASE_CONTRACT_SHA')
    base = json.loads(base_data)
    candidate = contract['candidate']
    mutable = {'adopted', 'status', 'decisions', 'source_pins', 'output_relative'}
    require(set(candidate) == set(base), 'CANDIDATE_FIELDS_CHANGED')
    for key in set(base) - mutable:
        require(candidate[key] == base[key], 'R1_FIXED_FIELD:'+key)
    require(candidate['adopted'] is True and candidate['decisions'] == {
        'angle': A6, 'seed': base['seed_proposal'], 'schedule_and_output': base['schedule_proposal'],
        'source_release': release['commit'], 'external_admission': 'DURABLE_ATOMIC_R1_V1'}, 'R1_POLICY_MISMATCH')
    require(candidate['source_pins'] == contract['source_pins'], 'DRIVER_PIN_SET_MISMATCH')
    roles = contract['source_roles']
    require(set(roles) == {'science', 'driver', 'guard', 'admission'} and
            roles['driver'] == [DRIVER] and roles['guard'] == [GUARD] and roles['admission'] == [ADMISSION] and
            set().union(*(set(v) for v in roles.values())) == set(contract['source_pins']), 'SOURCE_ROLE_COVERAGE')
    require(set(base['source_pins']) - {DRIVER} <= set(roles['science']), 'SCIENCE_PINS_MISSING')
    for name, pin in base['source_pins'].items():
        if name != DRIVER:
            require(contract['source_pins'].get(name) == pin, 'FIXED_SCIENCE_SOURCE_CHANGED')
    require(contract['input_pins'].get(base['saved_evidence']['path']) == base['saved_evidence']['sha256'],
            'SAVED_INPUT_PIN_MISSING')
    require(type(contract['scope']) is str and contract['scope'] and
            type(contract['claim_id']) is str, 'SCOPE_CLAIM_REQUIRED')
    if artificial:
        require(contract['scope'].startswith('ARTIFICIAL_') and contract['claim_id'].startswith('ARTIFICIAL_'),
                'ARTIFICIAL_NAMESPACE_ONLY')
    else:
        require(not contract['scope'].startswith('ARTIFICIAL_') and contract['claim_id'].startswith('B3_NEW_'),
                'FRESH_PRODUCTION_NAMESPACE_REQUIRED')
        require(contract['source_pins'][DRIVER] != OLD_DRIVER_SHA, 'F01_OLD_DRIVER_FORBIDDEN')
        require(Path(D.__file__).resolve() == root / DRIVER and Path(__file__).resolve() == root / ADMISSION,
                'LOADED_MODULE_ROOT_MISMATCH')
    output = exact_path(root, candidate['output_relative'])
    require(not output.exists(), 'OUTPUT_COLLISION_NO_OVERWRITE')
    return {'formal_digest': D.identity(contract), 'driver_contract_digest': D.identity(candidate),
            'release_commit': release['commit'], 'scope': contract['scope'], 'claim_id': contract['claim_id'],
            'output_path': str(output), 'adoption_sha256': contract['adoption']['record_sha256'],
            'source_pins_digest': D.identity(contract['source_pins']),
            'input_pins_digest': D.identity(contract['input_pins'])}


class _Admission:
    artificial = False

    def __init__(self, contract, root, ledger, authority=None, *, fault=lambda point: None):
        self.contract = deepcopy(contract)
        self.root, self.ledger, self.authority = Path(root), ledger, authority
        self.fault, self.state, self.nonce, self.binding = fault, 'NEW', None, None
        self.lock = threading.Lock()
        self.owner_pid = os.getpid()
        self._stop_report = None
        if self.artificial:
            p = private_directory(self.root)
            require(p.is_relative_to(Path('/tmp')) and p.name.startswith('b3-r1-artificial-'), 'PRIVATE_TEMP_ONLY')
            require(ledger.directory.is_relative_to(p), 'ARTIFICIAL_LEDGER_TEMP_ONLY')
            if authority is not None:
                for name in (authority.anchor.authorization_reference, authority.anchor.statement_reference):
                    require(Path(name).is_relative_to(p) and Path(name) == Path(name).resolve(), 'ARTIFICIAL_AUTH_TEMP_ONLY')

    def _check(self, driver_digest):
        require(self.root.is_absolute() and self.root == self.root.resolve(), 'SOURCE_ROOT_REQUIRED')
        require(type(self.authority) is PinnedApproval, 'INDEPENDENT_SCIENCE_APPROVAL_REQUIRED')
        binding = validate_r1(self.contract, self.root, self.artificial)
        require(driver_digest == binding['driver_contract_digest'], 'DRIVER_CONTRACT_DIGEST')
        anchor = self.authority.anchor
        require(anchor.ledger_directory == str(self.ledger.directory) and
                anchor.ledger_store_id == self.ledger.store_id, 'TRUSTED_LEDGER_IDENTITY')
        proof = self.authority.verify(binding, self.artificial)
        # Approval's document binds the formal payload; the spent row additionally
        # binds that document's own hash without a self-referential hash cycle.
        binding = dict(binding, authorization=proof, ledger_directory=str(self.ledger.directory),
                       ledger_store_id=self.ledger.store_id)
        return binding, proof

    def _admit(self, driver_digest):
        with self.lock:
            require(self.state == 'NEW' and os.getpid() == self.owner_pid, 'STOP_NO_RETRY_RESUME_OR_FORK_REUSE')
            self.state = 'STOPPED'
            binding, proof = self._check(driver_digest)
            self.binding = binding
            try:
                self.nonce = self.ledger.reserve(binding, self.fault)
                self.fault('before_binding_recheck')
                second, second_proof = self._check(driver_digest)
                require(second == binding and second_proof == proof, 'BINDING_CHANGED_AFTER_RESERVATION')
                row = self.ledger.row(binding['scope'])
                require(row and row[:5] == (binding['claim_id'], binding['formal_digest'],
                        encoded(binding).decode(), self.nonce, 'RESERVED'), 'RESERVATION_RECHECK')
                self.fault('partial_claim_detail_write')
                self.ledger.transition(binding['scope'], self.nonce, 'RESERVED', 'ADMITTED', fault=self.fault)
                self.fault('before_output_creation')
                self.state = 'ADMITTED'
                return {'contract_digest': driver_digest, 'approved': True, 'fresh_claim': True,
                        'previously_consumed': False, 'retry': 0, 'resume': False,
                        'claim_id': binding['claim_id'], 'scope': SCENARIO,
                        'logical_scope': binding['scope'], 'formal_contract_digest': binding['formal_digest'],
                        'authorization': proof, 'mandatory_STOP': True, 'artificial_only': self.artificial}
            except BaseException as exc:
                # Includes commit-ack uncertainty: never release, delete, or retry.
                self._stop('ADMISSION_FAILURE', {'type': type(exc).__name__, 'message': str(exc)})
                raise

    def _stop(self, outcome, detail):
        if self._stop_report is not None:
            return self._stop_report  # Never retry a failed metadata write.
        self.state = 'STOPPED'
        record = {'schema': 'b3_admission_stop_v1', 'outcome': outcome, 'detail': detail,
                  'mandatory_STOP': True, 'retry': 0, 'resume': False}
        errors = []
        if self.nonce is not None:
            try:
                self.fault('failure_record_save')
                row = self.ledger.row(self.binding['scope'])
                if row and row[4] in ('RESERVED', 'ADMITTED'):
                    self.ledger.transition(self.binding['scope'], self.nonce, row[4], 'STOPPED', record)
            except BaseException as exc:
                errors.append(type(exc).__name__ + ':' + str(exc))
            try:
                self.fault('stop_record_save')
                path = self.ledger.directory / (self.nonce + '.STOP.json')
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                try:
                    data = encoded(record)
                    while data:
                        n = os.write(fd, data)
                        require(n > 0, 'STOP_SHORT_WRITE')
                        data = data[n:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
                sync_directory(self.ledger.directory)
            except BaseException as exc:
                errors.append(type(exc).__name__ + ':' + str(exc))
        self._stop_report = {'mandatory_STOP': True, 'record_errors': errors, 'reservation_never_released': True}
        return self._stop_report

    def stop(self, outcome='STOP', detail=None):
        with self.lock:
            require(os.getpid() == self.owner_pid, 'FORK_OWNER_MISMATCH')
            return self._stop(outcome, detail)

    def __enter__(self):
        require(self.state == 'NEW', 'STOP_NO_RETRY_OR_RESUME')
        return self

    def __exit__(self, kind, exc, traceback):
        self.stop('RETURNED' if kind is None else 'FAILED',
                  None if exc is None else {'type': kind.__name__, 'message': str(exc)})


class ProductionAdmission(_Admission):
    """Boundary.enter-compatible port. No real trust anchor is supplied here.

    Future reviewed host must wrap execute in this context to record outer STOP.
    execute's own guard.stop is retained; it does not notify an external port.
    A process dying without __exit__ leaves a spent reservation, never a lease.
    """
    def admit_once(self, driver_contract_digest):
        return self._admit(driver_contract_digest)


class ArtificialAdmission(_Admission):
    """Same engine, private fake inputs only; deliberately no admit_once method."""
    artificial = True

    def admit_artificial(self, driver_contract_digest):
        return self._admit(driver_contract_digest)
