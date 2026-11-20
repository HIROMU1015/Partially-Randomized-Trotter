"""Controller-local durable Q3 reservations and host attempts.

SQLite transactions protect a scientific job against simultaneous claims and
reserve its budget before any worker can receive it.  A lost connection never
expires an assignment; recovery must reconcile the same host and attempt.
This stores execution state, not evidence of user permission.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid
from .rpe_q3_timing import stamp


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


_TRANSITIONS = {
    'ASSIGNED': {'STARTING', 'RUNNING', 'UNKNOWN', 'COMPLETED_PENDING_TRANSFER'},
    'STARTING': {'RUNNING', 'UNKNOWN', 'COMPLETED_PENDING_TRANSFER'},
    'RUNNING': {'UNKNOWN', 'COMPLETED_PENDING_TRANSFER'},
    'UNKNOWN': {'COMPLETED_PENDING_TRANSFER'},
    'COMPLETED_PENDING_TRANSFER': set(),
    'ACCEPTED': set(),
}


class JobStore:
    """One local database per controller run; safe across threads/processes.

    ``allowed_jobs`` in claim is the controller's ordered queue.  Omitting it
    selects reserve order.  Host prefix hints are accepted but never reorder
    this queue: candidate priority belongs to the existing optimizer.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    method TEXT NOT NULL,
                    request TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    contract TEXT NOT NULL,
                    state TEXT NOT NULL,
                    host_id TEXT,
                    attempt_id TEXT,
                    enqueued_epoch REAL NOT NULL,
                    assigned_epoch REAL,
                    started_epoch REAL,
                    completed_epoch REAL,
                    adopted_epoch REAL,
                    updated_epoch REAL NOT NULL,
                    detail TEXT NOT NULL DEFAULT '{}',
                    completion_sha256 TEXT,
                    eligibility TEXT NOT NULL DEFAULT 'null'
                );
                CREATE INDEX IF NOT EXISTS method_state ON jobs(method, state);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    epoch REAL NOT NULL,
                    detail TEXT NOT NULL
                );
            ''')
            if 'eligibility' not in {r['name'] for r in db.execute('PRAGMA table_info(jobs)')}:
                db.execute("ALTER TABLE jobs ADD COLUMN eligibility TEXT NOT NULL DEFAULT 'null'")

    @contextmanager
    def _connection(self, *, write=False):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            if write:
                db.execute('BEGIN IMMEDIATE')
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _decoded(row):
        if row is None:
            return None
        value = dict(row)
        for key in ('request', 'contract', 'detail', 'eligibility'):
            value[key] = json.loads(value[key])
        return value

    @staticmethod
    def _event(db, job_id, event, detail=None):
        db.execute('INSERT INTO events(job_id,event,epoch,detail) VALUES(?,?,?,?)',
                   (job_id, event, time.time(), canonical(dict(stamp(), **(detail or {})))))

    def row(self, job_id):
        with self._connection() as db:
            return self._decoded(db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone())

    def rows(self):
        with self._connection() as db:
            return [self._decoded(row) for row in db.execute('SELECT * FROM jobs ORDER BY enqueued_epoch,job_id')]

    def unadopted_results(self, method):
        with self._connection() as db:
            return [row['job_id'] for row in db.execute(
                "SELECT job_id FROM jobs WHERE method=? AND state='ACCEPTED' AND adopted_epoch IS NULL "
                "ORDER BY completed_epoch,job_id", (method,))]

    def events(self):
        with self._connection() as db:
            return [dict(row, detail=json.loads(row['detail'])) for row in db.execute('SELECT * FROM events ORDER BY sequence')]

    def counts(self, method, started_only=False):
        """Reserved scientific jobs; assigned attempts count as started.

        An ASSIGNED job cannot be reassigned even if its start notification was
        lost.  ``started_epoch`` records the stronger STARTING/RUNNING observation.
        """
        condition = ' AND assigned_epoch IS NOT NULL' if started_only else " AND state!='CANCELLED'"
        with self._connection() as db:
            return db.execute('SELECT count(*) FROM jobs WHERE method=?' + condition, (method,)).fetchone()[0]

    @staticmethod
    def artificial(request):
        route = request.get('route') or {}
        if route.get('kind') == 'preparation':
            return (request.get('preparation') or {}).get('artificial') is True
        if route.get('kind') is not None:
            return route['kind'] == 'artificial'
        return request.get('artificial') is True

    @staticmethod
    def eligible(request, eligibility, host_id=None, host_fingerprints=None, job_id=None):
        """Validate persistent membership and a fresh, request-specific host report."""
        if eligibility is None:
            return JobStore.artificial(request)
        if eligibility.get('request_sha256') != digest(request):
            raise ValueError('Host eligibility belongs to a different REQUEST')
        allowed = eligibility.get('eligible_host_ids')
        if not isinstance(allowed, list) or any(not isinstance(h, str) or not h for h in allowed):
            raise ValueError('Explicit eligible host IDs required')
        expected_by_host = eligibility.get('host_fingerprints') or {}
        if any(not isinstance(expected_by_host.get(h), str) or not expected_by_host[h] for h in allowed):
            raise ValueError('Every eligible host needs a current-facts fingerprint')
        if host_id is None:
            return bool(allowed)
        if host_id not in allowed:
            return False
        expected = expected_by_host.get(host_id)
        current = (host_fingerprints or {}).get(job_id)
        return isinstance(current, dict) and isinstance(expected, str) and digest(current) == expected

    def preparation_reservations(self):
        with self._connection() as db:
            return dict(db.execute("""SELECT substr(method,13),count(*) FROM jobs
                WHERE method LIKE 'PREPARATION:%' AND state!='CANCELLED' GROUP BY method"""))

    def reserve(self, method, job_id, request, budget, approved, contract=None, eligibility=None):
        if not isinstance(method, str) or not method or not isinstance(job_id, str) or not job_id:
            raise ValueError('Nonempty method and scientific job_id required')
        if budget not in (None, 'UNLIMITED') and (type(budget) is not int or budget < 0):
            raise ValueError('Budget must be a nonnegative integer or UNLIMITED')
        encoded, request_sha = canonical(request), digest(request)
        encoded_contract = canonical(contract)
        with self._connection(write=True) as db:
            old = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if old is not None:
                if (old['method'] != method or old['request_sha256'] != request_sha or
                        old['contract'] != encoded_contract):
                    raise ValueError('Scientific job identity/request/contract changed: ' + job_id)
                if old['state'] != 'CANCELLED':
                    return True
            # Existing assigned attempts retain their owner for recovery. New
            # real jobs need at least one matching dry-check PASS before they
            # consume a scientific reservation; artificial legacy jobs remain
            # usable without a host registry.
            if not self.eligible(request, eligibility):
                return False
            used = db.execute("SELECT count(*) FROM jobs WHERE method=? AND state!='CANCELLED'", (method,)).fetchone()[0]
            if not approved or (budget not in (None, 'UNLIMITED') and used >= budget):
                return False
            now = time.time()
            if old is None:
                db.execute('''INSERT INTO jobs(job_id,method,request,request_sha256,contract,state,enqueued_epoch,updated_epoch,eligibility)
                              VALUES(?,?,?,?,?,'QUEUED',?,?,?)''',
                           (job_id, method, encoded, request_sha, encoded_contract, now, now, canonical(eligibility)))
            else:
                db.execute("UPDATE jobs SET state='QUEUED',enqueued_epoch=?,updated_epoch=?,eligibility=? WHERE job_id=?",
                           (now, now, canonical(eligibility), job_id))
            self._event(db, job_id, 'RESERVE')
            return True

    def set_eligibility(self, job_id, eligibility):
        """Refresh only an unassigned decision; never reassign a prior attempt."""
        with self._connection(write=True) as db:
            old = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if old is None or old['state'] != 'QUEUED':
                return False
            available = self.eligible(json.loads(old['request']), eligibility)
            state = 'QUEUED' if available else 'CANCELLED'
            encoded = canonical(eligibility)
            if old['eligibility'] != encoded or old['state'] != state:
                db.execute('UPDATE jobs SET eligibility=?,state=?,updated_epoch=? WHERE job_id=?',
                           (encoded, state, time.time(), job_id))
                self._event(db, job_id, 'HOST_ELIGIBILITY_UPDATED' if available else 'HOST_ELIGIBILITY_HOLD',
                            dict(eligibility=eligibility))
            return available

    def cancel(self, method, job_id):
        with self._connection(write=True) as db:
            old = db.execute('SELECT method,state FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if old is None:
                return False
            if old['method'] != method:
                raise ValueError('Job method changed')
            if old['state'] != 'QUEUED':
                return False
            db.execute("UPDATE jobs SET state='CANCELLED',updated_epoch=? WHERE job_id=?", (time.time(), job_id))
            self._event(db, job_id, 'CANCEL_QUEUED')
            return True

    def claim(self, host_id, slots, cached_prefixes=(), allowed_jobs=None, host_fingerprints=None):
        if not isinstance(host_id, str) or not host_id:
            raise ValueError('Nonempty host_id required')
        if type(slots) is not int or slots < 0:
            raise ValueError('Nonnegative free slots required')
        if slots == 0:
            return []
        with self._connection(write=True) as db:
            queued = {row['job_id']: row for row in db.execute("SELECT * FROM jobs WHERE state='QUEUED' ORDER BY enqueued_epoch,job_id")}
            order = list(queued) if allowed_jobs is None else list(dict.fromkeys(allowed_jobs))
            selected = []
            for job_id in order:
                if job_id not in queued:
                    continue
                row = self._decoded(queued[job_id])
                contract = row.get('contract')
                dependencies = contract.get('dependencies', []) if isinstance(contract, dict) else []
                if any((lambda value: value is None or value['state'] != 'ACCEPTED')(
                        db.execute('SELECT state FROM jobs WHERE job_id=?', (dependency,)).fetchone())
                        for dependency in dependencies):
                    continue
                if not self.eligible(row['request'], row['eligibility'], host_id, host_fingerprints, job_id):
                    continue
                now, attempt = time.time(), uuid.uuid4().hex
                db.execute("UPDATE jobs SET state='ASSIGNED',host_id=?,attempt_id=?,assigned_epoch=?,updated_epoch=? WHERE job_id=?",
                           (host_id, attempt, now, now, job_id))
                self._event(db, job_id, 'ASSIGN', dict(host_id=host_id, attempt_id=attempt))
                selected.append(self._decoded(db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()))
                if len(selected) >= slots:
                    break
            return selected

    @staticmethod
    def _owned(db, host_id, job_id, attempt_id):
        row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None or not attempt_id or row['host_id'] != host_id or row['attempt_id'] != attempt_id:
            raise ValueError('Unknown job or different execution host/attempt')
        return row

    def update(self, host_id, job_id, attempt_id, state, detail=None):
        if state not in _TRANSITIONS:
            raise ValueError('Unsupported worker state: ' + str(state))
        with self._connection(write=True) as db:
            old = self._owned(db, host_id, job_id, attempt_id)
            if state != old['state'] and state not in _TRANSITIONS.get(old['state'], set()):
                raise ValueError('Worker state regression: ' + old['state'] + ' -> ' + state)
            now = time.time()
            started = old['started_epoch']
            if started is None and state in ('STARTING', 'RUNNING', 'COMPLETED_PENDING_TRANSFER'):
                started = now
            value = canonical(detail or {})
            db.execute('UPDATE jobs SET state=?,detail=?,updated_epoch=?,started_epoch=? WHERE job_id=?',
                       (state, value, now, started, job_id))
            if state != old['state'] or value != old['detail']:
                self._event(db, job_id, state, detail)
            return self._decoded(db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone())

    def complete(self, host_id, job_id, attempt_id, request_sha256, files):
        """Commit validated transferred artifacts exactly once.

        The caller must durably install and validate artifacts before this
        transaction.  Identical retransmissions return ``duplicate=True``;
        changed data under the same attempt is rejected.
        """
        completion_sha = digest(files)
        with self._connection(write=True) as db:
            old = self._owned(db, host_id, job_id, attempt_id)
            if old['request_sha256'] != request_sha256:
                raise ValueError('Completion request hash mismatch')
            if old['state'] == 'ACCEPTED':
                if old['completion_sha256'] != completion_sha:
                    raise ValueError('Duplicate completion changed artifacts')
                return self._decoded(old), True
            if old['state'] not in _TRANSITIONS:
                raise ValueError('Completion has no assigned attempt')
            now = time.time()
            db.execute("UPDATE jobs SET state='ACCEPTED',completion_sha256=?,completed_epoch=?,updated_epoch=? WHERE job_id=?",
                       (completion_sha, now, now, job_id))
            self._event(db, job_id, 'ACCEPT', dict(host_id=host_id, attempt_id=attempt_id))
            return self._decoded(db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()), False

    def adopt(self, job_id):
        with self._connection(write=True) as db:
            old = db.execute('SELECT state,adopted_epoch FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if old is None or old['state'] != 'ACCEPTED':
                raise ValueError('Only an accepted result can be adopted')
            if old['adopted_epoch'] is not None:
                return False
            db.execute('UPDATE jobs SET adopted_epoch=? WHERE job_id=?', (time.time(), job_id))
            self._event(db, job_id, 'ADOPT')
            return True
