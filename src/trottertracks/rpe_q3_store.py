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
                CREATE INDEX IF NOT EXISTS jobs_enqueue_order ON jobs(enqueued_epoch, job_id);
                CREATE INDEX IF NOT EXISTS jobs_state_enqueue ON jobs(state, enqueued_epoch, job_id);
                CREATE INDEX IF NOT EXISTS host_state_queue ON jobs(host_id, state, enqueued_epoch, job_id);
                CREATE INDEX IF NOT EXISTS host_completion ON jobs(host_id, completed_epoch);
                CREATE INDEX IF NOT EXISTS host_adoption ON jobs(host_id, adopted_epoch);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    epoch REAL NOT NULL,
                    detail TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS prelaunch_redispatch
                    ON events(job_id) WHERE event='REDISPATCH_AFTER_DRY_CHECK';
                CREATE TABLE IF NOT EXISTS preparation_readiness (
                    job_id TEXT PRIMARY KEY,
                    record TEXT
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

    def save_preparation_readiness(self, updates):
        values = [(job, None if record is None else canonical(record)) for job, record in updates.items()]
        with self._connection(write=True) as database:
            database.executemany('INSERT OR REPLACE INTO preparation_readiness VALUES (?, ?)', values)

    def attempt_owner(self, job_id):
        return self.attempt_owners([job_id]).get(job_id)

    def capacity_history(self, host, epoch, totals):
        with self._connection() as database:
            usable, = database.execute(
                "SELECT count(*) FROM jobs WHERE host_id=? AND completed_epoch>=? AND state='ACCEPTED'",
                (host, epoch)).fetchone()
            adopted, = database.execute(
                "SELECT count(*) FROM jobs WHERE host_id=? AND adopted_epoch>=?", (host, epoch)).fetchone()
            selected = database.execute(
                "SELECT job_id FROM jobs WHERE host_id=? AND job_id IN (SELECT value FROM json_each(?))",
                (host, json.dumps(list(totals))))
            waits = [totals[job_id] for job_id, in selected]
        return usable, adopted, sum(waits) if waits else None

    def attempt_owners(self, job_ids):
        ids = list(dict.fromkeys(job_ids))
        owners = {}
        with self._connection() as database:
            for start in range(0, len(ids), 500):
                block = ids[start:start+500]
                placeholders = ','.join('?' for _ in block)
                for row in database.execute(
                        'SELECT job_id,method,host_id,attempt_id,state,request_sha256,completion_sha256 '
                        f'FROM jobs WHERE job_id IN ({placeholders})', block):
                    owners[row['job_id']] = dict(row)
        return owners

    def rows_for(self, job_ids):
        """Read one queue snapshot for a scheduling pass."""
        ids = list(dict.fromkeys(job_ids))
        if not ids:
            return {}
        with self._connection() as db:
            db.execute('BEGIN')
            result = {}
            for start in range(0, len(ids), 500):
                group = ids[start:start+500]
                what = ','.join('?' for _ in group)
                result.update((row['job_id'], self._decoded(row)) for row in db.execute(
                    f'SELECT * FROM jobs WHERE job_id IN ({what})', group))
            return result

    def queue_rows_delta(self, job_ids, revision, known):
        """Read only queue rows whose durable state changed since the last poll.

        The event revision and returned rows are from the same SQLite read
        transaction. Claim still verifies the latest state independently.
        """
        ids = set(job_ids)
        with self._connection() as db:
            db.execute('BEGIN')
            current = db.execute('SELECT coalesce(max(sequence),0) FROM events').fetchone()[0]
            changed = (set(known)|ids if revision is None or current < revision else
                       {row[0] for row in db.execute('SELECT DISTINCT job_id FROM events WHERE sequence>?',(revision,))})
            needed = sorted((changed & ids) | (ids-set(known)))
            result = {}
            for start in range(0,len(needed),500):
                group=needed[start:start+500];what=','.join('?' for _ in group)
                result.update((row['job_id'],self._decoded(row)) for row in db.execute(
                    f'SELECT * FROM jobs WHERE job_id IN ({what})',group))
            return current,changed,result

    def accepted_host(self, job_id, method):
        """Read the host for reuse telemetry without opening the saved request."""
        with self._connection() as db:
            row = db.execute("SELECT host_id FROM jobs WHERE job_id=? AND method=? AND state='ACCEPTED'",
                             (job_id, method)).fetchone()
            return row[0] if row is not None else None

    def preparation_states(self, job_ids):
        """Read scheduling states without deserializing preparation payloads."""
        with self._connection() as db:
            return {job: (method, state) for job, method, state in db.execute(
                "SELECT job_id, method, state FROM jobs WHERE job_id IN (SELECT value FROM json_each(?))",
                (json.dumps(list(job_ids)),))}

    def rows(self):
        with self._connection() as db:
            return [self._decoded(row) for row in db.execute('SELECT * FROM jobs ORDER BY enqueued_epoch,job_id')]

    def resume_rows(self):
        """Restore outstanding requests without parsing historical accepted JSON.

        Accepted publication only needs the original method/job/attempt. The
        exact full request and host evidence remain in the durable ledger for
        validation, result reuse and science, but need not stay in RAM.
        """
        with self._connection() as db:
            db.execute('BEGIN')
            active=[self._decoded(row) for row in db.execute(
                "SELECT * FROM jobs WHERE state IN ('QUEUED','ASSIGNED','STARTING','RUNNING','UNKNOWN','COMPLETED_PENDING_TRANSFER') ORDER BY enqueued_epoch,job_id")]
            accepted=[dict(row) for row in db.execute(
                "SELECT job_id,method,attempt_id,state FROM jobs WHERE state='ACCEPTED' ORDER BY enqueued_epoch,job_id")]
            return active,accepted

    def metric_rows(self):
        # Requests and scientific evidence can be large. The driver only
        # needs routing and accounting fields when taking a progress snapshot.
        with self._connection() as db:
            return db.execute("""
                SELECT job_id, method, host_id, attempt_id, state, started_epoch, adopted_epoch,
                       (CASE WHEN json_extract(request, '$.route.kind') = 'preparation'
                             THEN json_type(request, '$.preparation') = 'object' ELSE 0 END) AS is_preparation
                FROM jobs ORDER BY enqueued_epoch, job_id
            """).fetchall()

    def prelaunch_rejections(self):
        with self._connection() as db:
            return [self._decoded(row) for row in db.execute("""
                SELECT * FROM jobs INDEXED BY host_state_queue WHERE state='UNKNOWN' AND completion_sha256 IS NULL
                    AND json_type(detail, '$.prelaunch_rejection') = 'object'
                    AND NOT EXISTS (SELECT 1 FROM events WHERE events.job_id=jobs.job_id
                        AND event='REDISPATCH_AFTER_DRY_CHECK')
                ORDER BY enqueued_epoch, job_id
            """)]

    def requeue_pbs_buffer(self, job_id, attempt_id):
        """Restore a reservation only after the PBS session proves no launch."""
        with self._connection(write=True) as db:
            row = self._decoded(db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone())
            if (not row or row['state'] != 'UNKNOWN' or row['attempt_id'] != attempt_id
                    or any(row[k] is not None for k in ('started_epoch', 'completed_epoch',
                                                        'adopted_epoch', 'completion_sha256'))):
                return False
            detail = row['detail']
            proof = detail.get('pbs_admission_exit') or {}
            if (detail.get('reason') != 'PBS_ADMISSION_DEADLINE_BEFORE_LAUNCH'
                    or detail.get('resource_released') is not True
                    or detail.get('scientific_execution_started') is not False
                    or detail.get('automatic_retry') is not False or not proof.get('token')
                    or any(proof.get(k) != row[k] for k in ('job_id', 'attempt_id', 'request_sha256'))):
                return False
            self._event(db, job_id, 'PBS_READY_BUFFER_REQUEUED',
                        dict(host_id=row['host_id'], attempt_id=attempt_id, detail=detail))
            db.execute("""UPDATE jobs SET state='QUEUED',host_id=NULL,attempt_id=NULL,
                          assigned_epoch=NULL,updated_epoch=?,detail=? WHERE job_id=?""",
                       (time.time(), canonical(dict(prelaunch_recovery=True)), job_id))
            return True

    def requeue_prelaunch(self, job_id, attempt_id, host, eligibility, facts):
        """Reuse the original reservation after a proved non-scientific exit."""
        with self._connection(write=True) as db:
            row = self._decoded(db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone())
            if not row or row['state'] != 'UNKNOWN' or row['attempt_id'] != attempt_id:
                return False
            detail = row['detail']
            proof = detail.get('prelaunch_rejection') or {}
            is_dry = (proof.get('kind') == 'FAILED_DRY_CHECK_BEFORE_EXECUTION'
                      and detail.get('reason') == 'ValueError: Existing adapter dry-check failed')
            is_supervisor = (proof.get('kind') == 'SUPERVISOR_FAILED_BEFORE_EXECUTION'
                             and proof.get('reason') == detail.get('reason')
                             and proof.get('scientific_execution_started') is False
                             and proof.get('returncode') == 2)
            if (not row['host_id'] or detail.get('automatic_retry') is not False
                    or detail.get('resource_released') is not True
                    or not (is_dry or is_supervisor)
                    or (proof.get('job_id'), proof.get('attempt_id'), proof.get('request_sha256'))
                       != (job_id, attempt_id, row['request_sha256'])
                    or type(proof.get('returncode')) is not int or proof['returncode'] <= 0
                    or proof.get('signal') is not None
                    or any(row[key] is not None for key in ('completion_sha256', 'completed_epoch', 'adopted_epoch'))
                    or db.execute("SELECT 1 FROM events WHERE job_id=? AND event='REDISPATCH_AFTER_DRY_CHECK'",
                                  (job_id,)).fetchone()):
                return False
            if (not isinstance(eligibility, dict) or not self.eligible(row['request'], eligibility, host,
                                                                      {job_id: facts}, job_id)):
                return False
            old_fingerprint = ((row['eligibility'] or {}).get('host_fingerprints') or {}).get(host)
            if host == row['host_id'] and (not old_fingerprint or old_fingerprint == digest(facts)):
                return False
            previous = dict(host_id=row['host_id'], attempt_id=attempt_id, assigned_epoch=row['assigned_epoch'],
                            started_epoch=row['started_epoch'], detail=detail)
            self._event(db, job_id, 'REDISPATCH_AFTER_DRY_CHECK',
                        dict(original=previous, replacement_host=host, replacement_fingerprint=digest(facts),
                             request_sha256=row['request_sha256']))
            db.execute("""UPDATE jobs SET state='QUEUED',host_id=NULL,attempt_id=NULL,assigned_epoch=NULL,
                          started_epoch=NULL,updated_epoch=?,detail=?,eligibility=? WHERE job_id=?""",
                       (time.time(), canonical(dict(prelaunch_recovery=True)), canonical(eligibility), job_id))
            return True

    def unadopted_results(self, method):
        with self._connection() as db:
            return [row['job_id'] for row in db.execute(
                "SELECT job_id FROM jobs WHERE method=? AND state='ACCEPTED' AND adopted_epoch IS NULL "
                "ORDER BY completed_epoch,job_id", (method,))]

    def unadopted_preparation_successes(self):
        """Compact, durable inbox of successful preparation receipts."""
        with self._connection() as db:
            return [row['job_id'] for row in db.execute("""
                SELECT job_id FROM jobs WHERE method IN ('PREPARATION:signal','PREPARATION:compile','PREPARATION:audit')
                AND state='ACCEPTED' AND adopted_epoch IS NULL
                AND (json_extract(detail,'$.outcome_status')='PREPARATION_COMPLETE' OR json_extract(detail,'$.outcome_status') IS NULL)
                ORDER BY completed_epoch,job_id""")]

    def preparation_completion_revision(self):
        with self._connection() as db:
            row=db.execute("SELECT sequence FROM events WHERE event IN ('ACCEPT','ADOPT') AND job_id LIKE 'prep-%' ORDER BY sequence DESC LIMIT 1").fetchone()
            return row[0] if row else 0

    def has_preparation_successor(self, job_id, provider):
        """Check for a durable audit owner even if a workspace marker was lost."""
        with self._connection() as db:
            return db.execute("""
                SELECT 1 FROM jobs
                WHERE method LIKE 'PREPARATION:%'
                  AND json_extract(request, '$.preparation.provider') = ?
                  AND (state != 'CANCELLED' OR attempt_id IS NOT NULL)
                  AND EXISTS (
                      SELECT 1 FROM json_each(request, '$.preparation.dependencies')
                      WHERE value = ?
                  )
                LIMIT 1
            """, (provider, job_id)).fetchone() is not None

    def events(self):
        with self._connection() as db:
            return [dict(row, detail=json.loads(row['detail'])) for row in db.execute('SELECT * FROM events ORDER BY sequence')]

    def counts(self, method, started_only=False, *, observed_only=False):
        """Reserved scientific jobs; assigned attempts count as started.

        An ASSIGNED job cannot be reassigned even if its start notification was
        lost.  ``started_epoch`` records the stronger STARTING/RUNNING observation.
        """
        condition = ' AND assigned_epoch IS NOT NULL' if started_only else " AND state!='CANCELLED'"
        if observed_only:
            condition = " AND (started_epoch IS NOT NULL OR completion_sha256 IS NOT NULL)"
        with self._connection() as db:
            return db.execute("SELECT count(*) FROM jobs WHERE method=? AND json_extract(detail,'$.prior_partition_reuse') IS NOT 1" + condition, (method,)).fetchone()[0]

    @staticmethod
    def artificial(request):
        route = request.get('route') or {}
        if route.get('kind') == 'preparation':
            return (request.get('preparation') or {}).get('artificial') is True
        if route.get('kind') is not None:
            return route['kind'] == 'artificial'
        return request.get('artificial') is True

    @staticmethod
    def eligible(request, eligibility, host_id=None, host_fingerprints=None, job_id=None, *, request_sha256=None):
        """Validate persistent membership and a fresh, request-specific host report."""
        if eligibility is None:
            return JobStore.artificial(request)
        if eligibility.get('request_sha256') != (request_sha256 or digest(request)):
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
                WHERE method LIKE 'PREPARATION:%' AND state!='CANCELLED' AND json_extract(detail,'$.prior_partition_reuse') IS NOT 1 GROUP BY method"""))

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
            used = db.execute("SELECT count(*) FROM jobs WHERE method=? AND state!='CANCELLED' AND json_extract(detail,'$.prior_partition_reuse') IS NOT 1", (method,)).fetchone()[0]
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

    def claim(self, host_id, slots, cached_prefixes=(), allowed_jobs=None, host_fingerprints=None, *, assignment_detail=None):
        if not isinstance(host_id, str) or not host_id:
            raise ValueError('Nonempty host_id required')
        if type(slots) is not int or slots < 0:
            raise ValueError('Nonnegative free slots required')
        if slots == 0:
            return []
        with self._connection(write=True) as db:
            if allowed_jobs is None:
                queued = {row['job_id']: row for row in db.execute(
                    "SELECT * FROM jobs WHERE state='QUEUED' ORDER BY enqueued_epoch,job_id")}
                order = list(queued)
            else:
                order = list(dict.fromkeys(allowed_jobs))
                queued = {row['job_id']: row for row in db.execute(
                    "SELECT * FROM jobs WHERE state='QUEUED' AND job_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(order),))}
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
                detail = row['detail'] if assignment_detail is None else dict(row['detail'], **assignment_detail)
                db.execute("UPDATE jobs SET state='ASSIGNED',host_id=?,attempt_id=?,assigned_epoch=?,updated_epoch=?,detail=? WHERE job_id=?",
                           (host_id, attempt, now, now, canonical(detail), job_id))
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
            if state == old['state'] == 'UNKNOWN':
                previous = json.loads(old['detail'])
                for field in ('prelaunch_rejection', 'pbs_admission_exit', 'pbs_attempt_exit'):
                    if previous.get(field) and not (detail or {}).get(field):
                        detail = previous  # an older poll cannot discard a verified exit
                        break
            # Preserve the exact operational envelope while replay is still
            # possible. A worker's status detail is not a new assignment.
            previous = json.loads(old['detail'])
            detail = dict(detail or {})
            if state == 'ASSIGNED':
                for field in ('preparation_workload', 'pbs_launch_resources'):
                    if field in previous:detail[field] = previous[field]
            if 'pbs_allocation_token' in previous:
                detail['pbs_allocation_token'] = previous['pbs_allocation_token']
            value = canonical(detail)
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
        return self.complete_hashed(host_id, job_id, attempt_id, request_sha256, digest(files))

    def complete_hashed(self, host_id, job_id, attempt_id, request_sha256, completion_sha):
        """Commit a receipt whose bytes the controller already validated and hashed."""
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

    def import_partition_preparation(self, job_id, method, request, contract, origin):
        """Register a previously accepted block result, without a new reservation."""
        if (not method.startswith('PREPARATION:') or request.get('request_id') != job_id
                or origin.get('request_sha256') != digest(request) or origin.get('contract') != contract
                or not origin.get('attempt_id') or not origin.get('completion_sha256')):
            return False
        with self._connection(write=True) as db:
            if db.execute('SELECT 1 FROM jobs WHERE job_id=?',(job_id,)).fetchone():
                return False  # never replace an existing attempt or STOP
            now = time.time()
            db.execute("""INSERT INTO jobs(job_id,method,request,request_sha256,contract,state,host_id,
                attempt_id,enqueued_epoch,completed_epoch,updated_epoch,detail,completion_sha256)
                VALUES(?,?,?,?,?,'ACCEPTED',?,?,?,?,?,?,?)""",
                (job_id,method,canonical(request),origin['request_sha256'],canonical(contract),
                 origin['host_id'],origin['attempt_id'],now,now,now,
                 canonical(dict(prior_partition_reuse=True,origin=origin)),origin['completion_sha256']))
            self._event(db,job_id,'REUSE_PARTITION_PREPARATION',origin)
            return True

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
