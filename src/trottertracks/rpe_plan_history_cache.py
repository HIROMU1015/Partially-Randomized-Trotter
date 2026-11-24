"""Persistent delta reader for already saved optimizer job evidence.

Only metadata and saved outcomes are read. The supplied Executor remains the
authority for job identity, route matching and ingestion; no launch, collect,
queue or scientific API is called. A local cache is disposable and is never
substituted for its original evidence locator or hashes.
"""
from collections import OrderedDict, defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import pickle
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
import zlib

from . import rpe_method_optimizer as O

SCHEMA = 1
# Keep recently reused exact values in the controller without retaining the
# whole on-disk cache in RAM. The budget counts uncompressed serialization.
DECODED_CACHE_BYTES = 64 * 1024**2
MARKERS = ('REQUEST.json', 'OUTCOME.json', 'COLLECTED_OUTCOME.json',
           'COLLECT_SUPERVISION.json', 'CASE_INPUTS.json', 'SOURCE.json',
           'PROCESS_RETURN.json', 'LAUNCH.json', 'STARTING.json',
           'SUPERVISION.json', 'DRY_CHECK.json', 'RESOLVED_REQUEST.json',
           'WORKER_STATE.json', 'ASSIGNMENT.json')
REUSE_HOOKS = ('job_dir', 'same_route', 'prior_route_fields', 'route', 'expected_for',
               'pf_template_source', 'ingest', 'scope_unknown')
DISTRIBUTED_INGEST_WRAPPER = '''\
def ingest(self, job, proposal, calls):
    answer = super().ingest(job, proposal, calls)
    if calls == 0:
        self.record_prior_result_reuse(job)
    return answer
'''
# This implementation only adds scope to UNKNOWN records, which the cache
# never accepts. A changed implementation falls back to full source hashing.
DISTRIBUTED_UNKNOWN_SCOPE_SHA256 = '92f6f5e832b03e5e323d594c7eab4e8a0ae10d2bcfd3543d91a3b0d66459547c'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _pack(value):
    # This database is generated locally by this class; never import a cache
    # received from another host. Binary integers avoid reparsing enormous
    # exact numerator/denominator decimal strings on every preparation run.
    return zlib.compress(pickle.dumps(value, protocol=5), 1)


def _cached_fraction(numerator, denominator):
    """Restore this cache's already normalized Fraction on older runtimes."""
    try:
        return Fraction(numerator, denominator, _normalize=False)
    except TypeError:
        # A future interpreter without either internal constructor remains
        # correct; it merely loses this optional decoding speedup.
        return Fraction(numerator, denominator)


class _LocalExactUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('fractions', 'Fraction'):
            # Original payloads were created from validated Fraction objects,
            # whose integer pairs are already coprime with positive divisor.
            # Avoid doing the same enormous GCD on every warm preparation.
            return getattr(Fraction, '_from_coprime_ints', _cached_fraction)
        return super().find_class(module, name)


def _unpack(value):
    # Compatible with existing local payloads; original JSON and validation
    # paths still construct and normalize Fraction in the usual way.
    return _LocalExactUnpickler(io.BytesIO(zlib.decompress(value))).load()


def _signature(job):
    signature = []
    for name in MARKERS:
        try:
            s = (job / name).stat()
            value = [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        except FileNotFoundError:
            value = None
        signature.append([name, value])
    return signature


def _hashes(job, signature):
    return {name: hashlib.sha256((job / name).read_bytes()).hexdigest()
            for name, value in signature if value is not None}


def _validator(executor):
    # Whole-file hashes also cover the canonical exact parser and the fixed
    # Git-blob reader used by route/template validation.
    paths = {Path(inspect.getfile(value)).resolve() for value in (O, O.Reader, O.frac, O.blob)}
    for cls in type(executor).__mro__:
        try:
            path = Path(inspect.getfile(cls)).resolve()
        except (TypeError, OSError):
            continue  # Builtin base classes have no source file.
        if cls.__module__ == __package__ + '.rpe_optimizer_concurrent':
            hooks = {name for name in REUSE_HOOKS if name in vars(cls)}
            # The known distributed wrapper logs validated reuse. The
            # ordinary Executor still decides which exact result is valid.
            if cls.__name__ == 'ConcurrentExecutor' and not hooks:
                continue
            if cls.__name__ == 'DistributedExecutor' and hooks in ({'ingest'}, {'ingest', 'scope_unknown'}):
                try:
                    wrapper = textwrap.dedent(inspect.getsource(cls.ingest))
                    scope = (textwrap.dedent(inspect.getsource(cls.scope_unknown))
                             if 'scope_unknown' in hooks else None)
                except (TypeError, OSError):
                    wrapper = None
                if wrapper == DISTRIBUTED_INGEST_WRAPPER and (scope is None or
                        hashlib.sha256(scope.encode()).hexdigest() == DISTRIBUTED_UNKNOWN_SCOPE_SHA256):
                    continue
        paths.add(path)
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def _validator_sources(paths):
    # A fixed overlay may be installed under a new deployment directory.
    return sorted((Path(path).parent.name, Path(path).name, sha) for path, sha in paths.items())


def _context(executor, cid):
    route = executor.route(cid) or {}
    result = dict(executor.prior_route_fields(route, cid), executor_root=str(Path(executor.root).resolve()),
                  expected=executor.expected_for(cid, route), contract=executor.contract)
    if route.get('kind') == 'pf_b_pf6':
        result['pf_template_source'] = executor.pf_template_source(route, cid)
    return result


def _current_unresolved(job, outcome_status=None):
    """Refresh current markers and /proc, never infer completion from a PID."""
    row = dict(job=str(job), outcome_status=outcome_status)
    for name in ('LAUNCH.json', 'PROCESS_RETURN.json', 'WORKER_STATE.json'):
        try:
            row[name] = json.loads((job / name).read_text())
        except (OSError, ValueError) as exc:
            row[name] = None
            if (job / name).exists():
                row[name + '_error'] = str(exc)
    launch = row['LAUNCH.json'] or {}
    pid = launch.get('pid')
    cmdline = None
    if isinstance(pid, int) and pid > 0:
        try:
            cmdline = (Path('/proc') / str(pid) / 'cmdline').read_bytes().decode(errors='replace')
        except OSError:
            pass
    alive = cmdline is not None and str(job) in cmdline
    row.update(pid=pid, matching_process_alive=alive,
               state='RUNNING' if alive else 'UNKNOWN_OUTCOME' if outcome_status else
               'RETURNED_WITHOUT_OUTCOME' if row['PROCESS_RETURN.json'] else
               'LAUNCHED_STATE_UNKNOWN' if launch else 'REQUEST_ONLY_NOT_LAUNCHED')
    return row


def _directory_version(path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns


def _request_configuration(job):
    try:
        cid = json.loads((job/'REQUEST.json').read_bytes()).get('configuration_id')
        return cid if isinstance(cid, str) else None
    except (OSError, ValueError, AttributeError):
        return None


@dataclass
class _DirectoryIndex:
    version: tuple | None
    names: dict
    configurations: dict
    pending: tuple

    @classmethod
    def build(cls, version, names):
        configurations = defaultdict(list)
        pending = []
        for name, (_, cid) in names.items():
            if cid is None:
                pending.append(name)
            else:
                configurations[cid].append(name)
        return cls(version, names, dict(configurations), tuple(pending))


class HistoryInventory:
    """Share lightweight history lookups and exact cache hits within an invocation."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending = {}
        self.directories = {}
        self.decoded = OrderedDict()
        self.decoded_bytes = 0

    def index(self, directory, history, *, stopped=None):
        version = _directory_version(directory)
        if version is None:
            return None
        with self.lock:
            current = self.directories.get(directory)
            if current is not None and current.version == version and not current.pending:
                return current
            future = self.pending.get(directory)
            if future is not None:
                owner = False
            else:
                future = self.pending[directory] = Future()
                owner = True
        if not owner:
            observed = future.result()
            return observed if observed is None or observed.version == version else self.index(
                directory, history, stopped=stopped)
        try:
            index = history.refresh_directory(directory, version, current, stopped)
        except BaseException as error:
            with self.lock:
                del self.pending[directory]
            future.set_exception(error)
            raise
        with self.lock:
            if index is not None:
                self.directories[directory] = index
            del self.pending[directory]
        future.set_result(index)
        return index

    def prior_paths(self, directories, job_name):
        """Use the same directory index for duplicate checks, with fresh
        lookups on an unindexed or changed directory."""
        for directory in directories:
            with self.lock:
                index = self.directories.get(directory)
            if index is None or job_name in index.names or index.version != _directory_version(directory):
                yield directory/job_name

    def load_payload(self, key, read_payload):
        with self.lock:
            previous = self.decoded.get(key)
            if previous is not None:
                self.decoded.move_to_end(key)
                return previous[0], True
        raw = zlib.decompress(read_payload())
        cached = _LocalExactUnpickler(io.BytesIO(raw)).load()
        if cached.get('record') is not None and len(raw) <= DECODED_CACHE_BYTES:
            with self.lock:
                if key not in self.decoded:
                    self.decoded[key] = cached, len(raw)
                    self.decoded_bytes += len(raw)
                    while self.decoded_bytes > DECODED_CACHE_BYTES:
                        _, (_, size) = self.decoded.popitem(last=False)
                        self.decoded_bytes -= size
        return cached, False


class HistoryCache:
    """Local SQLite cache of exact records previously validated by Executor.

    ``prior_records(executor_for_method, candidates)`` returns
    ``(known, refused, stats)`` with the same known/refused meaning as the
    existing Executor method. Repeated calls check the markers of matching jobs,
    refresh unresolved processes, and recalculate costs with current prices.
    """
    def __init__(self, path):
        if hasattr(sys, 'set_int_max_str_digits'):
            sys.set_int_max_str_digits(0)
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS evidence '
                        '(job TEXT PRIMARY KEY, signature TEXT NOT NULL, context TEXT NOT NULL, payload BLOB NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS history_directory '
                        '(directory TEXT PRIMARY KEY, version TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS history_job '
                        '(directory TEXT NOT NULL, job TEXT NOT NULL, device INTEGER NOT NULL, inode INTEGER NOT NULL, '
                        'configuration TEXT, PRIMARY KEY (directory, job)) WITHOUT ROWID')
        self.db.commit()
        self.stats_by_method = {}

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def refresh_directory(self, directory, version, current, stopped):
        key = str(directory)
        if current is None:
            row = self.db.execute('SELECT version FROM history_directory WHERE directory=?', (key,)).fetchone()
            if row is not None:
                try:
                    saved_version = tuple(json.loads(row[0]))
                except (ValueError, TypeError, AttributeError):
                    saved_version = None
                saved = self.db.execute(
                    'SELECT job, device, inode, configuration FROM history_job WHERE directory=?', (key,))
                current = _DirectoryIndex.build(saved_version, {name: ((dev, ino), cid) for name, dev, ino, cid in saved})
        previous = current.names if current is not None else {}
        names = previous.copy()
        modified = {}
        if current is None or current.version != version:
            names = {}
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if not entry.is_dir():
                            continue
                        info = entry.stat()
                    except FileNotFoundError:
                        continue
                    identity = info.st_dev, info.st_ino
                    old_identity, cid = previous.get(entry.name, (None, None))
                    # REQUEST identity is fixed by the job directory's key.
                    # Existing job bytes are still checked before reuse.
                    if identity != old_identity or cid is None:
                        if stopped is not None and stopped():
                            return None
                        cid = _request_configuration(directory/entry.name)
                    names[entry.name] = identity, cid
                    if names[entry.name] != previous.get(entry.name):
                        modified[entry.name] = identity, cid
        elif current.pending:
            for name in current.pending:
                if stopped is not None and stopped():
                    return None
                cid = _request_configuration(directory/name)
                if cid is not None:
                    names[name] = previous[name][0], cid
                    modified[name] = names[name]
        if current is not None and current.version == version and not modified:
            return current
        with self.db:
            self.db.executemany('DELETE FROM history_job WHERE directory=? AND job=?',
                                ((key, name) for name in previous.keys() - names.keys()))
            self.db.executemany('INSERT OR REPLACE INTO history_job VALUES (?, ?, ?, ?, ?)',
                                ((key, name, *identity, cid) for name, (identity, cid) in modified.items()))
            self.db.execute('INSERT OR REPLACE INTO history_directory VALUES (?, ?)', (key, _json(version)))
        return _DirectoryIndex.build(version, names)

    def prior_records(self, executor, candidates, *, details=True):
        started = time.monotonic()
        ready = {c['configuration_id']: c for c in candidates
                 if c['ready'] and c['configuration_id']}
        held = {(Path(row['parent_directory'])/'jobs'/row['method']/row['job_id']).resolve()
                for row in getattr(executor, 'prior_q3_holds', ())}
        record_reuse = None if details else getattr(executor, 'record_prior_result_reuse', None)
        stopped = None if details else getattr(executor, '_user_stop_requested', None)
        request_configurations = executor.prior_request_configurations
        inventory = None if details else getattr(executor, 'prior_history_inventory', None)
        validator = _validator(executor)
        contexts, comparisons = {}, {}
        known, refused = {}, []
        writes = {}

        def flush():
            if writes:
                with self.db:
                    self.db.executemany('INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?)',
                                        ((job, *row) for job, row in writes.items()))
                writes.clear()

        def save(key, signature, context, payload):
            writes[key] = (_json(signature), context, _pack(payload))
            if len(writes) >= 64:
                flush()

        stats = dict(cache_schema=SCHEMA, cache_path=str(self.path), scanned_jobs=0,
                     cache_hits=0, request_cache_hits=0, unrelated_request_hits=0,
                     parsed_jobs=0, validated_jobs=0, invalidated_jobs=0,
                     context_changes={}, current_unresolved=[], evidence=[], scientific_calls=0, collection_calls=0,
                     indexed_out_of_scope=0, memory_hits=0, interrupted=False)
        for base in executor.prior_run_dirs:
            directory = Path(base) / 'jobs' / executor.method
            if inventory is None and not directory.exists():
                continue
            if details:
                jobs = [(p, None) for p in sorted(directory.iterdir()) if p.is_dir()]
            elif inventory is not None:
                index = inventory.index(directory, self, stopped=stopped)
                if index is None:
                    if stopped is not None and stopped():
                        stats['interrupted'] = True
                        break
                    continue
                matching = {name for cid in ready for name in index.configurations.get(cid, ())}
                matching.update(job.name for job in held if job.parent == directory.resolve())
                stats['indexed_out_of_scope'] += len(index.names) - len(matching & index.names.keys())
                jobs = [(directory/name, None) for name in sorted(matching)
                        if (directory/name/'OUTCOME.json').is_file()]
            else:
                jobs = [(outcome.parent, None) for outcome in sorted(directory.glob('*/OUTCOME.json'))]
            for job, observed_request in jobs:
                if stopped is not None and stopped():
                    stats['interrupted'] = True
                    break
                request_path = job / 'REQUEST.json'
                previous = request_configurations.get(request_path)
                if (not details and inventory is None and previous is not None
                        and previous[1] not in ready and (not held or job.resolve() not in held)):
                    try:
                        info = request_path.stat()
                    except FileNotFoundError:
                        pass
                    else:
                        observed_request = (info.st_dev, info.st_ino, info.st_size,
                                            info.st_mtime_ns, info.st_ctime_ns)
                    if observed_request == previous[0]:
                        stats['scanned_jobs'] += 1
                        stats['unrelated_request_hits'] += 1
                        refused.append(dict(job=str(job), reason='CONFIGURATION_NOT_IN_THIS_ROUND'))
                        continue
                signature = _signature(job)
                present = {name for name, value in signature if value is not None}
                if not present.intersection(('REQUEST.json', 'LAUNCH.json', 'OUTCOME.json')):
                    continue
                stats['scanned_jobs'] += 1
                if 'OUTCOME.json' not in present:
                    stats['current_unresolved'].append(_current_unresolved(job))
                    continue
                if held and job.resolve() in held:
                    refused.append(dict(job=str(job), reason='PRIOR_Q3_PERSISTENCE_HOLD'))
                    continue
                request_signature = tuple(signature[0][1] or ())
                key = str(job.resolve())
                saved = writes.get(key)
                row_id = None
                if saved is None and inventory is not None:
                    header = self.db.execute('SELECT signature, context, rowid FROM evidence WHERE job=?', (key,)).fetchone()
                    if header is not None:
                        saved, row_id = header[:2], header[2]
                elif saved is None:
                    saved = self.db.execute('SELECT signature, context, payload FROM evidence WHERE job=?', (key,)).fetchone()
                cached = None
                if saved is not None and saved[0] == _json(signature):
                    try:
                        if row_id is None:
                            cached = _unpack(saved[2])
                        else:
                            def read_payload():
                                row = self.db.execute(
                                    'SELECT payload FROM evidence WHERE rowid=? AND job=? AND signature=? AND context=?',
                                    (row_id, key, saved[0], saved[1])).fetchone()
                                if row is None:
                                    raise ValueError('History cache entry replaced')
                                return row[0]
                            cached, memory_hit = inventory.load_payload(
                                (self.path, key, row_id, saved[0], saved[1]), read_payload)
                            stats['memory_hits'] += memory_hit
                    except (ValueError, TypeError, zlib.error, KeyError, pickle.UnpicklingError, EOFError):
                        cached = None
                if cached is not None:
                    request = cached['request']
                    stats['request_cache_hits'] += 1
                else:
                    stats['parsed_jobs'] += 1
                    try:
                        request = json.loads(request_path.read_text())
                    except (OSError, ValueError):
                        refused.append(dict(job=str(job), reason='NO_REQUEST'))
                        continue
                    cached = dict(request=request, record=None)
                    if signature != _signature(job):
                        refused.append(dict(job=str(job), reason='HISTORY_CHANGED_DURING_VALIDATION'))
                        continue
                    save(key, signature, '', cached)
                cid = request.get('configuration_id')
                if request_signature:
                    request_configurations[request_path] = request_signature, cid
                candidate = ready.get(cid)
                if candidate is None:
                    # Even a currently unavailable configuration may have an
                    # unresolved old owner. Discover its status once per delta,
                    # then refresh only the current markers/process each call.
                    if details and 'outcome_status' not in cached:
                        try:
                            cached['outcome_status'] = json.loads((job / 'OUTCOME.json').read_text()).get('status')
                        except (OSError, ValueError):
                            cached['outcome_status'] = 'UNKNOWN_BAD_OUTCOME'
                        if signature == _signature(job):
                            old_context = saved[1] if saved is not None and cached.get('record') is not None else ''
                            save(key, signature, old_context, cached)
                    if details and cached['outcome_status'] not in (O.PASS, O.FAIL, 'INCONCLUSIVE'):
                        stats['current_unresolved'].append(_current_unresolved(job, cached['outcome_status']))
                    refused.append(dict(job=str(job), reason='CONFIGURATION_NOT_IN_THIS_ROUND'))
                    continue
                try:
                    if cid not in contexts:
                        contexts[cid] = _json(dict(schema=SCHEMA, validator=validator,
                                                  context=_context(executor, cid)))
                    context = contexts[cid]
                    proposal = dict(configuration_id=cid, vector=O.interleave(request['N_X'], request['N_Y']),
                                    origin=request.get('origin', {}))
                    hit = cached.get('record') is not None and saved is not None and saved[1] == context
                    if not hit and saved is not None and cached.get('record') is not None and saved[1]:
                        pair = cid, saved[1]
                        if pair not in comparisons:
                            old, new = json.loads(saved[1]), json.loads(context)
                            old_context, new_context = old['context'], new['context']
                            changed = [field for field in old_context.keys() | new_context.keys()
                                       if field not in old_context or field not in new_context
                                       or _json(old_context[field]) != _json(new_context[field])]
                            if old['schema'] != new['schema']:
                                changed.append('schema')
                            if _validator_sources(old['validator']) != _validator_sources(new['validator']):
                                changed.append('validator')
                            comparisons[pair] = changed
                        hit = not comparisons[pair]
                        for field in comparisons[pair]:
                            stats['context_changes'][field] = stats['context_changes'].get(field, 0) + 1
                    if hit:
                        record = cached['record']
                        stats['cache_hits'] += 1
                    else:
                        if saved is not None:
                            stats['invalidated_jobs'] += 1
                        if executor.job_dir(proposal).name != job.name or not executor.same_route(job, cid, request=request):
                            refused.append(dict(job=str(job), reason='DIFFERENT_JOB_KEY_OR_ROUTE'))
                            continue
                        record = executor.ingest(job, proposal, calls=0)
                        stats['validated_jobs'] += 1
                        if record['status'] not in (O.PASS, O.FAIL, 'INCONCLUSIVE'):
                            refused.append(dict(job=str(job), reason=record['status'] + ':' + str(record.get('reason'))))
                            if details:
                                stats['current_unresolved'].append(_current_unresolved(job, record['status']))
                            continue
                        cached = dict(cached, record=record, outcome_status=record['status'])
                    if signature != _signature(job):
                        refused.append(dict(job=str(job), reason='HISTORY_CHANGED_DURING_VALIDATION'))
                        continue
                    updated = not hit
                    if details:
                        if 'provenance' not in cached:
                            cached['provenance'] = dict(job=key, sha256=_hashes(job, signature))
                            if signature != _signature(job):
                                refused.append(dict(job=str(job), reason='HISTORY_CHANGED_DURING_VALIDATION'))
                                continue
                            updated = True
                        stats['evidence'].append(cached['provenance'])
                    if hit and record_reuse is not None:
                        record_reuse(job)
                    if updated:
                        save(key, signature, context, cached)
                    known[(cid, proposal['vector'])] = dict(record, source='PRIOR_ROUND_JOB',
                                                          cost=None if candidate['prices'] is None else
                                                          O.cost(proposal['vector'], candidate['prices']))
                except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
                    refused.append(dict(job=str(job), reason='HISTORY_METADATA_ERROR:' + str(exc)))
            if stats['interrupted']:
                break
        flush()
        stats.update(known_records=len(known), refused_records=len(refused), wall_seconds=time.monotonic() - started)
        self.stats_by_method[executor.method] = stats
        return known, refused, stats
