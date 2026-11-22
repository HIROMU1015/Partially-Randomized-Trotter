"""Persistent delta reader for already saved optimizer job evidence.

Only metadata and saved outcomes are read. The supplied Executor remains the
authority for job identity, route matching and ingestion; no launch, collect,
queue or scientific API is called. A local cache is disposable and is never
substituted for its original evidence locator or hashes.
"""
from fractions import Fraction
import hashlib
import io
import inspect
import json
from pathlib import Path
import pickle
import sqlite3
import subprocess
import sys
import textwrap
import time
import zlib

from . import rpe_method_optimizer as O

SCHEMA = 1
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


class HistoryCache:
    """Local SQLite cache of exact records previously validated by Executor.

    ``prior_records(executor_for_method, candidates)`` returns
    ``(known, refused, stats)`` with the same known/refused meaning as the
    existing Executor method. Repeated calls still scan original stat markers,
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
        self.db.commit()
        self.stats_by_method = {}

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def prior_records(self, executor, candidates, *, details=True):
        started = time.monotonic()
        ready = {c['configuration_id']: c for c in candidates
                 if c['ready'] and c['configuration_id']}
        held = {(Path(row['parent_directory'])/'jobs'/row['method']/row['job_id']).resolve()
                for row in getattr(executor, 'prior_q3_holds', ())}
        record_reuse = None if details else getattr(executor, 'record_prior_result_reuse', None)
        stopped = None if details else getattr(executor, '_user_stop_requested', None)
        request_configurations = executor.prior_request_configurations
        validator = _validator(executor)
        contexts, comparisons = {}, {}
        known, refused = {}, []
        stats = dict(cache_schema=SCHEMA, cache_path=str(self.path), scanned_jobs=0,
                     cache_hits=0, request_cache_hits=0, unrelated_request_hits=0,
                     parsed_jobs=0, validated_jobs=0, invalidated_jobs=0,
                     context_changes={}, current_unresolved=[], evidence=[], scientific_calls=0, collection_calls=0,
                     interrupted=False)
        for base in executor.prior_run_dirs:
            directory = Path(base) / 'jobs' / executor.method
            if not directory.exists():
                continue
            jobs = (sorted(p for p in directory.iterdir() if p.is_dir()) if details else
                    [outcome.parent for outcome in sorted(directory.glob('*/OUTCOME.json'))])
            for job in jobs:
                if stopped is not None and stopped():
                    stats['interrupted'] = True
                    break
                request_path = job / 'REQUEST.json'
                previous = request_configurations.get(request_path)
                if not details and previous is not None and previous[1] not in ready and (not held or job.resolve() not in held):
                    try:
                        info = request_path.stat()
                    except FileNotFoundError:
                        pass
                    else:
                        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                        if identity == previous[0]:
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
                saved = self.db.execute('SELECT signature, context, payload FROM evidence WHERE job=?', (key,)).fetchone()
                cached = None
                if saved is not None and saved[0] == _json(signature):
                    try:
                        cached = _unpack(saved[2])
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
                    self.db.execute('INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?)',
                                    (key, _json(signature), '', _pack(cached)))
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
                            self.db.execute('INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?)',
                                            (key, _json(signature), old_context, _pack(cached)))
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
                        cached.update(record=record, outcome_status=record['status'])
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
                        self.db.execute('INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?)',
                                        (key, _json(signature), context, _pack(cached)))
                    known[(cid, proposal['vector'])] = dict(record, source='PRIOR_ROUND_JOB',
                                                          cost=O.cost(proposal['vector'], candidate['prices']))
                except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
                    refused.append(dict(job=str(job), reason='HISTORY_METADATA_ERROR:' + str(exc)))
            if stats['interrupted']:
                break
        self.db.commit()
        stats.update(known_records=len(known), refused_records=len(refused), wall_seconds=time.monotonic() - started)
        self.stats_by_method[executor.method] = stats
        return known, refused, stats
