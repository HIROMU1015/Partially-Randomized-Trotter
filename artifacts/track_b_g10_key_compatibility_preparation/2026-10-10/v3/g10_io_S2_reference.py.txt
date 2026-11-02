"""G10 v2 bounded output only; no scientific imports or calculations.

The supported tree is JSON scalars/string-key dict/list/tuple/Fraction.
Do not mutate it during validation/encoding. A final JSON alone is insufficient:
a matching successful terminal STOP receipt is required for scientific use.
"""
import hashlib
import json
import math
import os
import resource
import signal
import time
import traceback
from fractions import Fraction
from pathlib import Path

from .rte_reallocation.launch import BudgetGuard

CHUNK_CHARACTERS = 8192  # UTF-8 writes <=32768 bytes
TERMINAL_RESERVE = 32768
RECEIPT_CAP = 16384
TELEMETRY_CAP = 32768


def protected_check_streaming(root, contract):
    """Same G7 protected hash/prefix rule without a whole-file byte buffer."""
    root = Path(root)
    ledger = json.loads((root/contract['protected_ledger']).read_text())
    failures = []
    for relative, record in ledger.items():
        if relative.lower().endswith('.npz'):
            raise PermissionError('NPZ path forbidden before access')
        digest, size = hashlib.sha256(), 0
        limit = record['bytes'] if relative in contract['append_only_paths'] else None
        with (root/relative).open('rb') as stream:
            while limit is None or size < limit:
                chunk = stream.read(32768 if limit is None else min(32768, limit-size))
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != record['sha256']:
            failures.append(relative)
    return {'protected_paths': len(ledger), 'violations': failures,
            'old_sources_results_authorizations_markers_STOP_unchanged': not failures}


class FractionEncoder(json.JSONEncoder):
    def default(self, value):
        if isinstance(value, Fraction):
            return str(value)
        return super().default(value)


def validate_tree(value, check=lambda: None):
    """Depth-sized ancestor set, no graph copy; permit shared acyclic subtrees."""
    ancestors = set()
    visited = 0

    def walk(item):
        nonlocal visited
        visited += 1
        if visited % 1024 == 0:
            check()
        if isinstance(item, (dict, list, tuple)):
            ident = id(item)
            if ident in ancestors:
                raise ValueError('circular JSON tree')
            ancestors.add(ident)
            try:
                if isinstance(item, dict):
                    for key, child in item.items():
                        if not isinstance(key, str):
                            raise TypeError('G10 JSON keys must be strings')
                        walk(child)
                else:
                    for child in item:
                        walk(child)
            finally:
                ancestors.remove(ident)
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError('non-finite JSON scalar')
        elif item is not None and not isinstance(item, (str, int, bool, Fraction)):
            raise TypeError('unsupported G10 JSON type: '+type(item).__name__)
    check()
    walk(value)
    check()


def iter_json_bytes(value, check=lambda: None):
    validate_tree(value, check)
    encoder = FractionEncoder(indent=2, ensure_ascii=False, allow_nan=False)
    # iterencode does not materialize CPython encode()'s list(chunks)/join.
    # CPython may still construct one entire escaped string token. G10 fields
    # contain short strings; this is not a general bound on arbitrary tokens.
    buffer = bytearray()
    for index, token in enumerate(encoder.iterencode(value)):
        if index % 256 == 0:
            check()
        for start in range(0, len(token), CHUNK_CHARACTERS):
            piece = token[start:start+CHUNK_CHARACTERS].encode('utf-8')
            if len(buffer)+len(piece) > 32768:
                check()
                yield bytes(buffer)
                buffer.clear()
            buffer.extend(piece)
    if buffer:
        check()
        yield bytes(buffer)
    check()
    yield b'\n'


def memory_snapshot():
    current = None
    try:
        with open('/proc/self/statm', encoding='ascii') as stream:
            current = int(stream.read().split()[1])*os.sysconf('SC_PAGE_SIZE')//1024
    except (OSError, ValueError, IndexError):
        pass
    return {'current_RSS_KiB': current,
            'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


class IOBudgetGuard(BudgetGuard):
    """Same caps/per-key guard, with a bounded diagnostic-only failure window."""
    def suspend_for_failure_receipt(self):
        # ru_maxrss is sticky: a failed peak guard cannot pass after cleanup.
        # No science/large serialization may run in this window. __exit__ later
        # restores timers/limits; existing AS/CPU limits remain active meanwhile.
        signal.setitimer(signal.ITIMER_REAL, 0)


class OutputSession:
    """Exclusive files, bounded writes, incremental identity, terminal protocol.

    On failure retain .partial (or an uncommitted final) as technical evidence.
    Never overwrite an old marker/result/receipt. The aggregate output cap
    includes marker, telemetry, payload, terminal files; hardlinks count once.
    """
    def __init__(self, directory, cap, guard, provenance):
        self.directory = Path(directory)
        self.cap, self.guard, self.provenance = cap, guard, provenance
        self.bytes_used = sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())
        self.stage = 'before_science_imports'
        self.phase = 'not_started'
        self.payload_bytes = 0
        self.digest = hashlib.sha256()
        self.partial = self.directory/'result_v1.json.partial'
        self.final = self.directory/'result_v1.json'
        self.promoted = False
        self.completion = self.directory/'COMPLETED.v2'
        self.completion_created = False
        self.telemetry_bytes = 0
        self.telemetry = self.directory/'io_stages_v2.jsonl'

    def _room(self, count, terminal=False):
        reserve = 0 if terminal else TERMINAL_RESERVE
        if self.bytes_used+count+reserve > self.cap:
            raise RuntimeError('registered aggregate output byte cap; no retry')

    def snapshot(self, stage, monitored=True):
        self.stage = stage
        if monitored:
            self.guard.check()
        record = {'stage': stage, 'phase': self.phase, **self.guard.usage(),
                  **memory_snapshot(), 'output_bytes': self.bytes_used,
                  'payload_bytes': self.payload_bytes}
        raw = (json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n').encode()
        if self.telemetry_bytes+len(raw) > TELEMETRY_CAP:
            raise RuntimeError('bounded I/O telemetry cap')
        self._room(len(raw))
        mode = 'ab' if self.telemetry_bytes else 'xb'
        with self.telemetry.open(mode) as stream:
            written = stream.write(raw)
            self.bytes_used += written
            self.telemetry_bytes += written
            if written != len(raw):
                raise OSError('short telemetry write')
            stream.flush()
        if monitored:
            self.guard.check()

    def _small(self, name, value, monitored):
        raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
        if len(raw) > RECEIPT_CAP:
            raise RuntimeError('bounded receipt cap')
        self._room(len(raw), terminal=True)
        if monitored:
            self.guard.check()
        with (self.directory/name).open('xb') as stream:
            written = stream.write(raw)
            self.bytes_used += written
            if written != len(raw):
                raise OSError('short receipt write')
            stream.flush()
            os.fsync(stream.fileno())
            if monitored:
                self.guard.check()
        if monitored:
            self.guard.check()

    def write_result(self, result):
        self.phase = 'validate_encode_write'
        self.snapshot('before_result_stream')
        # Unbuffered writes give a precise successfully-written prefix identity.
        with self.partial.open('xb', buffering=0) as stream:
            for chunk in iter_json_bytes(result, self.guard.check):
                self._room(len(chunk))
                written = stream.write(chunk)
                if written is None:
                    written = 0
                self.bytes_used += written
                self.payload_bytes += written
                self.digest.update(chunk[:written])
                if written != len(chunk):
                    raise OSError('short result write')
                self.guard.check()
            self.phase = 'flush_fsync'
            self.snapshot('before_result_flush')
            stream.flush()
            os.fsync(stream.fileno())
            self.guard.check()
            self.phase = 'close'
        self.guard.check()
        self.phase = 'verify_disk_identity'
        digest, size = hashlib.sha256(), 0
        with self.partial.open('rb') as stream:
            while True:
                self.guard.check()
                chunk = stream.read(32768)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        if size != self.payload_bytes or digest.hexdigest() != self.digest.hexdigest():
            raise OSError('result disk identity mismatch')
        self.snapshot('result_closed_identity_verified')
        # Exclusive hardlink promotion: same-directory, no overwriting race.
        self.phase = 'promote'
        os.link(self.partial, self.final)
        self.promoted = True
        self.partial.unlink()  # only the newly-created temporary name
        self.guard.check()
        self.snapshot('result_promoted_pending_terminal')
        return {'path': self.final.name, 'bytes': size, 'sha256': digest.hexdigest()}

    def success(self, identity):
        self.phase = 'terminal_success'
        self.snapshot('before_success_STOP')
        self._small('STOP.json', {
            'status': 'G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE',
            'completion_protocol': 'g10-stream-and-terminal-v2',
            'result': identity, 'scientific_result_committed': True,
            'resource_after_result_io': self.guard.usage(),
            'memory_after_result_io': memory_snapshot(),
            'mandatory_STOP': True, 'next_science_authorized': False,
            'research_owner': 'GPT / user', 'retries': 0,
        }, monitored=True)
        # Commit token is created LAST, after STOP close and final guarded check.
        # Its existence plus verified matching STOP/result is required by readers.
        self.guard.check()
        with self.completion.open('xb') as stream:
            self.completion_created = True
            stream.flush()
            os.fsync(stream.fileno())
        self.guard.check()
        # No scientific work or large I/O follows the completion token.

    def failure(self, exc, row_count, new_calls, reused):
        self.guard.suspend_for_failure_receipt()
        # Invalidate only our own fresh commit token BEFORE fallible receipt I/O.
        # Existing markers/STOP/foreign tokens are never removed or overwritten.
        if self.completion_created:
            self.completion.unlink()
            self.completion_created = False
        reason = type(exc).__name__+': '+str(exc)[:2048]
        frames = [{'file': Path(frame.filename).name,
                   'line': frame.lineno, 'function': frame.name}
                  for frame in traceback.extract_tb(exc.__traceback__, limit=-4)]
        # A traceback can otherwise keep collect()'s old/pending/native frames
        # alive during failure reporting, even after result.clear().
        exc.__traceback__ = None
        exc.__context__ = None
        exc.__cause__ = None
        value = {
            'status': 'G10_TECHNICAL_INCONCLUSIVE', 'technical_reason': reason,
            'failed_stage': self.stage, 'failed_phase': self.phase,
            'provenance': self.provenance, 'resource': self.guard.usage(),
            **memory_snapshot(), 'rows_retained_diagnostic_only': row_count,
            'new_synthesis_calls': new_calls, 'reused_keys': reused,
            'partial_output': {'path': self.final.name if self.promoted else self.partial.name,
                               'written_bytes': self.payload_bytes,
                               'written_prefix_sha256': self.digest.hexdigest(),
                               'identity_reverified_after_failure': False},
            'prefix_rows_usable_for_final_research_decision': False,
            'scientific_result_committed': False, 'runs': 1, 'retries': 0,
            'mandatory_STOP': True, 'next_science_authorized': False,
            'failure_reporting': 'best effort bounded receipt; AS/CPU limits retained; no sticky-peak recheck',
            'exception_location': frames,
        }
        # Never re-encode result, never remove an old STOP/token or retry I/O.
        self._small('failure_receipt_v2.json', value, monitored=False)
        if not (self.directory/'STOP.json').exists():
            self._small('STOP.json', {
                'status': 'G10_TECHNICAL_INCONCLUSIVE',
                'scientific_result_committed': False, 'mandatory_STOP': True,
                'next_science_authorized': False, 'retries': 0,
                'failure_receipt': 'failure_receipt_v2.json',
            }, monitored=False)


def verify_completed(directory):
    """Stdlib read-only completion check; never classify a technical prefix."""
    directory = Path(directory)
    if not (directory/'COMPLETED.v2').is_file() or (directory/'failure_receipt_v2.json').exists():
        raise PermissionError('G10 v2 output is not scientifically committed')
    stop = json.loads((directory/'STOP.json').read_text())
    if (stop.get('completion_protocol') != 'g10-stream-and-terminal-v2'
            or stop.get('status') != 'G10_DEGREE_MATCHED_NATIVE_RESOURCE_MAP_COMPLETE'
            or stop.get('scientific_result_committed') is not True
            or stop.get('mandatory_STOP') is not True
            or stop.get('next_science_authorized') is not False):
        raise PermissionError('G10 v2 terminal receipt invalid')
    identity = stop['result']
    if identity['path'] != 'result_v1.json':
        raise PermissionError('unexpected result path')
    digest, size = hashlib.sha256(), 0
    with (directory/identity['path']).open('rb') as stream:
        for chunk in iter(lambda: stream.read(32768), b''):
            size += len(chunk)
            digest.update(chunk)
    if size != identity['bytes'] or digest.hexdigest() != identity['sha256']:
        raise PermissionError('committed result identity mismatch')
    return identity
