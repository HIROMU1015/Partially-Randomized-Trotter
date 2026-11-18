"""Lossless optimizer-only storage for sidecars embedded in a completed Q3 result."""
from contextlib import closing, contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from unittest.mock import patch
import zlib

CHECKPOINTS = 'Q3_CHECKPOINTS.sqlite3'
WITNESS = 'Q3_COMPACT_EVIDENCE.json'
SCHEMA = 'RPE_Q3_COMPACT_EVIDENCE_V1'
SIDECARS = frozenset(('API_RETURN.json', 'PARTIAL_COARSE.json',
                     'PARTIAL_LEMMA.json', 'PARTIAL_TOTAL_DP.json'))
COMPLETE = frozenset(('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',
                      'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD', 'INCONCLUSIVE'))


def encode(value):
    # Same bytes as the pinned writer, including order, signed zero and giant integers.
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_sidecars(result):
    record = result['record']
    diagnostic = record['diagnostic']
    bounds = diagnostic['deterministic_risk']['bounds']
    return {'API_RETURN.json': [record], 'PARTIAL_COARSE.json': diagnostic['partial_decoder_risk'],
            'PARTIAL_LEMMA.json': bounds['lemma'], 'PARTIAL_TOTAL_DP.json': bounds['total_state_dp']}


@contextmanager
def capture(writer, folder):
    """Use one durable transaction per early write in the existing forked worker."""
    original = writer.write_new
    checkpoint = folder / CHECKPOINTS

    def write_new(path, value):
        path = Path(path)
        if path.parent != folder or path.name not in SIDECARS:
            return original(path, value)
        if path.exists():
            raise FileExistsError(path)
        raw = encode(value)
        with closing(sqlite3.connect(checkpoint)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS artifacts (name TEXT PRIMARY KEY, sha TEXT, gzip BLOB)')
            db.execute('INSERT INTO artifacts VALUES (?, ?, ?)',
                       (path.name, sha(raw), gzip.compress(raw, compresslevel=1, mtime=0)))

    with patch.object(writer, 'write_new', write_new):
        yield


def checkpoints(folder):
    result = {}
    path = folder / CHECKPOINTS
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True)) as db:
        for name, digest, compressed in db.execute('SELECT name, sha, gzip FROM artifacts'):
            try:
                data = gzip.decompress(compressed)
            except (EOFError, zlib.error) as exc:
                raise ValueError('Q3 checkpoint compression is corrupt') from exc
            if name not in SIDECARS or sha(data) != digest:
                raise ValueError('Q3 checkpoint contents differ')
            result[name] = data
    return result


def verified_sidecars(result_bytes, witness):
    if witness.get('schema') != SCHEMA or witness.get('result_sha256') != sha(result_bytes):
        raise ValueError('Q3 compact evidence does not match the original RESULT')
    payloads = {name: encode(obj) for name, obj in source_sidecars(json.loads(result_bytes)).items()}
    if witness.get('sha256') != {name: sha(raw) for name, raw in payloads.items()}:
        raise ValueError('Q3 compact sidecar checksum differs')
    return payloads


class ReadableArtifact:
    def __init__(self, data):
        self.data = data

    def read_text(self):
        return self.data.decode()


class ValidatorDirectory:
    """Serve worker-emitted bytes to the pinned validator through its existing reads."""
    def __init__(self, folder):
        self.folder = folder
        self.payloads = {}
        if (folder / CHECKPOINTS).is_file():
            self.payloads = checkpoints(folder)
        elif (folder / WITNESS).is_file():
            self.payloads = verified_sidecars((folder / 'RESULT.json').read_bytes(),
                                             json.loads((folder / WITNESS).read_bytes()))

    def __truediv__(self, filename):
        path = self.folder / filename
        if path.is_file() or filename not in self.payloads:
            return path
        return ReadableArtifact(self.payloads[filename])


def compact(folder, outcome_path):
    """Remove the staging database only after durable result, verdict and byte proof."""
    db = folder / CHECKPOINTS
    if not db.is_file():
        return False
    result_path = folder / 'RESULT.json'
    result = result_path.read_bytes()
    actual = checkpoints(folder)
    recovered = source_sidecars(json.loads(result))
    if actual.keys() != SIDECARS or any(encode(value) != actual[name] for name, value in recovered.items()):
        # Preserve the independently written evidence if the result layout differs.
        return False
    receipt = dict(schema=SCHEMA, result_sha256=sha(result),
                   sha256={name: sha(raw) for name, raw in actual.items()})
    marker = folder / WITNESS
    if marker.exists() and json.loads(marker.read_bytes()) != receipt:
        raise ValueError('Existing Q3 compact witness differs')
    if not marker.exists():
        with marker.open('xb') as file:
            file.write(encode(receipt))
            file.flush()
            os.fsync(file.fileno())
    for path in (result_path, outcome_path, folder/'PERSISTED.json', folder/'RESOURCE_FINAL.json',
                 outcome_path.parent/'SOURCE.json', outcome_path.parent/'SUPERVISION.json'):
        with path.open('rb') as file:
            os.fsync(file.fileno())
    # Persist the file entries before removing evidence. A crash may leave both
    # copies; then collection still uses the independent worker checkpoints.
    for directory in (folder, outcome_path.parent):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    db.unlink()
    return True


def export(folder, destination):
    """Materialize original sidecars explicitly, including partial-only recoveries."""
    view = ValidatorDirectory(folder)
    if not view.payloads:
        raise ValueError('No compact Q3 artifacts to export')
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in sorted(view.payloads.items()):
        with (destination / name).open('xb') as file:
            file.write(data)
