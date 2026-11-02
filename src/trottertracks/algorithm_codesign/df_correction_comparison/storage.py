"""Lossless review-sized JSON storage, without changing JSON value semantics.

Small artifacts retain their original path. Large ones have a .review.json index
and UTF-8 byte chunks; readers use load_json at the original logical path. The
serialization remains json.dump(default=str, indent=2) plus a final newline.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

REVIEW_BYTES = 512 * 1024
CHUNK_BYTES = 256 * 1024


def index_path(path):
    return Path(str(path) + '.review.json')


def _encoded_parts(data, ensure_ascii):
    encoder = json.JSONEncoder(default=str, indent=2, ensure_ascii=ensure_ascii)
    for fragment in encoder.iterencode(data):
        # Bound encoding memory even for a long rational/string field.
        for start in range(0, len(fragment), CHUNK_BYTES // 4):
            yield fragment[start:start + CHUNK_BYTES // 4].encode('utf-8')
    yield b'\n'


def _top_level(data):
    if not isinstance(data, dict):
        return {'type': type(data).__name__}
    return {str(k): {'type': type(v).__name__,
                     **({'items': len(v)} if isinstance(v, (dict, list)) else {})}
            for k, v in data.items()}


def dump_json(path, data, *, replace_existing=False, ensure_ascii=True):
    """Atomically publish one raw JSON or a lossless chunk/index generation.

    No raw+chunk duplication is retained. Prefix replacements preserve the last
    complete generation until the new index is published. Temporary files also
    count towards the unchanged supervisor output limit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = index_path(path)
    if not replace_existing and (path.exists() or idx.exists()):
        raise FileExistsError(path)
    old = json.loads(idx.read_text()) if idx.exists() else None
    generation = path.parent / (path.name + '.chunks') / uuid.uuid4().hex
    generation.mkdir(parents=True)
    records, buffer = [], bytearray()
    total, whole = 0, hashlib.sha256()

    def flush():
        nonlocal total
        if not buffer:
            return
        b = bytes(buffer)
        b.decode('utf-8')  # Every review file is independently valid UTF-8.
        p = generation / ('chunk-%05d.txt' % len(records))
        with p.open('xb') as stream:
            stream.write(b)
        records.append({'order': len(records), 'path': str(p.relative_to(path.parent)),
                        'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()})
        whole.update(b)
        total += len(b)
        buffer.clear()

    for part in _encoded_parts(data, ensure_ascii):
        if len(buffer) + len(part) > CHUNK_BYTES:
            flush()
        buffer.extend(part)
    flush()
    digest = whole.hexdigest()
    # Verify disk bytes, order, byte count and digest before publishing readers.
    disk_hash, disk_bytes = hashlib.sha256(), 0
    for row in records:
        b = (path.parent / row['path']).read_bytes()
        assert len(b) == row['bytes'] and hashlib.sha256(b).hexdigest() == row['sha256']
        disk_hash.update(b); disk_bytes += len(b)
    if disk_bytes != total or disk_hash.hexdigest() != digest:
        raise ValueError('JSON chunk reconstruction identity failed')
    receipt = {'logical_path': path.name, 'original_bytes': total,
               'original_sha256': digest, 'encoding': 'UTF-8',
               'serialization': 'json default=str indent=2 plus newline',
               'ensure_ascii': ensure_ascii, 'chunk_limit_bytes': CHUNK_BYTES,
               'review_limit_bytes': REVIEW_BYTES, 'top_level': _top_level(data),
               'chunks': records, 'reconstruction_verified': True}
    if total <= REVIEW_BYTES:
        if len(records) == 1:
            os.replace(path.parent / records[0]['path'], path)
        else:
            tmp = generation / 'small.partial'
            with tmp.open('xb') as stream:
                for row in records:
                    stream.write((path.parent / row['path']).read_bytes())
            os.replace(tmp, path)
        idx.unlink(missing_ok=True)
        shutil.rmtree(generation)
        receipt['storage'] = 'raw'
    else:
        receipt['storage'] = 'chunks'
        b = (json.dumps(receipt, indent=2) + '\n').encode()
        if len(b) > REVIEW_BYTES:
            raise ValueError('Review index itself exceeds 512 KiB')
        tmp = generation / 'index.partial'
        with tmp.open('xb') as stream:
            stream.write(b)
        os.replace(tmp, idx)
        path.unlink(missing_ok=True)
    # Delete only a previous generation belonging to this logical NEW-run path.
    if old:
        for row in old['chunks']:
            previous = _chunk_path(path, row['path'])
            if previous.parent != generation:
                previous.unlink(missing_ok=True)
        for parent in {_chunk_path(path, row['path']).parent for row in old['chunks']}:
            if parent != generation:
                parent.rmdir()
    return receipt


def _chunk_path(logical, relative):
    path = (logical.parent / relative).resolve()
    permitted = (logical.parent / (logical.name + '.chunks')).resolve()
    if not path.is_relative_to(permitted) or path == permitted:
        raise ValueError('Review chunk escapes its logical artifact directory')
    return path


def iter_verified_bytes(path):
    path = Path(path)
    idx = index_path(path)
    # An index is the atomic commit point for a large generation. During an
    # interrupted small->large prefix replacement, an older raw file may remain;
    # it must not override the fully verified committed index (or hide corruption).
    if not idx.exists():
        with path.open('rb') as stream:
            while b := stream.read(CHUNK_BYTES):
                yield b
        return
    record = json.loads(idx.read_text())
    if record['logical_path'] != path.name or idx.stat().st_size > REVIEW_BYTES:
        raise ValueError('Review index identity/size failed')
    digest, count = hashlib.sha256(), 0
    for i, row in enumerate(record['chunks']):
        if row['order'] != i or not 0 < row['bytes'] <= CHUNK_BYTES:
            raise ValueError('Review chunk order/size failed')
        b = _chunk_path(path, row['path']).read_bytes()
        if len(b) != row['bytes'] or hashlib.sha256(b).hexdigest() != row['sha256']:
            raise ValueError('Review chunk identity failed')
        b.decode('utf-8')
        digest.update(b); count += len(b)
        yield b
    if count != record['original_bytes'] or digest.hexdigest() != record['original_sha256']:
        raise ValueError('Review reconstruction identity failed')


def load_json(path):
    """Only return a value after the COMPLETE reconstruction has been checked."""
    return json.loads(b''.join(iter_verified_bytes(path)))


def artifact_exists(path):
    return Path(path).is_file() or index_path(path).is_file()


def artifact_identity(path):
    digest, count = hashlib.sha256(), 0
    for b in iter_verified_bytes(path):
        digest.update(b); count += len(b)
    return {'logical_path': str(path), 'bytes': count, 'sha256': digest.hexdigest()}
