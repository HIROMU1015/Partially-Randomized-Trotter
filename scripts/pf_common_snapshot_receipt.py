"""Read-only original H6 NPY receipt validation, without NumPy or science calls."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

# Same input archive cap as the original H6 completion loader.
SNAPSHOT_CAP = 16 * 2**20
HEADER_CAP = 65536


def require(condition, label):
    if not condition:
        raise ValueError('H6_SNAPSHOT_PREFLIGHT:' + label)


def npy_record(data):
    """Describe original NPY bytes, preserving scalar shape () and dtype."""
    require(data[:6] == b'\x93NUMPY', 'NPY_MAGIC')
    version = tuple(data[6:8])
    require(version in ((1, 0), (2, 0), (3, 0)), 'NPY_VERSION')
    length_bytes = 2 if version == (1, 0) else 4
    offset = 8 + length_bytes
    require(len(data) >= offset, 'NPY_HEADER_LENGTH')
    header_length = int.from_bytes(data[8:offset], 'little')
    require(0 < header_length <= HEADER_CAP and offset + header_length <= len(data), 'NPY_HEADER_SIZE')
    encoding = 'utf-8' if version == (3, 0) else 'latin1'
    try:
        header = ast.literal_eval(data[offset:offset + header_length].decode(encoding))
    except (ValueError, SyntaxError, UnicodeError, TypeError) as exc:
        raise ValueError('H6_SNAPSHOT_PREFLIGHT:NPY_HEADER') from exc
    require(isinstance(header, dict) and set(header) == {'descr', 'fortran_order', 'shape'}, 'NPY_HEADER_KEYS')
    shape = header['shape']
    require(isinstance(shape, tuple) and all(type(n) is int and n >= 0 for n in shape), 'NPY_SHAPE')
    require(header['fortran_order'] is False, 'ORIGINAL_C_ORDER')
    # Original fixed H6 layout: little-endian numeric and Unicode arrays only.
    dtype = header['descr']
    require(isinstance(dtype, str), 'NPY_DTYPE')
    match = re.fullmatch(r'<([ficU])([1-9][0-9]*)', dtype)
    require(match is not None, 'NPY_DTYPE')
    kind, width = match.groups()
    width = int(width)
    require(kind == 'U' or (kind, width) in (('f', 8), ('i', 8), ('c', 16)), 'ORIGINAL_DTYPE')
    payload = data[offset + header_length:]
    itemsize = width * (4 if kind == 'U' else 1)
    require(len(payload) == math.prod(shape) * itemsize, 'NPY_PAYLOAD_SIZE')
    return dict(shape=list(shape), dtype=dtype, sha256=hashlib.sha256(payload).hexdigest())


def check_original_snapshot(snapshot, receipt, *, expected_sha256=None):
    """Check archive and each receipt entry before a pack or grant is created."""
    snapshot = Path(snapshot)
    require(snapshot.stat().st_size <= SNAPSHOT_CAP, 'ARCHIVE_SIZE')
    raw = snapshot.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(isinstance(receipt, dict) and isinstance(receipt.get('arrays'), dict) and bool(receipt['arrays']), 'RECEIPT_ARRAYS')
    require(digest == receipt.get('sha256'), 'ARCHIVE_SHA256')
    if expected_sha256 is not None:
        require(digest == expected_sha256, 'BINDING_SHA256')
    require(receipt.get('bytes', len(raw)) == len(raw), 'ARCHIVE_RECEIPT_SIZE')
    records = {}
    with zipfile.ZipFile(snapshot) as archive:
        members = archive.infolist()
        require(len({e.filename for e in members}) == len(members), 'DUPLICATE_MEMBER')
        expected = {name + '.npy' for name in receipt['arrays']}
        require({e.filename for e in members} == expected, 'ARRAY_NAMES')
        require(sum(e.file_size for e in members) <= SNAPSHOT_CAP, 'UNCOMPRESSED_SIZE')
        for e in members:
            record = npy_record(archive.read(e))
            name = e.filename[:-4]
            require(record == receipt['arrays'][name], 'ARRAY_RECEIPT:' + name)
            records[name] = record
    return dict(status='PASS', snapshot_sha256=digest, bytes=len(raw), arrays=records,
                array_count=len(records), numpy_imported=False, snapshot_restore_calls=0,
                new_grants=0, scientific_actions=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    receipt = json.loads(Path(args.receipt).read_text())
    print(json.dumps(check_original_snapshot(args.snapshot, receipt,
        expected_sha256=args.expected_sha256), indent=2, sort_keys=True))
