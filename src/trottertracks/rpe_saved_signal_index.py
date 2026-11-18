"""Disposable local indexes for immutable, provider-validated signal evidence."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zlib


def fixed_locator(ref):
    commit = ref.get('commit') if isinstance(ref, dict) else None
    return isinstance(commit, str) and bool(ref.get('path')) and re.fullmatch('[0-9a-f]{40}', commit) is not None


def load_or_build(directory, identity, build):
    """The builder returns (serializable index, immutable dependencies)."""
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(directory) / (key + '.json.gz')
    try:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            saved = json.load(stream)
        if saved['key'] == key:
            return saved['index']
    except (OSError, EOFError, ValueError, KeyError, TypeError, zlib.error):
        pass
    index, immutable = build()
    if immutable:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
                try:
                    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as stream:
                        json.dump(dict(key=key, index=index), stream, separators=(',', ':'), allow_nan=False)
                    temporary.flush()
                    Path(temporary.name).replace(path)
                finally:
                    Path(temporary.name).unlink(missing_ok=True)
        except OSError:
            pass
    return index


def pauli_index(entry, directory, source, root, packets):
    """Keep the fixed entry's first-hit ordering and its carrier/kernel checks."""
    index = {}
    for packet in packets:
        if not fixed_locator(packet):
            partial = entry.saved_index(root, [packet])
        else:
            def build():
                verified = entry.saved_index(root, [packet])
                handoff = entry.read_ref(root, packet)
                # A skipped row may later become usable if any referenced
                # locator is mutable. In that case always keep the old scan.
                immutable = all(fixed_locator(row.get('mean_source')) and
                                fixed_locator(row['mean_record'].get('source'))
                                for item in handoff['packets'] for row in item['rounds']
                                if row.get('mean_record') and row.get('mean_source'))
                return verified, immutable

            partial = load_or_build(directory, ['pauli-packet-v1', source, packet], build)
        for physical, item in partial.items():
            index.setdefault(physical, item)
    return index
