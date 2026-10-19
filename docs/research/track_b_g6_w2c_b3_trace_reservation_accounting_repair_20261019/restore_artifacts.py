"""Reassemble published artificial-evidence archives; never run the producer."""
from hashlib import sha256
import json
from pathlib import Path
import sys


def main():
    here = Path(__file__).resolve().parent
    archives = json.loads((here / 'archive_parts.json').read_text())['archives']
    if len(sys.argv) != 2 or sys.argv[1] not in {a['path'] for a in archives}:
        raise SystemExit('Usage: python3 restore_artifacts.py ARCHIVE_NAME.tar.gz')
    archive = next(a for a in archives if a['path'] == sys.argv[1])
    blocks = []
    for part in archive['parts']:
        raw = (here / part['path']).read_bytes()
        if len(raw) != part['bytes'] or sha256(raw).hexdigest() != part['sha256']:
            raise SystemExit('Part integrity mismatch: ' + part['path'])
        blocks.append(raw)
    raw = b''.join(blocks)
    if len(raw) != archive['bytes'] or sha256(raw).hexdigest() != archive['sha256']:
        raise SystemExit('Archive integrity mismatch')
    target = here / archive['path']
    if target.exists():
        if target.read_bytes() != raw:
            raise SystemExit('Refusing to overwrite different existing archive')
    else:
        with target.open('xb') as output:
            output.write(raw)
    print(json.dumps(dict(path=str(target), bytes=len(raw), sha256=archive['sha256'])))


if __name__ == '__main__':
    main()
