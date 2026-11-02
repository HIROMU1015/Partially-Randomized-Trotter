"""Verify the H8 B0 acquisition proposal only; no runtime or scientific entry."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
from h8_b0_anchor_plan import validate

BUNDLE = ROOT/'artifacts/resource_applicability/track_a_h8_b0_anchor_readiness/2026-10-16'


def check(bundle=BUNDLE):
    manifest = json.loads((bundle/'publication_files_v1.json').read_text())
    for row in manifest['files']:
        p = Path(row['path'])
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('PUBLIC_RELATIVE_PATH')
        raw = (ROOT/p).read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('PUBLIC_BYTES_CHANGED:'+str(p))
    result = validate(json.loads((bundle/'acquisition_plan_UNAPPROVED_v1.json').read_text()))
    result['public_files_verified'] = len(manifest['files'])
    return result


if __name__=='__main__':
    print(json.dumps(check(), sort_keys=True))
