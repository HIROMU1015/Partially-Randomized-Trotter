"""Verify the public metadata-only H8 B1 bundle. Does not fit or compile."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Load this stdlib guard without running trotterlib's plotting-package __init__.
sys.path.insert(0, str(ROOT / 'src' / 'trotterlib'))
from h8_phase_b1_contract import inspect_anchors, select_b2, validate_draft

BUNDLE = ROOT / 'artifacts/resource_applicability/track_a_h8_phase_b1_readiness/2026-10-16'


def check(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'publication_files_v1.json').read_text())
    for row in manifest['files']:
        p = Path(row['path'])
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('PUBLIC_PATH')
        raw = (ROOT / p).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('CHANGED_PUBLIC_FILE:' + str(p))
    read = lambda name: json.loads((bundle / name).read_text())
    validate_draft(read('execution_contract_DRAFT_v1.json'))
    result = inspect_anchors(read('anchor_registry_v1.json')['anchors'])
    if result != read('anchor_gate_v1.json'):
        raise ValueError('ANCHOR_GATE_CHANGED')
    if select_b2(read('S4_eligibility_metadata_v1.json')['cells']) != read('B2_selection_v1.json')['axes']:
        raise ValueError('REGISTERED_B2_SELECTION_CHANGED')
    return dict(check='PASS_METADATA_ONLY', scientific_readiness=result['status'],
                files_verified=len(manifest['files']), actual_fit=0,
                new_compile=0, cost_reaccount=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    print(json.dumps(check(parser.parse_args().bundle), sort_keys=True))
