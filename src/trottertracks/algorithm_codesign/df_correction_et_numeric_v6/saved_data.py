"""Read-only saved-substage acquisition. No recipes, arrays or solvers."""
import hashlib
import json
import math
import struct
from pathlib import Path
from ..df_correction_comparison.storage import artifact_identity, load_json


def verify_file(root, expected):
    path = Path(root)/expected['path']
    raw = path.read_bytes()
    if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
        raise PermissionError('Saved evidence identity mismatch: '+expected['path'])
    return path


def frame_identity(frame, index, r):
    if not isinstance(frame, list) or len(frame) != r or any(len(row) != 2 for row in frame):
        raise PermissionError('Saved frame shape/rank changed')
    raw = bytearray()
    for row in frame:
        for pair in row:
            if len(pair) != 2 or not all(isinstance(v, str) for v in pair):
                raise PermissionError('Saved frame hex encoding changed')
            a, b = map(float.fromhex, pair)
            if not math.isfinite(a) or not math.isfinite(b):
                raise PermissionError('Nonfinite frame')
            raw.extend(struct.pack('<dd', a, b))
    return {'index': index, 'dtype_str': '<c16', 'shape': [r, 2],
            'order': 'C', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def load_saved_inputs(root, contract):
    saved = contract['resumption']
    path = verify_file(root, saved['saved_E_inputs'])
    inputs = json.loads(path.read_text())
    if set(inputs) != {'3', '6', '_process_receipt'}:
        raise PermissionError('Saved E input domains changed')
    for r in (3, 6):
        frames = inputs[str(r)]
        if len(frames) != 3 or [frame_identity(f, i, r) for i, f in enumerate(frames)] != saved['frame_identities'][str(r)]:
            raise PermissionError('Saved frame ordered bytes mismatch')
    return {str(r): inputs[str(r)] for r in (3, 6)}


def saved_cd_receipt(root, contract):
    saved = contract['resumption']
    verify_file(root, saved['old_marker'])
    terminal = json.loads(verify_file(root, saved['old_terminal']).read_text())
    summary = json.loads(verify_file(root, saved['saved_CD_summary']).read_text())
    logical = Path(root)/saved['saved_CD_logical_path']
    actual = artifact_identity(logical)  # Verifies all UTF-8 chunks before acceptance.
    if actual['bytes'] != saved['saved_CD_bytes'] or actual['sha256'] != saved['saved_CD_sha256']:
        raise PermissionError('Saved CD original bytes changed')
    cd = load_json(logical)
    expected = [(arm, delta, n) for delta, n in zip(contract['CD']['deltas'], contract['CD']['steps']) for arm in ('A', 'B')]
    if ([(x['arm'], x['delta'], x['n']) for x in cd['rows']] != expected
        or terminal['classification'] != 'TECHNICAL_INCONCLUSIVE'
        or 'CD' not in terminal['completed_stages']
        or summary['interpretation'] != 'INCOMPLETE_RUN_DIAGNOSTIC_ONLY'
        or summary['CD_identity']['sha256'] != actual['sha256']):
        raise PermissionError('Saved CD completed substage/source status mismatch')
    return {'old_result_commit': saved['old_science_result_commit'],
            'old_source': saved['old_source'], 'old_authorization': saved['old_authorization'],
            'old_run_classification': terminal['classification'], 'saved_rows': len(cd['rows']),
            'logical_path': saved['saved_CD_logical_path'], 'bytes': actual['bytes'],
            'sha256': actual['sha256'], 'summary': saved['saved_CD_summary'],
            'use': 'CONDITIONAL_COMPLETED_SAVED_SUBSTAGE_SYMBOLIC_COST_ONLY',
            'CD_recomputed': False, 'old_run_reclassified': False,
            'E_cost_multiplication_allowed': False, 'PR_total_winner': None}

