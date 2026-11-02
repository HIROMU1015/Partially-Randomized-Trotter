"""Lossless S1 coefficient/target selection, never runs a diagnostic or PSD."""
import hashlib
import json
from pathlib import Path

from .n3_old_family_contract_v1 import decode, fingerprint

IDS = ('S1_edge_pair', 'S1_distributed_three')
PREFIX = 'artifacts/n3_old_family_v1/2026-10-12/'
BASE = 'ce1e9512ac225a234a5ce0f8bce03950dae1f29c'
PAYLOAD = '652667187ce83c7c36041820a0a752669134865f'
PACK = 'artifacts/n3_occupation_redesign_preparation/2026-10-12/proposed_transfer_inputs.json'
LEDGER = 'artifacts/n3_occupation_transfer_v1/2026-10-12/run1/primary_ledger.json'
PACK_SHA = 'c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4'
LEDGER_SHA = '8b1e75e89da78c13b82f24896db387adec6e76597ec12669a7549279592d0d69'
WIDTH_PREFIX = 'artifacts/n3_saved_width_v1/2026-10-12/'
OLD_FP = '41b951dae4968a2b679e61c9e61e761dc8253babeb2c610ccc1886fe3646fcb6'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(root):
    """Select stored values and validate their schema; no new matrix verdict."""
    root = Path(root)
    if digest(root / PACK) != PACK_SHA or digest(root / LEDGER) != LEDGER_SHA:
        raise ValueError('ORIGINAL_INPUT_OR_LEDGER_IDENTITY')
    cases = json.loads((root / PACK).read_bytes())['cases']
    ledger = json.loads((root / LEDGER).read_bytes())
    answer = []
    for name in IDS:
        index = next(i for i, case in enumerate(cases) if case['id'] == name)
        case, primary = cases[index], ledger[index]
        if (case['particles'] != 4 or len(case['active']) != 9 or case['modes'] != 10
                or fingerprint(case) != primary['case_sha256'] or primary['input_id'] != name):
            raise ValueError('FROZEN_S1_CASE_IDENTITY')
        record_path = WIDTH_PREFIX + 'evaluation1/' + name + '.json'
        summary_path = WIDTH_PREFIX + 'evaluation_audit/case_summaries/' + name + '.json'
        record = json.loads((root / record_path).read_bytes())
        summary = json.loads((root / summary_path).read_bytes())
        width = record['widths']['OCC_RAW_ALL_SAVED_SPECTRA']
        if width != summary['widths']['OCC_RAW_ALL_SAVED_SPECTRA']:
            raise ValueError('PUBLISHED_WIDTH_RECORD_IDENTITY')
        winner_index = next(i for i, row in enumerate(record['candidates']) if row['id'] == width['upper_witness_id'])
        winner = record['candidates'][winner_index]
        if (width['positive_proof_delta'] != width['high'] or winner['kind'] != 'OCC'
                or record['U'] != primary['upper']['U'] or record['c_star'] != primary['bounds']['c_star']):
            raise ValueError('FIXED_OCC_TARGET_OR_COMMON_BOUND_IDENTITY')
        active, d = case['active'], case['external']
        instance = {'id': name, 'h_A': [[case['h'][i][j] for j in active] for i in active],
                    'v': [case['h'][i][d] for i in active], 'nu': case['particles'],
                    'U': record['U'], 'c': record['c_star'], 'delta': width['positive_proof_delta']}
        decode(instance)  # Shape, exact rational domain and common bound only.
        answer.append({'instance': instance, 'instance_sha256': fingerprint(instance),
                       'provenance': {'pack_path': PACK, 'pack_sha256': PACK_SHA, 'case_index': index,
                           'case_sha256': primary['case_sha256'], 'active_indices': active, 'external': d,
                           'ledger_path': LEDGER, 'ledger_sha256': LEDGER_SHA,
                           'result_payload_commit': PAYLOAD, 'saved_result_path': record_path,
                           'saved_result_sha256': digest(root / record_path), 'summary_path': summary_path,
                           'summary_sha256': digest(root / summary_path), 'saved_OCC_winner_id': winner['id'],
                           'winner_pointer': '/candidates/' + str(winner_index),
                           'target_policy': 'stored positive_proof_delta = stored OCC high; no rounding or retuning'}})
    return answer


def verify_inputs(root):
    root = Path(root); expected = select(root)
    for row in expected:
        path = root / PREFIX / 'inputs' / (row['instance']['id'] + '.json')
        if json.loads(path.read_bytes()) != row: raise ValueError('EXTRACTED_INSTANCE_LOSSLESS_IDENTITY')
    index = json.loads((root / PREFIX / 'inputs/index.json').read_bytes())
    if [r['input_id'] for r in index['cases']] != list(IDS): raise ValueError('TWO_CASE_ORDER')
    for row in index['cases']:
        if digest(root / row['path']) != row['sha256']: raise ValueError('EXTRACTED_INPUT_FILE_IDENTITY')
    return {'status': 'TWO_S1_LOSSLESS_IDENTITY_SCHEMA_PASS_NO_DIAGNOSTICS', 'cases': list(IDS),
            'new_diagnostic_calls': 0, 'new_PSD_calls': 0, 'new_eigensolver_calls': 0,
            'new_certificate_proposals': 0, 'new_U_c_computations': 0}


def audit_saved_winner(root, row):
    """Authorized saved verification ONLY, not called by selection/preflight."""
    from .n3_occupation_transfer_check_v1 import Checker
    from .n3_old_family_contract_v1 import rational
    root = Path(root); p = row['provenance']; case = json.loads((root / PACK).read_bytes())['cases'][p['case_index']]
    primary = json.loads((root / LEDGER).read_bytes())[p['case_index']]
    saved = json.loads((root / p['saved_result_path']).read_bytes())
    winner = next(r for r in saved['candidates'] if r['id'] == p['saved_OCC_winner_id'])
    checker = Checker(); checker.audit_case(case, primary, OLD_FP)
    A, v, nu, U, c, delta = decode(row['instance']); k = rational(winner['kappa'])
    K = checker.plus(checker.scale(A, k), checker.scale(checker.outer(v), -1))
    s, valid = checker.spectrum(K, nu, winner['parameters'])
    if (not valid or winner['s'] != [s.numerator, s.denominator]
            or k > c - U + delta or k * U - s > delta * (c - U + delta)):
        raise ValueError('SAVED_OCC_TARGET_CERTIFICATE')
    return {'status': 'FROZEN_U_C_AND_SAVED_OCC_TARGET_CERTIFICATE_PASS', 'input_id': row['instance']['id'],
            'original_checker_conditions': checker.checks, 'max_rational_bits': checker.max_bits,
            'new_eigensolver_calls': 0, 'new_U_c_bounds': 0, 'new_OCC_proposals': 0}
