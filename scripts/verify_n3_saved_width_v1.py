#!/usr/bin/env python3
"""Independent saved posthoc audit: provenance, PSD and polynomial endpoints."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration.n3_occupation_transfer_check_v1 import Checker, hash_value
from trottertracks.representation_exploration.n3_saved_width_v1 import check_interval, decode, encode


def verify_case(case, primary, secondary, record, source_fingerprint, seal):
    c = Checker()
    c.audit_case(case, primary, source_fingerprint)
    primary_cache = c.cache
    c.secondary(case, primary, secondary, seal)
    c.pool, c.cache = primary['proof_pool'], primary_cache
    h, _, active, d, nu, _ = c.decode(case)
    A, v = [[h[i][j] for j in active] for i in active], [h[i][d] for i in active]
    U, bound = c.f(primary['upper']['U']), c.f(primary['bounds']['c_star'])
    if record['U'] != encode(U) or record['c_star'] != encode(bound) or record['g'] != encode(bound - U):
        raise ValueError('common U/c identity')
    if (record['input_id'] != case['id'] or record['base'] != case['base']
            or record['case_sha256'] != primary['case_sha256'] or record['primary_seal'] != seal
            or record['source_fingerprint'] != source_fingerprint):
        raise ValueError('case/seal identity')
    expected = {}

    def add(kind, kappa, s, parameters, origin, selected=False):
        identity = hash_value({'input': case['id'], 'kind': kind, 'kappa': encode(kappa), 's': encode(s), 'parameters': parameters})
        if identity not in expected:
            expected[identity] = {'kind': kind, 'kappa': kappa, 's': s, 'parameters': parameters, 'origins': [], 'selected': False}
        expected[identity]['origins'].append(origin)
        expected[identity]['selected'] |= selected

    for ri, row in enumerate(primary['raw_rows']):
        for mi, method in enumerate(row['methods']):
            for ai, w in enumerate(method['attempts']):
                origin = {'row': row['row'], 'method': method['method'], 'attempt': ai,
                          'pointer': f'/raw_rows/{ri}/methods/{mi}/attempts/{ai}',
                          'original_delta_goal': row['delta'], 'original_accepted': w.get('accepted', False)}
                if w['kind'] == 'norm':
                    add('NORM', F(0), -sum(x * x for x in v), {'definition': 'saved coefficient norm-only identity'}, origin)
                elif w['kind'] == 'old' and w.get('PSD_valid'):
                    kappa, mu = c.f(w['kappa']), c.f(w['mu'])
                    matrix = c.plus(c.shift(c.scale(A, kappa), mu), c.scale(c.outer(v), -1))
                    if not c.proof(matrix, w['proof'])[0]: raise ValueError('old proof')
                    add('OLD', kappa, nu * mu, {'mu': w['mu'], 'proof': w['proof']}, origin)
                elif w['kind'] == 'occ':
                    kappa = c.f(w['kappa'])
                    matrix = c.plus(c.scale(A, kappa), c.scale(c.outer(v), -1))
                    saved_pairs = list(enumerate(w.get('spectral_attempts', [])))
                    if w.get('spectral') is not None: saved_pairs.append(('selected', w['spectral']))
                    for si, s in saved_pairs:
                        lower, valid = c.spectrum(matrix, nu, s)
                        if not valid: continue
                        ref = dict(origin, spectral_entry=si)
                        ref['pointer'] += '/spectral' if si == 'selected' else '/spectral_attempts/' + str(si)
                        add('OCC', kappa, lower, s, ref, si == 'selected')
    observed = {r['id']: r for r in record['candidates']}
    if len(observed) != len(record['candidates']) or set(observed) != set(expected):
        raise ValueError('incomplete or augmented finite saved set')
    for identity, truth in expected.items():
        r = observed[identity]
        if r['kind'] != truth['kind'] or r['origins'] != truth['origins'] or r['parameters'] != truth['parameters']:
            raise ValueError('candidate provenance')
        if r['kappa'] != encode(truth['kappa']) or r['s'] != encode(truth['s']) or r['q'] != encode(truth['kappa'] * U - truth['s']):
            raise ValueError('fixed witness scalar parameters')
        if r['selected_by_generator'] != truth['selected']: raise ValueError('selected subset attribution')
        check_interval(bound - U, truth['kappa'], truth['kappa'] * U - truth['s'], r['width'])
    membership = {'BASE': [key for key,t in expected.items() if t['kind'] != 'OCC'],
                  'NORM': [key for key,t in expected.items() if t['kind'] == 'NORM'],
                  'FIVE': [key for key,t in expected.items() if any(o['method'] == 'FIVE' for o in t['origins'])],
                  'OLD_DIRECT': [key for key,t in expected.items() if any(o['method'] == 'OLD_DIRECT' for o in t['origins'])],
                  'OCC_RAW_ALL_SAVED_SPECTRA': [key for key,t in expected.items() if t['kind'] == 'OCC'],
                  'OCC_SELECTED_SPECTRA': [key for key,t in expected.items() if t['kind'] == 'OCC' and t['selected']]}
    for name, keys in membership.items():
        width = record['widths'][name]
        if not keys:
            if width is not None: raise ValueError('empty pool width')
            continue
        lower = min(decode(observed[key]['width']['low']) for key in keys)
        upper_key = min(keys, key=lambda key: (decode(observed[key]['width']['high']), key))
        upper = decode(observed[upper_key]['width']['high'])
        if width != {'low': encode(lower), 'high': encode(upper), 'upper_witness_id': upper_key,
                     'positive_proof_delta': observed[upper_key]['width']['positive_proof_delta'], 'candidate_count': len(keys)}:
            raise ValueError('saved set minimum envelope')
    occ, base = record['widths']['OCC_RAW_ALL_SAVED_SPECTRA'], record['widths']['BASE']
    if occ is None or base is None: outcome = 'MISSING_SAVED_COVERAGE'
    elif decode(occ['high']) < decode(base['low']): outcome = 'OCC_STRICTLY_NARROWER_IN_SAVED_SET'
    elif decode(base['high']) < decode(occ['low']): outcome = 'BASE_STRICTLY_NARROWER_IN_SAVED_SET'
    elif decode(occ['low']) == decode(occ['high']) == decode(base['low']) == decode(base['high']): outcome = 'EXACT_EQUAL_WIDTH'
    else: outcome = 'OVERLAPPING_INTERVALS_UNDETERMINED'
    if record['comparison'] != outcome: raise ValueError('width comparison')
    if any(record[k] != 0 for k in ('optimizer_calls', 'numerical_eigensolver_calls', 'new_certificate_proposals')):
        raise ValueError('scope')
    if record['original_primary_result_changed'] or not record['mandatory_stop']: raise ValueError('STOP/result preservation')
    return {'status': 'SAVED_POSTHOC_CASE_POLYNOMIAL_AND_PSD_PASS', 'input_id': case['id'],
            'fixed_candidates': len(expected), 'old_checker_conditions': c.checks, 'new_eigensolver_calls': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads((ROOT / 'docs/research/n3_saved_width_v1_execution_plan_20261012.json').read_bytes())
    for path, sha in dict(plan['source_sha256'], **plan['saved_input_sha256']).items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha: raise ValueError('frozen bytes')
    pack = json.loads((ROOT / plan['input_path']).read_bytes())
    primary = json.loads((ROOT / plan['ledger_path']).read_bytes())
    secondary = json.loads((ROOT / plan['secondary_path']).read_bytes())
    rows = [verify_case(case, record, s, json.loads((args.directory / (case['id'] + '.json')).read_bytes()),
                        plan['original_source_fingerprint'], plan['ledger_sha256']) for case, record, s in zip(pack['cases'], primary, secondary)]
    print(json.dumps({'status': 'INDEPENDENT_SAVED_POSTHOC_WIDTHS_PASS', 'cases': rows, 'new_eigensolver_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
