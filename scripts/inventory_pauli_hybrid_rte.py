#!/usr/bin/env python3
"""Saved evidence inventory only. No G arithmetic, minima, RNG or science imports."""
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/pauli_hybrid_rte_preparation/2026-10-25'
OLD = ROOT / 'artifacts/pauli_h2_h6_comparison/2026-10-25'
EPS = ('0.05', '0.01', '0.005', '0.001', '0.0005')
M = {'H2': 14, 'H4': 184, 'H6': 918}


def read(path):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(raw, parse_float=str, parse_int=str)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csvsave(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def group(molecule, prefix):
    n = int(prefix)
    if not 0 <= n <= M[molecule]:
        raise ValueError('PREFIX_DOMAIN')
    return 'RTE' if n == 0 else 'Hybrid' if n < M[molecule] else 'FULL_RETENTION'


def base_row(molecule, cid, cfg, eps, cohort):
    return dict(molecule=molecule, epsilon_sig=eps, candidate_id=cid,
                group=group(molecule, cfg['L_D']), M=str(M[molecule]),
                L_D=cfg['L_D'], q=cfg['q'], r=cfg['r'], K=cfg['K'], cohort=cohort,
                eligibility=None, G_exact=None, G_SE=None, B_exact=None, B_saved_decimal=None,
                bias_cos_exact=None, bias_sin_exact=None, u_cos_exact=None, u_sin_exact=None,
                N_cos=None, N_sin=None, C_cos_exact=None, C_sin_exact=None,
                price_n=None, C_cos_SE=None, C_sin_SE=None,
                shot_cos_reason=None, shot_sin_reason=None,
                exploration_n=None, existing_confirmation_n=None,
                selection_role=None, source_commit=None, source_file=None,
                detail_source=None, price_evidence_locator=None,
                G_availability=None, actual_compiled_rho='UNKNOWN',
                population_price_tail='UNGUARANTEED', hypothetical_rho='0')


def availability(row):
    # Classify stored statuses; do not turn missing G into 0/infinity or a winner.
    if row['G_exact'] is not None:
        return 'SAVED_FINITE_G'
    if row['eligibility'] == 'BIAS_INELIGIBLE':
        return 'BIAS_INELIGIBLE'
    if row['eligibility'] == 'NUMERICALLY_UNDETERMINED':
        return 'NUMERICALLY_UNDETERMINED'
    if 'SAFE_INTEGER_LIMIT' in (row['shot_cos_reason'], row['shot_sin_reason']):
        return 'SHOT_ACCOUNTING_SAFE_INTEGER_LIMIT'
    if row['price_n'] is None:
        return 'PRICE_MISSING'
    return 'ACCOUNTING_G_MISSING_SEE_SAVED_REASONS'


def saved_csv_value(value):
    # H4 writes JSON null as the literal CSV cell "null"; H2/H6 use empty cells.
    # Preserve missing semantics, never classify either spelling as finite G.
    return None if value in ('', 'null') else value


def h26_rows():
    result = []
    for n in (2, 6):
        molecule = f'H{n}'
        stored = read(OLD / molecule / 'candidate_results.json.gz')
        confirmation_path = OUT / 'evidence' / molecule / 'confirmation01/results.json'
        confirmed = read(confirmation_path)
        plan = read(confirmation_path.with_name('plan.json'))
        cn = {r['candidate_id']: r['price_statistics']['n'] for r in confirmed}
        en = {r['candidate_id']: (r['price_statistics']['n'] if r['price_statistics'] else None)
              for r in stored if r['config']['method'] == 'PR'}
        for original in stored + confirmed:
            if original['config']['method'] != 'PR':
                continue
            x = original
            row = base_row(molecule, x['candidate_id'], x['config'], x['epsilon_sig'], x['cohort'])
            for key in ('eligibility', 'G_exact', 'G_SE', 'N_cos', 'N_sin'):
                row[key] = x.get(key)
            row['B_exact'] = x['target']['B_saved_exact']
            st = x['price_statistics']
            for axis, short in (('cosine', 'cos'), ('sine', 'sin')):
                target = x['target']['axes'][axis]
                row[f'bias_{short}_exact'] = target['signed_bias_exact']
                row[f'u_{short}_exact'] = target['u_enc_exact']
                row[f'shot_{short}_reason'] = x['axes'][axis]['rho_zero'].get('reason')
                if st:
                    row[f'C_{short}_exact'] = st['mean_exact'][axis]
                    row[f'C_{short}_SE'] = st['SE'][axis]
            row.update(price_n=st['n'] if st else None, exploration_n=en[x['candidate_id']],
                       existing_confirmation_n=cn.get(x['candidate_id']),
                       selection_role=('EXISTING_EXPLORATION_NOMINEE' if
                         x['candidate_id'] == plan['nominees'][x['epsilon_sig']][0]
                         else 'EXISTING_FROZEN_FINALIST' if x['candidate_id'] in plan['finalists']
                         else 'REGISTERED_EXPLORATION_CANDIDATE'),
                       source_commit='8fd1ceb283309a7e515ff4a533a253260bdd0f78',
                       source_file=str((OLD / molecule / 'candidate_results.json.gz').relative_to(ROOT))
                         if x['cohort'] == 'exploration' else str(confirmation_path.relative_to(ROOT)),
                       detail_source=x['case_path'],
                       price_evidence_locator='data_archives.json members: ' + molecule + '/batch*/cases/'
                           + x['candidate_id'] + '/prices/' + x['cohort'] + '/*/result.json')
            row['G_availability'] = availability(row)
            result.append(row)
    return result


def h4_rows(paths):
    catalog = {r['candidate_id']: r for r in read(paths['h4_catalog'])}
    noms = read(paths['h4_nominations'])
    with paths['h4_precision'].open() as f:
        originals = list(csv.DictReader(f))
    result = []
    for original in originals:
        x = {k: saved_csv_value(v) for k, v in original.items()}
        c = catalog[x['candidate_id']]
        row = base_row('H4', x['candidate_id'], c['parameters'], x['epsilon_sig'], x['cohort'])
        for key in ('eligibility', 'G_exact', 'G_SE', 'N_cos', 'N_sin'):
            row[key] = x[key]
        st = c['costs'][x['cohort']]
        for axis, short in (('cosine', 'cos'), ('sine', 'sin')):
            row[f'bias_{short}_exact'] = c['bias_axes'][axis]['signed_bias_exact']
            row[f'u_{short}_exact'] = c['bias_axes'][axis]['u_enc_exact']
            row[f'C_{short}_exact'] = x['C_' + short]
            row[f'C_{short}_SE'] = st['axes'][axis]['standard_error']
            row[f'shot_{short}_reason'] = 'N_SAVED' if x['N_' + short] else 'NOT_IN_CSV_SEE_TARGET'
        if st['n'] != x['n']:
            raise ValueError('H4_COHORT_N_MISMATCH')
        row.update(B_saved_decimal=x['B'], price_n=x['n'],
                   exploration_n=c['costs'].get('exploration', {}).get('n'),
                   existing_confirmation_n=c['costs'].get('confirmation', {}).get('n'),
                   selection_role=('EXISTING_EXPLORATION_NOMINEE' if
                      x['candidate_id'] == noms['by_epsilon'][x['epsilon_sig']][0]
                      else 'EXISTING_FROZEN_FINALIST' if x['candidate_id'] in noms['finalists']
                      else 'HISTORICAL_BASELINE_ONLY' if x['cohort'] == 'historical_baseline'
                      else 'REGISTERED_EXPLORATION_CANDIDATE'),
                   source_commit='d3fdfad880902b7ff35b710d22e1c9759f7087c2',
                   source_file='artifacts/non_df_h4_pauli_pr_search/2026-10-25/candidate_precision_table.csv',
                   detail_source=c['target_path'], price_evidence_locator=c['law_path'] + '; catalog.costs.' + x['cohort'])
        row['G_availability'] = availability(row)
        result.append(row)
    return result


def coverage(rows):
    result = []
    for molecule in M:
        for eps in EPS:
            x = dict(molecule=molecule, epsilon_sig=eps)
            for family in ('RTE', 'Hybrid'):
                for cohort in ('exploration', 'confirmation', 'historical_baseline'):
                    rr = [r for r in rows if (r['molecule'], r['epsilon_sig'], r['group'], r['cohort'])
                          == (molecule, eps, family, cohort)]
                    x[family + '_' + cohort] = dict(
                        candidates=len(rr), saved_statuses=dict(Counter(r['eligibility'] for r in rr)),
                        G_availability=dict(Counter(r['G_availability'] for r in rr)),
                        candidate_ids=[r['candidate_id'] for r in rr])
                ex = x[family + '_exploration']
                cf = x[family + '_confirmation']
                x[family + '_confirmation_gap'] = (
                    'CONFIRMATION_MISSING' if
                    ex['G_availability'].get('SAVED_FINITE_G', 0) and
                    not cf['G_availability'].get('SAVED_FINITE_G', 0)
                    else 'SAVED_BOTH_COHORTS_AVAILABLE' if cf['G_availability'].get('SAVED_FINITE_G', 0)
                    else 'NO_COMPLETE_SAVED_EXPLORATION_G_SEE_STATUS')
            result.append(x)
    return result


def main():
    paths = {}
    for source in read(OUT / 'source_manifest.json'):
        path = ROOT / source['local_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError('FIXED_INPUT_CHANGED ' + source['key'])
        paths[source['key']] = path
    for member in read(OUT / 'archive_members.json'):
        if hashlib.sha256((ROOT / member['local_path']).read_bytes()).hexdigest() != member['sha256']:
            raise ValueError('RECOVERED_SAVED_MEMBER_CHANGED')
    rows = h26_rows() + h4_rows(paths)
    keys = [(r['molecule'], r['candidate_id'], r['epsilon_sig'], r['cohort']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('DUPLICATE_CANDIDATE_EPS_COHORT')
    # Join the existing H2/H6 canonical CSV by ID/epsilon only; no G comparisons.
    original = {}
    with paths['h26_precision'].open() as f:
        for r in csv.DictReader(f):
            if r['method'] == 'PR':
                original[(r['molecule'], r['candidate_id'], r['epsilon_sig'], r['cohort'])] = r
    for row in rows:
        key = (row['molecule'], row['candidate_id'], row['epsilon_sig'], row['cohort'])
        if key in original:
            for field in ('G_exact', 'G_SE', 'N_cos', 'N_sin', 'B_exact', 'bias_cos_exact', 'bias_sin_exact'):
                value = original[key][field] or None
                if row[field] != value:
                    raise ValueError('CANONICAL_STRING_MISMATCH ' + str(key) + field)
    with paths['h4_precision'].open() as f:
        h4 = {(r['candidate_id'], r['epsilon_sig'], r['cohort']): r for r in csv.DictReader(f)}
    for row in rows:
        if row['molecule'] == 'H4':
            original = h4[(row['candidate_id'], row['epsilon_sig'], row['cohort'])]
            for field in ('eligibility', 'G_exact', 'G_SE', 'N_cos', 'N_sin'):
                if row[field] != saved_csv_value(original[field]):
                    raise ValueError('H4_CANONICAL_STRING_OR_NULL_MISMATCH ' + field)
    csvsave(OUT / 'candidate_evidence_inventory.csv', rows)
    save(OUT / 'coverage_by_precision.json', coverage(rows))
    counts = {m: {g: len({r['candidate_id'] for r in rows if r['molecule'] == m and r['group'] == g})
                  for g in ('RTE', 'Hybrid')} for m in M}
    checks = dict(status='PASS', inventory_rows=len(rows), registered_candidates=counts,
                  duplicate_keys=0, canonical_H2_H6_strings_match=True, fixed_input_SHA256_match=True,
                  canonical_H4_strings_and_nulls_match=True,
                  saved_cohorts_separate=True, H4_historical_baseline_separate=True,
                  new_scientific_calls=0, new_scientific_ranking=0, new_G_or_ratio_or_SE_calculation=0,
                  stage='INVENTORY_AND_DISABLED_EXECUTION_PLAN_ONLY',
                  validator='Primary Codex technical checks; no independent scientific review')
    save(OUT / 'inventory_verification.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
