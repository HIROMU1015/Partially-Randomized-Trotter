#!/usr/bin/env python3
"""Write the Full PF saved-evidence inventory and coverage matrix (metadata only).

Reads pinned Git carriers (SHA-checked) and the read-only private context files.
No value is recomputed; no evolution, reference, N/RMSE, compile or fit.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import full_pf_saved_inventory as I  # noqa: E402

COLUMNS = ('system', 'formula', 'formula_id', 'T', 'q_PF', 'actual_width', 'scope', 'physical_candidate_id', 'group',
           'signal_status', 'signal_kind', 'signal_carrier', 'reference_status', 'reference_kind', 'independent_pair',
           'numerical_endpoints', 'X_status', 'X_rz_count', 'Y_status', 'Y_rz_count', 'price_kind', 'price_carrier',
           'compiler', 'mismatch_or_conditional_reasons', 'missing')


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def flat(row):
    price = row['prices'][0] if row['prices'] else {}
    reasons = sorted(set(row['physical_reasons'] + row['signal'].get('reasons', [])
                         + [x for s in row['prices'] for a in s['axes'].values() for x in a['reasons']]))
    def carrier(c):
        return None if not c else c['commit'][:12] + ':' + c['path'].split('/')[-1] + '#' + c['key']
    return dict(system=row['system'], formula=row['formula'], formula_id=row['formula_id'], T=row['T'], q_PF=row['q_PF'],
        actual_width=row['actual_width'], scope=row['scope'], physical_candidate_id=row['physical_candidate_id'], group=row['group'],
        signal_status=row['signal']['status'], signal_kind=row['signal'].get('kind'), signal_carrier=carrier(row['signal'].get('carrier')),
        reference_status=row['reference']['status'], reference_kind=row['reference'].get('kind'),
        independent_pair=row['reference'].get('independent_pair_status'),
        numerical_endpoints=json.dumps(row['numerical'].get('statuses'), sort_keys=True),
        X_status=row['X'], X_rz_count=price.get('axes', {}).get('X', {}).get('rz_count'),
        Y_status=row['Y'], Y_rz_count=price.get('axes', {}).get('Y', {}).get('rz_count'),
        price_kind=price.get('kind'), price_carrier=carrier(price.get('carrier')), compiler=price.get('compiler'),
        mismatch_or_conditional_reasons=';'.join(reasons), missing=';'.join(row['missing']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    k = I.load_carriers()
    cov = I.coverage(k)
    write(args.output_dir/'INVENTORY.json', dict(status='FULL_PF_SAVED_SOURCE_INVENTORY', carrier=I.CARRIER,
        carrier_files={n: dict(path=I.LONG + n, **v) for n, v in k['sha'].items()}, provenance=k['provenance'],
        study_binding={s: k['science']['inputs'][s]['binding'] for s in ('H4', 'H6')}, compiler=k['science']['compiler'],
        cost_policy=k['science'].get('cost_policy'), pf_common=k['pf_common'],
        private_context={s: dict(path=str(I.PRIVATE / (s + '_context.json')), sha256=k['context_sha'].get(s),
                                 new_reference_times=k['context'].get(s, {}).get('new_reference_times'),
                                 reused_reference_times=k['context'].get(s, {}).get('reused_reference_times'),
                                 reference_expm_multiply_check=k['context'].get(s, {}).get('reference_expm_multiply_check'))
                         for s in ('H4', 'H6')},
        other_series=I.other_saved_series(), values_recomputed=False, mandatory_stop=True))
    write(args.output_dir/'COVERAGE.json', cov)
    with (args.output_dir/'COVERAGE.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator='\n'); w.writeheader()
        for row in cov['rows']:
            w.writerow(flat(row))


if __name__ == '__main__':
    main()
