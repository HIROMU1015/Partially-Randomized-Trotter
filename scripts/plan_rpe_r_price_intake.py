#!/usr/bin/env python3
"""Write C1 R-price intake readiness and static N/C/G coverage metadata.

Reads pinned Git JSON only. No sampling, compile, fit, price prediction,
signal/N, N*C or R selection. --publication-commit/--run-dir read one
explicit public C1 run and copy its price records into INTAKE.json.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import rpe_r_price_intake as ci  # noqa: E402

p, b = ci.p, ci.b
A0_INTAKE = dict(commit='d5d9823e17ecf78a82216fc02b7fcc13a21bb2cb',
                 path='docs/research/rpe_normalization_results_intake_20261111/INTAKE.json')
EXAMPLE_MENUS = {'H4': {3: [4, 8], 4: [4, 8], 12: [4, 8192]}, 'H6': {4: [16, 32], 5: [16, 32], 12: [16, 8192]}}
# A's original finite R menu (86 PR descriptors) plus the C1 cells; no row is removed.
A_MENU_COMMIT = 'f98982118324bcc8402ed513b5c720b37bb03e87'
FIELDS = ('m', 'candidate_id', 'arm', 'eligible', 'reasons', 'N_status', 'signal_phase_radius_N', 'C_status',
          'R_price_status', 'C1_cell', 'normalization', 'G', 'long_q_or_new_R_transfer_qualified')


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def full_menus(store):
    menus = {}
    for system in ('H4', 'H6'):
        ref = dict(commit=A_MENU_COMMIT, path='docs/research/pr_r_screening_20261111/'+system+'_METADATA_PROPOSAL.json')
        proposal = store.read(ref)
        mapping = {int(m): list(v) for m, v in proposal['menu']['R_menu_values'].items()}
        for m, values in ci.C1_R_MENU[system].items():
            mapping[m] = sorted(set(mapping.get(m, [])) | set(values))
        menus[system] = dict(mapping=mapping, A_menu=ref, R_upper_bound=deepcopy(proposal['R_upper_bound']),
                             range_complete=proposal['range_complete'])
    return menus


def static_coverage(root, intake, accepted=None, *, full_menu=False):
    store = p.SavedArtifacts(root)
    a0 = store.read(A0_INTAKE)
    inputs = b.saved.load_inputs(root, store)
    menus = full_menus(store) if full_menu else {s: dict(mapping=v) for s, v in EXAMPLE_MENUS.items()}
    systems = {}
    for system, spec in menus.items():
        mapping = spec['mapping']
        packet = b.prepare_menu(root, system, mapping, inputs=inputs)
        prices = packet['prices'] + ([] if accepted is None else ci.records_for(accepted, packet['menu']))
        out = ci.coverage(packet['menu'], packet['shots'], prices, cohort_selection={}, intake=intake,
                          accepted=accepted, a0=a0)
        rows = []
        for item in out['rows']:
            c = item['candidate']
            rows.append(dict({k: item[k] for k in FIELDS}, q_PF=c['q'], T=c['T'], R_outer=c['R_outer'],
                             method=c['method'], price_scenarios=[dict(axis=v['axis'], cohort=v['cohort'],
                             classification=v['classification'], value_available=v['value_available'])
                             for v in item['price_contexts']]))
        systems[system] = dict({k: v for k, v in spec.items() if k != 'mapping'},
                               R_menu={str(k): v for k, v in sorted(mapping.items())}, rows=rows,
                               saved_N_records=len(packet['shots']), price_records=len(prices),
                               C1_price_records=len(prices)-len(packet['prices']))
    return dict(status='STATIC_R_PRICE_COVERAGE_METADATA_ONLY', A0_intake=A0_INTAKE, C1_plan=ci.C_PLAN,
                C1_completion=None if accepted is None else accepted['completion'], systems=systems,
                menu='A_FULL_MENU_PLUS_C1_CELLS' if full_menu else 'EXAMPLE',
                menu_is_example_not_adoption=True, costs=None, G=None, recommended_R=None,
                scientific_R_adopted=False, full_RPE_qualified=False, mandatory_stop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--publication-commit')
    parser.add_argument('--run-dir')
    parser.add_argument('--full-menu', action='store_true', help="coverage over A's full R menu plus the C1 cells")
    args = parser.parse_args()
    if bool(args.publication_commit) != bool(args.run_dir):
        parser.error('--publication-commit and --run-dir go together')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    intake = ci.C1PriceIntake(ROOT)
    write(args.output_dir/'READINESS.json', intake.readiness())
    accepted = None
    if args.publication_commit:
        accepted = intake.read_published(args.publication_commit, args.run_dir)
        write(args.output_dir/'INTAKE.json', accepted)
    write(args.output_dir/'COVERAGE.json', static_coverage(ROOT, intake, accepted, full_menu=args.full_menu))


if __name__ == '__main__':
    main()
