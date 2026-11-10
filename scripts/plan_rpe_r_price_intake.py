#!/usr/bin/env python3
"""Write C1 R-price intake readiness and static N/C/G coverage metadata.

Reads pinned Git JSON only. No sampling, compile, fit, price prediction,
signal/N, N*C or R selection. --publication-commit/--run-dir read one
explicit public C1 run and copy its price records into INTAKE.json.
"""
import argparse
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
FIELDS = ('m', 'candidate_id', 'arm', 'eligible', 'reasons', 'N_status', 'signal_phase_radius_N', 'C_status',
          'R_price_status', 'C1_cell', 'normalization', 'G', 'long_q_or_new_R_transfer_qualified')


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def static_coverage(root, intake, accepted=None):
    store = p.SavedArtifacts(root)
    a0 = store.read(A0_INTAKE)
    inputs = b.saved.load_inputs(root, store)
    systems = {}
    for system, mapping in EXAMPLE_MENUS.items():
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
        systems[system] = dict(R_menu={str(k): v for k, v in mapping.items()}, rows=rows,
                               saved_N_records=len(packet['shots']), price_records=len(prices))
    return dict(status='STATIC_R_PRICE_COVERAGE_METADATA_ONLY', A0_intake=A0_INTAKE, C1_plan=ci.C_PLAN,
                C1_completion=None if accepted is None else accepted['completion'], systems=systems,
                menu_is_example_not_adoption=True, costs=None, G=None, recommended_R=None,
                scientific_R_adopted=False, full_RPE_qualified=False, mandatory_stop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--publication-commit')
    parser.add_argument('--run-dir')
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
    write(args.output_dir/'COVERAGE.json', static_coverage(ROOT, intake, accepted))


if __name__ == '__main__':
    main()
