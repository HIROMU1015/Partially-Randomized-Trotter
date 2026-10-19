"""Explicit artificial manifest -> new external report directory. No discovery."""
import argparse
import csv
from fractions import Fraction
import io
import json
from pathlib import Path
from .g6_w2c_b3_offline_result import verify_bundle, plain, safe_path, require
from .g6_w2c_b3_offline_cost import aggregate
from .g6_w2c_b3_integration_acceptance import build_report


def write_json(path, obj):
    def encode(value):
        if type(value) is Fraction:
            return {'exact_fraction': [hex(value.numerator), hex(value.denominator)]}
        raise TypeError(type(value).__name__)
    with path.open('x', encoding='utf-8') as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=True, default=encode) + '\n')


def run(manifest, output, *, source_root, acceptance_baseline, unit_evidence=None):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    input_root = manifest.parent
    require(not output.is_relative_to(input_root) and not input_root.is_relative_to(output), 'OUTPUT_MUST_BE_OUTSIDE_INPUT')
    require(not output.exists(), 'OUTPUT_EXISTS_NO_OVERWRITE')
    verification = verify_bundle(manifest)
    costs = aggregate(verification)
    acceptance = build_report(source_root, acceptance_baseline['candidate'], acceptance_baseline['source_pins'],
                              unit_evidence=unit_evidence, verification=verification, costs=costs)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'verification_report.json', verification)
    write_json(output/'cost_summary.json', costs)
    write_json(output/'integration_acceptance_report.json', acceptance)
    (output/'verification_report.md').write_text(
        '# Artificial serial R1 verification\n\nStatus: **' + verification['status'] + '**.\n\n'
        'Envelope chain/file bytes and trial identities/counts are independently checked. '
        'Full retained documents permit event-trace recomputation. Compact per-event hashes permit cross-file '
        'hash comparison, not recovery of concatenated event bytes or independent full-trace SHA. '
        'A technical completion is not scientific correctness, physical precision or an algorithm ranking.\n', encoding='utf-8')
    fields = ['pass_name', 'arm', 'layer', 'total_cpu_ns', 'online_cpu_ns', 'setup_cpu_ns', 'output_cpu_ns',
              'cpu_per_trial_ns', 'cpu_per_accepted_ns', 'wall_ns', 'output_bytes', 'output_rows', 'memory']
    with (output/'comparison_table.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for row in costs['comparison']:
            writer.writerow(dict(row, memory=json.dumps(row['memory'], sort_keys=True)))
    table = io.StringIO()
    table.write('# Artificial descriptive cost comparison\n\nAll times are artificial nanoseconds. X/Y share setup once per pair.\n\n')
    table.write('|Pass|Arm|Layer|CPU|Online|Setup|Output|CPU/trial|CPU/accepted|Wall|Bytes|RSS/HWM|\n')
    table.write('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n')
    for r in costs['comparison']:
        table.write('|'+'|'.join(str(r.get(k)) for k in fields[:-2])+'|'+str(r['memory'])+'|\n')
    table.write('\nFine functions are an alternative partition. Parent wall overlaps child wall; parent shared '
                'setup and final summary output are separate. Receipt self-write cost remains null. '
                'RSS values are observations, not summed unique physical memory. No pure observer overhead certificate or ranking.\n')
    (output/'cost_comparison.md').write_text(table.getvalue(), encoding='utf-8')
    lines = ['# Artificial integration acceptance\n',
             'No scientific adoption. Parallel adapter: PENDING_SERVER_CONTRACT.\n',
             '|Check|Judgment|Target|Unresolved|', '|---|---|---|---|']
    lines += ['|' + '|'.join(str(r[k]).replace('|', '\\|').replace('\n', ' ') for k in
                            ('check_id', 'judgment', 'target_SOURCE', 'unresolved')) + '|'
              for r in acceptance['checks']]
    (output/'integration_acceptance_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return verification, costs, acceptance


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artificial-manifest', type=Path, required=True)
    p.add_argument('--new-output', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--acceptance-baseline', type=Path, required=True,
                   help='Explicit fixed technical candidate JSON; read only')
    p.add_argument('--artificial-test-evidence', type=Path)
    args = p.parse_args(argv)
    t = plain(args.acceptance_baseline.read_text())['formal_v2_template_not_executable']
    ev = plain(args.artificial_test_evidence.read_text()) if args.artificial_test_evidence else None
    v, c, _ = run(args.artificial_manifest, args.new_output, source_root=args.source_root,
                   acceptance_baseline=t, unit_evidence=ev)
    print(json.dumps({'verification': v['status'], 'cost': c['status'], 'mandatory_STOP': True}))
    return 0 if v['status'] == 'VERIFIED_COMPLETE' and c['status'] == 'COMPLETE_ARTIFICIAL_DESCRIPTIVE_COSTS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
