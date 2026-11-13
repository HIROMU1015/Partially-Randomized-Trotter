#!/usr/bin/env python3
"""Trace reused T=0.8 Full PF price receipts; join B V2 requests to the saved coverage.

Metadata only: pinned Git carriers and the read-only private files of the saved
long-time study. No evolution, reference, N/RMSE, build/compile/transpile, RZ
recount, fit or G. The V2 join uses synthetic RMSE-policy fixtures (as in the D
tests) and saved H/state bindings; it is a technical join, not a menu.
"""
import argparse
import csv
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from trottertracks import full_baseline_v2_requests as D  # noqa: E402
from trottertracks import full_pf_saved_inventory as I  # noqa: E402
from trottertracks import full_pf_price_receipts as R  # noqa: E402
from inventory_full_pf_saved_evidence import COLUMNS, flat, write  # noqa: E402

EXTRA = ('physical_formula_label', 'X_previous_status', 'Y_previous_status', 'receipt_evidence', 'receipt_native_SOURCE')
LIGHT = {'H4': (4, dict(q=16, R_outer=16)), 'H6': (6, dict(q=32, R_outer=32))}   # saved light-study layouts at T=0.8


def flat_v2(row):
    out = flat(row); price = row['prices'][0] if row['prices'] else {}
    trace = price.get('receipt_trace') or {}
    out.update(physical_formula_label=row.get('physical_formula_label'),
               X_previous_status=price.get('axes', {}).get('X', {}).get('previous_status'),
               Y_previous_status=price.get('axes', {}).get('Y', {}).get('previous_status'),
               receipt_evidence=';'.join(sorted({e['kind'] for e in trace.get('evidence', [])})),
               receipt_native_SOURCE=';'.join(sorted({e['native_SOURCE'][:12] for e in trace.get('evidence', []) if e.get('native_SOURCE')})))
    return out


def base(system, M):
    from check_pr_rpe_rmse_controller import spec
    s = spec(M=M); prep = D.saved_full_h_preparation(system)
    s['binding'].update({k: prep['binding'][k] for k in D.BINDING_KEYS if k in prep['binding']}, rank=prep['binding']['rank'])
    s['h'] = D.B.a.exact_value(0.1); s['profile'] = deepcopy(D.p.PROFILE)
    return s, prep


def join_example(cov):
    tenth = D.B.a.exact_value(0.1); rows = {r['physical_candidate_id']: r for r in cov['rows']}
    out = dict(status='ARTIFICIAL_TECHNICAL_JOIN_NOT_A_MENU_OR_ADOPTED_CONDITION', b_commit=D.B_LATEST,
               note='h=0.1, delta_PF=0.1, M=4 (Full PF) and the saved light layouts at m=3 (Full-H RTE); synthetic RMSE '
                    'policy/decomposition fixtures, real compiler profile and saved H/state bindings. Physical and V2 identities '
                    'are kept separately; no candidate is registered into a comparison menu.',
               full_pf=[], full_h_rte=[])
    for system in ('H4', 'H6'):
        for formula in ('2nd', '4th', '4th(m5_best)'):
            s, _ = base(system, 4)
            packet = D.bind(D.full_pf_spec(s, formula=formula, delta_pf=tenth, M=4, rank=s['binding']['rank']))
            joined = I.for_packet(packet, cov)
            for j in joined:
                saved = rows.get(j['physical_candidate_id'])
                j['mapping_source'] = None if saved is None else dict(
                    physical=dict(signal=saved['signal'].get('carrier'), price=[dict(kind=v['kind'], carrier=v['carrier']) for v in saved['prices']]),
                    v2=dict(commit=D.B_LATEST, formula_id=j['V2_formula_id'], configuration_id=j['V2_configuration_id']),
                    newly_mapped_by_B_formula_hook=formula == '4th')
            out['full_pf'].append(dict(system=system, formula=formula, V2_configuration_id=packet['v2_configuration']['id'],
                                       resolved_pending=packet['resolved_pending'], saved_join=joined))
        K, layout = LIGHT[system]; s, prep = base(system, 3)
        packet = D.bind(D.full_h_rte_spec(s, M=3, finite_K=K, preparation=prep), layout_by_round={3: layout})
        out['full_h_rte'].append(dict(system=system, K=K, layout_m3=layout, V2_configuration_id=packet['v2_configuration']['id'],
            event_mapping_kind=packet['v2_configuration']['spec']['event_mapping']['kind'], resolved_pending=packet['resolved_pending'],
            identity_map=[{k: r[k] for k in ('m', 'physical_candidate_id', 'V2_candidate_id', 'V2_resolution', 'b_commit')}
                          for r in packet['identity_map']],
            saved_asset_status=[r['saved_asset_status'] for r in packet['physical']['rounds']]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    k = I.load_carriers(); ctx = R.load(k)
    cov = R.update(I.coverage(k), k, ctx)
    write(args.output_dir/'RECEIPTS.json', dict(status='REUSED_T0.8_PRICE_RECEIPT_TRACE_METADATA_ONLY',
        previous_coverage_commit=cov['previous_coverage_commit'], source_checks=cov['receipt_source_checks'],
        private_preparation_check=cov['private_preparation_check'], changes=cov['changes'], traces=cov['receipt_traces'],
        pilot=dict(commit=R.PILOT[0], path=R.PILOT[1], SOURCE=R.PILOT_SOURCE),
        pf_common={s: dict(commit=v['commit'], path=v['path'], SOURCE=v['identity']['SOURCE'],
                           inherited_SOURCE=v['end_audit'].get('inherited_SOURCE'), native_records=len(v['by_sha']),
                           endpoint_priced_candidates=len(v['prices']), endpoint_price_conflicts=v['price_conflicts'])
                   for s, v in ctx['systems'].items()},
        values_recomputed=False, rz_recount=0, new_compile=0, mandatory_stop=True))
    write(args.output_dir/'COVERAGE_V2.json', {f: v for f, v in cov.items() if f != 'receipt_traces'})
    with (args.output_dir/'COVERAGE_V2.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, COLUMNS + EXTRA, lineterminator='\n'); w.writeheader()
        for row in cov['rows']:
            w.writerow(flat_v2(row))
    write(args.output_dir/'V2_SAVED_JOIN_EXAMPLE.json', join_example(cov))


if __name__ == '__main__':
    main()
