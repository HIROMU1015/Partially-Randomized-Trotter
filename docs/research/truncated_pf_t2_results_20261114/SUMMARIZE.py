"""Deterministic summary of T2 run01 + A's saved Full PF cells. No new numerics on H4.

Inputs: run01 cell results (this study), PLAN_METADATA (c3ae31f3, schedules),
A's published Full PF cell summaries (69ee0ff8 / 3ae055fa / 71f04be7, reused,
not recomputed). Outputs CELLS.csv, CONFIGURATIONS.csv and T3_CANDIDATES.json.
Proxies are structural ordering aids only (not compiled price, N, MSE or G).
"""
import csv
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EPS_TROT, EPS_E = 0.00079680005099520, 0.00159360010199040
H = Fraction(.1)
A_SCAN = ('3ae055fa7', 'docs/research/h4_full_pf_heff_scan_results_20261114/CELL_SUMMARY.csv')
A_FOLLOW = ('71f04be7bf94e7773a32270f36ae29eee1b903d5', 'docs/research/h4_full_pf_followup_results_20261114/CELL_SUMMARY.csv')
PLAN_META = ('c3ae31f353d27a9f720f51fa0b089acd3d57e211', 'docs/research/truncated_pf_optimization_plan_20261114/PLAN_METADATA.json')
FULL_PF_SHORTLIST = {('2nd', '16/7'), ('4th', '16/3'), ('4th(m5_best)', '16')}


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT).decode()


def full_pf_cells():
    """A's overlap-selected Full PF shifts keyed by (formula, actual/h)."""
    out = {}
    for row in csv.DictReader(io.StringIO(blob(*A_SCAN))):
        key = (row['formula'], str(Fraction(float(row['actual_delta']))/H))
        out[key] = dict(delta_E_ha=float(row['delta_E_ha']), source=dict(commit=A_SCAN[0], path=A_SCAN[1], label=row['label']),
                        origin=row['origin'])
    for row in csv.DictReader(io.StringIO(blob(*A_FOLLOW))):
        formula = row['label'].split(' @ ')[0]
        key = (formula, str(Fraction(int(row['actual_delta_numerator']), int(row['actual_delta_denominator']))/H))
        out[key] = dict(delta_E_ha=float(row['overlap_selected_shift_ha']),
                        source=dict(commit=A_FOLLOW[0], path=A_FOLLOW[1], label=row['label']), origin='NEW_SCIENTIFIC_DIAGNOSTIC')
    return out


def label_of(trunc, pf, total, lam):
    L2, L3 = abs(trunc)+abs(pf), abs(total)
    return ('CANCELLATION_DEPENDENT' if L3 <= EPS_TROT < L2 else 'WITHIN_L2_ROBUST' if L2 <= EPS_TROT else 'EXCEEDS_EPS_TROT_L3'), L2, L3, lam+abs(pf)


def main():
    result = json.loads((HERE/'run01/RESULT.json').read_text())
    if result['status'] != 'T2_HEFF_CELLS_COMPLETE_NOT_RPE_QUALIFIED':
        raise SystemExit('run01 not complete')
    t1 = {p: json.loads((ROOT/('docs/research/truncated_pf_t1_results_20261114/run01/prefix_%d.json' % p)).read_text()) for p in (7, 8, 9)}
    cells = {(c['prefix'], c['formula'], c['actual_delta_over_h']): c for c in result['cells']}
    full = full_pf_cells()
    rows = []
    for (p, f, w), c in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], Fraction(kv[0][2]))):
        k = c['classification']; ref = full.get((f, w))
        rows.append(dict(prefix=p, formula=f, actual_delta_over_h=w, status=c['status'],
            delta_E_trunc_ha=repr(c['delta_E_trunc_ha']), delta_E_PF_ret_ha=repr(c['delta_E_PF_ret_ha']),
            delta_E_total_ha=repr(c['delta_E_total_ha']), L1_ha=repr(k['L1_weyl_plus_abs_pf_ha']),
            L2_ha=repr(k['L2_abs_sum_ha']), L3_ha=repr(k['L3_signed_total_ha']), label=k['label'],
            full_pf_delta_E_ha=repr(ref['delta_E_ha']) if ref else '',
            pf_ret_minus_full_pf_ha=repr(c['delta_E_PF_ret_ha']-ref['delta_E_ha']) if ref else '',
            max_trace_overlap=repr(c['max_trace_overlap_original']), state_weight=repr(c['saved_state_weight_on_matched_group']),
            mixture_phase_bound_rad=repr(c['mixture_phase_bound_rad']),
            branch_mode=c['branch']['mode'], selected_cut_clearance_rad=repr(c['evaluator']['diagnostics']['selected_cut_clearance_rad']),
            window_edge_clearance_rad=repr(c['evaluator']['diagnostics'].get('window_edge_clearance_rad', '')),
            max_two_time_gap=repr(max(x['absolute_difference'] for x in c['two_time_checks'])),
            decomposition_identity_error_ha=repr(c['decomposition_identity_error_ha'])))
    with (HERE/'CELLS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    meta = json.loads(blob(*PLAN_META))
    configs = []
    for r in meta['configurations_table']:
        p = r['prefix']
        if p not in (7, 8, 9, 10):
            continue
        used = sorted({x['actual_delta_over_h'] for x in r['rounds']}, key=Fraction)
        vals = []
        for u in used:
            if p == 10:
                ref = full[(r['formula'], u)]; trunc, pf = 0., ref['delta_E_ha']; total = pf; lam = 0.
            else:
                c = cells[(p, r['formula'], u)]; trunc, pf, total = c['delta_E_trunc_ha'], c['delta_E_PF_ret_ha'], c['delta_E_total_ha']
                lam = t1[p]['lambda_R']
            vals.append((u,)+label_of(trunc, pf, total, lam)+(total,))
        nominal = [v for v in vals if v[0] == r['requested_delta_over_h']][0]
        labels = {v[1] for v in vals}
        config_label = ('EXCEEDS_EPS_TROT_L3' if 'EXCEEDS_EPS_TROT_L3' in labels else
                        'CANCELLATION_DEPENDENT' if 'CANCELLATION_DEPENDENT' in labels else 'WITHIN_L2_ROBUST')
        blocks = p+1
        q2 = 1/(1-(nominal[5]/EPS_E)**2) if abs(nominal[5]) < EPS_E else None
        configs.append(dict(prefix=p, method='FULL_PF' if p == 10 else 'TRUNCATED_B0I', formula=r['formula'],
            requested_delta_over_h=r['requested_delta_over_h'], configuration_id=r['configuration_id'],
            used_actual_widths_over_h=' '.join(used), config_label=config_label,
            nominal_delta_E_total_ha=repr(nominal[5]), nominal_L2_ha=repr(nominal[2]), nominal_L1_ha=repr(nominal[4]),
            max_L3_over_used_cells_ha=repr(max(v[3] for v in vals)), m12_q_PF=r['m12_q_PF'], m12_signed_pieces=r['m12_signed_pieces'],
            retained_blocks=blocks, structural_proxy_m12=r['m12_signed_pieces']*blocks,
            structural_proxy_all_rounds=r['total_signed_pieces']*blocks,
            q2_heuristic_N_factor=repr(q2) if q2 else '', full_pf_shortlist_member=p == 10 and (r['formula'], r['requested_delta_over_h']) in FULL_PF_SHORTLIST))
    with (HERE/'CONFIGURATIONS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(configs[0]), lineterminator='\n'); w.writeheader(); w.writerows(configs)
    # T3 ordering: per formula, eligible (WITHIN_L2_ROBUST) truncated configs by m12 structural proxy.
    t3 = {}
    for f in ('2nd', '4th', '4th(m5_best)'):
        eligible = sorted([c for c in configs if c['formula'] == f and c['method'] == 'TRUNCATED_B0I'
                           and c['config_label'] == 'WITHIN_L2_ROBUST'], key=lambda c: (c['structural_proxy_m12'], c['prefix']))
        full_ref = [c for c in configs if c['formula'] == f and c['full_pf_shortlist_member']][0]
        t3[f] = dict(ranked=[dict(prefix=c['prefix'], width=c['requested_delta_over_h'], configuration_id=c['configuration_id'],
                                  structural_proxy_m12=c['structural_proxy_m12'],
                                  proxy_ratio_to_full_pf_shortlist=c['structural_proxy_m12']/full_ref['structural_proxy_m12'],
                                  nominal_delta_E_total_ha=float(c['nominal_delta_E_total_ha']),
                                  q2_heuristic_N_factor=float(c['q2_heuristic_N_factor'])) for c in eligible],
                     full_pf_shortlist=dict(width=full_ref['requested_delta_over_h'], configuration_id=full_ref['configuration_id'],
                                            structural_proxy_m12=full_ref['structural_proxy_m12'],
                                            nominal_delta_E_total_ha=float(full_ref['nominal_delta_E_total_ha'])))
    (HERE/'T3_CANDIDATES.json').write_text(json.dumps(dict(scope='STRUCTURAL_ORDERING_ONLY_NOT_PRICE_N_MSE_OR_G',
        proxy='m12 signed S2 pieces x retained blocks (one-body + fragments); hypothesis to be tested by compiled prices',
        q2_heuristic='1/(1-(nominal total shift/eps_E)^2); diagnostic only, Q1/Q2 not adopted', per_formula=t3), indent=1)+'\n')
    print(json.dumps({f: [(x['prefix'], x['width'], x['structural_proxy_m12'], round(x['proxy_ratio_to_full_pf_shortlist'], 3))
                          for x in v['ranked'][:4]] for f, v in t3.items()}, indent=1))


if __name__ == '__main__':
    main()
