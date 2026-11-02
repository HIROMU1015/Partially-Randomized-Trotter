#!/usr/bin/env python3
"""Display saved totals only. No scientific providers, cost accounting or fitting.

H6/H8: take the first family/order entry in an already saved ordinal ranking.
H2/H4: reduce saved eligible scalar rows to the requested display-group minimum.
PR: copy the existing A selection. Never pool cohorts or read P observations.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
EPS = ('0.05', '0.005', '0.0005', '0.01', '0.001')
SYSTEMS = ('H2', 'H4', 'H6', 'H8')
SERIES = ('PR', 'full PF, S2', 'full PF, S4', 'B0, S2', 'B0, S4',
          'B0+I, S2', 'B0+I, S4')
PREFIX = 'artifacts/resource_applicability/'
PF_DIR = PREFIX + 'track_a_pf_selection_cost_comparison/2026-10-20/'
A_DIR = PREFIX + 'track_a_pr_new_a_total_cost_results/2026-10-18/'
SPECS = {
    'conditions': ('c0808b25', PREFIX + 'track_a_protocol_v2_draft/2026-10-19/system_conditions.json'),
    'protocol': ('c0808b25', 'docs/research/track_a_pr_resource_evaluation_protocol_v2_DRAFT.md'),
    'h2_a': ('829d697a', PREFIX + 'track_a_h2_pr_resource/2026-10-19/A_completed_run01/result.json'),
    'h2_d': ('829d697a', PREFIX + 'track_a_h2_pr_resource/2026-10-19/D_completed_run02/result.json'),
    'h2_a_audit': ('829d697a', PREFIX + 'track_a_h2_pr_resource/2026-10-19/A_completed_run01/completion_audit.json'),
    'h2_d_audit': ('829d697a', PREFIX + 'track_a_h2_pr_resource/2026-10-19/D_completed_run02/completion_audit.json'),
    'h4_a': ('85cb5b4d', PREFIX + 'track_a_h4_tol_a_completed/2026-10-19/result.json'),
    'h4_d': ('406f7057', PREFIX + 'track_a_h4_tol_d_completed/2026-10-19/result.json'),
    'h4_a_audit': ('85cb5b4d', PREFIX + 'track_a_h4_tol_a_completed/2026-10-19/completion_audit.json'),
    'h4_d_audit': ('406f7057', PREFIX + 'track_a_h4_tol_d_completed/2026-10-19/completion_audit.json'),
    'h68_a': ('64c85f7a', A_DIR + 'comparisons.csv'),
    'h68_a_points': ('64c85f7a', A_DIR + 'new_pr_cost_points.csv'),
    'h68_a_completion': ('64c85f7a', A_DIR + 'completion_summary.json'),
    'pf_totals': ('3983fe4f', PF_DIR + 'candidate_totals.csv'),
    'pf_ranks': ('3983fe4f', PF_DIR + 'candidate_ranks.csv'),
    'pf_selection': ('3983fe4f', PF_DIR + 'selection_comparison.csv'),
    'pf_prices': ('3983fe4f', PF_DIR + 'used_price_evidence.json'),
    'pf_contract': ('3983fe4f', PF_DIR + 'sealed_contract.json'),
    'pf_readme': ('3983fe4f', PF_DIR + 'README.md'),
    'pf_verification': ('3983fe4f', PF_DIR + 'structural_validation.json'),
    'pf_replay': ('3983fe4f', PF_DIR + 'baseline_replay_gate.json'),
    'pf_stop': ('3983fe4f', PF_DIR + 'STOP.json'),
    'old_absolute_data': ('c5ad7945', PREFIX + 'track_a_pr_pf_absolute_cost_v2/2026-10-19/data/comparisons.csv'),
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n')


def rows(b):
    return list(csv.DictReader(io.StringIO(b.decode())))


def write_csv(name, data):
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n')
        w.writeheader()
        for row in data:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in row.items()})


def freeze_sources():
    (HERE / 'sources').mkdir(exist_ok=True)
    manifest = []
    for label, (commit, path) in SPECS.items():
        full = subprocess.check_output(['git', 'rev-parse', commit], cwd=ROOT).decode().strip()
        b = subprocess.check_output(['git', 'show', full + ':' + path], cwd=ROOT)
        dest = HERE / 'sources' / (label + Path(path).suffix)
        if dest.exists():
            assert dest.read_bytes() == b, f'Existing snapshot differs: {label}'
        else:
            dest.write_bytes(b)
        last = subprocess.check_output(['git', 'log', '-1', '--format=%H', full, '--', path], cwd=ROOT).decode().strip()
        manifest.append(dict(label=label, commit=full, last_change_commit=last,
                             source_path=path, local_path=str(dest.relative_to(HERE)),
                             bytes=len(b), sha256=sha(b),
                             url=f'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/{full}/{path}'))
    dump('input_sources.json', manifest)


def load_sources():
    manifest = json.loads((HERE / 'input_sources.json').read_text())
    data = {}
    for m in manifest:
        b = (HERE / m['local_path']).read_bytes()
        assert sha(b) == m['sha256'] and len(b) == m['bytes'], m['label']
        data[m['label']] = rows(b) if m['source_path'].endswith('.csv') else (
            json.loads(b) if m['source_path'].endswith('.json') else b.decode())
    return data, {m['label']: m for m in manifest}


def family(cid):
    method = cid.split('_')[0]
    return {'B1': 'full PF', 'B0': 'B0', 'B0I': 'B0+I', 'B2': 'PR'}[method]


def parameters(cid):
    m = re.fullmatch(r'(B0I|B0|B1|B2)_p(\d+)_q(\d+)_(2nd|4th)(?:_r(\d+)_K(\d+))?', cid)
    assert m, cid
    method, p, q, order, r, k = m.groups()
    return dict(L_D=int(p), q=int(q), delta=str(Fraction(4, 5) / int(q)),
                PF_order={'2nd': 'S2', '4th': 'S4'}[order],
                r=int(r) if r else None, K=int(k) if k else None)


def series_for(cid):
    f = family(cid)
    return f if f == 'PR' else f + ', ' + parameters(cid)['PF_order']


def source_ref(index, label, locator):
    m = index[label]
    return dict(label=label, commit=m['commit'], path=m['source_path'],
                sha256=m['sha256'], locator=locator)


def build_data():
    d, index = load_sources()
    cond = {c['system']: c for c in d['conditions']['systems']}
    retained = []
    group_audit = []
    original = []

    def add(system, eps, series, token, cid, ref, science_source, *,
            cost_class, n, cohort, ties=None, saved_se=None, selection='', extra=None):
        c = cond[system]
        p = parameters(cid) if cid else dict(L_D=None, q=None, delta=None, PF_order=None, r=None, K=None)
        retained.append(dict(epsilon=eps, system=system, hydrogen_atoms=int(system[1:]),
            series=series, method=series.split(',')[0], **p,
            cost_exact=token, cost_display=float(Fraction(token)) if token else None,
            status='SAVED_POINT' if token else 'MISSING_NO_ELIGIBLE_SAVED_CANDIDATE',
            candidate_id=cid, exact_ties=ties or ([cid] if cid else []),
            n=n, cohort=cohort, saved_PR_SE=saved_se, cost_class=cost_class,
            selection=selection, model='DF electronic Hamiltonian',
            geometry='linear chain, adjacent 1 Angstrom', basis=c['basis'],
            DF_policy=c['DF_policy'], DF_rank=c['actual_rank'], T='0.8', P=0,
            bias_mode='USUAL_SIGNED', allowance_factor='1.0', alpha_axis='0.025',
            cost_scope='shot-weighted full cosine/sine Hadamard wrapper compiled RZ; no preparation',
            control=c['control'], scalar=c['scalar'], reference_rotated=c['reference_rotated'],
            hamiltonian_hash=c['hamiltonian_hash'], state_hash=c['state_hash'],
            snapshot_hash=c['snapshot_hash'], compiler_settings_hash=c['compiler_settings_hash'],
            scientific_SOURCE=science_source, result_source=ref,
            evidence_details=extra or {}))

    # Completed D rows are deterministic baseline accounting, not P observations.
    for system in ('H2', 'H4'):
        a_label, pf_label = system.lower() + '_a', system.lower() + '_d'
        a, pf = d[a_label], d[pf_label]
        assert a['status'] == pf['status'] == 'COMPLETE_MANDATORY_STOP'
        for name in ('hamiltonian', 'state', 'snapshot', 'compiler'):
            assert a['binding'][name] == pf['binding'][name]
        binding = a['binding']
        assert binding['rank'] == cond[system]['actual_rank']
        assert binding['hamiltonian'] == cond[system]['hamiltonian_hash']
        assert binding['state'] == cond[system]['state_hash']
        aa, dd = a.get('accounting', a), pf.get('accounting', pf)
        for eps in EPS:
            ci, comparison = next((i, x) for i, x in enumerate(aa['comparisons']) if str(x['epsilon']) == eps)
            pr = comparison['PR']
            assert pr['eligible'] is True and pr['n'] == 8 and pr['cohort'] == 'EXPLORATION_N8'
            locator = ('/accounting' if 'accounting' in a else '') + f'/comparisons/{ci}/PR'
            add(system, eps, 'PR', pr['G_RZ_exact'], pr['cell_id'], source_ref(index, a_label, locator),
                a['SOURCE'], cost_class='SAVED_EXPLORATION_N8_MEAN', n=8, cohort=pr['cohort'],
                saved_se=pr['G_RZ_SE'], selection='COPY_SAVED_A_SELECTION')
            original.append(dict(system=system, epsilon=eps, original_A_PF_id=comparison['PF']['cell_id'],
                                 original_A_PF_cost_exact=comparison['PF']['G_RZ_exact']))
            for series in SERIES[1:]:
                registered = [(i, x) for i, x in enumerate(dd['rows'])
                              if str(x['epsilon']) == eps and series_for(x['cell_id']) == series]
                usable = [(i, x) for i, x in registered if x['eligible'] is True and x['G_RZ_exact'] is not None]
                assert not any(x['eligible'] is None or x['eligible'] is True and x['G_RZ_exact'] is None
                               for _, x in registered), (system, eps, series, 'unresolved')
                smallest = min((Fraction(x['G_RZ_exact']) for _, x in usable), default=None)
                ties = [(i, x) for i, x in usable if Fraction(x['G_RZ_exact']) == smallest]
                ties.sort(key=lambda x: x[1]['cell_id'])
                group_audit.append(dict(system=system, epsilon=eps, series=series,
                    registered=len(registered), eligible_saved=len(usable), unresolved=0,
                    rule='MINIMUM_OF_UNCHANGED_SAVED_SCALARS_FOR_DISPLAY_ONLY',
                    all_eligible=[dict(candidate_id=x['cell_id'], cost_exact=x['G_RZ_exact'],
                                      source_row=i) for i, x in usable]))
                i, best = ties[0] if ties else (None, None)
                loc = ('/accounting' if 'accounting' in pf else '') + f'/rows/{i}' if best else 'no eligible row'
                add(system, eps, series, best['G_RZ_exact'] if best else None,
                    best['cell_id'] if best else None, source_ref(index, pf_label, loc), pf['SOURCE'],
                    cost_class='DIRECT_MEASURED' if best else 'MISSING', n=1 if best else None,
                    cohort='DETERMINISTIC', ties=[x['cell_id'] for _, x in ties],
                    selection='DISPLAY_GROUP_MINIMUM_OF_SAVED_ELIGIBLE_TOTALS')

    # Already saved rankings under one explicitly named PF price policy.
    totals = {(x['system'], x['bias_mode'], x['epsilon'], x['allowance_factor'], x['candidate_id']): (i, x)
              for i, x in enumerate(d['pf_totals'])}
    assert len(totals) == len(d['pf_totals'])
    scopes = {'H6': ('fine_grid_S2_plus_truncated_S4', 'signed_signal', 'strengthened_S2_S4', 136),
              'H8': ('strengthened184', 'USUAL_SIGNED_SAVED', 'strengthened184', 184)}
    pf_source = d['pf_contract']['SOURCE']
    assert pf_source == '19f2c5a9409db69fd8bd6500eb4228fefa00145c'
    price_index = {x['task_id']: x for x in d['pf_prices']['rows']}
    for system, (scope, mode, a_scope, candidate_count) in scopes.items():
        comp = d['h68_a_completion']['systems'][system]
        assert comp['status'] == 'COMPLETE_MANDATORY_STOP' and comp['native_exit_code'] == 0 and comp['reaped']
        for eps in EPS:
            ai, a = next((i, x) for i, x in enumerate(d['h68_a']) if x['system'] == system and
                         x['epsilon'] == eps and x['bias_mode'] == 'USUAL_SIGNED' and
                         x['factor'] == '1.0' and x['deterministic_scope'] == a_scope)
            assert a['new_domain_complete'] == 'True' and a['unresolved_new_domain_ids'] == ''
            point = next(x for x in d['h68_a_points'] if x['system'] == system and x['epsilon'] == eps
                         and x['factor'] == '1.0' and x['bias_mode'] == 'USUAL_SIGNED'
                         and x['cell_id'] == a['new_best_id'])
            assert point['cohort'] == 'EXPLORATION_N8' and point['n_used'] == '8'
            assert point['G_RZ_exact'] == a['new_n8_G_exact'] and point['independent_confirmation'] == 'False'
            add(system, eps, 'PR', a['new_n8_G_exact'], a['new_best_id'],
                source_ref(index, 'h68_a', f'CSV data row {ai + 1}'), a['SOURCE'],
                cost_class='SAVED_EXPLORATION_N8_MEAN', n=8, cohort='EXPLORATION_N8',
                saved_se=a['new_n8_G_SE'], selection='COPY_SAVED_A_SELECTION')
            original.append(dict(system=system, epsilon=eps, original_A_PF_id=a['PF_best_available_id'],
                                 original_A_PF_cost_exact=a['PF_G_exact']))
            rank_rows = [(i, x) for i, x in enumerate(d['pf_ranks']) if x['system'].startswith(system + '_')
                         and x['bias_mode'] == mode and x['epsilon'] == eps
                         and x['allowance_factor'] == '1.0' and x['original_scope'] == scope]
            pool = [x for x in d['pf_totals'] if x['system'].startswith(system + '_') and x['bias_mode'] == mode
                    and x['epsilon'] == eps and x['allowance_factor'] == '1.0']
            assert len(pool) == candidate_count
            assert {x['candidate_id'] for _, x in rank_rows} == {x['candidate_id'] for x in pool if x['eligible'] == 'True'}
            for series in SERIES[1:]:
                rr = [(i, x) for i, x in rank_rows if series_for(x['candidate_id']) == series]
                # Read the old ordinal; do not recompute a candidate ranking.
                chosen = min(rr, key=lambda ix: json.loads(ix[1]['replacement'])['ordinal']) if rr else None
                pp = [x for x in pool if series_for(x['candidate_id']) == series]
                group_audit.append(dict(system=system, epsilon=eps, series=series,
                    registered=len(pp), eligible_saved=len(rr), unresolved=0,
                    rule='FIRST_ENTRY_BY_ALREADY_SAVED_REPLACEMENT_ORDINAL',
                    all_eligible=[dict(candidate_id=x['candidate_id'],
                        cost_exact=json.loads(x['replacement'])['G_exact'],
                        saved_ordinal=json.loads(x['replacement'])['ordinal'], source_row=i) for i, x in rr]))
                if chosen:
                    ri, best = chosen
                    token = json.loads(best['replacement'])['G_exact']
                    ti, total = totals[(best['system'], mode, eps, '1.0', best['candidate_id'])]
                    assert total['eligible'] == 'True' and total['replacement_G_exact'] == token
                    ties = [x['candidate_id'] for _, x in rr if Fraction(json.loads(x['replacement'])['G_exact']) == Fraction(token)]
                    unit_evidence = []
                    for axis in ('cosine', 'sine'):
                        unit = price_index[best['system'] + ':' + best['candidate_id'] + ':' + axis]
                        binding = unit['direct_input_binding']
                        assert binding['hamiltonian'] == cond[system]['hamiltonian_hash']
                        assert binding['state'] == cond[system]['state_hash']
                        assert binding['input'] == cond[system]['snapshot_hash']
                        assert unit['compiler_hash'] == cond[system]['compiler_settings_hash']
                        unit_evidence.append({k: unit.get(k) for k in ('task_id', 'axis',
                            'replacement_RZ_literal', 'replacement_SOURCE', 'replacement_result_sha256',
                            'replacement_evidence_sha_kind', 'old_profile_pending_48', 'comparison_condition')})
                    add(system, eps, series, token, best['candidate_id'],
                        source_ref(index, 'pf_totals', f'CSV data row {ti + 1}; replacement_G_exact'), pf_source,
                        cost_class='SAVED_ALL_SAVED458_DIRECT_PRICE_POLICY', n=1, cohort='DETERMINISTIC',
                        ties=ties, selection='COPY_FIRST_FAMILY_ORDER_IN_SAVED_REPLACEMENT_RANK',
                        extra=dict(pf_scope=scope, saved_ordinal=json.loads(best['replacement'])['ordinal'],
                                   saved_rank_source=source_ref(index, 'pf_ranks', f'CSV data row {ri + 1}'),
                                   original_accounting_source_id=total['source_id'],
                                   original_accounting_locator=total['source_locator'],
                                   original_accounting_sha256=total['source_file_sha256'],
                                   unit_provenance=json.loads(total['unit_provenance']),
                                   unit_evidence=unit_evidence,
                                   saved_baseline_total=total['baseline_G_exact'],
                                   saved_shots=json.loads(total['saved_shots']),
                                   price_policy='ALL_SAVED458 replacement; retained old measured axes allowed',
                                   runtime_profile_equivalence_verified=False))
                else:
                    add(system, eps, series, None, None, source_ref(index, 'pf_totals', 'no eligible row'),
                        pf_source, cost_class='MISSING', n=None, cohort='DETERMINISTIC')

    retained.sort(key=lambda x: (EPS.index(x['epsilon']), SYSTEMS.index(x['system']), SERIES.index(x['series'])))
    assert len(retained) == len(EPS) * len(SYSTEMS) * len(SERIES) == 140
    # Only display normalization is new arithmetic: exact ratio to displayed best full PF.
    for row in retained:
        full = [x for x in retained if x['system'] == row['system'] and x['epsilon'] == row['epsilon']
                and x['method'] == 'full PF' and x['cost_exact']]
        denom = min(full, key=lambda x: Fraction(x['cost_exact']))
        ratio = Fraction(row['cost_exact']) / Fraction(denom['cost_exact']) if row['cost_exact'] else None
        row.update(relative_cost_exact=str(ratio) if ratio is not None else None,
                   relative_cost_display=float(ratio) if ratio is not None else None,
                   denominator_full_PF_cost_exact=denom['cost_exact'],
                   denominator_full_PF_candidate=denom['candidate_id'],
                   denominator_full_PF_series=denom['series'])
    pr_old = {(x['system'], x['epsilon']): x for x in d['old_absolute_data'] if x['panel'] == 'exploratory'}
    for row in retained:
        if row['series'] == 'PR':
            assert row['cost_exact'] == pr_old[(row['system'], row['epsilon'])]['G_PR_exact']
            assert row['candidate_id'] == pr_old[(row['system'], row['epsilon'])]['PR_id']
    dump('figure_data.json', dict(schema='SAVED_ALL_METHODS_HCHAIN_DISPLAY_V1',
        date_jst='2026-10-22', epsilons=list(EPS), primary_epsilons=list(EPS[:3]),
        series=list(SERIES), comparison_stage='EXPLORATION_A_ONLY_PR_PLUS_DETERMINISTIC_D',
        PF_H6_H8_price_policy='ALREADY_SAVED_ALL_SAVED458_REPLACEMENT',
        no_independent_P_observations=True, rows=retained))
    write_csv('figure_data.csv', retained)
    dump('group_selection_audit.json', group_audit)
    # Differences relative to the historical A comparator, never used to change PR.
    for x in original:
        pool = [r for r in retained if r['system'] == x['system'] and r['epsilon'] == x['epsilon'] and r['series'] != 'PR' and r['cost_exact']]
        best = min(pool, key=lambda r: Fraction(r['cost_exact']))
        x.update(displayed_PF_id=best['candidate_id'], displayed_PF_cost_exact=best['cost_exact'],
                 candidate_unchanged=best['candidate_id'] == x['original_A_PF_id'],
                 cost_unchanged=Fraction(best['cost_exact']) == Fraction(x['original_A_PF_cost_exact']))
    write_csv('historical_A_PF_comparison.csv', original)
    return retained


def render(data):
    os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
        'axes.labelsize': 16, 'axes.titlesize': 20, 'legend.fontsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    styles = {
        'PR': ('#0072B2', 'o', '-', True),
        'full PF, S2': ('#4A5568', 's', '-', True),
        'full PF, S4': ('#4A5568', 's', '--', False),
        'B0, S2': ('#D55E00', '^', '-', True),
        'B0, S4': ('#D55E00', '^', '--', False),
        'B0+I, S2': ('#009E73', 'D', '-', True),
        'B0+I, S4': ('#009E73', 'D', '--', False),
    }
    offsets = dict(zip(SERIES, (-.105, -.07, -.035, 0, .035, .07, .105)))
    display = []
    for eps in EPS:
        for kind, field in (('absolute', 'cost_display'), ('relative', 'relative_cost_display')):
            fig, ax = plt.subplots(figsize=(10.8, 6.5))
            fig.subplots_adjust(left=.125, right=.965, bottom=.28, top=.86)
            for series in SERIES:
                rr = [next(r for r in data if r['epsilon'] == eps and r['system'] == s and r['series'] == series) for s in SYSTEMS]
                y = [r[field] if r[field] is not None else math.nan for r in rr]
                c, marker, ls, filled = styles[series]
                ax.plot([i + offsets[series] for i in range(4)], y, label=series,
                        color=c, marker=marker, linestyle=ls, linewidth=2.4 if series == 'PR' else 1.6,
                        markersize=8.2 if series == 'PR' else 7.4, markeredgewidth=1.6,
                        markerfacecolor=c if filled else 'white', zorder=7 if series == 'PR' else 4)
            ax.set_yscale('log')
            ax.set_xticks(range(4), SYSTEMS)
            ax.tick_params(axis='x', length=0, pad=11, labelsize=17)
            ax.tick_params(axis='y', labelsize=13)
            ax.set_xlim(-.34, 3.34)
            ax.set_xlabel('H-chain', labelpad=13)
            if kind == 'absolute':
                ax.set_ylabel('Total compiled RZ cost (including shots)', labelpad=12)
                vals = [r[field] for r in data if r['epsilon'] == eps and r[field] is not None]
                lo, hi = min(vals), max(vals)
                ax.set_ylim(lo / 1.5, hi * 1.5)
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.,), numticks=12))
                ax.yaxis.set_major_formatter(LogFormatterSciNotation())
            else:
                ax.set_ylabel('Cost / best full PF cost (S2 or S4)', labelpad=12)
                ax.axhline(1, color='#9AA1AB', linewidth=1, zorder=1)
                vals = [r[field] for r in data if r[field] is not None]
                ax.set_ylim(min(vals) / 1.4, max(vals) * 1.4)
                from matplotlib.ticker import FixedLocator, FuncFormatter
                ticks = [x for x in (.1, .2, .3, .5, 1, 2, 3, 5, 10) if min(vals)/1.4 <= x <= max(vals)*1.4]
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis='y', which='major', color='#D9DEE5', linewidth=.7)
            ax.set_axisbelow(True)
            fig.text(.125, .94, 'Exploration A', fontsize=12, color='#536171')
            ax.set_title(r'$\epsilon = ' + eps + '$', loc='left', pad=16)
            from matplotlib.lines import Line2D
            handles, labels = ax.get_legend_handles_labels()
            handles.insert(1, Line2D([], [], linestyle='none'))
            labels.insert(1, '')
            legend = fig.legend(handles, labels, loc='lower center',
                                bbox_to_anchor=(.54, .012), ncol=4, frameon=False,
                                handlelength=2.6, columnspacing=1.5, labelspacing=.9)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bounds = fig.bbox
            visible_ylabels = [tick.label1 for tick in ax.yaxis.get_major_ticks()
                               if ax.get_ylim()[0] <= tick.get_loc() <= ax.get_ylim()[1]]
            checked = [ax.xaxis.label, ax.yaxis.label, ax.title, legend,
                       *ax.get_xticklabels(), *visible_ylabels, *fig.texts]
            clipped = [t.get_text() if hasattr(t, 'get_text') else 'legend' for t in checked
                       if t.get_visible() and not bounds.contains(*t.get_window_extent(renderer).get_points()[0])
                       or t.get_visible() and not bounds.contains(*t.get_window_extent(renderer).get_points()[1])]
            assert not clipped, clipped
            name = f'{kind}_cost_eps_{eps.replace(".", "p")}'
            for fmt in ('png', 'svg', 'pdf'):
                opts = dict(dpi=220)
                if fmt == 'svg': opts['metadata'] = {'Date': None}
                if fmt == 'pdf': opts['metadata'] = {'CreationDate': None, 'ModDate': None}
                fig.savefig(HERE / (name + '.' + fmt), **opts)
            display.append(dict(figure=name, epsilon=eps, kind=kind, series_count=7,
                                missing_points=sum(r[field] is None for r in data if r['epsilon']==eps),
                                axes_scale='log', x_offsets=offsets, layout_clipping=[]))
            plt.close(fig)
    dump('display_layout_check.json', dict(status='PASS', figures=display,
        scope='programmatic canvas bounds; visual review recorded separately'))
    dump('render_environment.json', dict(python=sys.version, platform=platform.platform(),
        matplotlib=matplotlib.__version__, executable=sys.executable,
        available_cpus=os.cpu_count(), affinity_cpus=len(os.sched_getaffinity(0)),
        resource_policy='No CPU/RAM/worker cap; ten small independent display renders, no scientific workload',
        gpu='NVIDIA T1000 4 GiB available; raster/vector plot export uses CPU',
        scientific_calls=0))


def verify(data):
    from PIL import Image
    import xml.etree.ElementTree as ET
    checks = []
    d, _ = load_sources()
    for row in data:
        ref = row['result_source']
        if row['cost_exact'] is None:
            continue
        if ref['label'].startswith(('h2_', 'h4_')):
            source = d[ref['label']]
            for part in ref['locator'].strip('/').split('/'):
                source = source[int(part)] if isinstance(source, list) else source[part]
            assert source['eligible'] is True
            assert row['candidate_id'] == source['cell_id']
            assert row['cost_exact'] == source['G_RZ_exact']
            assert row['epsilon'] == str(source['epsilon'])
        elif ref['label'] == 'h68_a':
            source = d['h68_a'][int(re.search(r'row (\d+)', ref['locator'])[1])-1]
            assert row['candidate_id'] == source['new_best_id']
            assert row['cost_exact'] == source['new_n8_G_exact']
            assert source['bias_mode'] == 'USUAL_SIGNED' and source['factor'] == '1.0'
        else:
            assert ref['label'] == 'pf_totals'
            source = d['pf_totals'][int(re.search(r'row (\d+)', ref['locator'])[1])-1]
            assert row['candidate_id'] == source['candidate_id']
            assert row['cost_exact'] == source['replacement_G_exact']
            assert source['eligible'] == 'True' and source['allowance_factor'] == '1.0'
            assert source['bias_mode'] in ('signed_signal', 'USUAL_SIGNED_SAVED')
    expected_csv = io.StringIO(newline='')
    writer = csv.DictWriter(expected_csv, fieldnames=list(data[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                     for k, v in row.items()} for row in data)
    assert (HERE / 'figure_data.csv').read_text() == expected_csv.getvalue()
    assert len(data) == 140
    assert {(r['epsilon'], r['system'], r['series']) for r in data} == {
        (e, s, m) for e in EPS for s in SYSTEMS for m in SERIES}
    for row in data:
        if row['cost_exact'] is not None:
            assert Fraction(row['cost_exact']) > 0
            assert row['cost_display'] == float(Fraction(row['cost_exact']))
            assert Fraction(row['relative_cost_exact']) == Fraction(row['cost_exact']) / Fraction(row['denominator_full_PF_cost_exact'])
        if row['series'] == 'PR':
            assert row['n'] == 8 and row['cohort'] == 'EXPLORATION_N8'
    for audit in json.loads((HERE / 'group_selection_audit.json').read_text()):
        row = next(r for r in data if (r['system'], r['epsilon'], r['series']) ==
                   (audit['system'], audit['epsilon'], audit['series']))
        available = audit['all_eligible']
        if available:
            assert all(Fraction(row['cost_exact']) <= Fraction(x['cost_exact']) for x in available)
            assert set(row['exact_ties']) == {x['candidate_id'] for x in available if Fraction(row['cost_exact']) == Fraction(x['cost_exact'])}
        else: assert row['cost_exact'] is None
    for eps in EPS:
        for kind in ('absolute', 'relative'):
            name = f'{kind}_cost_eps_{eps.replace(".", "p")}'
            im = Image.open(HERE / (name + '.png')); im.verify()
            svg = ET.parse(HERE / (name + '.svg')).getroot()
            pdf = (HERE / (name + '.pdf')).read_bytes()
            assert pdf.startswith(b'%PDF-') and pdf.rstrip().endswith(b'%%EOF')
            text = ''.join(svg.itertext())
            assert all(s in text for s in SERIES)
            checks.append(dict(figure=name, png_pixels=list(Image.open(HERE / (name+'.png')).size),
                               svg_valid=True, pdf_signature_valid=True, seven_legend_entries=True))
    dump('verification.json', dict(status='PASS', data_rows=140,
        source_snapshots_verified=len(SPECS), PR_points_equal_previous_saved_A=20,
        saved_group_minimum_and_all_exact_ties=True, selected_tokens_equal_pinned_sources=True,
        csv_json_identical=True,
        display_ratios_verified=True, P_observations_used=0,
        new_signal=0, sampling=0, compile=0, fit=0, bootstrap=0,
        shot_recalculation=0, total_cost_recalculation=0, scientific_reranking=0,
        PF_H6_H8_selection='Reuse saved ordinals; no rank recomputation',
        PF_H2_H4_selection='Requested display minimum of unchanged eligible saved total scalars',
        exports=checks, independent_scientific_review=False,
        verification_scope='Codex local technical data/export verification'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-sources', action='store_true', help='copy pinned Git blobs once')
    parser.add_argument('--check', action='store_true', help='verify saved sources/data/exports without rendering')
    args = parser.parse_args()
    if args.freeze_sources: freeze_sources()
    if args.check:
        load_sources()
        data = json.loads((HERE / 'figure_data.json').read_text())['rows']
    else:
        data = build_data()
        render(data)
    verify(data)
    print(json.dumps(dict(status='PASS', points=len(data), missing=sum(r['cost_exact'] is None for r in data),
                          figures=10, exports=30)))


if __name__ == '__main__':
    main()
