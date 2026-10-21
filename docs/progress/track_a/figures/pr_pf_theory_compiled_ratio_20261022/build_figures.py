"""Read saved totals/selections; divide saved totals for display, and draw.

No circuit, signal, shot, theory-cost, total-cost, statistical or ranking code.
The standard-library --check path neither imports Matplotlib nor changes files.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPS = ['0.05', '0.01', '0.005', '0.001', '0.0005']
SYSTEMS = ['H2', 'H4', 'H6', 'H8']
PF_SERIES = ['full PF, S2', 'B0, S2', 'B0+I, S2']
FIGURES = ['reference_A_H4_legacy_DF12_eps_0p05',
           'reference_B_H4_legacy_DF12_eps_0p05']


def read(label):
    path = next(e['local_path'] for e in inputs() if e['label'] == label)
    text = (HERE / path).read_text()
    return json.loads(text) if path.endswith('.json') else list(csv.DictReader(io.StringIO(text)))


def inputs():
    return json.loads((HERE / 'input_sources.json').read_text())


def reference(label, locator):
    item = next(e for e in inputs() if e['label'] == label)
    return {k: item[k] for k in ['label', 'commit', 'source_path', 'local_path', 'sha256']} | {'locator': locator}


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def data():
    for entry in inputs():
        blob = (HERE / entry['local_path']).read_bytes()
        assert len(blob) == entry['bytes']
        assert hashlib.sha256(blob).hexdigest() == entry['sha256'], entry['label']
    legacy = read('legacy_figure_data')['rows']
    ranks = read('pf_ranks')
    totals = read('pf_totals')
    historic_totals = read('stage3_shot_weighted_totals')
    historic_pairs = read('stage3_pf_pr_comparisons')
    shots = read('stage3_preserved_shots')
    models = read('stage3_theory_rows')
    current = []
    methods = []
    for system in SYSTEMS:
        for epsilon in EPS:
            old = [(i, r) for i, r in enumerate(legacy)
                   if r['system'] == system and r['epsilon'] == epsilon]
            pri, pr = next((i, r) for i, r in old if r['series'] == 'PR')
            assert pr['PF_order'] == 'S2' and pr['status'] == 'SAVED_POINT'
            pf_choices = [(i, r) for i, r in old if r['series'] in PF_SERIES]
            assert len(pf_choices) == 3
            winner = None
            selection = None
            if system == 'H2':
                w = read('h2_d')['accounting']['winners'][epsilon]
                assert w['cell_id'].endswith('_2nd')
                winner = next(r for _, r in pf_choices if r['candidate_id'] == w['cell_id'])
                assert winner['cost_exact'] == w['G_RZ_exact']
                selection = reference('h2_d', '/accounting/winners/' + epsilon)
                selection['rule'] = 'COPY_SAVED_WINNER_ALREADY_S2'
            elif system == 'H4':
                ws = read('h4_d')['winners']
                wi, wrow = next((i, r) for i, r in enumerate(ws) if str(r['epsilon']) == epsilon)
                w = wrow['best']
                if w['cell_id'].endswith('_2nd'):
                    winner = next(r for _, r in pf_choices if r['candidate_id'] == w['cell_id'])
                    assert winner['cost_exact'] == w['G_RZ_exact']
                    selection = reference('h4_d', f'/winners/{wi}/best')
                    selection['rule'] = 'COPY_SAVED_WINNER_ALREADY_S2'
                else:
                    assert epsilon == '0.001' and w['cell_id'].endswith('_4th')
                    selection = reference('h4_d', f'/winners/{wi}/best')
                    selection.update(rule='S2_ONLY_WINNER_NOT_SAVED_NO_NEW_ARGMIN', excluded_saved_winner=w['cell_id'])
            else:
                scope, bias = ('fine_grid_S2_plus_truncated_S4', 'signed_signal') if system == 'H6' else ('strengthened184', 'USUAL_SIGNED_SAVED')
                # Reuse saved ordinal metadata. Never sort or compare costs.
                ordered = [(i, r, json.loads(r['replacement'])) for i, r in enumerate(ranks)
                           if r['system'].startswith(system) and r['epsilon'] == epsilon
                           and r['allowance_factor'] == '1.0' and r['bias_mode'] == bias
                           and r['original_scope'] == scope]
                ordered.sort(key=lambda x: x[2]['ordinal'])
                ri, ranked, saved_rank = next(x for x in ordered if x[1]['candidate_id'].endswith('_2nd'))
                winner = next(r for _, r in pf_choices if r['candidate_id'] == ranked['candidate_id'])
                assert Fraction(winner['cost_exact']) == Fraction(saved_rank['G_exact'])
                selection = reference('pf_ranks', f'CSV data row {ri + 1}; replacement')
                selection.update(rule='COPY_FIRST_S2_IN_SAVED_REPLACEMENT_ORDINAL_ORDER', saved_rank=saved_rank,
                                 condition_id=ranked['condition_id'], PR_from_this_rank_packet_used=False)
            # Copy saved shots where actually present; never reconstruct them.
            if system in ['H2', 'H4']:
                pa = read(system.lower() + '_a')
                comps = pa.get('accounting', pa)['comparisons']
                cr = next(c for c in comps if str(c['epsilon']) == epsilon)
                pr_shots = cr['PR']['shots']
                pr_shot_status = 'SAVED_AXIS_SHOTS'
                pf_shots = None
                if winner:
                    pd = read(system.lower() + '_d')
                    dr = pd.get('accounting', pd)['rows']
                    wr = next(r for r in dr if r['cell_id'] == winner['candidate_id'] and str(r['epsilon']) == epsilon)
                    pf_shots = wr['shots']
            else:
                pr_shots = None
                pr_shot_status = 'NOT_PUBLIC_IN_SELECTED_NEW_A_CSV_EXCERPT'
                pf_shots = winner['evidence_details']['saved_shots']
            reason = ['NO_MATCHED_CURRENT_PR_THEORY_TOTAL_IN_INSPECTED_PINNED_SOURCES',
                      'NO_MATCHED_CURRENT_PF_THEORY_TOTAL_IN_INSPECTED_PINNED_SOURCES']
            if not winner:
                reason.append('S2_ONLY_PF_SELECTION_NOT_SAVED_NO_NEW_ARGMIN')
            if system == 'H4':
                reason.append('LEGACY_DF12_CANNOT_JOIN_CURRENT_DF10_EVEN_IF_CANDIDATE_ID_MATCHES')
            if system in ['H6', 'H8']:
                reason.extend(['OLD_PR_N8_N32_NOT_SUBSTITUTED_FOR_CURRENT_A',
                               'ALL_SAVED458_PRICE_NOT_NATIVE_THEORY_COST',
                               'RUNTIME_PROFILE_EQUIVALENCE_NOT_CERTIFIED'])
            record = dict(scope='CURRENT_BEST_S2', system=system, epsilon=epsilon,
                          PR_candidate_id=pr['candidate_id'], PF_candidate_id=winner['candidate_id'] if winner else None,
                          PF_method=winner['method'] if winner else None,
                          G_PR_theory_lower=None, G_PR_theory_upper=None,
                          G_PF_theory_lower=None, G_PF_theory_upper=None,
                          G_PR_compiled=pr['cost_exact'], G_PF_compiled=winner['cost_exact'] if winner else None,
                          rho_theory_lower=None, rho_theory_upper=None,
                          rho_compiled=str(Fraction(pr['cost_exact']) / Fraction(winner['cost_exact'])) if winner else None,
                          PR_reduction_percent_lower=None, PR_reduction_percent_upper=None,
                          PF_reduction_percent_lower=None, PF_reduction_percent_upper=None,
                          PR_shots=pr_shots, PR_shot_status=pr_shot_status, PF_shots=pf_shots,
                          status='NOT_COMPARABLE_MISSING_MATCHED_THEORY', reasons=reason,
                          PR_saved_record=pr, PF_saved_record=winner,
                          PR_selection_source=reference('legacy_figure_data', f'/rows/{pri}'),
                          PF_selection_source=selection, PF_method_minima=[r for _, r in pf_choices],
                          theory_model=None, new_argmin_executed=False)
            current.append(record)
            for i, r in old:
                if r['series'] not in ['PR'] + PF_SERIES:
                    continue
                attempts = []
                for j, h in enumerate(historic_totals):
                    if h['system'] == system and h['candidate_id'] == r['candidate_id'] and str(h['epsilon']) == epsilon and h['factor'] == '1.0' and h['metric'] == 'RZ' and h['model'] == 'NATIVE_MIXED':
                        assert system == 'H4' and r['DF_rank'] == 10
                        matched_pair = next(hp for hp in historic_pairs if r['candidate_id'] in hp['input_and_compiled_provenance'])
                        old_identity = matched_pair['input_and_compiled_provenance'][r['candidate_id']]
                        assert old_identity['hamiltonian_hash'] != r['hamiltonian_hash']
                        assert old_identity['state_hash'] != r['state_hash']
                        attempts.append(dict(source=reference('stage3_shot_weighted_totals', f'/{j}'),
                                             accepted=False, reason='H4_LEGACY_DF12_INPUT_AND_COHORT_DO_NOT_MATCH_CURRENT_DF10',
                                             legacy_identity=old_identity,
                                             current_identity={k: r[k] for k in ['hamiltonian_hash', 'state_hash', 'DF_rank', 'cohort', 'n']}))
                methods.append(dict(system=system, epsilon=epsilon, method=r['series'], candidate_id=r['candidate_id'],
                                    compiled_G=r['cost_exact'], theory_G_lower=None, theory_G_upper=None,
                                    compiled_status='SAVED_POINT', theory_status='MISSING_MATCHED_TOTAL',
                                    status='NOT_COMPARABLE', DF_rank=r['DF_rank'], cohort=r['cohort'],
                                    source=reference('legacy_figure_data', f'/rows/{i}'), rejected_ID_only_matches=attempts))
    historical = []
    for i, row in enumerate(historic_totals):
        if row['metric'] != 'RZ' or row['model'] != 'NATIVE_MIXED' or row['factor'] != '1.0' or '_2nd' not in row['candidate_id']:
            continue
        j, shot = next((j, s) for j, s in enumerate(shots) if s['candidate_id'] == row['candidate_id'] and s['epsilon'] == row['epsilon'] and s['factor'] == row['factor'] and s['system'] == row['system'])
        historical.append(dict(scope='HISTORICAL_FIXED_CANDIDATE_S2', system=row['system'], epsilon=str(row['epsilon']),
                               candidate_id=row['candidate_id'], theory_model='NATIVE_MIXED / explicit_rz_sx_x_cx_v1',
                               G_theory_lower=row.get('theory_G_lower'), G_theory_upper=row.get('theory_G_upper'),
                               G_compiled=row.get('compiled_G_saved'), status=row['status'], shots=shot.get('shots'),
                               cohort=shot['sample_phase'], n=shot['sample_count'], source=reference('stage3_shot_weighted_totals', f'/{i}'),
                               shot_source=reference('stage3_preserved_shots', f'/{j}')))
    references = []
    # Both saved pairs are retained. No best candidate is selected in this panel.
    for i, pair in enumerate(historic_pairs):
        if pair['system'] != 'H4' or pair['epsilon'] != 0.05 or pair['factor'] != '1.0' or pair['metric'] != 'RZ' or pair['model'] != 'NATIVE_MIXED' or pair['PF'] != 'B1_p12_q1_2nd':
            continue
        assert pair['status'] == 'CONDITIONAL_SAVED_COST_COMPARISON'
        pr = next(h for h in historical if h['system'] == 'H4' and h['epsilon'] == '0.05' and h['candidate_id'] == pair['PR'])
        pf = next(h for h in historical if h['system'] == 'H4' and h['epsilon'] == '0.05' and h['candidate_id'] == pair['PF'])
        assert pr['status'] == pf['status'] == 'ELIGIBLE_SAVED_EMPIRICAL_SHOTS'
        identities = pair['input_and_compiled_provenance']
        for key in ['hamiltonian_hash', 'state_hash', 'input_sha256', 'DF_receipt_sha256']:
            assert identities[pair['PR']][key] == identities[pair['PF']][key]
        for candidate in [pr, pf]:
            assert candidate['n'] == identities[candidate['candidate_id']]['mean_sample_size']
            mr = [m for m in models if m['candidate_id'] == candidate['candidate_id'] and m['model'] == 'NATIVE_MIXED']
            assert len(mr) == 2 and {m['axis'] for m in mr} == {'cosine', 'sine'}
            assert all(m['control'] == 'symmetric_directional' and m['theory_profile'] == 'explicit_rz_sx_x_cx_v1' for m in mr)
        theory_lo = Fraction(pr['G_theory_lower']) / Fraction(pf['G_theory_upper'])
        theory_hi = Fraction(pr['G_theory_upper']) / Fraction(pf['G_theory_lower'])
        compiled = Fraction(pr['G_compiled']) / Fraction(pf['G_compiled'])
        assert theory_lo == Fraction(pair['theory_PR_PF_lower'])
        assert theory_hi == Fraction(pair['theory_PR_PF_upper'])
        assert compiled == Fraction(pair['compiled_PR_PF'])
        reduction = lambda c, t: str(100 * (1 - Fraction(c) / Fraction(t)))
        references.append(dict(scope='REFERENCE_H4_LEGACY_DF12_FIXED_S2_PAIR', system='H4', epsilon='0.05',
                               PR_candidate_id=pair['PR'], PF_candidate_id=pair['PF'], PF_method='full PF S2 (fixed; not best across truncations)',
                               rho_theory_lower=str(theory_lo), rho_theory_upper=str(theory_hi), rho_compiled=str(compiled),
                               PR_reduction_percent_lower=reduction(pr['G_compiled'], pr['G_theory_lower']),
                               PR_reduction_percent_upper=reduction(pr['G_compiled'], pr['G_theory_upper']),
                               PF_reduction_percent_lower=reduction(pf['G_compiled'], pf['G_theory_lower']),
                               PF_reduction_percent_upper=reduction(pf['G_compiled'], pf['G_theory_upper']),
                               PR=pr, PF=pf, provenance=identities,
                               SOURCE_accounting=pair['SOURCE_accounting'], SOURCE_counter_evaluation=pair['SOURCE_counter_evaluation'],
                               source=reference('stage3_pf_pr_comparisons', f'/{i}'),
                               comparable='CONDITIONAL_SAVED_FIXED_PAIR_ONLY', same_saved_shots_per_candidate=True,
                               same_event_expectation_vs_mean=False, population_mean_certified=False,
                               old_compiler_runtime_bit_identity_certified=False))
    assert len(references) == 2 and len(current) == 20 and len(methods) == 80 and len(historical) == 25
    return dict(schema='SAVED_S2_THEORY_COMPILED_DISPLAY_V1', epsilons=EPS, current=current,
                references=references, method_coverage=methods, historical_S2_coverage=historical,
                current_complete_four_total_pairs=0, current_compiled_pairs_with_saved_S2_selection=19,
                figures=FIGURES, arithmetic='Division of saved total costs and 100*(1-compiled/theory) only; exact Fraction then display float',
                forbidden_scientific_calls=0)


def exported_rows(d):
    rows = []
    for r in d['references']:
        for role in ['PR', 'PF']:
            c = r[role]
            rows.append(dict(scope=r['scope'], system=r['system'], epsilon=r['epsilon'], pair_PR=r['PR_candidate_id'], role=role,
                             candidate_id=c['candidate_id'], G_theory_lower=c['G_theory_lower'], G_theory_upper=c['G_theory_upper'],
                             G_compiled=c['G_compiled'], rho_theory_lower=r['rho_theory_lower'], rho_theory_upper=r['rho_theory_upper'],
                             rho_compiled=r['rho_compiled'], reduction_percent_lower=r[role + '_reduction_percent_lower'],
                             reduction_percent_upper=r[role + '_reduction_percent_upper'], shots=c['shots'], cohort=c['cohort'], n=c['n'],
                             theory_model=c['theory_model'], status=r['comparable'], source=c['source'], shot_source=c['shot_source'],
                             pair_source=r['source'], identities=r['provenance'][c['candidate_id']]))
    return rows


def coverage_rows(d):
    keys = ['scope', 'system', 'epsilon', 'PR_candidate_id', 'PF_candidate_id', 'PF_method',
            'G_PR_theory_lower', 'G_PR_theory_upper', 'G_PF_theory_lower', 'G_PF_theory_upper',
            'G_PR_compiled', 'G_PF_compiled', 'rho_theory_lower', 'rho_theory_upper', 'rho_compiled',
            'PR_shots', 'PR_shot_status', 'PF_shots', 'status', 'reasons', 'PR_selection_source', 'PF_selection_source']
    return [{k: r[k] for k in keys} for r in d['current']]


def render(d):
    os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'svg.hashsalt': 'saved-s2-theory-compiled-20261022'})
    coordinates = []
    for kind, stem in zip(['A', 'B'], FIGURES):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.8), sharey=True)
        for ax, record in zip(axes, d['references']):
            title = 'PR q1' if '_q1_' in record['PR_candidate_id'] else 'PR q8'
            ax.set_title(title, fontsize=12, pad=13)
            ax.set_xticks([0], ['H4'])
            ax.set_xlim(-0.6, 0.6)
            ax.grid(axis='y', alpha=0.2)
            if kind == 'A':
                ax.axhline(1, color='#4b5563', lw=1.1, linestyle='--', zorder=1)
                for x, field, label, marker, color in [(-.11, 'rho_theory_lower', 'Theory', 'D', '#8064a2'),
                                                       (.11, 'rho_compiled', 'Compiled', 'o', '#008c95')]:
                    value = float(Fraction(record[field]))
                    ax.plot(x, value, marker=marker, markersize=9, color=color, markerfacecolor='white' if marker == 'D' else color,
                            markeredgewidth=1.8, linestyle='none', label=label, zorder=3)
                    if label == 'Theory':
                        upper = float(Fraction(record['rho_theory_upper']))
                        ax.vlines(x, value, upper, colors=color, lw=2)
                    coordinates.append(dict(figure=stem, panel=title, series=label, value_exact=record[field], value_display=value))
                ax.set_ylim(0, 2.85)
                ax.yaxis.set_major_locator(MultipleLocator(.5))
            else:
                ax.axhline(0, color='#4b5563', lw=1.1, linestyle='--', zorder=1)
                for x, role, label, color in [(-.16, 'PR', 'PR S2', '#008c95'), (.16, 'PF', 'Full PF S2', '#4275a8')]:
                    lower = float(Fraction(record[role + '_reduction_percent_lower']))
                    upper = float(Fraction(record[role + '_reduction_percent_upper']))
                    assert lower == upper  # Only these exact, saved bounds coincide.
                    ax.bar(x, lower, width=.27, color=color, label=label, zorder=3)
                    ax.text(x, lower + 1.5, f'{lower:.1f}', ha='center', va='bottom', fontsize=11)
                    coordinates.append(dict(figure=stem, panel=title, series=label,
                                            value_exact=record[role + '_reduction_percent_lower'], value_display=lower))
                ax.set_ylim(-3, 65)
                ax.yaxis.set_major_locator(MultipleLocator(10))
        axes[0].set_ylabel('PR / PF total RZ cost' if kind == 'A' else 'Cost reduction (%)')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .86), ncol=2, frameon=False, fontsize=11)
        fig.suptitle(r'H4 legacy DF12  |  $\epsilon=0.05$', y=.98, fontsize=14)
        fig.subplots_adjust(left=.12, right=.98, bottom=.13, top=.67, wspace=.18)
        for ext in ['png', 'svg', 'pdf']:
            metadata = {'Creator': 'Saved-data display only'} if ext == 'pdf' else {}
            fig.savefig(HERE / (stem + '.' + ext), dpi=220, metadata=metadata)
        plt.close(fig)
    dump('display_coordinates.json', coordinates)


def check(d):
    assert json.loads((HERE / 'figure_data.json').read_text()) == d
    audit = json.loads((HERE / 'coverage_audit.json').read_text())
    assert audit['current'] == d['current']
    assert audit['method_coverage'] == d['method_coverage']
    assert audit['historical_S2_coverage'] == d['historical_S2_coverage']
    assert len(list(csv.DictReader((HERE / 'coverage_matrix.csv').open()))) == 20
    assert len(list(csv.DictReader((HERE / 'figure_data.csv').open()))) == 4
    for stem in FIGURES:
        for ext in ['png', 'svg', 'pdf']:
            assert (HERE / (stem + '.' + ext)).stat().st_size > 1000
    coords = json.loads((HERE / 'display_coordinates.json').read_text())
    assert len(coords) == 8
    for c in coords:
        assert c['value_display'] == float(Fraction(c['value_exact']))
    return dict(status='PASS', fixed_input_blobs=len(inputs()), current_conditions=20, method_coverage_rows=80,
                current_complete_four_total_pairs=0, current_S2_selected_compiled_pairs=19,
                current_S2_PF_selection_unresolved=1, historical_S2_candidate_conditions=25,
                historical_reference_pairs=2, figures=2, exports=6, display_coordinates=8,
                saved_ratio_tokens_equal_recomputed_display_ratios=True,
                new_theory_costs=0, new_shots=0, new_total_accounting=0, new_cost_rankings=0,
                new_compilation=0, new_sampling=0, independent_scientific_review=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--check', action='store_true')
    parsed = args.parse_args()
    result = data()
    if parsed.check:
        print(json.dumps(check(result)))
    else:
        dump('figure_data.json', result)
        write_csv('figure_data.csv', exported_rows(result))
        write_csv('coverage_matrix.csv', coverage_rows(result))
        write_csv('coverage_methods.csv', result['method_coverage'])
        write_csv('historical_coverage.csv', result['historical_S2_coverage'])
        dump('coverage_audit.json', {k: result[k] for k in ['current', 'method_coverage', 'historical_S2_coverage']} |
             {'scope': 'Specified pinned sources only; no claim that unsaved costs exist nowhere',
              'no_theory_total_for_B0_B0I_in_inspected_sources': True,
              'no_current_main_A_or_B_figures': 'No four-total pair matches; avoid empty main figures',
              'H6_PF_S4_reference_excluded': 'Task is S2 only',
              'H4_epsilon_0p001': 'Saved overall winner is S4; no saved S2-only winner copied and no new argmin executed'})
        render(result)
        dump('verification_data.json', check(result))
        print(json.dumps(check(result)))
