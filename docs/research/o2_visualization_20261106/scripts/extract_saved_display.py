"""Extract fixed saved fields for display, without numerical arithmetic.

No float, Fraction, scientific modules, statistics, ranking or report runners.
The explicit selectors copy prior selections, including their result versions.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
BASE = '59f7083dacf6fdf25b6b593608d60c1e816846a3'
CAT = ROOT / 'docs/research/o2_data_catalog_20261105'
PLAN = ROOT / 'docs/research/o2_figure_planning_20261105'
EPS = ['0.05', '0.01', '0.005', '0.001', '0.0005']
TIMES = ['0.8', '1.2', '1.6', '2.0']
TARGETS = ['DF03', 'DF04', 'DF07', 'DF09', 'PA01', 'PA02', 'PA03',
           'XP02', 'XP04', 'ME01', 'ME03', 'ME04', 'ME05', 'ME06']
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], env=ENV)


def scalar(value):
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    assert isinstance(value, str)
    return value


def pointer(d, p):
    for key in p.split('/')[1:]:
        key = key.replace('~1', '/').replace('~0', '~')
        d = d[int(key)] if isinstance(d, list) else d[key]
    return d


def write_csv(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False, separators=(',', ':'))
                         if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in records)


def main():
    refs = json.loads((CAT / 'SOURCE_INDEX.json').read_text())['references']
    refs.update(json.loads((PLAN / 'VALIDATION.json').read_text())['supplemental_fixed_refs'])
    for key, commit, path in [
        ('VIS_P3_PRICES', 'e44e87a306d05c8549f1a044b7fd056c2e1055a3',
         'artifacts/df_pauli_matched_partition_validation/2026-11-02/stage3/run04/price_statistics.json'),
        ('VIS_MATCHING', '2779710a86d380bb484accc51d3dccba183370b8',
         'docs/research/o2_data_catalog_20261105/df_pauli_matching.csv'),
    ]:
        blob = git('rev-parse', commit + ':' + path).decode().strip()
        refs[key] = dict(commit=commit, path=path, blob=blob, bytes=int(git('cat-file', '-s', blob)),
                         url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/' + commit + '/' + path)
    cache = {}

    def raw(key):
        if key not in cache:
            e = refs[key]
            cache[key] = git('show', e['commit'] + ':' + e['path'])
        return cache[key]

    def table(key):
        result = []
        for i, row in enumerate(csv.DictReader(io.StringIO(raw(key).decode('utf-8-sig'))), 1):
            result.append((dict(type='csv_row', data_row=i), row))
        return result

    def document(key):
        return json.loads(raw(key), parse_int=str, parse_float=str)

    def select(key, **where):
        return [(dict(s, where=where), r) for s, r in table(key)
                if all(r[k] == v for k, v in where.items())]

    def one(key, **where):
        result = select(key, **where)
        assert len(result) == 1, (key, where, len(result))
        return result[0]

    plans = {r['figure_candidate_id']: r for r in csv.DictReader((PLAN / 'figure_data_plan.csv').open())}
    studies = {r['study_id']: r for r in csv.DictReader((CAT / 'study_catalog.csv').open())}
    points, manifest = [], []
    figure_candidates = {
        'V01': ['DF03'], 'V02': ['DF04'], 'V03': ['DF07'], 'V04': ['DF09'],
        'V05': ['PA01', 'PA02', 'PA03'], 'V06': ['XP02'], 'V07': ['XP04'],
        'V08': ['ME01'], 'V09': ['ME03', 'ME04'], 'V10': ['ME05'], 'V11': ['ME06'],
    }

    def point(fig, cid, panel, series, ref, source, field, *, xfield='', x='', err='',
              candidate='', cohort='', kind='DIRECT_SAVED_COMPILED_PRICE', unit='RZ×shot',
              status='', display='numeric', metadata=None):
        selector, row = source
        identity = candidate or row.get('candidate_id', row.get('id', row.get('arm', '')))
        saved_cohort = row.get('cohort', cohort)
        datum = dict(point_id='saved-mark-' + str(len(points) + 1), figure_id=fig,
            figure_candidate_id=cid, panel_id=panel, series_id=series, ref_id=ref,
            source_selector=selector, y_field=field, y_literal=scalar(row[field]),
            x_field=xfield, x_literal=scalar(row[xfield]) if xfield else x,
            se_field=err, se_literal=scalar(row[err]) if err else '',
            candidate_id=identity, source_cohort=saved_cohort, price_kind=kind,
            n_literal=scalar(row.get('price_n', row.get('n', None))),
            shared_price_group=ref + ':' + identity + ':' + saved_cohort,
            unit=unit, original_status=status or row.get('status', row.get('eligibility', '')),
            display_kind=display, metadata_literals=metadata or {})
        points.append(datum)
        return datum

    def jsource(key, ptr):
        return dict(type='json_pointer', pointer=ptr), pointer(document(key), ptr)

    # All four saved comparison scopes; no min/ranking is repeated.
    for source in table('PRI_PF60'):
        _, r = source
        panel = r['system'] + '|' + r['scope']
        point('V01', 'DF03', panel, 'PF', 'PRI_PF60', source, 'PF_total_RZ', xfield='epsilon',
              candidate=r['PF_id'], cohort='DETERMINISTIC', metadata={'scope': r['scope']})
        point('V01', 'DF03', panel, 'PR', 'PRI_PF60', source, 'PR_mean_exact', xfield='epsilon',
              err='PR_SE', candidate=r['PR_candidate_id'], cohort='SAVED_EXPLORATION',
              metadata={k: r[k] for k in ['PR_n', 'PR_formula', 'PR_q', 'direction']})

    # ALL is a stored scope label, not a new selection. Retain the original 40 cells.
    for source in select('PRI_LONGBEST', formula_group='ALL'):
        _, r = source
        point('V02', 'DF04', r['system'] + '|' + r['epsilon'], r['family'] + '_exploration',
              'PRI_LONGBEST', source, 'G', xfield='T', err='G_SE',
              kind='DIRECT_SAVED_' + r['cohort'], metadata={k: r[k] for k in ['formula', 'q', 'prefix', 'R_outer', 'K', 'n']})
    confirmations = select('DF_LONG_COMPARE', kind='PR_VS_PF')
    assert len(confirmations) == 33  # Locator coverage, not a recomputed success count.
    for source in confirmations:
        _, r = source
        panel = r['system'] + '|' + r['epsilon']
        common = {k: r[k] for k in ['left_n', 'right_n', 'difference_SE', 'lower_cost_point_reproduced']}
        point('V02', 'DF04', panel, 'PR_fixed_confirmation', 'DF_LONG_COMPARE', source,
              'left_G', xfield='T', candidate=r['left'], cohort='FIXED_CONFIRMATION', metadata=common)
        point('V02', 'DF04', panel, 'PF_saved_comparator', 'DF_LONG_COMPARE', source,
              'right_G', xfield='T', candidate=r['right'], cohort='SAVED_COMPARATOR', metadata=common)

    # B3 aliases repeat across the three baseline families; draw B3 once per cell.
    for source in table('PRI_B3CSV'):
        _, r = source
        panel = r['system'] + '|' + r['epsilon']
        common = {k: r[k] for k in ['B3_n', 'baseline_n', 'difference_SE', 'B3_lower', 'baseline_uses_independent_confirmation']}
        if r['family'] == 'B2':
            point('V03', 'DF07', panel, 'B3', 'PRI_B3CSV', source, 'B3_G', xfield='T',
                  candidate=r['B3_id'], cohort='FIXED_NOMINEE_CONFIRMATION', metadata=common)
        else:
            _, same = one('PRI_B3CSV', system=r['system'], T=r['T'], epsilon=r['epsilon'], family='B2')
            assert same['B3_id'] == r['B3_id'] and same['B3_G'] == r['B3_G']
        point('V03', 'DF07', panel, r['family'], 'PRI_B3CSV', source, 'baseline_G', xfield='T',
              candidate=r['baseline_id'], cohort='SOURCE_BASELINE_NOT_RELABELED', metadata=common)

    h8 = document('H8_C_RAW')
    for arm, ptr, candidate in [('PR', '/PR_summary', h8['scope']['comparison_ids']['PR']),
                               ('PF', '/fixed_PF', h8['scope']['comparison_ids']['PF'])]:
        point('V04', 'DF09', 'G', arm, 'H8_C_RAW', jsource('H8_C_RAW', ptr), 'mean_G_exact',
              x=arm, err='sample_SE_G', candidate=candidate,
              metadata={'n': pointer(h8, ptr)['n'], 'cohort': pointer(h8, ptr)['cohort']})
    assert '24.58909439%' in raw('H8_C').decode()

    # Explicit mappings are copied from the saved report and the preceding plan.
    # Do not minimize over the candidate tables.
    h2_rte = {'0.05': 'PR_p0000_q1_r4_K2', '0.01': 'PR_p0000_q1_r4_K2',
              '0.005': 'PR_p0000_q2_r2_K4', '0.001': 'PR_p0000_q2_r2_K4', '0.0005': 'PR_p0000_q2_r2_K4'}
    for e in EPS:
        h2 = [('Hybrid', 'PA_BASE', one('PA_BASE', molecule='H2', epsilon_sig=e, family='PR_confirmed_best')),
              ('RTE', 'PRI_PA-E4', one('PRI_PA-E4', candidate_id=h2_rte[e], epsilon_sig=e, cohort='confirmation'))]
        h4 = []
        for role in ['Hybrid', 'RTE']:
            choices = [(s, r) for s, r in select('PRI_PA-E5', molecule='H4', epsilon_sig=e) if r['role'] == role]
            assert len(choices) == 1
            h4.append((role, 'PRI_PA-E5', choices[0]))
        comp = one('PRI_PA-E6compare', epsilon_sig=e)
        hr = one('PA_H6_CAND', candidate_id=comp[1]['new_Hybrid_id'], cohort='confirmation', epsilon_sig=e)
        rr = one('PRI_PA-E5', molecule='H6', role='RTE', epsilon_sig=e)
        assert hr[1]['G_exact'] == comp[1]['new_Hybrid_G_exact']
        assert rr[1]['G_exact'] == comp[1]['RTE_G_exact']
        h6 = [('Hybrid', 'PA_H6_CAND', hr), ('RTE', 'PRI_PA-E5', rr)]
        for system, cid, selected in [('H2', 'PA01', h2), ('H4', 'PA02', h4), ('H6', 'PA03', h6)]:
            for role, key, source in selected:
                for quantity, field, error, unit in [
                    ('G', 'G_exact', 'G_SE', 'RZ×shot'),
                    ('N_cos', 'N_cos', '', 'analytic quantum shots'),
                    ('N_sin', 'N_sin', '', 'analytic quantum shots'),
                    ('C_cos', 'C_cos_exact', '', 'compiled RZ/shot'),
                    ('C_sin', 'C_sin_exact', '', 'compiled RZ/shot')]:
                    panel = system + '|' + quantity.split('_')[0]
                    series = role if quantity == 'G' else role + '|' + quantity.split('_')[1]
                    point('V05', cid, panel, series, key, source, field, xfield='epsilon_sig',
                          err=error, unit=unit,
                          metadata={'selection_policy': 'ORIGINAL_REPORT_CHOICE;NO_NEW_MIN',
                                    'original_cohort_blank_retained': source[1].get('cohort', '') == ''})

    # P3 price statistics are the exact Stage4 input, not a newly computed mean.
    binding = document('P3_STAGE4_BINDINGS')
    stats = refs['VIS_P3_PRICES']
    assert binding['inputs'][stats['path']]['ref'] == stats['commit']
    assert hashlib.sha256(raw('VIS_P3_PRICES')).hexdigest() == binding['inputs'][stats['path']]['sha256']
    for source in table('P3_ACCOUNT'):
        _, r = source
        for quantity, field, error, unit in [
            ('G', 'RZ_G_exact', 'RZ_price_SE', 'RZ×shot'),
            ('N_cos', 'N_cosine', '', 'analytic quantum shots'),
            ('N_sin', 'N_sine', '', 'analytic quantum shots')]:
            panel = quantity.split('_')[0]
            series = r['arm'] if quantity == 'G' else r['arm'] + '|' + quantity.split('_')[1]
            point('V06', 'XP02', panel, series, 'P3_ACCOUNT', source, field,
                  xfield='epsilon_sig', err=error, cohort='STAGE3_RUN04_PRICES_REUSED', unit=unit,
                  metadata={'n': '32', 'matching': 'CONDITIONAL_MATCHED'})
    for arm in ['P', 'DF']:
        for axis in ['cosine', 'sine']:
            ptr = '/arms/' + arm + '/axes/' + axis + '/RZ'
            point('V06', 'XP02', 'C', axis, 'VIS_P3_PRICES', jsource('VIS_P3_PRICES', ptr),
                  'mean_exact', x=arm, err='standard_error', candidate=arm,
                  cohort='STAGE3_RUN04_PRICES', unit='compiled RZ/shot', metadata={'n': '32'})

    for source in table('CNG'):
        _, r = source
        rowkey = '|'.join(r[k] for k in ['system', 'family', 'T', 'higher_formula'])
        high = one('PRI_LONGBEST', system=r['system'], T=r['T'], epsilon=r['epsilon'],
                   family=r['family'], formula_group=r['higher_formula'])
        low = one('PRI_LONGBEST', system=r['system'], T=r['T'], epsilon=r['epsilon'],
                  family=r['family'], formula_group='2nd')
        identity = [dict(ref_id='PRI_LONGBEST', selector=s, fields={'id': a['id']}) for s, a in [high, low]]
        for q, field in [('N', 'higher_over_S2_N'), ('C_eff', 'higher_over_S2_C_eff'), ('G', 'higher_over_S2_G')]:
            point('V08', 'ME01', q, rowkey, 'CNG', source, field, xfield='epsilon',
                  candidate=high[1]['id'], cohort='SAVED_EXPLORATION', unit='saved higher/S2 ratio',
                  metadata=dict({k: r[k] for k in ['S2_q', 'higher_q', 'S2_prefix', 'higher_prefix', 'finite_sample_point_comparison']},
                                saved_candidate_identity_bindings=identity))

    for cid, system in [('ME03', 'H4'), ('ME04', 'H6')]:
        selectors = json.loads(plans[cid]['candidate_selectors'])[0]['selector']
        catalog_literals = [r for r in csv.DictReader((CAT / 'candidate_results.csv').open())
                            if r['record_kind'] == 'LITERAL_EXTRACT' and r['ref_id'] == 'BOUNDARY_RESULTS']
        fixed_sources = []
        for candidate_id in selectors['where_in']['id']:
            saved = next(r for r in catalog_literals if r['candidate_id'] == candidate_id)
            fixed_sources.append(one('BOUNDARY_RESULTS', **json.loads(saved['row_selector'])['where']))
        for source in fixed_sources:
            _, r = source
            assert r['system'] == system and r['epsilon'] == '0.0005'
            for quantity, field, error, unit in [
                ('G', 'G', 'G_SE', 'RZ×shot'), ('N_cos', 'N_cos', '', 'analytic quantum shots'),
                ('N_sin', 'N_sin', '', 'analytic quantum shots'),
                ('C_cos', 'C_cos', '', 'compiled RZ/shot'), ('C_sin', 'C_sin', '', 'compiled RZ/shot')]:
                panel = system + '|' + quantity.split('_')[0]
                series = 'G' if quantity == 'G' else quantity.split('_')[1]
                point('V09', cid, panel, series, 'BOUNDARY_RESULTS', source, field,
                      xfield='R_outer', err=error, unit=unit,
                      metadata={k: r[k] for k in ['diagnostic', 'point_lower', 'n', 'record_kind', 'R_outer', 'formula', 'prefix', 'K']})

    p1 = document('P1_DIFFERENCES')
    for label in ['H16', 'H32', 'H64', 'R64']:
        for component in ['outer', 'hold', 'trunc', 'grid']:
            ptr = '/rows/' + label + '/differences/' + component
            for series, field in [('Absolute difference', 'absolute'), ('Diagnostic resolution', 'absolute_resolution')]:
                point('V10', 'ME05', component, series, 'P1_DIFFERENCES', jsource('P1_DIFFERENCES', ptr),
                      field, x=label, candidate=p1['rows'][label]['candidate'],
                      cohort='SAVED_SIGNAL_DIAGNOSTIC', kind='ENGINEERING_DIAGNOSTIC', unit='dimensionless signal',
                      status=p1['rows'][label]['differences'][component]['status'],
                      metadata={'structural_zero_terms': p1['rows'][label].get('structural_zero_terms', []),
                                'uncertainty_kind': p1['rows'][label]['differences'][component]['uncertainty_kind']})
    for label in ['H32', 'H64', 'R64']:
        for axis in ['cosine', 'sine']:
            ptr = '/interactions/' + label + '/' + axis + '/RZ'
            for field in ['full', 'retained', 'event', 'empty', 'Delta_interaction']:
                point('V11', 'ME06', label + '|' + axis, field, 'P2_INTERACTIONS',
                      jsource('P2_INTERACTIONS', ptr), field, x=field, candidate=label,
                      cohort='CONFIRMATION_INDEX_0000', unit='compiled RZ/shot',
                      display='annotation' if field == 'Delta_interaction' else 'numeric')

    fields = list(points[0])
    assert all(list(p) == fields for p in points)
    write_csv(OUT / 'DISPLAY_DATA.csv', points)

    titles = {
        'V01': 'DF: strong PF comparison across saved precision conditions',
        'V02': 'DF: time and precision, exploration and fixed confirmation',
        'V03': 'DF B3: saved comparisons against B2 and deterministic baselines',
        'V04': 'H8 Stage C: fixed PR confirmation and saved PF comparator',
        'V05': 'Pauli-JW: saved Hybrid and pure-RTE precision comparisons',
        'V06': 'H4 P3: conditional matched-partition N / C / G comparison',
        'V07': 'DF B3 and Pauli pure RTE: definitions and comparison limits',
        'V08': 'Saved higher-order / S2 ratios: N, C_eff and G',
        'V09': 'DF R boundaries: saved N / C / G at fixed T and precision',
        'V10': 'H6 P1: saved signal differences and diagnostic resolution',
        'V11': 'H6 P2: fixed-program Full / R / E / Empty compiled prices',
    }
    used = set()
    for fid, cids in figure_candidates.items():
        pp = [p for p in points if p['figure_id'] == fid]
        rr = list(dict.fromkeys(p['ref_id'] for p in pp))
        if fid == 'V07':
            rr = ['VIS_MATCHING', 'SRC_B3', 'PAULISYNTH']
        rr += list(dict.fromkeys(ref for cid in cids for ref in json.loads(plans[cid]['source_refs']) if ref not in rr))
        used.update(rr)
        ss = list(dict.fromkeys(s for cid in cids for s in json.loads(plans[cid]['study_ids'])))
        if fid == 'V05' and 'pauli_h2_h6_comparison' not in ss:
            ss.append('pauli_h2_h6_comparison')  # Original H2 Hybrid carrier.
        for study_id in ss:
            for key in json.loads(studies[study_id]['identity_refs']):
                if key not in rr:
                    rr.append(key)
                    used.add(key)
        original_comparisons = list(dict.fromkeys(c for cid in cids for c in json.loads(plans[cid]['comparison_ids'])))
        manifest.append(dict(figure_id=fid, figure_candidate_ids=cids, title=titles[fid],
            svg_path='figures/' + fid + '.svg', png_path='figures/' + fid + '.png',
            study_ids=ss, comparison_ids=original_comparisons, source_refs=rr,
            scientific_SOURCE_roles={s: json.loads(studies[s]['source_commit']) for s in ss},
            fixed_conditions={cid: plans[cid]['fixed_conditions'] for cid in cids},
            inherited_limitations={cid: plans[cid]['limitations'] for cid in cids},
            selectors=list({json.dumps(dict(ref_id=p['ref_id'],selector=p['source_selector']),sort_keys=True):
                            dict(ref_id=p['ref_id'],selector=p['source_selector']) for p in pp}.values()),
            saved_fields=list(dict.fromkeys(p['y_field'] for p in pp)),
            cohorts=list(dict.fromkeys(p['source_cohort'] for p in pp)),
            price_kinds=list(dict.fromkeys(p['price_kind'] for p in pp)),
            units=list(dict.fromkeys(p['unit'] for p in pp)),
            comparison_class={cid: plans[cid]['comparison_class'] for cid in cids},
            assumptions={cid: plans[cid]['matching_record'] for cid in cids if plans[cid]['matching_id']},
            existing_figure_reused='false', reuse='Saved fields; display-only log axes, Fraction conversion and panel/marker patterns adapted from prior plotting code; no old report runner executed',
            new_numeric_analysis='false', status='CREATED_FROM_SAVED_DATA',
            held_panel_or_limit='V02: confirmation CSV covers 33 of the 40 exploration cells; no other confirmation imputed. V04: PR mean C not saved, G-only display. V06: C from original Stage4 input.' if fid in {'V02','V04','V06'} else 'NONE',
            data_rows=len(pp)))
    write_csv(OUT / 'FIGURE_MANIFEST.csv', manifest)
    reuse_refs=['CODE_LONG_TIME_0','CODE_PA_ZOOM_0','CODE_OLD_H4_0']
    used.update(reuse_refs)
    meta = dict(schema='O2-SAVED-VISUALIZATION-1', date_JST='2026-11-06', base_plan=BASE,
        catalog_commit='2779710a86d380bb484accc51d3dccba183370b8',
        synthesis_commit='3380f6de0d86c0ac1db9aeb6d97ab5d328dd43bd',
        target_candidates=TARGETS, inherited_claim_scope='CURRENT_CLAIM_SCOPE_EVIDENCE_SUFFICIENT',
        fixed_refs={k:refs[k] for k in sorted(used)}, figures=list(figure_candidates),
        figure_count=len(manifest), display_rows=len(points),
        scope=dict(new_scientific_execution=False,numeric_reanalysis=False,new_G_N_C_SE_ratio=False,
            candidate_reselection=False,sampling=False,circuit_build_compile=False,
            detailed_scientific_review=False,manuscript=False,formal_claim_adoption=False),
        permitted_display='Saved-field extraction; string-to-number conversion; log axes; saved ±1SE errorbars as geometric display, not new confidence intervals; no new statistical endpoints exported',
        reuse=dict(static_code_refs=reuse_refs,
            adapted='Matplotlib Agg/SVG text, faceted axes, markers, log scales and float(Fraction(saved_literal)) from existing display code; missing values remain gaps',
            excluded='Old report generators recompute G/SE/ratios/select minima; old 2*SE plotting and missing-SE-to-zero are not reused',
            new_code_reason='New cross-study figures and explicit cohorts need a display-only extraction/render split; existing full report runners cannot safely be called'),
        P3_price_input_sha256=hashlib.sha256(raw('VIS_P3_PRICES')).hexdigest(),
        missing_policy='CSV blanks and JSON null preserved; absent side-specific confirmation SE is not replaced by comparison-difference SE; no mean C reconstructed for H8',
        technical_status='EXTRACTED_PENDING_RENDER_VALIDATION',new_scientific_PASS=False,
        publication='Dedicated branch push, remote SHA and all public file bodies/blobs checked, then mandatory STOP')
    (OUT / 'FIGURE_VALIDATION.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(figures=len(manifest),candidate_ids=TARGETS,display_rows=len(points),
                          fixed_refs=len(used),new_numeric_analysis=False),ensure_ascii=False))


if __name__ == '__main__':
    main()
