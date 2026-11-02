#!/usr/bin/env python3
"""Copy pinned public evidence into document proposals; stdlib/Git metadata only.

No scientific modules, raw array decoding, seed generation, statistics, cost
arithmetic, ranking, grant activation, or execution entry points are used.
Run --build to generate the H2-owned additive packet, then --verify after edits.
"""
import argparse
import csv
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'f2d1a86c431f835db700f348ce618d8604e7ee23'
PAPER = '1923fce8f7461513d1bc47b5061e87cd70917e87'
V1 = '9036d8a87ad1c38e83b9ff654b412dde7041e9d3'
V2_DRAFT = 'c0808b254aa02a94c7482f5c6e353b54532a54f8'
V2_ADOPTED = '1f001cfc4e8e0531279a26613be29a52c7c0640c'
FIGURE = 'eb33de67475e7d33eed56801b74a2a277147e4be'
PUB = 'artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/'
OUT = ROOT / 'artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19'
PAPER_DIR = 'artifacts/resource_applicability/track_a_h468_paper_evidence_integration/2026-10-19/'
DISPLAY = 'artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/'
FIG_DIR = 'artifacts/resource_applicability/track_a_mechanism_figure2_a82_readiness/2026-10-19/'
URL = 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'
BASELINE = Path('/tmp/h2-results-integration-20261019-preservation.json')
DOC = 'docs/research/track_a_h2_results_integration_20261019.md'
AMENDMENT = 'docs/research/track_a_pr_resource_protocol_v2_h2_update_DRAFT_20261019.md'
NOTE = 'docs/research/研究ノート/2026-10-19_h2_results_integration.md'
INDEXES = ['PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'scripts/README.md',
           'src/trotterlib/README.md', 'docs/README.md', 'docs/research/研究概要・現状.md']
LIMITS = ('FINITE_SAMPLE_POINT;NO_POPULATION_MEAN_GUARANTEE;UNOBSERVED_RARE_COST_UNKNOWN;'
          'NO_COHORT_POOLING;EMPIRICAL_BIAS_ALLOWANCE;ANALYTIC_SHOTS;COMPILED_RZ_NOT_FT;'
          'P0;NO_GLOBAL_OPTIMUM;INPUT_RULE_EQUIVALENCE_NOT_VERIFIED;NOT_FINAL_FREEZE')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


@lru_cache(None)
def blob(ref, path):
    return git('show', ref + ':' + path)


def parsed(ref, path, literals=False):
    return json.loads(blob(ref, path), **({'parse_float': str} if literals else {}))


def write_json(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_json(name):
    return json.loads((OUT / name).read_text())


def rows(data):
    return list(csv.DictReader(io.StringIO(data.decode())))


def pointer(value, path):
    for part in path.strip('/').split('/'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def write_csv(name, values, fields=None):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def reference(ref, path):
    return {'commit': ref, 'path': path, 'sha256': sha(blob(ref, path)),
            'url': URL + ref + '/' + path}


def prepare_baseline(create=False):
    """Only the known H2 private directory is read. Paths remain local in /tmp."""
    if not BASELINE.exists():
        if not create:
            raise FileNotFoundError('Original H2 preservation snapshot required: ' + str(BASELINE))
        private = ROOT / '.h2_pr_resource_private/2026-10-19'
        records = [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p.read_bytes())}
                   for p in sorted(private.rglob('*')) if p.is_file()]
        BASELINE.write_text(json.dumps({'base': BASE, 'private': records}))
    return json.loads(BASELINE.read_text())


def stage_evidence():
    receipt = read_json('github_read_receipt.json')
    evidence = []
    for stage in receipt['stages']:
        ref, directory = stage['completion_commit'], stage['path']
        for entry in stage['files']:
            data = blob(ref, entry['path'])
            identity = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert identity == entry['git_blob_sha'], entry['path']
            assert len(data) == entry['bytes']
            assert (ROOT / entry['path']).read_bytes() == data
        binding = parsed(ref, directory + '/execution_binding.json')
        assert binding['SOURCE'] == stage['SOURCE']
        source_files = []
        for path, digest in binding['source_hashes'].items():
            assert sha(blob(stage['SOURCE'], path)) == digest, (stage['stage'], path)
            assert sha((ROOT / path).read_bytes()) == digest, path
            source_files.append({'path': path, 'sha256': digest})
        inherited = []
        for path, digest in binding.get('inherited_source_hashes', {}).items():
            assert sha(blob(binding['original_SOURCE'], path)) == digest
            inherited.append({'path': path, 'sha256': digest})
        manifest = parsed(ref, directory + '/artifact_manifest.json')['files']
        manifest = ([{'path': key, 'sha256': value} for key, value in manifest.items()]
                    if isinstance(manifest, dict) else manifest)
        for item in manifest:
            assert sha(blob(ref, directory + '/' + item.get('path', item.get('file')))) == item['sha256']
        stage_id = stage['stage']
        result_name = 'confirmation_result.json' if stage_id == 'P' else 'result.json'
        audit_name = 'completion_audit.json' if stage_id == 'S_STATS' else 'native_completion_audit.json'
        result = parsed(ref, directory + '/' + result_name)
        audit = parsed(ref, directory + '/' + audit_name)
        native = parsed(ref, directory + '/native_exit.json')
        assert result['SOURCE'] == stage['SOURCE'] == audit['SOURCE']
        assert audit['status'] == 'COMPLETE_MANDATORY_STOP'
        assert native['exit_code'] == 0 and native['wait4_complete'] is True and native['same_parent_reaped'] is True
        result_digest = sha(blob(ref, directory + '/' + result_name))
        assert audit['result_sha256'] == (result['scientific_result_sha256'] if stage_id == 'P' else result_digest)
        evidence.append({
            'stage': stage_id, 'SOURCE': stage['SOURCE'], 'completion_commit': ref,
            'status': 'COMPLETE_MANDATORY_STOP', 'directory': directory,
            'result': reference(ref, directory + '/' + result_name),
            'scientific_result_sha256': audit['result_sha256'],
            'public_projection_differs_from_private_result': stage_id == 'P',
            'execution_binding': reference(ref, directory + '/execution_binding.json'),
            'spec_digest': binding['spec_digest'],
            'grant_sha256_or_digest': binding.get('grant_sha256', binding.get('grant_digest')),
            'completion_audit': reference(ref, directory + '/' + audit_name),
            'native_exit': reference(ref, directory + '/native_exit.json'),
            'artifact_manifest': reference(ref, directory + '/artifact_manifest.json'),
            'public_files_verified': len(stage['files']), 'source_files_verified': source_files,
            'inherited_source_files_verified': inherited,
            'exit_code': native['exit_code'], 'same_parent_reaped': native['same_parent_reaped'],
            'wait4_complete': native['wait4_complete'],
            'scope_note': {
                'F': 'New tol-only DF3, spin sector4; not legacy rank3/sector6.',
                'FP': 'Two diagnostic trajectories n1 each; not A/P statistical samples.',
                'D': 'Accepted D_run02 only. Original interrupted D_run01 and nonrefundable reservations retained.',
                'A': '66 candidates, new n8 each; FP diagnostic costs not pooled.',
                'S_STATS': 'Saved-summary design only; n80/12 proposal subsequently adopted before P.',
                'P': 'Fixed independent n80/12, 92 trajectories/184 compiled axes/256 RTE events; finite points only.'
            }[stage_id],
        })
    return evidence


def build():
    prepare_baseline(create=True)
    evidence = stage_evidence()
    write_json('stage_evidence.json', {'status': 'PUBLIC_SOURCE_BINDING_RESULT_EXIT_RECEIPTS_VERIFIED',
        'method': 'Pinned Git bytes matched to GitHub directory blob IDs; saved audits inspected without rerunning science.',
        'private_raw_reexecution_or_independent_review': False, 'stages': evidence})
    write_csv('stage_evidence.csv', [{
        'stage': x['stage'], 'SOURCE': x['SOURCE'], 'completion_commit': x['completion_commit'],
        'result_file': x['result']['path'], 'result_sha256': x['result']['sha256'],
        'scientific_result_sha256': x['scientific_result_sha256'], 'audit_file': x['completion_audit']['path'],
        'status': x['status'], 'scope_note': x['scope_note']} for x in evidence])
    ppath = PUB + 'P_completed_run01/confirmation_result.json'
    p = parsed(BASE, ppath, True)
    f = parsed(BASE, PUB + 'F_completed_run01/result.json', True)
    saved = p['accounting']['comparisons']
    main_bytes = blob(PAPER, DISPLAY + 'main_table.csv')
    main = rows(main_bytes)
    (OUT / 'H468_main_table_UNCHANGED.csv').write_bytes(main_bytes)
    (OUT / 'H468_methods_UNCHANGED.md').write_bytes(blob(PAPER, DISPLAY + 'methods.md'))
    additional = []
    for index, c in enumerate(saved):
        row = dict.fromkeys(main[0], '')
        row.update(comparison_id='H2_DF3_P_' + str(index + 1), system='H2',
            geometry='linear_1Angstrom', basis='STO-3G', DF_rank='3', DF_policy='tol_only_1e-8',
            T='0.8', P='0', epsilon=c['epsilon'], L_D='1', delta=['0.8', '0.2'][index],
            PR_id=c['cell_id'], PF_id=c['PF']['cell_id'], n=str(c['n']),
            G_PR_exact=c['G_exact'], paired_SE=c['paired_SE'], G_PF_exact=c['PF']['G_RZ_exact'],
            saved_direction_status=c['status'], lower_saved_point_cost='PR',
            PF_cost_class='DIRECT_MEASURED', cohort=p['scope'],
            audit_status='PUBLIC_COMPLETION_RECEIPTS_VERIFIED', result_commit=BASE,
            source_file=ppath, source_record='/accounting/comparisons/' + str(index),
            interpretation_limits=LIMITS)
        additional.append(row)
    write_csv('H2_adopted_points.csv', additional)
    write_csv('H2468_comparison_separate.csv', main + additional)
    write_json('adoption_record.json', {
        'schema': 'H2_FINITE_SAMPLE_RESULT_ADOPTION_V1', 'date': '2026-10-19',
        'authority': 'USER_EXPLICIT_CURRENT_CHAT_INSTRUCTION',
        'instruction_quote': 'H2の独立確認Pの2条件を、有限標本の結果として正式採用します。',
        'status': 'ADOPTED_FINITE_SAMPLE_POINTS_ONLY', 'result_commit': BASE, 'P_SOURCE': p['SOURCE'],
        'points': additional, 'main_manuscript_inclusion': 'PENDING_SEPARATE_USER_DECISION',
        'v1_scope': 'H2_ADDITIONAL_SERIES_OUTSIDE_V1',
        'protocol_amendment_adopted_by_this_task': False, 'population_mean_guaranteed': False,
        'rare_event_cost_variance': None, 'full_input_procedure_equivalence': 'NOT_VERIFIED',
        'new_scientific_execution_authorized': False, 'final_evidence_freeze': False,
        'detailed_scientific_review': False, 'mandatory_STOP': True})
    (OUT / 'H2468_comparison_separate.md').write_text(
        '# H4/H6/H8主表6条件とH2追加2条件の照合\n\n'
        '文書上の並列表。既存主表の置換・cohort統合・系サイズ効果の解析ではない。'
        '全行linear 1 Å/STO-3G、T=.8/P=0、通常signed/倍率1、tol-only1e-8、'
        'full Hadamard wrapperのcompiled RZ×保存analytic shots。exact費用・SEは保存値の転記。\n\n'
        '|区分・系/DF|ε|PR / PF|L_D / δ|n|保存G_PR|paired SE|保存G_PF|保存方向判定|低い点費用|\n'
        '|---|---|---|---|---|---|---|---|---|---|\n' + ''.join(
            '|{} {} DF{}|{}|{} / {}|{} / {}|{}|{}|{}|{}|{}|{}|\n'.format(
                '既存主表' if x['system'] != 'H2' else '追加系列・本文未採否',
                x['system'], x['DF_rank'], x['epsilon'], x['PR_id'], x['PF_id'], x['L_D'], x['delta'],
                x['n'], x['G_PR_exact'], x['paired_SE'], x['G_PF_exact'], x['saved_direction_status'],
                x['lower_saved_point_cost']) for x in main + additional) +
        '\nH4粗精度のNOT_REPRODUCEDとH6粗精度のPF低費用を保持する。REPRODUCEDはPR低費用の同義語ではない。'
        'H8の状態はREPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTSのまま。'
        'H8主表2条件はFigure2B/Mechanism H8_CONFIRM_1/3と同じ観測で、追加cohortではない。\n\n'
        'H2は旧v1対象外。Fはdense eigh/sector4で、H4 dense sector36、H6/H8の記録上eigsh/sector400/4900と'
        '生成・正規化・残差gateの完全同一性を認定しない。H4 legacy DF12と新DF10も分離する。'
        '各行のSOURCE・結果・公開監査状態はCSVとstage_evidence.jsonおよび固定H468 packetに結合する。\n\n'
        '有限標本点、保存SE、経験的bias allowanceに限定する。母平均優位、実power、希少層の費用/分散、'
        '厳密bias、全域最適、FT T資源、実量子shots、全energy/RPE費用は保証しない。'
        '新しい比・削減率・CI・統計量は作成していない。H2の元Pに比/削減率は保存されていないため空欄を保持する。\n')
    write_json('protocol_v2_H2_amendment_DRAFT.json', {
        'status': 'DRAFT_H2_STATUS_AMENDMENT_NOT_ADOPTED_BY_THIS_TASK',
        'original_draft': reference(V2_DRAFT, 'docs/research/track_a_pr_resource_evaluation_protocol_v2_DRAFT.md'),
        'separately_adopted_protocol': reference(V2_ADOPTED, 'docs/research/track_a_pr_resource_evaluation_protocol_v2.md'),
        'v1': reference(V1, 'docs/research/track_a_pr_resource_evaluation_protocol_v1.md'),
        'historical_snapshot': '829d697ae7905b3ea90a61da9d2d4da15d8b757a',
        'current_H2_snapshot': BASE, 'new_doc': AMENDMENT,
        'F_generation_policy_copied': f['generation_policy'], 'input_binding_copied': p['binding'],
        'stages': [{'stage': x['stage'], 'SOURCE': x['SOURCE'], 'completion_commit': x['completion_commit'],
                    'status': x['status']} for x in evidence],
        'P_fixed_n': p['n_fixed'], 'P_calls_reserved_copied': p['calls_reserved'],
        'P_seed_plan_digest_copied': p['seed_plan_digest'],
        'P_spec_digest_copied': p['spec_digest'], 'P_results_adopted': True,
        'P_seal_commit': '6eacd7f82afb1c6d3fa6b66987df8dcfa1ed8bcb',
        'old_v2_pending_fields': 'Historical snapshot only; do not retroactively overwrite old nulls or approval status.',
        'paper_scope': 'H468_MAIN6_UNCHANGED_H2_ADDITIONAL2_MAIN_TEXT_PENDING',
        'new_scientific_calls': 0, 'new_statistical_analysis': False,
        'v1_or_existing_adopted_v2_changed': False, 'formal_amendment_adoption': False})
    cem_path = PAPER_DIR + 'claim_evidence_matrix.csv'
    data_path = PAPER_DIR + 'paper_data_manifest.csv'
    ledger_path = PAPER_DIR + 'evidence_ledger_v3.csv'
    cem_base, data_base, ledger_base = [rows(blob(PAPER, x)) for x in [cem_path, data_path, ledger_path]]
    cem, data_add, ledger_add = [], [], []
    for index, (row, c) in enumerate(zip(additional, saved)):
        pointer = row['source_record']
        conditions = ('linear_1Angstrom;STO-3G;tol-only DF3;4qubits;Nalpha=Nbeta=1;sector4;'
            'T=.8;P=0;L_D=1;delta=' + row['delta'] + ';n=' + row['n'] + ';' + row['PR_id'] + ';PF=' + row['PF_id'])
        cem.append(dict(zip(cem_base[0], [
            'CE_' + row['comparison_id'], 'H2 ε=' + row['epsilon'] + 'の独立固定標本ではPR低費用、REPRODUCED。',
            BASE, ppath, p['SOURCE'], conditions,
            'G_PR=' + c['G_exact'] + ';paired_SE=' + c['paired_SE'] + ';G_PF=' + c['PF']['G_RZ_exact'],
            'PAIRED_MEASURED_RZ_AND_DIRECT_PF;ANALYTIC_SHOTS', p['scope'], LIMITS,
            'ADOPTED_FINITE_POINT_ONLY;PUBLIC_COMPLETION_RECEIPTS_VERIFIED;MAIN_TEXT_PENDING',
            PUB + 'P_completed_run01/native_completion_audit.json'])))
        for role, id_key, value_key, field in [
            ('CONFIRMATION_PR', 'PR_id', 'G_PR_exact', '/G_exact'),
            ('FIXED_PF', 'PF_id', 'G_PF_exact', '/PF/G_RZ_exact')]:
            item = dict.fromkeys(data_base[0], 'null')
            item.update(record_id=row['comparison_id'] + '_' + role, record_type='H2_ADDITIONAL_FIXED_CONFIRMATION',
                panel='H2_ADDITIONAL_SERIES_MAIN_TEXT_PENDING', system='H2', geometry='linear_1Angstrom',
                basis='STO-3G', DF_policy='tol_only_1e-8', DF_rank='3', T='0.8', P='0', epsilon=row['epsilon'],
                bias_mode='USUAL_SIGNED', allowance_factor='1', alpha_total='0.05', alpha_axis='0.025',
                deterministic_comparison_scope='D70_REGISTERED_BEST_FIXED_BEFORE_P',
                PR_cohort=p['scope'] if role == 'CONFIRMATION_PR' else 'SAVED_DETERMINISTIC_FIXED_CONTROL',
                sample_size=row['n'] if role == 'CONFIRMATION_PR' else 'NOT_APPLICABLE',
                sample_role='INDEPENDENT_FIXED_CANDIDATE_CONFIRMATION' if role == 'CONFIRMATION_PR' else 'FIXED_DETERMINISTIC_CONTROL',
                role=role, candidate_id=row[id_key], p='1' if role == 'CONFIRMATION_PR' else '3',
                q=['1', '4'][index], r='2' if role == 'CONFIRMATION_PR' else 'null',
                K='4' if role == 'CONFIRMATION_PR' else 'null', outer_PF_order='2',
                saved_cost_exact=row[value_key], saved_SE=row['paired_SE'] if role == 'CONFIRMATION_PR' else 'null',
                cost_class='DIRECT_MEASURED', saved_cost_status=row['saved_direction_status'],
                eligibility='FIXED_CONTRACT_NOT_REEVALUATED', missing='REDUCTION_RATIO_NOT_SAVED',
                unresolved='POPULATION_RARE_SUPPORT_INPUT_RULE_EQUIVALENCE_MAIN_TEXT_DECISION',
                full_domain_state='D70_FIXED_CONTROL_P_TWO_CELLS_ONLY', registered_domain='H2_A66_D70',
                boundary_context='FIXED_CONFIRMATION_NOT_NEW_SEARCH;BOUNDARY_NOT_RECLASSIFIED',
                result_commit=BASE, source_file=ppath, source_record=pointer, source_field=pointer + field,
                source_file_sha256=sha(blob(BASE, ppath)), metadata_source_commit=BASE,
                metadata_source_file=ppath, metadata_source_record=pointer, science_SOURCE=p['SOURCE'],
                acquisition_SOURCE=p['SOURCE'] if role == 'CONFIRMATION_PR' else evidence[2]['SOURCE'],
                compiler='Qiskit 1.3.0;rz/sx/x/cx;optimization_level=1;seed_transpiler=17;backend/coupling=None',
                compiler_settings_hash=p['binding']['compiler'],
                profile_evidence_kind='PUBLIC_H2_P_BINDING_AND_D70_RESULT;NO_CROSS_SYSTEM_IDENTITY_ASSERTION',
                figure_table_ids='H2_ADDITIONAL2;H468_MAIN6_AND_FIGURE2_UNCHANGED', interpretation_limits=LIMITS)
            data_add.append(item)
        for metric, literal, suffix in [('G_PR', row['G_PR_exact'], '/G_exact'),
                                        ('paired_SE', row['paired_SE'], '/paired_SE'),
                                        ('G_PF', row['G_PF_exact'], '/PF/G_RZ_exact'),
                                        ('n', row['n'], '/n'),
                                        ('direction_status', row['saved_direction_status'], '/status')]:
            item = dict.fromkeys(ledger_base[0], '')
            item.update(evidence_row_id='H2_ADOPTED_' + str(index + 1) + '_' + metric,
                study_id=row['comparison_id'], metric=metric, value_literal=literal,
                unit='trajectory' if metric == 'n' else 'compiled_RZ_x_analytic_shots' if metric.startswith('G_') or metric == 'paired_SE' else 'saved_status',
                comparison_scope=conditions, baseline_denominator='SAVED_D70_FIXED_PF',
                cost_evidence='DIRECT_MEASURED_AXES_WITH_ANALYTIC_SHOTS', scientific_guarantee=LIMITS,
                evidence_id='CE_' + row['comparison_id'], source_line='JSON_POINTER', source_excerpt='COPIED_SAVED_LITERAL',
                source_field=pointer + suffix, SOURCE_commit=p['SOURCE'], result_commit=BASE,
                evidence_path=ppath, evidence_file_sha256=sha(blob(BASE, ppath)), reported_precision='EXACT_SOURCE_TOKEN_NO_RECALCULATION')
            ledger_add.append(item)
    write_csv('paper_data_manifest_H2_additions_DRAFT.csv', data_add)
    write_csv('claim_evidence_matrix_H2_additions_DRAFT.csv', cem)
    write_csv('evidence_ledger_H2_additions_DRAFT.csv', ledger_add)
    write_json('paper_manifest_update_DRAFT.json', {
        'status': 'ADDITIVE_UPDATE_PROPOSAL_NOT_APPLIED_TO_CANONICAL_PAPER',
        'bases': {k: reference(PAPER, path) for k, path in {
            'paper_manifest': PAPER_DIR + 'paper_manifest.json', 'publication_manifest': DISPLAY + 'publication_manifest.json',
            'paper_data_manifest': data_path, 'claim_evidence_matrix': cem_path, 'evidence_ledger': ledger_path,
            'main_table': DISPLAY + 'main_table.csv', 'methods': DISPLAY + 'methods.md'}.items()},
        'existing_rows': {'paper_data_manifest': len(data_base), 'claim_evidence_matrix': len(cem_base), 'evidence_ledger': len(ledger_base)},
        'proposed_additions': {'paper_data_manifest': 'paper_data_manifest_H2_additions_DRAFT.csv',
            'claim_evidence_matrix': 'claim_evidence_matrix_H2_additions_DRAFT.csv',
            'evidence_ledger': 'evidence_ledger_H2_additions_DRAFT.csv'},
        'addition_counts': {'paper_data_manifest': len(data_add), 'claim_evidence_matrix': len(cem), 'evidence_ledger': len(ledger_add)},
        'application_rule': 'Append H2 unique IDs only if adopted; preserve original rows verbatim. No replacement of historical CEM cells or main table.',
        'H2_finite_point_results_adopted': True, 'H2_main_text_inclusion': 'PENDING_SEPARATE_USER_DECISION',
        'mechanism32_or_64_axes_changed': False, 'H2_mechanism_factors_available': False,
        'Figure2_H8_aliases': ['H8_CONFIRM_1', 'H8_CONFIRM_3'], 'extra_independent_cohorts_from_aliases': False,
        'v1_scope': 'H2_OUTSIDE_V1', 'protocol_v2_H2_amendment': 'DRAFT_NOT_ADOPTED_BY_THIS_TASK',
        'H8_audit_status': 'REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS',
        'new_ratios_SE_CI_shots_rankings_or_statistics': False, 'final_evidence_freeze': False,
        'detailed_scientific_review': False})
    write_json('publication_scope.json', {
        'authority': 'USER_CURRENT_TASK: dedicated branch commit and push approved public scope only',
        'allowed': ['New H2 document-only adoption, literal public-result comparison, DRAFT amendment and additive paper/CEM/ledger proposals',
            'Public GitHub metadata receipts and own structural validation', 'Six H2-owned index notices and additive validation_manifest entry'],
        'excluded': ['Private input NPZ, seed/event lists, valid grant bodies, raw worker data',
            'Other worktrees/branches, science SOURCE or results, canonical H468 table/figure, v1 and existing adopted v2'],
        'scope_does_not_adopt_drafts_or_authorize_science': True})


def verify():
    evidence = stage_evidence()
    comparison = rows((OUT / 'H2468_comparison_separate.csv').read_bytes())
    main = rows(blob(PAPER, DISPLAY + 'main_table.csv'))
    assert comparison[:6] == main and len(comparison) == 8
    assert (OUT / 'H468_main_table_UNCHANGED.csv').read_bytes() == blob(PAPER, DISPLAY + 'main_table.csv')
    assert (OUT / 'H468_methods_UNCHANGED.md').read_bytes() == blob(PAPER, DISPLAY + 'methods.md')
    assert comparison[0]['saved_direction_status'] == 'NOT_REPRODUCED'
    assert comparison[2]['lower_saved_point_cost'] == 'PF'
    assert all(x['audit_status'] == 'REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS' for x in comparison[4:6])
    result = parsed(BASE, PUB + 'P_completed_run01/confirmation_result.json', True)
    additions = rows((OUT / 'H2_adopted_points.csv').read_bytes())
    assert comparison[6:] == additions
    for row, saved in zip(additions, result['accounting']['comparisons']):
        for target, value in [('G_PR_exact', saved['G_exact']), ('paired_SE', saved['paired_SE']),
                              ('G_PF_exact', saved['PF']['G_RZ_exact']), ('n', str(saved['n'])),
                              ('saved_direction_status', saved['status'])]:
            assert row[target] == value
        assert row['saved_reduction_percent_exact'] == '' and row['mechanism_core_alias'] == ''
    proposal = read_json('paper_manifest_update_DRAFT.json')
    for name, file in proposal['proposed_additions'].items():
        added = rows((OUT / file).read_bytes())
        old = rows(blob(PAPER, proposal['bases'][name]['path']))
        assert len(added) == proposal['addition_counts'][name]
        assert list(added[0]) == list(old[0])
        id_key = list(added[0])[0]
        ids = [x[id_key] for x in added]
        assert len(set(ids)) == len(ids) and not set(ids).intersection(x[id_key] for x in old)
        assert proposal['bases'][name]['sha256'] == sha(blob(PAPER, proposal['bases'][name]['path']))
        if name == 'paper_data_manifest':
            for row in added:
                assert row['saved_cost_exact'] == pointer(result, row['source_field'])
                assert row['source_file_sha256'] == sha(blob(BASE, row['source_file']))
                if row['role'] == 'CONFIRMATION_PR':
                    assert row['saved_SE'] == pointer(result, row['source_record'])['paired_SE']
        if name == 'evidence_ledger':
            for row in added:
                assert row['value_literal'] == str(pointer(result, row['source_field']))
                assert row['evidence_file_sha256'] == sha(blob(BASE, row['evidence_path']))
    baseline = prepare_baseline()
    for item in baseline['private']:
        assert sha((ROOT / item['path']).read_bytes()) == item['sha256'], item['path']
    current_private = {str(p.relative_to(ROOT)) for p in (ROOT / '.h2_pr_resource_private/2026-10-19').rglob('*') if p.is_file()}
    assert current_private == {x['path'] for x in baseline['private']}
    changed = git('diff', '--name-only', '-z', BASE).decode().rstrip('\0').split('\0')
    allowed_existing = set(INDEXES + ['artifacts/validation_manifest.json'])
    base_paths = set(git('ls-tree', '-r', '--name-only', '-z', BASE).decode().rstrip('\0').split('\0'))
    assert set(changed).intersection(base_paths) <= allowed_existing, changed
    allowed_new = {DOC, AMENDMENT, NOTE, 'scripts/integrate_h2_research_documents.py'}
    for path in set(changed) - base_paths:
        assert path in allowed_new or path.startswith(str(OUT.relative_to(ROOT)) + '/'), path
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().rstrip('\0').split('\0')
    for path in filter(None, untracked):
        assert path in allowed_new or path.startswith(str(OUT.relative_to(ROOT)) + '/'), path
    for path in INDEXES:
        assert (ROOT / path).read_bytes().endswith(blob(BASE, path)), path
    manifest_old = parsed(BASE, 'artifacts/validation_manifest.json')
    manifest_now = json.loads((ROOT / 'artifacts/validation_manifest.json').read_text())
    assert all(manifest_now[k] == v for k, v in manifest_old.items())
    assert set(manifest_now) - set(manifest_old) == {'track_a_h2_results_integration_20261019'}
    protected = []
    for ref, prefix in [(V1, 'docs/research/track_a_pr_resource_'),
                        (V1, 'artifacts/resource_applicability/track_a_protocol_v1/'), (FIGURE, FIG_DIR)]:
        names = git('ls-tree', '-r', '--name-only', '-z', ref,
                    'docs/research' if prefix.startswith('docs/') else prefix).decode().rstrip('\0').split('\0')
        if prefix.startswith('docs/'):
            names = [x for x in names if x.startswith(prefix) and ('v1' in x)]
        for path in names:
            data = blob(ref, path)
            if (ROOT / path).exists():
                assert (ROOT / path).read_bytes() == data, path
            protected.append(reference(ref, path))
    assert len([x for x in protected if x['commit'] == FIGURE]) == 34
    for path in [DOC, AMENDMENT, NOTE, *INDEXES]:
        assert (ROOT / path).is_file()
        text = (ROOT / path).read_text()
        if path in INDEXES:
            text = text[:len(text) - len(blob(BASE, path).decode())]
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https://', 'http://', '#')):
                continue
            assert ((ROOT / path).parent / target.split('#')[0]).exists(), (path, target)
    for path in OUT.glob('*.json'):
        json.loads(path.read_text())
    assert parsed(BASE, PUB + 'P_completed_run01/confirmation_result.json')['n_fixed'] == read_json('protocol_v2_H2_amendment_DRAFT.json')['P_fixed_n']
    assert read_json('adoption_record.json')['points'] == additions
    assert read_json('protocol_v2_H2_amendment_DRAFT.json')['formal_amendment_adoption'] is False
    report = {'status': 'PASS_DOCUMENT_STRUCTURE_AND_PRESERVATION',
        'scientific_reexecution_or_analysis': False, 'independent_scientific_review': False,
        'checks': {'GitHub_commits': len(read_json('github_read_receipt.json')['commits']),
            'GitHub_public_stage_files': sum(x['public_files_verified'] for x in evidence),
            'source_hash_entries': sum(len(x['source_files_verified']) for x in evidence),
            'stages_with_exit0_wait4_reaped': len(evidence), 'existing_main_rows_unchanged': 6,
            'H2_adopted_additional_rows': 2, 'original_figure_files_unchanged': 34,
            'private_files_unchanged': len(baseline['private']), 'existing_manifest_entries_unchanged': len(manifest_old),
            'index_historical_bytes_retained': len(INDEXES), 'new_scientific_calls': 0,
            'new_statistical_analysis': 0, 'new_sampling_compile': 0,
            'new_grants_seeds_or_private_outputs': 0},
        'D_run01_preserved_not_reclassified': True, 'H8_audit_gap_retained': True,
        'population_mean_or_rare_cost_guarantee': False, 'H2_main_text_decision_pending': True,
        'protocol_amendment_formally_adopted_by_this_task': False,
        'final_evidence_freeze_or_detailed_scientific_review': False, 'mandatory_STOP': True}
    write_json('preservation.json', {'status': 'PASS', 'base': BASE,
        'existing_tracked_changes_limited_to_document_indexes_and_additive_manifest': [x for x in changed if x in base_paths],
        'private_file_count': len(baseline['private']), 'private_snapshot_digest': sha(json.dumps(baseline['private'], sort_keys=True).encode()),
        'private_paths_or_payloads_published': False, 'fixed_v1_and_figure': protected,
        'H468_main_table': reference(PAPER, DISPLAY + 'main_table.csv'),
        'adopted_v2_reference_only': reference(V2_ADOPTED, 'docs/research/track_a_pr_resource_evaluation_protocol_v2.md')})
    write_json('structural_validation.json', report)
    write_json('MANDATORY_STOP.json', {'status': 'DOCUMENT_INTEGRATION_COMPLETE_MANDATORY_STOP',
        'new_scientific_calls': 0, 'new_statistics_sampling_compile': 0,
        'next_scientific_stage_authorized': False, 'protocol_amendment_adopted': False,
        'final_evidence_freeze': False, 'detailed_scientific_review': False})
    paths = sorted(x for x in OUT.rglob('*') if x.is_file() and x.name != 'artifact_manifest.json')
    write_json('artifact_manifest.json', {'files': [{'path': str(x.relative_to(OUT)), 'sha256': sha(x.read_bytes()), 'bytes': x.stat().st_size} for x in paths],
        'external_documents': [{'path': x, 'sha256': sha((ROOT / x).read_bytes())} for x in [DOC, AMENDMENT, NOTE, *INDEXES, 'artifacts/validation_manifest.json', 'scripts/integrate_h2_research_documents.py']]})
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--build', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    build() if args.build else verify()
