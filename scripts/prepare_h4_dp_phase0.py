"""Bounded metadata extraction. No science imports, cost arithmetic or statistics.

Read saved decisions/values, compare IDs and bytes, and prepare INVALID contracts.
Private origins stay in an ignored directory; public projections contain no seeds.
"""
import argparse
import csv
import gzip
import io
from pathlib import Path
import subprocess

import h4_dp_contract as c

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19'
PRIVATE = ROOT / '.h4_dp_phase0_private'
NATIVE = Path('/home/AbeHiromu/projects/h4-native-validation/20261018/private')
BATCH = Path('/home/AbeHiromu/projects/track-a-pr-cost-batch-seal/20261018/private/sealed03')
REVS = dict(native='2588bb1ff8a03120050fdaa67b9cd30739131bec',
            followup='97d25cc431516d720ce91c7cb4f148e061d8ea34',
            geometry='98cdb35cd08469714f20d6cc3b982d9c80908f46',
            total='64c85f7a833b5950ef2ccae623708b73081edcd6',
            audit='bf8c19057fd82c900c6234152e152e79cb52e707',
            h8='e6eea5387a6e84d1e39a9adb3b61975acc74e200',
            h6='321cd8c3066194fe98741cbebbcdba1515d1deaa')
ART = 'artifacts/resource_applicability/'
NATIVE_ART = ART + 'track_a_h4_native_validation/2026-10-18/'
TOTAL_ART = ART + 'track_a_pr_new_a_total_cost_results/2026-10-18/'
SOURCE_ART = ART + 'track_a_final_technical_audit/2026-10-19/'


class Store:
    def __init__(self):
        self.public, self.private = [], []

    def blob(self, revision, path):
        raw = subprocess.check_output(['git', '--no-optional-locks', '-c', 'gc.auto=0',
                                       'show', revision + ':' + path], cwd=ROOT)
        target = PRIVATE / 'git' / revision / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        self.public.append(dict(role='git_blob', commit=revision, path=path,
                                sha256=c.sha(target), bytes=len(raw)))
        return target

    def file(self, role, path, expected=None):
        path = Path(path)
        h = c.sha(path)
        c.require(expected is None or h == expected, 'ORIGINAL_BYTES:' + role)
        target = PRIVATE / 'store' / h
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
        c.require(c.sha(target) == h, 'PRIVATE_SNAPSHOT')
        row = dict(role=role, path=str(path), copy=str(target), sha256=h, bytes=path.stat().st_size)
        self.private.append(row)
        self.public.append({k: v for k, v in row.items() if k not in ('path', 'copy')})
        return target


def load(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return __import__('json').loads(raw)


def table(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stream.getvalue())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-preservation', action='store_true')
    args = parser.parse_args()
    if args.verify_preservation:
        rows = c.read(PRIVATE / 'preservation_baseline.json')
        for row in rows:
            c.require(c.sha(row['path']) == row['sha256'] and c.sha(row['copy']) == row['sha256'],
                      'PRESERVATION_CHANGED:' + row['role'])
        print('PRESERVATION_PASS', len(rows))
        return
    c.require(not (PRIVATE / 'preservation_baseline.json').exists(), 'DO_NOT_RESET_BASELINE')
    PUBLIC.mkdir(parents=True, exist_ok=True)
    store = Store()
    docs = [('native', 'track_a_h4_native_validation_20261018.md'),
            ('followup', 'track_a_h4_followup_20261018.md'),
            ('geometry', 'track_a_h4_b0i_map_results_20261014.md'),
            ('total', 'track_a_pr_new_a_total_cost_results_20261018.md'),
            ('audit', 'track_a_final_technical_audit_20261019.md'),
            ('h8', 'track_a_h8_new_pr_confirmation_results_20261019.md'),
            ('h8', 'track_a_h8_new_pr_confirmation_design_20261018.md'),
            ('h8', 'track_a_h8_phase_c_run01_results_20261017.md'),
            ('h6', 'track_a_h6_independent_confirmation_result_v1.md')]
    for key, name in docs:
        store.blob(REVS[key], 'docs/research/' + name)
    for name in ['B_domain/report.md', 'B_domain/registered_candidate_domains.csv',
                 'B_domain/winner_boundary_flags.csv', 'A_rare_event/report.md',
                 'E_claim_evidence/claim_evidence_matrix.md']:
        store.blob(REVS['audit'], SOURCE_ART + name)
    # Exact blobs at each requested commit are acquired, including evidence status.
    for key in REVS:
        store.blob(REVS[key], 'AGENTS.md')
    for name in ['PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'artifacts/validation_manifest.json']:
        store.blob(REVS['native'], name)
    plan = load(store.blob(REVS['native'], NATIVE_ART + 'plan_v1.json'))
    public_binding = load(store.blob(REVS['native'], NATIVE_ART + 'input_binding.json'))
    result_d = load(store.blob(REVS['native'], NATIVE_ART + 'D_result.json.gz'))
    receipt_d = load(store.blob(REVS['native'], NATIVE_ART + 'D_completion_receipt.json'))
    result_n = load(store.blob(REVS['native'], NATIVE_ART + 'N_result.json.gz'))
    result_a = load(store.blob(REVS['native'], NATIVE_ART + 'A_result.json.gz'))
    for name in ['source_closure.json', 'execution_bindings.json', 'A_completion_receipt.json',
                 'N_completion_receipt.json']:
        store.blob(REVS['native'], NATIVE_ART + name)
    for name in ['input_binding.json', 'plan_v1.json', 'final_completion.json', 'T_result.json.gz',
                 'FP_result.json.gz', 'B_result.json.gz']:
        store.blob(REVS['followup'], ART + 'track_a_h4_followup/2026-10-18/' + name)
    comparisons = list(csv.DictReader(store.blob(REVS['total'], TOTAL_ART + 'comparisons.csv').open()))
    new_points = list(csv.DictReader(store.blob(REVS['total'], TOTAL_ART + 'new_pr_cost_points.csv').open()))
    for name in ['completion_summary.json', 'verification.json']:
        store.blob(REVS['total'], TOTAL_ART + name)
    for name in ['comparisons.csv', 'paired_metrics.csv']:
        store.blob(REVS['h8'], ART + 'track_a_h8_new_pr_confirmation_results/2026-10-19/' + name)
    native = load(store.file('native_config', NATIVE / 'config_SEALED_PRIVATE.json'))
    c.require(native['binding'] == public_binding['binding'], 'PUBLIC_INPUT_BINDING')
    for row in native['inputs']:
        store.file('native_input:' + row['role'], row['path'], row['sha256'])
    profile = native['profile_reference']
    profile_copy = store.blob(profile['commit'], profile['path'])
    c.require(c.sha(profile_copy) == profile['sha256'], 'COMPILER_PROFILE_BYTES')
    for name, h in native['source_hashes'].items():
        blob = store.blob(receipt_d['SOURCE'], name)
        c.require(c.sha(blob) == h and c.sha(ROOT / name) == h, 'ORIGINAL_NATIVE_SOURCE')
    # Only the known provider closure is checked; no project/tmp/history-wide scan.
    for name, h in native['provider_hashes'].items():
        c.require(c.sha(Path(native['provider_root']) / name) == h, 'ORIGINAL_PROVIDER_SOURCE')
    for stage in ['D', 'A', 'N']:
        for name in ['grant.json', 'one_shot_lock.json', 'native_started.json', 'native_exit.json']:
            store.file('old_' + stage + ':' + name, NATIVE / (stage + '_run01') / name)
        for name in ['result.json', 'terminal.json', 'final_gate.json']:
            store.file('old_' + stage + ':' + name, NATIVE / (stage + '_run01/output') / name)
    store.file('D_original_result_bound_to_public_receipt', NATIVE / 'D_run01/output/result.json',
               receipt_d['uncompressed_sha256'])
    signal_path = store.file('D96_saved_signals', NATIVE / 'D_run01/output/signals.json')
    records = load(signal_path)['signals']
    c.require({r['cell']['id'] for r in records} == {x['id'] for x in plan['D']['cells']}
              and len(records) == 96, 'EXACT_REGISTERED_D96')
    old_signals = load(native['mapping']['old_signals'])['signals']
    full = [r for r in result_n['signals'] if r['cell']['method'] == 'B1']
    c.require(len(full) == 8 and {r['cell']['id'] for r in full} <= {r['cell']['id'] for r in old_signals},
              'FULL8_REGISTRATION')
    tasks = c.deterministic_tasks(records, result_d['selected'])
    measured = {x['cell_id']: {r['axis']: r for r in x['rows']} for x in result_d['measurements']}
    with Path(native['mapping']['old_metrics']).open() as stream:
        old_metrics = list(csv.DictReader(stream))
    for row in old_metrics:
        if row['control'] == 'symmetric_directional' and row['cell_id'] in {r['cell']['id'] for r in full}:
            c.require(row['compiler_hash'] == native['binding']['compiler'],
                      'SAVED_FULL_PF_COMPILER_CONTROL')
            measured.setdefault(row['cell_id'], {})[row['axis']] = row
    c.require(len(measured) == 16 and all(set(x) == set(c.AXES) for x in measured.values()), '16_DIRECT_PF_PAIRS')
    endpoints, catalog, coverage = [], [], []
    for record in records + full:
        cell = record['cell']; cid = cell['id']
        catalog.append(dict(cell=cell, record_sha256=c.digest(record), signals=record['signals'],
                            reference=record['reference'], error_decomposition=record['error_decomposition'],
                            allowance=record['allowance'], log_B=record['log_B'],
                            scalar_receipt=record.get('scalar_receipt'),
                            result_origin='D96_PRIVATE_SAVED_SIGNALS' if cell['method'] != 'B1' else 'N_PUBLIC_RESULT'))
        for mode in c.FIELDS:
            for eps in c.EPS:
                a = c.saved_endpoint(record, mode, eps)
                endpoints.append(dict(cell_id=cid, bias_mode=mode, epsilon=eps, factor=1.,
                                      saved_accounting=a, saved_accounting_sha256=c.digest(a)))
        axes = measured.get(cid, {})
        coverage.append(dict(candidate_id=cid, prefix=cell['prefix'], q=cell['q'], formula=cell['formula'],
            control='symmetric_directional', status='DIRECT_MEASURED_REUSABLE' if axes else
                'UNMEASURED_INELIGIBLE_ALL_REGISTERED_ENDPOINTS' if cid in tasks['excluded'] else 'UNMEASURED_COMPILE_REQUIRED',
            cosine_RZ=axes.get('cosine', {}).get('RZ'), sine_RZ=axes.get('sine', {}).get('RZ'),
            cosine_fingerprint=axes.get('cosine', {}).get('fingerprint'),
            sine_fingerprint=axes.get('sine', {}).get('fingerprint'),
            compiler_hash=native['binding']['compiler'], signal_record_sha256=c.digest(record),
            measurement_commit=REVS['native'] if cid in result_d['selected'] else
                '0f96a2e45ba58bd916508e1b1f32e6ff1ff65f41' if axes else None,
            measurement_path=NATIVE_ART + 'D_result.json.gz' if cid in result_d['selected'] else
                'PRIVATE_ORIGINAL_NATIVE_METRICS' if axes else None,
            measurement_sha256=receipt_d['compressed_sha256'] if cid in result_d['selected'] else
                next(x['sha256'] for x in native['inputs'] if x['role'] == 'old_metrics') if axes else None,
            measurement_SOURCE=receipt_d['SOURCE'] if cid in result_d['selected'] else
                load(native['mapping']['old_summary'])['source_commit'] if axes else None,
            scientific_input_binding_digest=c.digest(native['binding'])))
    c.save(PUBLIC / 'saved_signal_catalog.json', catalog)
    c.save(PUBLIC / 'saved_endpoints.json', endpoints)
    c.save(PUBLIC / 'saved_measurements.json', {cid: {axis: {k: v for k, v in row.items()
             if k in ('RZ', 'CX', 'depth', 'RZ_depth', 'CX_depth', 'size', 'compiler_hash', 'fingerprint')}
             for axis, row in axes.items()} for cid, axes in measured.items()})
    table(PUBLIC / 'deterministic_coverage.csv', coverage)
    c.save(PUBLIC / 'D_tasks.json', tasks)
    c.save(PUBLIC / 'scientific_binding.json', dict(**public_binding['binding'],
           profile_sha256=profile['sha256'], compiler_options=native['compiler_options'],
           target='linear_H4_1Angstrom_STO3G_legacy_DF12', T=.8, P=0,
           control='symmetric_directional', qiskit='1.3.0', backend=None, coupling_map=None))
    main_comparisons = [r for r in comparisons if r['system'] == 'H4' and r['factor'] == '1.0'
                        and r['bias_mode'] == 'USUAL_SIGNED' and r['deterministic_scope'] == 'old_plus_direct_subset']
    c.require(len(main_comparisons) == 5, 'FIVE_SAVED_PRIMARY_COMPARISONS')
    assignments = []
    for row in main_comparisons:
        for key, cohort in [('old_PR_best_id', 'OLD_FIXED_ANCHOR_N32'), ('new_best_id', 'EXPLORATION_N8')]:
            assignments.append(dict(cell_id=row[key], epsilon=float(row['epsilon']), role='PRIMARY',
                                    cohort=cohort, selected_by='SAVED_WINNER_ID_NO_RERANKING'))
    for cid in ['B2_p3_q8_2nd_r2_K4', 'B2_p2_q8_2nd_r4_K4']:
        for eps in [.001, .0005]:
            assignments.append(dict(cell_id=cid, epsilon=eps, role='BOUNDARY_K_PARTNER_DIAGNOSTIC',
                cohort='OLD_FIXED_ANCHOR_N32' if '_p3_' in cid else 'EXPLORATION_N8',
                selected_by='STRUCTURAL_K_PARTNER_AT_REGISTERED_Q_R_BOUNDARY'))
    signal_by_id = {r['cell']['id']: r for r in result_n['signals'] + result_a['signals']}
    saved_stats = []
    for item in assignments:
        cid, eps = item['cell_id'], item['epsilon']
        signal = signal_by_id[cid]
        shot = c.saved_endpoint(signal, 'USUAL_SIGNED', eps)
        if item['cohort'] == 'OLD_FIXED_ANCHOR_N32':
            point = next(r for r in result_n['rows'] if r['cell_id'] == cid and r['epsilon'] == eps
                         and r['bias_mode'] == 'USUAL_SIGNED' and r['allowance_factor'] == 1.)
            mean, se, n = point.get('G_RZ_exact'), point.get('G_RZ_SE'), point.get('n')
        else:
            point = next(r for r in new_points if r['system'] == 'H4' and r['cell_id'] == cid
                         and r['epsilon'] == str(eps) and r['bias_mode'] == 'USUAL_SIGNED' and r['factor'] == '1.0')
            mean, se, n = point['G_RZ_exact'], point['G_RZ_SE'], point['n_used']
        saved_stats.append(dict(**item, saved_mean_exact=mean, saved_paired_SE=se, saved_n=n,
            saved_shots={axis: shot['axes'][channel]['shots'] for axis, channel in [('cosine', 'real'), ('sine', 'imag')]},
            saved_SD='NOT_EXTRACTED_OR_NOT_SAVED', saved_axis_covariance='NOT_EXTRACTED_OR_NOT_SAVED',
            numeric_required_n=None, n_status='PENDING_D_RESULT_AND_S_STATS_APPROVAL'))
    table(PUBLIC / 'P_saved_statistics.csv', [{**r, 'saved_shots': str(r['saved_shots'])} for r in saved_stats])
    c.save(PUBLIC / 'P_registration.json', dict(assignments=assignments,
        candidates=[signal_by_id[cid]['cell'] for cid in sorted({r['cell_id'] for r in assignments})],
        primary_family_size=10, diagnostic_family_size=4, saved_statistics=saved_stats,
        P_pending=['D_RESULT', 'S_STATS_NUMERICAL_DESIGN_APPROVAL', 'FINAL_N', 'SEED_SEAL', 'P_EXECUTION_APPROVAL']))
    c.save(PUBLIC / 'P_signal_bindings.json', {cid: dict(cell=signal_by_id[cid]['cell'],
        record_sha256=c.digest(signal_by_id[cid]), log_B=signal_by_id[cid]['log_B'],
        log_B_tolerance=signal_by_id[cid]['allowance']['log_B_margin'])
        for cid in sorted({r['cell_id'] for r in assignments})})
    # Existing batch inputs are located through its fixed, bounded metadata pointers.
    a_spec = load(store.file('old_A_cost_spec', BATCH / 'contracts/H4_A_COST/technical_seal.json'))
    a_payload = load(store.file('old_A_cost_payload', a_spec['payload']))
    store.file('old_A_cost_result', Path(a_spec['output']) / 'result.json')
    store.file('old_A_cost_terminal', Path(a_spec['output']) / 'terminal.json')
    if Path(a_spec['lock']).is_file():
        store.file('old_A_cost_one_shot', a_spec['lock'])
    c.save(PRIVATE / 'A_native_adapter.json', a_payload['native_adapter'])
    registry = a_payload['new_seed_registry']
    c.save(PRIVATE / 'existing_A_seed_registry.json', registry)
    c.save(PUBLIC / 'seed_inventory_status.json', dict(new_confirmation_seeds_generated=0,
        existing_A_registry_obtained=True, old_exclusion_inventory_obtained=True,
        final_collision_check='PENDING_FINAL_SOURCE_N_AND_COMPLETE_GEOMETRY_PILOT_INVENTORY',
        collision_zero_proves_mathematical_independence=False))
    c.save(PRIVATE / 'native_config_pointer.json', dict(path=str(NATIVE / 'config_SEALED_PRIVATE.json')))
    counts = dict(truncated_registered=96, full_registered=8, total_registered=104,
        truncated_direct_measured=len(result_d['selected']), truncated_unmeasured=88,
        full_direct_measured=8, total_direct_measured=16, eligible_truncated=len(tasks['eligible']),
        excluded_by_frozen_ineligibility=len(tasks['excluded']),
        new_compile_candidates=len(tasks['compile_ids']), new_compile_axes=2*len(tasks['compile_ids']),
        maximum_native_control_probes=4*len(tasks['compile_ids']),
        P_unique_candidates=len({r['cell_id'] for r in assignments}),
        P_primary_endpoints=10, P_diagnostic_endpoints=4,
        new_scientific_calls_executed=0, new_statistics_executed=0)
    c.save(PUBLIC / 'coverage_counts.json', counts)
    c.save(PUBLIC / 'source_inventory.json', store.public)
    c.save(PRIVATE / 'preservation_baseline.json', store.private)
    c.save(PRIVATE / 'git_blob_receipts.json', [r for r in store.public if r.get('commit')])
    print('METADATA_EXTRACTION_PASS', counts)


if __name__ == '__main__':
    main()
