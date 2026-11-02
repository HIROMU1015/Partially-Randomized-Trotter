"""Public metadata/structure checks only; never imports a numerical module."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import h46_m5_cost_contract as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report')
    args = parser.parse_args()
    root = c.ROOT
    bundle = root / 'artifacts/resource_applicability/track_a_h46_m5_cost_sampling_readiness/2026-10-20'
    checks = []

    def check(name, condition):
        c.require(condition, name)
        checks.append(dict(check=name, PASS=True))

    def rows(name):
        with (bundle / name).open(newline='') as stream:
            return list(csv.DictReader(stream))

    manifest = c.manifest()
    summary = c.read(bundle / 'summary.json')
    source = summary['SOURCE']
    domain = {r['candidate']['id']: r for r in manifest['candidate_table']}
    table = rows('candidate_cost_targets_88.csv')
    check('EXACT88_IDS_NO_EXTRA_OR_DUPLICATE', len(table) == 88 and {r['id'] for r in table} == set(domain))
    for row in table:
        item = domain[row['id']]
        check('DOMAIN_ROW_' + row['id'], row['proposal'] == item['proposal'] and all(
            row[k] == str(item['candidate'][k]) for k in ('system', 'method', 'formula', 'prefix', 'q')))
    check('DET92_NEW_AXIS_BUDGET_NOT_COUNTING_INACTIVE_SAVED8', sum(int(r['additional_compile_axes_if_adopted']) for r in table if r['method'] != 'PR') == 92)
    check('PR24_N_REMAINS_NULL', len([r for r in table if r['method'] == 'PR']) == 24 and all(r['n_E'] == '' for r in table if r['method'] == 'PR'))
    endpoints = rows('candidate_endpoint_join_440.csv')
    check('ENDPOINT440_BIJECTION', len(endpoints) == 440 and len({(r['id'], r['epsilon']) for r in endpoints}) == 440)
    check('SAVED_AXIS_LABELS_EXACT_NOT_RECOMPUTED', all(
        r['real_status'] == domain[r['id']]['saved_axis_status'][r['epsilon']]['real'] and
        r['imag_status'] == domain[r['id']]['saved_axis_status'][r['epsilon']]['imag'] and
        r['both_saved_ELIGIBLE'] == str(float(r['epsilon']) in domain[r['id']]['eligible_epsilons']) for r in endpoints))
    saved = rows('reuse_cost_axes_12.csv')
    expected = {(r['candidate_id'], r['axis']): r for r in manifest['reuse_axes']}
    check('REUSE12_EXACT_IDENTITIES_AND_SAVED_INTEGERS', len(saved) == len(expected) == 12 and
          all((r['candidate_id'], r['axis']) in expected and int(r['compiled_RZ']) == expected[r['candidate_id'], r['axis']]['compiled_RZ'] and
              r['receipt_sha256'] == expected[r['candidate_id'], r['axis']]['receipt_sha256'] for r in saved))
    check('ACTIVE_SAVED_AXES_ONLY4', sum(r['eligible_in_current_five_eps'] == 'True' for r in saved) == 4)
    check('ALL24_PR_PROPOSALS_EXACT_AND_UNADOPTED', c.read(bundle / 'PR_sampling_design_UNAPPROVED.json') == manifest['sampling_proposal'])
    closure = c.read(bundle / 'source_closure.json')
    check('SOURCE_PATHS_INCLUDE_SAVED_NORMALIZATION', closure['SOURCE'] == source and {r['path'] for r in closure['files']} == set(c.SOURCE_PATHS) and c.SIGNAL_RECORDS in c.SOURCE_PATHS)
    for item in closure['files']:
        data = subprocess.check_output(['git', 'show', source + ':' + item['path']], cwd=root)
        check('SOURCE_GIT_' + item['path'], data == (root / item['path']).read_bytes() and
              len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'])
    for stage in c.STAGES:
        plan = c.read(bundle / (stage + '_contract_DRAFT.json'))
        grant = c.read(bundle / (stage + '_grant_INVALID.json'))
        check(stage + '_DRAFT_SOURCE_INVALID_UNADOPTED', plan['SOURCE'] == source == grant['SOURCE'] and
              plan['authorized'] is False and plan['valid_grant'] is False and grant['valid'] is False and
              all(v is False for k, v in plan['decisions'].items() if k != 'n_by_candidate'))
        check(stage + '_CT01_AND_ORIGINAL_NATIVE_COMPILER', plan['compile_tolerance'] == 1e-7 and
              plan['other_tolerance'] == 1e-9 and plan['compiler_settings_hash'] == c.design.COMPILER_HASH and
              plan['native_policy'] == c.design.NATIVE_POLICY and plan['AD03'] == 'DEFERRED')
        check(stage + '_NO_RETRY_OR_CROSS_APPROVAL', plan['retry'] is False and grant['retry'] is False and
              plan['next_phase_authorized'] is False and grant['next_phase_authorized'] is False)
        check(stage + '_CALL_SCOPE_BOUND', plan['call_scope'] == (c.design.call_scope(plan['tasks'], 'DET') if stage.endswith('DET') else None))
    audit = c.read(bundle / 'metadata_audit.json')
    check('METADATA84_ALL_PASS', len(audit['checks']) == 84 and all(x['PASS'] for x in audit['checks']))
    check('PROTECTED3137_UNCHANGED', c.read(bundle / 'preservation_receipt.json')['files'] == 3137 and c.read(bundle / 'preservation_receipt.json')['changed'] == [])
    for name, count in [('test_h46_m5_cost_readiness.py.txt', 18), ('test_h46_m5_signal.py.txt', 18)]:
        log = (bundle / name).read_text()
        check('NON_SKIPPED_TESTS_' + name, 'Ran ' + str(count) + ' tests' in log and log.rstrip().endswith('OK') and 'skipped' not in log)
    check('NO_SCIENTIFIC_DISPATCH_OR_AD03_ADOPTION', summary['valid_grants'] == 0 and summary['AD03'] == 'DEFERRED' and
          all(v == 0 for v in summary['new_molecular_science'].values()) and summary['PR']['compile_axes'] is None)
    base_manifest = json.loads(subprocess.check_output(['git', 'show', c.design.BASE + ':artifacts/validation_manifest.json'], cwd=root))
    current = c.read(root / 'artifacts/validation_manifest.json')
    check('ALL_HISTORICAL_MANIFEST_ENTRIES_PRESERVED', all(current[k] == v for k, v in base_manifest.items()))
    report = 'track_a_h46_m5_cost_sampling_readiness_20261020.md'
    for name in ('PROJECT_MAP.md', 'VALIDATION_STATUS.md', 'docs/README.md', 'scripts/README.md', 'src/trotterlib/README.md', 'docs/research/研究概要・現状.md'):
        check('DISCOVERABLE_' + name, report in (root / name).read_text()[:2000])
    output = dict(status='PASS_PUBLIC_STRUCTURE_ONLY', checks=checks, SOURCE=source, numerical_analysis=False, scientific_execution=False)
    if args.report:
        c.save(args.report, output)
    print(json.dumps(dict(status=output['status'], checks=len(checks), SOURCE=source)))


if __name__ == '__main__':
    main()
