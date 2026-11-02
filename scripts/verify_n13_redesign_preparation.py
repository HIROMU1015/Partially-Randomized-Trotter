#!/usr/bin/env python3
"""Read-only stdlib check of N1/N3 design records; never import or run science."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'ae7434452450e80ec007d65417335e18a063fa1e'
CONTRACT = 'docs/research/n13_redesign_limited_scope_v1_20261011.json'


def check_preparation(root: Path) -> dict:
    checks = 0

    def require(condition, description):
        nonlocal checks
        if not condition:
            raise ValueError(description)
        checks += 1

    def digest(path):
        return hashlib.sha256((root / path).read_bytes()).hexdigest()

    c = json.loads((root / CONTRACT).read_text())
    require(c['schema_version'] == 1, 'scope schema')
    require(c['base_commit'] == BASE, 'base identity')
    require(c['status'] == 'DESIGN_PREPARED_SCIENCE_NOT_AUTHORIZED', 'STOP status')
    require(c['science_execution_authorized'] is False, 'science not authorized')
    require(c['next_scientific_stage_authorized'] is False, 'next science not authorized')
    require(c['mandatory_stop'] is True, 'mandatory STOP')
    require(c['central_hypothesis_adopted'] is None, 'theme not adopted')
    require(c['prior_gpt_scientific_review_complete'] is True and
            c['repeat_prior_review'] is False, 'do not re-review old science')
    require(all(v == 0 for v in c['counts_this_preparation'].values()), 'zero science counters')
    require(digest(c['review_input']) == c['review_sha256'], 'provided review byte identity')
    n1 = c['N1']
    require(digest(n1['domain_source']) == n1['domain_source_sha256'], 'old domain identity')
    saved = json.loads((root / n1['domain_source']).read_text())
    metadata = {row['context']: [len(xs) for xs in row['generator_candidates']]
                for row in saved['contexts'] if row['track'] == 'N1'}
    require(metadata == {row['id']: row['factor_candidate_counts'] for row in n1['contexts']},
            'saved candidate counts; no coefficient evaluation')
    require(n1['runtime_selector_may_load_old_full_result'] is False, 'selector input isolation')
    require(digest(n1['selector_input']) == n1['selector_input_sha256'], 'candidate-only input hash')
    isolated = json.loads((root / n1['selector_input']).read_text())
    expected = []
    keys = ('family', 'eta', 'approx_eta', 'frame', 'ungauged_frame', 'ungauged_eta', 'g', 'groups')
    for row in saved['contexts']:
        if row['track'] != 'N1':
            continue
        copied = {k: row[k] for k in ('context', 'modes', 'particles', 'df_rank', 'weights',
                  'one_body', 'scalar', 'input_factors', 'time', 'model_budget', 'q_grid')}
        copied['generator_candidates'] = [[dict(
            {'id': f"{row['context']}/factor{i}/candidate{j}"},
            **{k: candidate[k] for k in keys}) for j, candidate in enumerate(xs)]
            for i, xs in enumerate(row['generator_candidates'])]
        expected.append(copied)
    require(isolated['contexts'] == expected and isolated['scientific_evaluation_performed'] is False,
            'only whitelisted coefficient/candidate fields copied')
    require(sum(sum(row['factor_candidate_counts']) for row in n1['contexts']) ==
            n1['factor_candidates'] == 44, 'factor count cap')
    pairs = sum(row['factor_candidate_counts'][0] * row['factor_candidate_counts'][1]
                for row in n1['contexts'])
    require(pairs == n1['pair_combinations'] == 178, 'finite pair count')
    require(n1['q_grid'] == [1, 2, 4, 8] and n1['axes'] == ['X', 'Y'], 'old fixed q/axes')
    require(n1['new_centers_or_angles'] == n1['new_q_or_precision_points'] == 0,
            'no expanded N1 scan')
    n3 = c['N3']
    require(len(n3['inputs']) == n3['input_count'] == 4, 'four N3 inputs')
    require(n3['comparators_count'] * n3['input_count'] * n3['active_stages_per_method_max'] ==
            n3['certificate_records_max'] == 36, 'N3 record cap')
    require(len(n3['lambda_grid_fractions']) == n3['lambda_shifts_max'] == 5, 'fixed lambda grid')
    require(all(n3[x] is False for x in ('uses_true_active_ground', 'uses_true_gap', 'forces_five_modes')),
            'N3 has no truth/target-result selector')
    n2 = c['N2']
    require(n2['fixture_count'] == 1 and n2['off_diagonal_weight'] == 2 and
            n2['diagonal_weight'] == 1, 'fixed-S weighted L1 design')
    require(n2['large_S_searches'] == n2['legacy_820_candidate_lp_rescans'] ==
            n2['new_circuit_compilations'] == 0, 'N2 no scan')
    r = c['resources']
    require(r['wall_seconds'] is None and r['cpu_seconds'] is None, 'no scientific time caps')
    require(r['workers_max'] == 2 and r['threads_per_worker'] == 1 and
            r['transpile_num_processes'] == 1 and r['qubits_initially_zero'] is False,
            'bounded resource and arbitrary-system compiler settings')
    wrappers = 2 * pairs * len(n1['q_grid']) * len(n1['axes']) + 24 + 48
    require(wrappers <= r['wrapper_compiles_max'] == 3072, 'wrapper call cap')
    require(r['wrapper_compiles_max'] + r['additional_basis_compiles_max'] ==
            r['total_compiles_max'] == 3116, 'total compile cap')
    require(c['implementation']['status'] == 'not_implemented', 'design-only state')
    require(all(c['implementation'][x] is None for x in
                ('source_commit', 'source_bound_plan', 'execution_authorization')), 'no implicit launch')
    require(all(not (root / c['implementation'][x]).exists() for x in
                ('module', 'runner', 'test', 'saved_verifier')), 'future science paths absent')
    previous = json.loads(subprocess.check_output(
        ['git', 'show', BASE + ':artifacts/validation_manifest.json'], cwd=root))
    current = json.loads((root / 'artifacts/validation_manifest.json').read_text())
    count = len(previous['result_sets'])
    require(count == 86 and current['result_sets'][:count] == previous['result_sets'],
            'all 86 old result entries preserved')
    require(len(current['result_sets']) == count + 1 and
            current['result_sets'][-1]['id'] == c['id'], 'one new preparation entry')
    for key, value in previous.items():
        if key != 'result_sets':
            require(current[key] == value, 'old manifest field preserved: ' + key)
    protected = ['artifacts/hamiltonian_algorithm_design',
                 'src/trottertracks/representation_exploration/algorithm_design.py',
                 'scripts/run_hamiltonian_algorithm_design.py',
                 'scripts/finalize_hamiltonian_algorithm_design.py',
                 'scripts/verify_hamiltonian_algorithm_design.py',
                 'scripts/audit_hamiltonian_algorithm_design_construction.py',
                 'tests/test_hamiltonian_algorithm_design.py',
                 'docs/research/hamiltonian_algorithm_design_scope.md',
                 'docs/research/hamiltonian_algorithm_design_results_20261011.md']
    changed = subprocess.check_output(['git', 'diff', BASE, '--name-only', '--', *protected],
                                      cwd=root, text=True)
    require(not changed.strip(), 'old frozen science tree untouched')
    return {'status': 'PREPARATION_METADATA_PASS', 'checks': checks,
            'science_execution_authorized': False, 'scientific_calculations_run': 0,
            'old_manifest_entries_preserved': count, 'new_scientific_tests_run': 0,
            'scope_sha256': digest(CONTRACT), 'review_sha256': c['review_sha256'],
            'claim_scope': 'Metadata, identities, limits and preservation only; no mathematical or scientific result validation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('output already exists; no overwrite')
    result = check_preparation(ROOT)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
