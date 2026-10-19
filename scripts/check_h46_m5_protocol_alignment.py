"""Read-only validation of plan metadata; no scientific imports or dispatch."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = 'artifacts/resource_applicability/track_a_h46_m5_protocol_alignment/2026-10-19'
DOC = 'docs/research/track_a_h46_m5_protocol_alignment_20261019.md'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def validate(root, *, verify_git=True):
    root = Path(root)
    bundle = root / BUNDLE
    load = lambda p: json.loads((bundle / p).read_text())
    plan = load('aligned_plan_DRAFT.json')
    require(plan['status'] == 'PLAN_REVISION_ONLY_NOT_LAUNCH_READY'
            and plan['AD03_main_comparison_status'] == 'DEFERRED_ADDITIONAL_PF', 'AD03_STILL_HELD')
    for key in ('new_science_authorized', 'numerical_reanalysis_authorized', 'push_authorized',
                'automatic_contract_application', 'old_four_contracts_modified'):
        require(plan[key] is False, 'PLAN_ONLY_NO_AUTHORIZATION:' + key)
    require(all(plan[k] is None for k in ('scientific_SOURCE', 'final_seeds', 'final_sample_sizes')),
            'NO_SCIENTIFIC_SEAL')
    inputs = load('target_input_bindings_DRAFT.json')
    require(inputs['actual_snapshot_bytes_verified'] is False
            and inputs['input_rule_equivalence'] == 'NOT_VERIFIED', 'RECEIPT_NOT_RAW_REVALIDATION')
    require(inputs['systems']['H4']['series'] == 'H4_TOL_ONLY_DF10'
            and inputs['systems']['H4']['legacy_preserved_separately'] is True, 'H4_TOL_LEGACY_SEPARATE')
    for system, entry in inputs['systems'].items():
        require(entry['rank_policy'] == 'tol_only_1e-8'
                and entry['rank_is_observation_not_generator_argument'] is True, 'TOL_ONLY_POLICY')
        require(entry['preparation_hash'] is None and entry['raw_snapshot_path'] is None,
                'PREPARATION_NOT_YET_SEALED')
        require(all(re.fullmatch('[0-9a-f]{64}', v) for v in entry['binding'].values()), 'INPUT_HASH_FORMAT')

    eval_ = plan['evaluation']
    require(eval_['epsilon'] == ['0.05', '0.01', '0.005', '0.001', '0.0005']
            and eval_['bias'] == 'USUAL_SIGNED' and eval_['allowance_factor'] == 1,
            'COMMON_FIVE_ENDPOINTS')
    require(eval_['rank_auto'] is True and eval_['control'] == 'symmetric_directional', 'COMMON_INPUT_CONTROL')
    pilot = plan['technical_pilot']
    require(pilot['shot_accounting'] is False and pilot['ranking'] is False
            and pilot['independent_confirmation'] is False
            and pilot['allowance_adopted_for_shots'] is False, 'PILOT_NOT_CORE_RESULT')
    lanes = pilot['lanes']
    require(len({l['id'] for l in lanes}) == len(lanes) == 4
            and sum(l['compile_axes'] for l in lanes) == 60
            and sum(l['trajectories'] for l in lanes) == 12, 'FOUR_LANE_PILOT_SCOPE')
    for lane in lanes:
        require(lane['authorized'] is False and lane['technical_sealed'] is False
                and all(lane[k] is None for k in ('scientific_SOURCE', 'plan_digest', 'grant_id',
                                                 'output_path', 'seed_registry')), 'LANE_UNSEALED')
        require(lane['full_prefix'] == inputs['systems'][lane['system']]['observed_rank'], 'PILOT_SERIES_RANK')

    domain = load('candidate_domain_PROPOSED.json')
    require(domain['status'] == 'PROPOSED_NOT_PREREGISTERED_FOR_EXECUTION'
            and domain['approval'] is False and domain['numerical_eligibility_evaluated'] is False
            and domain['cost_winners_selected'] is False, 'DOMAIN_IS_PROPOSAL')
    rows = domain['rows']
    require(len(rows) == len({r['proposal_id'] for r in rows}) == 124, 'PROPOSED_DOMAIN_UNIQUE')
    semantic = {tuple(sorted((k, str(v)) for k, v in row.items() if k != 'proposal_id')) for row in rows}
    require(len(semantic) == len(rows), 'NO_DUPLICATE_SCIENTIFIC_CONDITION')
    for row in rows:
        rank = inputs['systems'][row['system']]['observed_rank']
        require(0 < row['prefix'] <= rank and row['q'] in (1, 2, 4, 8), 'SPARSE_PREFIX_Q_RANGE')
        if row['method'] == 'FULL':
            require(row['prefix'] == rank, 'FULL_USES_SAME_SERIES_RANK')
        else:
            require(row['prefix'] < rank, 'CUT_PREFIX_STRICTLY_BELOW_FULL')
        if row['kind'] == 'PR_SIGNAL':
            require(row['r_policy'] in ('PER_SIGNED_S2_PIECE', 'ABS_TIME_WEIGHTED_OUTER_TOTAL')
                    and row['K'] in (2, 4), 'RTE_POLICY_EXPLICIT')

    stages = {s['id']: s for s in plan['stages']}
    require(len(stages) == len(plan['stages']), 'UNIQUE_STAGES')
    done = set()
    while True:
        ready = {k for k, s in stages.items() if set(s['depends_on']) <= done} - done
        if not ready:
            break
        done |= ready
    require(done == set(stages), 'DAG_DEPENDENCIES_CLOSED_ACYCLIC')
    for stage in stages.values():
        require(stage['execution_approved'] is False and stage['mandatory_STOP'] is True
                and stage['auto_transition'] is False, 'EVERY_STAGE_SEPARATE_APPROVAL_STOP')
    require(plan['parallel']['unrelated_worktree_freeze'] is False
            and plan['parallel']['transient_PSI_CPU_only_stop'] is False
            and plan['parallel']['AI_subagents'] is False, 'PARALLEL_NONINTERFERENCE')

    catalog = load('evidence_catalog.json')
    evidence = {e['id']: e for e in catalog['entries']}
    require(len(evidence) == len(catalog['entries']) == 23, 'FIXED_EVIDENCE_COUNT')
    with (bundle / 'protocol_gap_matrix.csv').open(newline='') as f:
        gaps = list(csv.DictReader(f))
    require(len(gaps) == len({g['requirement'] for g in gaps}) == 18, 'GAP_TABLE_COVERAGE')
    for gap in gaps:
        require(all(k in evidence for k in gap['evidence_ids'].split(';'))
                and gap['limit'] and gap['next_action'], 'GAP_REFERENCE_LIMIT')
    for e in evidence.values():
        require(e['retrieved'] is True and e['science_executed'] is False, 'STATIC_EVIDENCE_ONLY')
        if verify_git:
            value = subprocess.check_output(['git', 'show', e['commit'] + ':' + e['path']], cwd=root)
            require(len(value) == e['bytes'] and hashlib.sha256(value).hexdigest() == e['sha256'], 'FIXED_EVIDENCE_BYTES')
    text = (root / DOC).read_text()
    for term in ('PLAN_REVISION_ONLY_NOT_LAUNCH_READY', 'H4 legacy', 'AD03',
                 'mandatory STOP', '別チャット', '母平均保証', 'ユーザー'):
        require(term in text, 'DOCUMENT_SCOPE:' + term)
    return dict(status='PASS',scope='PLAN_METADATA_STRUCTURE_ONLY',proposal_rows=124,
                gap_rows=18,evidence_files=23,scientific_execution=False,numerical_reanalysis=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    print(json.dumps(validate(p.parse_args().root), ensure_ascii=False))
