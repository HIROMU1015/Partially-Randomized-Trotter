"""Seal an inactive fresh-22 recovery specification using saved evidence only.

No activation, grant creation, molecular import, circuit creation, compilation,
shot evaluation or ranking is implemented here. Budget approval is still needed.
"""
import argparse
import json
from pathlib import Path

import pf_common_run_contract as c
from audit_h6_pf_interruption import standard_fresh_budget

BUNDLE = 'artifacts/resource_applicability/track_a_h6_pf_cost_interruption/2026-10-22'
NEW_SOURCE = ('scripts/prepare_h6_pf_cost_recovery.py', 'scripts/run_detached_local_job.py',
              'tests/test_h6_pf_recovery_preparation.py')


def validate_inactive(plan):
    c.require(plan['status'] == 'UNAPPROVED_NOT_A_GRANT', 'INACTIVE_ONLY')
    c.require(plan['execution_authorized'] is False and plan['additional_budget_approved'] is False, 'NO_BUDGET_APPROVAL')
    c.require(plan['new_grants_created'] == 0 and plan['execution_command'] is None, 'NO_EXECUTION')
    c.require(plan['retry'] is False and plan['resume'] is False, 'NO_RETRY_RESUME')
    rows = plan['fresh_budget']['candidates']
    c.require(len(rows) == len({c.cid(r) for r in rows}) == 22, 'EXACT22')
    c.require(all(r['system'] == 'H6' and r['q'] in (1, 2, 4, 8) and r['formula'] in c.FORMULAS
        and ((r['method'] == 'FULL' and r['prefix'] == 19)
             or (r['method'] in ('B0', 'B0I') and r['prefix'] in range(5, 16))) for r in rows), 'H6_REGISTERED_ONLY')
    c.require(sorted(c.cid(r) for r in rows) == plan['expected22_ids'], 'PINNED22_IDENTITIES')
    caps = plan['fresh_budget']['fresh_caps']
    c.require(caps['native_evolution'] == 22 and caps['native_piece'] == 1520
        and caps['wrapper_compile'] == caps['wrapper_build'] == 44, 'FIXED_FRESH_INTERNAL_CAPS')
    c.require(plan['fresh_budget']['exceeds_fixed_internal_budget'] is True, 'EXPLICIT_BUDGET_GAP')
    c.require(plan['stages']['ACCOUNT']['requires_new_COST_success'] is True, 'NO_PREMATURE_ACCOUNT')
    return True


def verify_seals(plan):
    validate_inactive(plan)
    for entry in plan['seals']:
        path = Path(entry['path'])
        c.require(path.stat().st_size == entry['bytes'] and c.sha(path) == entry['sha256'], 'SEALED:'+entry['role'])
    return True


def prepare(private, output):
    private, output = Path(private).resolve(), Path(output).resolve()
    c.require(not output.exists(), 'FRESH_INACTIVE_PACK_REQUIRED')
    parent = c.read(private/'plan_APPROVED.json')
    c.verify(parent)
    c.require(parent['system'] == 'H6' and (private/'COST/STOP.json').exists()
        and not (private/'COST/completion.json').exists(), 'INTERRUPTED_PARENT_ONLY')
    b = c.ROOT/BUNDLE
    summary = c.read(b/'summary.json')
    c.require(summary['complete_native_pairs'] == 187 and summary['missing_compiled_pairs'] == 22, 'PINNED_AUDITED_PARTITION')
    selected = c.read(private/'COST/selected_candidates.json')
    missing = [r for r in selected if not (private/'COST/output'/c.cid(r)/'result.json').exists()]
    c.require(missing == c.read(b/'unresolved_candidates_22.json'), 'EXACT_AUDITED_MISSING_IDS')
    c.require({c.cid(r) for r in missing} <= set(parent['compile_pool']), 'OLD_REGISTERED_POOL_ONLY')
    consumed = c.read(private/'COST/reserved_counts.json')
    fresh = standard_fresh_budget(missing, consumed, parent['cost_cap_template'])
    c.require(fresh == c.read(b/'fresh_completion_contract_UNAPPROVED.json'), 'UNCHANGED_BUDGET_PROPOSAL')
    seals = list(parent['closure'])
    for rel in ('plan_APPROVED.json', 'approval_receipt.json', 'saved_prices.json', 'cost_identity_audit.json',
                'COST/STOP.json', 'COST/grant.json', 'COST/CONSUMED.json', 'COST/reserved_counts.json',
                'COST/selected_candidates.json', 'SIGNAL/completion.json', 'SIGNAL/output/all_signals.json',
                'QUALIFY/completion.json', 'QUALIFY/output/endpoints.json', 'QUALIFY/output/eligible_ids.json'):
        seals.append(c.seal(private/rel, 'parent:'+rel))
    for name in ('summary.json', 'ending_preservation_checks.json', 'raw_archive_inventory.json',
                 'complete_native_results_187.json.gz', 'available_prices_UNACCOUNTED.json.gz',
                 'fresh_completion_contract_UNAPPROVED.json', 'candidate_status_588.json'):
        seals.append(c.seal(b/name, 'audited_evidence:'+name))
    for entry in c.read(b/'raw_archive_inventory.json'):
        p = private/'COST/output'/entry['candidate']/entry['path']
        c.require(c.sha(p) == entry['archive_sha256'], 'ACCEPTED_ARCHIVE_UNCHANGED')
        seals.append(c.seal(p, 'accepted_raw_archive:'+entry['candidate']+'/'+entry['path']))
    for row in selected:
        path = private/'COST/output'/c.cid(row)/'result.json'
        if path.exists():
            seals.append(c.seal(path, 'accepted_complete_pair:'+c.cid(row)))
    for stage in ('SIGNAL', 'QUALIFY', 'COST'):
        for path in sorted((private/stage/'ledger').glob('*.json')):
            seals.append(c.seal(path, 'parent_nonrefundable:'+stage+'/'+path.name))
    for rel in NEW_SOURCE:
        p = c.ROOT/rel
        c.require(p.read_bytes() == c.gitbytes(c.source(), rel), 'COMMITTED_PREPARATION_SOURCE')
        seals.append(c.seal(p, 'recovery_preparation_SOURCE:'+rel))
    plan = dict(status='UNAPPROVED_NOT_A_GRANT', preparation_SOURCE=c.source(),
        original_scientific_SOURCE=parent['SOURCE'], original_plan_sha256=c.sha(private/'plan_APPROVED.json'),
        original_private=str(private), execution_authorized=False, additional_budget_approved=False,
        new_grants_created=0, execution_command=None, retry=False, resume=False,
        scientific_binding=parent['binding'], scientific_conditions={k:parent[k] for k in ('T', 'P', 'compiler', 'native')},
        fresh_budget=fresh, expected22_ids=sorted(c.cid(r) for r in missing), seals=seals,
        accepted_completed_pairs=187, saved_eligible_pairs=30,
        parent_COST_success=False, parent_native_exit_reap_verified=False,
        inherited_SIGNAL_QUALIFY_are_references_not_new_stages=True,
        stages=dict(COST=dict(fresh_grant_required=True, caps=fresh['fresh_caps'],
                original_numeric_worker='scripts/pf_common_run_native.py:cost_one',
                new_private_required=True, original_private_forbidden_as_output=True,
                complete_pairs_only=True, partial_axes_accepted=False),
            ACCOUNT=dict(fresh_grant_required=True, requires_new_COST_success=True,
                caps=dict(weighted_PF_endpoint=1380, comparison=20),
                inherited_qualification_not_recomputed=True, old_PR_statistics_unchanged=True),
            AUDIT=dict(new_worker_exit_reap_required=True, old_parent_exit_reap_must_remain_unknown=True,
                cumulative_failed_and_new_ledgers_required=True, global_minimum_requires_all276=True),
            JOIN=dict(H2_H4_reference_only=True, registered_candidates=588, new_PR_sampling=0,
                local_commit=True, mandatory_STOP=True, GitHub_push=False)),
        activation_requirements=['Explicit approval of the stated internal-call increase',
            'New committed execution SOURCE and a separately sealed approved pack',
            'Driver handling inherited stages and multi-SOURCE accepted prices; original full-run driver cannot resume',
            'Fresh resource checks and distinct one-shot COST and ACCOUNT grants'])
    validate_inactive(plan)
    verify_seals(plan)
    output.mkdir(parents=True)
    c.save(output/'recovery_contract_UNAPPROVED.json', plan)
    c.save(output/'preparation_validation.json', dict(status='PASS_INACTIVE', sealed_files=len(seals),
        SOURCE=plan['preparation_SOURCE'], scientific_calls=0, new_grants=0,
        inactive_contract_sha256=c.sha(output/'recovery_contract_UNAPPROVED.json'),
        old_grant_reused=False, execution_authorized=False, all_sealed_files_match=True))
    return dict(status='PASS_INACTIVE', sealed_files=len(seals), scientific_calls=0, new_grants=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--private', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.private, a.output)))
