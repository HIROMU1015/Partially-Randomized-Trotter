"""Audit and publish saved S_STATS results without another numerical-design call."""
import argparse
import csv
from pathlib import Path

import h4_dp_contract as c
import h4_s_stats as stats

ROOT = Path(__file__).resolve().parents[1]


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    spec, grant = c.read(run/'spec_SEALED_PRIVATE.json'), c.read(run/'grant_PRIVATE.json')
    output = Path(spec['output'])
    native, terminal = c.read(run/'native_exit.json'), c.read(output/'terminal.json')
    ledger, result = c.read(output/'calls_ledger.json'), c.read(output/'result.json')
    sha = c.sha(output/'result.json')
    c.require(native['exit_code'] == 0 and native['same_parent_reaped'] is True
        and native['result_sha256'] == terminal['result_sha256'] == sha
        and terminal['status'] == 'COMPLETE_MANDATORY_STOP' and terminal['final_closure_checked'] is True,
        'S_STATS_NATIVE_COMPLETION')
    c.require(not (output/'STOP.json').exists() and Path(spec['lock']).exists(), 'S_STATS_ONE_SHOT_COMPLETED')
    c.require(spec['SOURCE'] == result['SOURCE'] == native['SOURCE'] == grant['SOURCE']
        and c.digest(spec) == grant['spec_digest'], 'SOURCE_AND_GRANT_DIGEST')
    c.require(c.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'APPROVAL_BYTES')
    stats.verify_bytes(spec)  # Hash verification only; no normal_n_formula/analyze call.
    previous = c.read(run/'previous_preservation_manifest_PRIVATE.json')
    for item in previous:
        c.require(c.sha(item['path']) == item['sha256'] and c.sha(item['copy']) == item['sha256'],
                  'OLD_D_RECORD_BYTES')
    c.require(ledger['numeric_design_pass'] == result['actual_calls']['numeric_design_pass'] == 1
              and result['actual_calls']['normal_scenarios'] == 84
              and result['actual_calls']['fixed_reference_sign_comparisons'] == 14
              and all(result['actual_calls'][k] == 0 for k in stats.FORBIDDEN), 'ONE_APPROVED_PASS')
    scenarios = result['normal_scenarios']; directions = result['reference_directions']
    c.require(len(scenarios) == len({(x['cell_id'], x['epsilon'], x['f'], x['eta']) for x in scenarios}) == 84
              and len(directions) == len({(x['cell_id'], x['epsilon']) for x in directions}) == 14,
              'FIXED_RESULT_CARDINALITY')
    recommended = result['recommended']
    c.require(len(recommended['n_fixed']) == 9 and recommended == next(x for x in result['sensitivity']
              if x['f'] == 4 and x['eta'] == .02), 'FIXED_RECOMMENDATION_SCENARIO')
    c.require(all(type(n) is int and n >= 2 for n in recommended['n_fixed'].values()), 'INTEGER_N')
    public = ROOT/'artifacts/resource_applicability/track_a_h4_s_stats/2026-10-19'
    c.require(not (public/'completion_audit.json').exists(), 'UNUSED_STATS_PUBLICATION')
    c.save(public/'result.json', result)
    c.save(public/'native_exit_projection.json', native)
    c.save(public/'terminal_projection.json', terminal)
    c.save(public/'source_input_binding.json', dict(SOURCE=spec['SOURCE'], fixed_result_commit=stats.FIXED_RESULT,
        source_hashes=spec['source_hashes'], inputs=[{k: v for k, v in x.items() if k != 'path'} for x in spec['inputs']],
        private_spec_digest=c.digest(spec), user_approval_receipt_sha256=grant['approval_receipt_sha256'],
        python_sha256=spec['python_sha256'], runtime_files=len(spec['runtime_hashes']),
        runtime_manifest_digest=c.digest(spec['runtime_hashes'])))
    table(public/'normal_scenarios.csv', scenarios)
    table(public/'candidate_n.csv', [dict(cell_id=x['cell']['id'], role=x['role'],
        own_endpoint_max_n=x['own_endpoint_max_n'], K2_partner=x['K2_partner'], K2_partner_n=x['K2_partner_n'],
        recommended_n=x['recommended_n']) for x in recommended['candidates']])
    # Derive a reviewable P template from the saved result. No seeds, events, or
    # scientific law are generated, and no P execution grant is created.
    n_fixed = recommended['n_fixed']
    endpoints = [dict(x, n=n_fixed[x['cell_id']]) for x in directions]
    proposal = dict(status='DRAFT_UNAUTHORIZED', stage='P', approved_by_user=False, runtime_authorization=False,
        proposed_P_SOURCE=stats.FIXED_RESULT, P_SOURCE_status='PROPOSED_EXISTING_FROZEN_KERNEL_PENDING_FINAL_SEAL',
        S_STATS_SOURCE=spec['SOURCE'], S_STATS_result_sha256=sha,
        S_STATS_approval_receipt_sha256=grant['approval_receipt_sha256'],
        D_result_sha256=result['D_result_sha256'], recommendation=dict(f=4, eta=.02, n_cap=None),
        n_fixed=n_fixed, candidates=[x['cell'] for x in recommended['candidates']],
        evaluation_endpoints=endpoints, proposed_scientific_calls=recommended['proposed_P_calls'],
        seed_contract=dict(namespace='H4_PR_INDEPENDENT_CONFIRMATION_V1',
            rule='SHA256([namespace,SOURCE,seed_plan_digest,candidate_id,replica]) first 64 bits; original direct_master_to_step_seeds_v1',
            seed_slots=recommended['proposed_P_calls']['seed_slots'], seeds_generated=0,
            seed_list_status='PENDING_PRIVATE_FINAL_SEED_SEAL',
            collision_status='PENDING_COMPLETE_OLD_SERIES_METADATA_INVENTORY',
            cross_candidate_common_random_numbers=False, cosine_sine_share_same_trajectory=True),
        pending=['FINAL_P_SOURCE_AND_EXECUTION_CLOSURE_SEAL', 'COMPLETE_OLD_SEED_METADATA_INVENTORY',
                 'PRIVATE_SEED_PLAN_AND_COLLISION_RECEIPT', 'FRESH_P_RESOURCE_ADMISSION', 'SEPARATE_P_EXECUTION_APPROVAL'],
        P_executed=False, population_mean_certified=False, next_stage_authorized=False)
    c.save(public/'P_contract_after_S_STATS_DRAFT.json', proposal)
    c.save(public/'P_grant_INVALID.json', dict(stage='P', status='INVALID_NO_P_APPROVAL',
        approved_by_user=False, runtime_authorization=False, spec_digest=c.digest(proposal), SOURCE=None,
        one_shot=True, retry=False, next_stage_authorized=False))
    audit = dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'], result_sha256=sha,
        numeric_design_passes=1, normal_scenarios=84, fixed_reference_sign_comparisons=14,
        unique_candidates=9, native_exit=0, same_parent_reaped=True, final_closure_rechecked=True,
        old_records_and_copies_preserved=len(previous), raw_aggregation=0, bootstrap=0, sampling=0, compile=0,
        candidate_reselection=0, new_seeds_generated=0, P_executed=False,
        n_capped=False, numerical_design_reexecuted_by_audit=False, remote_push=False,
        next_stage_authorized=False)
    c.save(public/'completion_audit.json', audit)
    c.save(run/'completion_audit_PRIVATE.json', audit)
    paths = [ROOT/name for name in spec['source_hashes']]+[Path(x['path']) for x in spec['inputs']]
    paths += [run/name for name in ['user_approval_PRIVATE.json', 'resources_PRIVATE.json',
        'spec_SEALED_PRIVATE.json', 'grant_PRIVATE.json', 'preflight_PRIVATE.json',
        'ONE_SHOT.json', 'stdout.log', 'stderr.log', 'native_exit.json', 'completion_audit_PRIVATE.json']]
    paths += [output/name for name in ['calls_ledger.json', 'result.json', 'terminal.json']]
    store=run/'preservation_store';store.mkdir(exist_ok=False);records=[]
    for path in paths:
        h=c.sha(path);copy=store/h
        if not copy.exists():copy.write_bytes(path.read_bytes())
        c.require(c.sha(copy)==h,'S_STATS_SNAPSHOT_COPY')
        records.append(dict(path=str(path),sha256=h,copy=str(copy)))
    c.save(run/'preservation_manifest_PRIVATE.json',records)
    c.save(public/'preservation_projection.json',dict(old_records=len(previous),new_records=len(records),
        copies_verified=True,private_manifest_sha256=c.sha(run/'preservation_manifest_PRIVATE.json')))
    print('S_STATS_AUDIT_PASS; NUMERIC_DESIGN_NOT_REEXECUTED; P_UNAUTHORIZED; MANDATORY_STOP')


if __name__=='__main__':main()
