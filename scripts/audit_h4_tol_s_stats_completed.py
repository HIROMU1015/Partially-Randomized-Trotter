"""DF10 S_STATS finished-result audit. No numeric kernel or design recomputation."""
from copy import deepcopy
from pathlib import Path

import h4_tol_s_stats_contract as s
import prepare_h4_tol_s_stats as prep
c=s.c
ACT=prep.PRIVATE/'activation01'
PUBLIC=s.ROOT/'artifacts/resource_applicability/track_a_h4_tol_s_stats_completed/2026-10-19'


def audit():
    spec_path=ACT/'S_STATS_contract_EXECUTION_PRIVATE.json';grant_path=ACT/'S_STATS_grant_APPROVED_PRIVATE.json'
    spec,grant=c.read(spec_path),c.read(grant_path)
    output=Path(spec['output']);run=output.parent
    result,terminal,native=c.read(output/'result.json'),c.read(output/'terminal.json'),c.read(run/'native_exit.json')
    pending=c.read(prep.PRIVATE/'S_STATS_contract_SEALED_PENDING_APPROVAL_PRIVATE.json')
    c.require(spec['status']=='SEALED_FOR_EXECUTION' and spec['pending']==[] and spec['plan']==s.plan(),'FIXED_EXECUTION_SCOPE')
    for name in ('SOURCE','plan','source_hashes','summary','runtime_hashes','python','python_sha256','output','lock'):
        c.require(spec[name]==pending[name],'ORIGINAL_SEAL_RETAINED')
    c.require(spec['inputs'][:len(pending['inputs'])]==pending['inputs'],'ORIGINAL_INPUTS_RETAINED')
    c.require(grant['approved_by_user'] and grant['runtime_authorization'] and grant['one_shot']
        and not grant['automatic_retry'] and not grant['next_stage_authorized'],'EXPLICIT_ONE_SHOT_APPROVAL')
    c.require(spec['SOURCE']==grant['SOURCE']==result['SOURCE']==terminal['SOURCE']==native['SOURCE'],'SOURCE_IDENTITY')
    c.require(c.digest(spec)==grant['spec_digest']==result['spec_digest']
        ==c.read(run/'ONE_SHOT.json')['spec_digest'],'ONE_SHOT_SPEC_BINDING')
    c.require(c.sha(grant['approval_receipt'])==grant['approval_receipt_sha256'],'APPROVAL_RECEIPT_SHA')
    c.require(native['exit_code']==0 and native['same_parent_reaped'] and native['result_sha256']
        ==terminal['result_sha256']==c.sha(output/'result.json'),'NATIVE_REAPED_RESULT_BYTES')
    c.require(result['status']==terminal['status']=='COMPLETE_MANDATORY_STOP'
        and terminal['final_closure_checked'] and not terminal['automatic_retry']
        and not terminal['next_stage_authorized'] and not (output/'STOP.json').exists(),'CLEAN_TERMINAL')
    summary=s.verify_bytes(spec)
    c.require(result['input_summary_sha256']==c.sha(spec['summary'])
        and result['fixed_A_commit']==s.A_COMMIT and result['fixed_D_commit']==s.D_COMMIT,'ACCEPTED_FIXED_INPUTS')
    c.require(result['plan']==s.plan() and result['actual_calls']==s.plan()['calls'],'EXACT_ACTUAL_CALLS')
    c.require(c.read(output/'calls_ledger.json')==dict(reserved=s.plan()['calls'],nonrefundable=True,automatic_retry=False),
        'NONREFUNDABLE_SOLE_PASS')
    c.require(not result['P_executed'] and not result['raw_samples_read'] and not result['old_FP_or_legacy_pooled']
        and not result['population_mean_certified'] and not result['actual_power_certified']
        and not result['CI_or_p_values_computed'] and result['new_seeds_generated']==0
        and not result['next_stage_authorized'],'NUMERIC_APPROVAL_BOUNDARY')
    endpoints={e['cell']['id']:e for e in summary['endpoints']}
    scenarios=result['normal_scenarios'];sensitivities=result['sensitivity'];directions=result['reference_directions']
    expected={(cid,f,eta) for cid in endpoints for f in s.plan()['SD_inflation_sensitivity']
        for eta in s.plan()['relative_gap_sensitivity']}
    c.require(len(scenarios)==24 and {(r['cell_id'],r['SD_inflation'],r['relative_gap']) for r in scenarios}==expected,
        'ALL_24_SCENARIOS')
    for row in scenarios:
        e=endpoints[row['cell_id']]
        c.require(row['epsilon']==e['epsilon'] and row['saved_n']==8 and row['saved_paired_SE']==e['PR']['G_RZ_SE']
            and row['PF_cell_id']==e['PF']['cell_id'] and row['PF_G_exact']==e['PF']['G_RZ_exact']
            and type(row['n']) is int and row['n']>=2,'SAVED_SCENARIO_SCHEMA')
    keys={(f,eta) for f in s.plan()['SD_inflation_sensitivity'] for eta in s.plan()['relative_gap_sensitivity']}
    c.require(len(sensitivities)==12 and {(x['SD_inflation'],x['relative_gap']) for x in sensitivities}==keys,
        'ALL_12_SENSITIVITIES')
    for row in sensitivities:
        same=[r for r in scenarios if (r['SD_inflation'],r['relative_gap'])==(row['SD_inflation'],row['relative_gap'])]
        c.require(row['n_fixed']=={r['cell_id']:r['n'] for r in same},'SAVED_N_SCENARIO_JOIN')
        budget=row['proposed_P_calls'];fixed=row['n_fixed']
        # Integer call-accounting identities only; no n formula, SD, power or quantile rerun.
        c.require(budget['trajectory']==budget['native_evolution']==sum(fixed.values())
            and budget['wrapper_compile']==2*budget['trajectory']
            and budget['occurrence']==sum(e['cell']['q']*fixed[cid] for cid,e in endpoints.items())
            and budget['RTE_event']==sum(e['cell']['R']*fixed[cid] for cid,e in endpoints.items())
            and budget['seed_slots']==sum((1 if e['cell']['q']==1 else 1+e['cell']['q'])*fixed[cid] for cid,e in endpoints.items())
            and budget['control_probe_maximum']==8 and budget['independent_event_oracle_maximum']==2
            and not budget['P_executed'],'PROPOSED_P_CALL_IDENTITIES')
    reference=next(x for x in sensitivities if x['SD_inflation']==4 and x['relative_gap']==.02)
    c.require(result['recommended']==reference,'PREREGISTERED_REFERENCE_SCENARIO')
    c.require(len(directions)==2 and [d['cell_id'] for d in directions]==[e['cell']['id'] for e in summary['endpoints']],
        'TWO_FIXED_DIRECTIONS')
    for row in directions:
        e=endpoints[row['cell_id']]
        c.require(row['epsilon']==e['epsilon'] and row['PR_mean_exact']==e['PR']['G_RZ_exact']
            and row['saved_n']==8 and row['saved_paired_SE']==e['PR']['G_RZ_SE']
            and row['PF_cell_id']==e['PF']['cell_id'] and row['PF_G_exact']==e['PF']['G_RZ_exact']
            and row['expected_direction']==e['saved_point_direction'],'SAVED_DIRECTION_INPUT_BINDING')
    c.require(len(result['rare_information'])==2,'TWO_SAVED_RARE_SUMMARIES')
    for row in result['rare_information']:
        c.require(row['saved_counts']==endpoints[row['cell_id']]['saved_rare_counts']
            and row['law_probabilities'] is None and row['n_visibility'] is None and row['unseen_cost'] is None
            and not row['unobserved_strata_zero_filled'],'UNSEEN_TAIL_REMAINS_UNKNOWN')
    prior=c.read(prep.PRIVATE/'prior_preservation_manifest_PRIVATE.json')
    for row in prior:c.require(c.sha(row['path'])==row['sha256'],'OLD_D_A_F_FP_PRESERVED')
    seal=c.read(prep.PRIVATE/'technical_seal_preservation_manifest_PRIVATE.json')
    for row in seal:c.require(c.sha(row['path'])==row['sha256'],'PREPARATION_SEAL_PRESERVED')
    for name,h in c.read(prep.PUBLIC/'artifact_manifest.json')['files'].items():
        c.require(c.sha(prep.PUBLIC/name)==h,'PREPARATION_ARTIFACT_PRESERVED')
    c.require(not PUBLIC.exists(),'NEW_STATS_COMPLETION_ARTIFACT_DIRECTORY')
    PUBLIC.mkdir(parents=True)
    for name in ('result.json','terminal.json','calls_ledger.json'):
        (PUBLIC/name).write_bytes((output/name).read_bytes())
    (PUBLIC/'native_exit.json').write_bytes((run/'native_exit.json').read_bytes())
    c.save(PUBLIC/'approval_receipt.json',c.read(grant['approval_receipt']))
    resources=c.read(spec['fresh_resources']['receipt'])
    c.save(PUBLIC/'execution_binding.json',dict(SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),
        preparation_commit='92780e253a8483679fd03e8fa7a1641aad3db6bb',grant_sha256=c.sha(grant_path),
        approval_receipt_sha256=grant['approval_receipt_sha256'],source_hashes=spec['source_hashes'],
        input_files=len(spec['inputs']),runtime_files=len(spec['runtime_hashes']),
        input_manifest=[{k:v for k,v in row.items() if k!='path'} for row in spec['inputs']],
        runtime_binding_digest=c.digest(spec['runtime_hashes']),python_sha256=spec['python_sha256'],
        fresh_resource_receipt_sha256=spec['fresh_resources']['sha256'],
        allowed_CPUs_observed=len(resources['allowed_cpus']),memory_available_bytes=resources['memory_available_bytes'],
        agent_imposed_CPU_RAM_worker_wall_caps=None,processes=1,one_shot=True,automatic_retry=False,
        scientific_conditions_changed=False,new_public_push_authorized=False))
    c.save(PUBLIC/'P_after_S_STATS_DRAFT.json',dict(status='PENDING_USER_ACCEPTANCE_AND_FINAL_P_SEAL',
        fixed_primary_targets=s.plan()['targets'],recommended_scenario=deepcopy(reference),
        S_STATS_SOURCE=spec['SOURCE'],S_STATS_result_sha256=c.sha(output/'result.json'),
        P_n_final=None,P_SOURCE=None,P_seeds=None,P_grant=None,P_science_authorized=False,P_executed=False,
        population_mean_certified=False,rare_tail_limits=deepcopy(result['rare_information']),
        next_action='User decides whether to adopt this normal planning proposal and instruct final P contract preparation',
        mandatory_STOP=True))
    record_files=set(p for directory in (run,ACT) for p in directory.rglob('*') if p.is_file())
    record_files.add(Path(__file__).resolve())
    manifest=prep.PRIVATE/'completed_S_STATS_preservation_manifest_PRIVATE.json'
    c.require(not manifest.exists(),'NEW_STATS_COMPLETION_PRESERVATION')
    c.save(manifest,[dict(path=str(p),sha256=c.sha(p),bytes=p.stat().st_size) for p in sorted(record_files)])
    final=dict(status='COMPLETE_MANDATORY_STOP',SOURCE=spec['SOURCE'],result_sha256=c.sha(output/'result.json'),
        actual_calls=result['actual_calls'],recommended_n=reference['n_fixed'],proposed_P_calls=reference['proposed_P_calls'],
        source_files=len(spec['source_hashes']),input_files=len(spec['inputs']),runtime_files=len(spec['runtime_hashes']),
        native_exit_code=0,same_parent_reaped=True,technical_interruptions=0,retries=0,
        wall_seconds=native['wall_seconds'],numeric_wall_seconds=terminal['numeric_wall_seconds'],
        prior_records_preserved=len(prior),preparation_records_preserved=len(seal),new_records_preserved=len(record_files),
        private_preservation_manifest_sha256=c.sha(manifest),numeric_analysis_recomputed_by_auditor=False,
        raw_aggregation_bootstrap_sampling_compile_P=0,new_seeds=0,population_mean_certified=False,
        rare_visibility_or_unseen_cost_invented=False,auditor_sha256=c.sha(__file__),
        independent_scientific_review=False,new_public_push=False,mandatory_STOP=True,next_stage_authorized=False)
    c.save(PUBLIC/'completion_audit.json',final)
    c.save(PUBLIC/'artifact_manifest.json',dict(files={p.name:c.sha(p) for p in sorted(PUBLIC.glob('*.json'))
        if p.name!='artifact_manifest.json'},numeric_analysis_recomputed=False))
    return final


if __name__=='__main__':
    import json
    print(json.dumps(audit()))
