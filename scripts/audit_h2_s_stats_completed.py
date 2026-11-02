"""Saved H2 S_STATS result audit. Never calls the numeric design kernel."""
from copy import deepcopy
from pathlib import Path
import shutil

import h2_s_stats_contract as s
from prepare_h2_panels import row
c=s.c


def check_result(result,summary):
    p=s.plan();endpoints={e['cell']['id']:e for e in summary['endpoints']}
    c.require(result['plan']==p and result['actual_calls']==p['calls'], 'EXACT_APPROVED_STATS_SCOPE')
    c.require(not result['P_executed'] and not result['raw_samples_read'] and not result['old_FP_or_legacy_pooled']
        and not result['population_mean_certified'] and not result['actual_power_certified']
        and not result['CI_or_p_values_computed'] and result['new_seeds_generated']==0
        and not result['next_stage_authorized'], 'STATS_ONLY_NO_EXTRA_ANALYSIS')
    scenarios=result['normal_scenarios'];settings=result['sensitivity'];directions=result['reference_directions']
    expected={(cid,f,eta) for cid in endpoints for f in p['SD_inflation_sensitivity'] for eta in p['relative_gap_sensitivity']}
    c.require(len(scenarios)==24 and {(v['cell_id'],v['SD_inflation'],v['relative_gap']) for v in scenarios}==expected,
        'COMPLETE_24_SCENARIOS')
    for v in scenarios:
        ep=endpoints[v['cell_id']]
        c.require(v['epsilon']==ep['epsilon'] and v['saved_n']==ep['PR']['n']==8
            and v['saved_paired_SE']==ep['PR']['G_RZ_SE'] and v['PF_cell_id']==ep['PF']['cell_id']
            and v['PF_G_exact']==ep['PF']['G_RZ_exact'] and type(v['n']) is int and v['n']>=2,
            'SAVED_SCENARIO_INPUTS_AND_INTEGER_N')
    expected_settings={(f,eta) for f in p['SD_inflation_sensitivity'] for eta in p['relative_gap_sensitivity']}
    c.require(len(settings)==12 and {(v['SD_inflation'],v['relative_gap']) for v in settings}==expected_settings,
        'COMPLETE_12_SENSITIVITY_SETTINGS')
    for setting in settings:
        rows=[v for v in scenarios if (v['SD_inflation'],v['relative_gap'])==
            (setting['SD_inflation'],setting['relative_gap'])]
        fixed=setting['n_fixed'];budget=setting['proposed_P_calls']
        c.require(fixed=={v['cell_id']:v['n'] for v in rows}, 'SAVED_SCENARIO_N_JOIN')
        # These integer identities audit proposed call counts, never rederive n/SD/power.
        c.require(budget['trajectory']==budget['native_evolution']==sum(fixed.values())
            and budget['wrapper_compile']==2*budget['trajectory']
            and budget['occurrence']==sum(ep['cell']['q']*fixed[cid] for cid,ep in endpoints.items())
            and budget['RTE_event']==sum(ep['cell']['R']*fixed[cid] for cid,ep in endpoints.items())
            and budget['seed_slots']==sum((1 if ep['cell']['q']==1 else 1+ep['cell']['q'])*fixed[cid] for cid,ep in endpoints.items())
            and budget['control_probe_maximum']==8 and budget['independent_event_oracle_maximum']==2
            and not budget['P_executed'], 'SAVED_PROPOSED_P_CALL_IDENTITIES')
    reference=next(v for v in settings if v['SD_inflation']==4 and v['relative_gap']==.02)
    c.require(result['recommended']==reference, 'PREREGISTERED_F4_GAP02_REFERENCE')
    c.require(len(directions)==2 and [v['cell_id'] for v in directions]==
        [ep['cell']['id'] for ep in summary['endpoints']], 'TWO_SAVED_REFERENCE_DIRECTIONS')
    for v in directions:
        ep=endpoints[v['cell_id']];ratio=ep['saved_PR_to_PF_ratio']
        c.require(v['epsilon']==ep['epsilon'] and v['PR_mean_exact']==ep['PR']['G_RZ_exact']
            and v['saved_n']==8 and v['saved_paired_SE']==ep['PR']['G_RZ_SE']
            and v['PF_cell_id']==ep['PF']['cell_id'] and v['PF_G_exact']==ep['PF']['G_RZ_exact']
            and v['expected_direction']==(ratio<1)-(ratio>1), 'SAVED_DIRECTION_INPUT_JOIN')
    c.require(len(result['rare_information'])==2, 'TWO_SAVED_RARE_SUMMARIES')
    for v in result['rare_information']:
        c.require(v['saved_counts']==endpoints[v['cell_id']]['saved_rare_counts'] and v['law_probabilities'] is None
            and v['n_visibility'] is None and v['unseen_cost'] is None
            and not v['unobserved_strata_zero_filled'], 'UNKNOWN_RARE_TAIL_NOT_ZERO_FILLED')
    return reference


def audit():
    spec=c.read(s.ACTIVATION/'S_STATS_spec_EXECUTION.json');grant=c.read(s.ACTIVATION/'S_STATS_grant_APPROVED_PRIVATE.json')
    pending=c.read(spec['pending_spec']);output=Path(spec['output']);attempt=Path(spec['attempt'])
    c.require(spec['status']=='SEALED_FOR_EXECUTION' and spec['pending']==[] and spec['plan']==s.plan(), 'FIXED_STATS_EXECUTION')
    allowed={'status','pending','pending_spec','pending_spec_sha256','fresh_resources'}
    c.require({k:v for k,v in spec.items() if k not in allowed}==
        {k:v for k,v in pending.items() if k not in allowed}
        and c.sha(spec['pending_spec'])==spec['pending_spec_sha256'], 'ORIGINAL_STATS_SEAL_PRESERVED')
    c.require(grant['scope']==s.SCOPE and grant['stage']=='S_STATS' and grant['approved_by_user']
        and grant['runtime_authorization'] and grant['one_shot'] and not grant['automatic_retry']
        and not grant['next_stage_authorized'] and grant['spec_digest']==c.digest(spec), 'BOUND_EXPLICIT_STATS_GRANT')
    c.require(c.sha(grant['approval_receipt'])==grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT_BYTES')
    user=c.read(grant['approval_receipt'])
    c.require(user['role']=='user' and user['stage']=='S_STATS' and user['SOURCE']==spec['SOURCE']
        and user['plan_digest']==c.digest(s.plan()) and user['verbatim_user_message'], 'USER_STATS_APPROVAL')
    resource=spec['fresh_resources']
    c.require(resource['admitted'] and c.sha(resource['receipt'])==resource['sha256'], 'RESOURCE_RECEIPT_PRESERVED')
    summary=s.verify_bytes(spec)
    result=c.read(output/'result.json');terminal=c.read(output/'terminal.json');native=c.read(attempt/'native_exit.json')
    started=c.read(attempt/'native_started.json');marker=c.read(spec['lock'])
    c.require(result['SOURCE']==terminal['SOURCE']==native['SOURCE']==marker['SOURCE']==grant['SOURCE']==spec['SOURCE']
        and marker['spec_digest']==result['spec_digest']==c.digest(spec)
        and marker['grant_digest']==c.digest(grant), 'RESULT_SOURCE_SPEC_GRANT_MARKER')
    c.require(native['exit_code']==0 and native['same_parent_reaped'] and native['wait4_complete']
        and native['pid']==started['pid'] and native['result_sha256']==terminal['result_sha256']==c.sha(output/'result.json'),
        'NATIVE_EXIT_WAIT4_RESULT_BYTES')
    c.require(result['status']==terminal['status']=='COMPLETE_MANDATORY_STOP' and terminal['final_closure_checked']
        and not (output/'STOP.json').exists() and not terminal['automatic_retry']
        and not terminal['next_stage_authorized'], 'CLEAN_STATS_TERMINAL')
    c.require(result['fixed_A_commit']==s.A_COMMIT and result['fixed_D_commit']==s.D_COMMIT
        and result['input_summary_sha256']==c.sha(spec['summary']), 'FIXED_STATS_INPUT_SUMMARY')
    c.require(c.read(output/'calls_ledger.json')==dict(reserved=s.plan()['calls'],nonrefundable=True,automatic_retry=False),
        'NONREFUNDABLE_SOLE_NUMERIC_PASS')
    reference=check_result(result,summary)
    for manifest in (s.STORE/'technical_seal_preservation_manifest_PRIVATE.json',s.ACTIVATION/'activation_preservation_manifest.json'):
        for v in c.read(manifest):c.require(c.sha(v['path'])==v['sha256'], 'STATS_SEAL_AND_ACTIVATION_PRESERVED')
    c.require(not (s.PRIVATE/'attempts/P_run01').exists(), 'P_NOT_EXECUTED')
    s.COMPLETED.mkdir(exist_ok=False)
    for name in ('result.json','terminal.json','calls_ledger.json'):shutil.copyfile(output/name,s.COMPLETED/name)
    for name in ('native_exit.json','native_started.json'):shutil.copyfile(attempt/name,s.COMPLETED/name)
    shutil.copyfile(grant['approval_receipt'],s.COMPLETED/'approval_receipt.json')
    shutil.copyfile(spec['summary'],s.COMPLETED/'selected_saved_summaries.json')
    c.save(s.COMPLETED/'execution_binding.json',dict(SOURCE=spec['SOURCE'],scope=s.SCOPE,spec_digest=c.digest(spec),
        grant_sha256=c.sha(s.ACTIVATION/'S_STATS_grant_APPROVED_PRIVATE.json'),
        pending_spec_sha256=spec['pending_spec_sha256'],source_hashes=spec['source_hashes'],inputs=spec['inputs'],
        runtime_files=len(spec['runtime_hashes']),runtime_digest=c.digest(spec['runtime_hashes']),
        python_sha256=spec['python_sha256'],fresh_resources=spec['fresh_resources'],
        original_D_A_unchanged=True,valid_grant_body_exported=False,new_public_push=False))
    c.save(s.COMPLETED/'P_after_S_STATS_DRAFT.json',dict(status='PENDING_USER_ADOPTION_AND_FINAL_P_SEAL',
        fixed_primary_targets=s.plan()['targets'],recommended_scenario=deepcopy(reference),
        S_STATS_SOURCE=spec['SOURCE'],S_STATS_result_sha256=c.sha(output/'result.json'),
        P_n_final=None,P_SOURCE=None,P_seeds=None,P_grant=None,P_science_authorized=False,P_executed=False,
        rare_tail_limits=deepcopy(result['rare_information']),population_mean_certified=False,mandatory_STOP=True))
    rows=[row(p,str(p.relative_to(s.PRIVATE))) for base in (attempt,s.ACTIVATION)
        for p in sorted(base.rglob('*')) if p.is_file()]
    manifest=s.STORE/'completed_preservation_manifest_PRIVATE.json';c.save(manifest,rows)
    audit=dict(status='COMPLETE_MANDATORY_STOP',stage='S_STATS',SOURCE=spec['SOURCE'],
        result_sha256=c.sha(output/'result.json'),actual_calls=result['actual_calls'],recommended_n=reference['n_fixed'],
        proposed_P_calls=reference['proposed_P_calls'],source_files=len(spec['source_hashes']),
        input_files=len(spec['inputs']),runtime_files=len(spec['runtime_hashes']),native_exit_code=0,
        same_parent_reaped=True,wait4_complete=True,wall_seconds=native['wall_seconds'],
        numeric_wall_seconds=terminal['numeric_wall_seconds'],native_peak_RSS_bytes=native['peak_RSS_bytes'],
        prior_records_preserved=len(c.read(spec['prior_manifest'])),completed_private_records=len(rows),
        completed_manifest_sha256=c.sha(manifest),numeric_analysis_recomputed_by_auditor=False,
        raw_aggregation_bootstrap_sampling_compile_P=0,new_seeds=0,population_mean_certified=False,
        actual_power_certified=False,independent_scientific_review=False,automatic_retry=False,
        next_stage_authorized=False,new_public_push=False)
    c.save(s.COMPLETED/'completion_audit.json',audit)
    c.save(s.COMPLETED/'artifact_manifest.json',dict(files=[dict(path=p.name,sha256=c.sha(p),bytes=p.stat().st_size)
        for p in sorted(s.COMPLETED.iterdir()) if p.is_file()],manifest_self_excluded=True))
    return audit


if __name__=='__main__':print(__import__('json').dumps(audit(),indent=2))
