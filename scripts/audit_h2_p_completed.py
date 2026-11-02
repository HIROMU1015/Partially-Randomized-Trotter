"""H2 P completion audit and public projections; never samples or reaccounts.

The approved runner performs the sole statistics pass. This author's metadata
audit checks saved joins, identities, ledgers and bytes, not population claims.
"""
import argparse
from collections import Counter
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil

import h2_p_contract as c
from prepare_h2_panels import row
from audit_h2_fp_completed import original_process_alive


def check_saved_accounting(accounting,endpoints):
    c.require(not accounting['candidate_reselected'] and not accounting['old_cohorts_pooled']
        and not accounting['population_mean_certified'] and not accounting['actual_power_certified']
        and accounting['law_probabilities'] is None and accounting['unseen_event_cost'] is None
        and accounting['unseen_event_variance'] is None,'P_POINT_COMPARISON_WITH_UNKNOWN_TAIL')
    rows=accounting['comparisons'];c.require(len(rows)==len(endpoints)==2,'EXACT_TWO_P_COMPARISONS')
    for r,ep in zip(rows,endpoints):
        c.require((r['cell_id'],r['epsilon'],r['n'])==(ep['cell_id'],ep['epsilon'],ep['n'])
            and r['PF']==ep['PF'] and r['shots']==ep['saved_shots'],'SAVED_FIXED_ENDPOINT_PF_SHOT_JOIN')
        c.require(isinstance(r['G_exact'],str) and isinstance(r['sample_variance_exact'],str)
            and all(type(r[k]) in (float,int) and math.isfinite(r[k]) and r[k]>=0
                for k in ('sample_SD','paired_SE')) and not r['population_mean_certified'],
            'SAVED_FINITE_SAMPLE_STATISTICS_SCHEMA')
        direction=r['point_direction'];expected=ep['expected_direction']
        c.require(type(direction) is int and direction in (-1,0,1),'POINT_DIRECTION_SCHEMA')
        status='INCONCLUSIVE' if direction==0 or expected==0 else 'REPRODUCED' if direction==expected else 'NOT_REPRODUCED'
        c.require(r['status']==status,'SAVED_DIRECTION_STATUS_CONSISTENCY')


def public_pair(value):
    return {k:({a:{key:v for key,v in axis.items() if key!='seed'} for a,axis in val.items()}
        if k=='axes' else deepcopy(val)) for k,val in value.items() if k!='seed'}


def audit():
    activation=c.PRIVATE/'P_activation01';spec=c.read(activation/'P_spec_EXECUTION.json')
    grant=c.read(activation/'P_grant_APPROVED_PRIVATE.json');c.authorize(spec,grant)
    for manifest in (c.STORE/'seal_preservation_manifest_PRIVATE.json',activation/'activation_preservation_manifest.json'):
        for r in c.read(manifest):c.require(c.sha(r['path'])==r['sha256'],'P_SEAL_ACTIVATION_PRESERVED')
    attempt=c.owned(spec['attempt']);output=attempt/'output'
    result=c.read(output/'result.json');terminal=c.read(output/'terminal.json')
    native=c.read(attempt/'native_exit.json');started=c.read(attempt/'native_started.json')
    marker=c.read(attempt/'ONE_SHOT.json');done=c.read(attempt/'completion_audit.json')
    c.require(result['SOURCE']==spec['SOURCE']==marker['SOURCE']==done['SOURCE']
        and result['spec_digest']==terminal['spec_digest']==marker['spec_digest']==c.digest(spec)
        and marker['grant_digest']==c.digest(grant),'P_RESULT_SOURCE_SPEC_GRANT_MARKER')
    c.require(result['stage']==terminal['stage']==done['stage']=='P' and result['scope']==c.SCOPE
        and result['binding']==c.base.BINDING and result['seed_plan_digest']==spec['seed_plan_digest'], 'P_RESULT_BINDING')
    c.require(result['status']==terminal['status']==done['status']=='COMPLETE_MANDATORY_STOP'
        and native['exit_code']==terminal['exit_code']==done['native_exit_code']==0
        and all(v['same_parent_reaped'] and v['wait4_complete'] for v in (native,done))
        and started['pid']==native['pid'] and started['parent_pid']==marker['parent_pid'], 'P_EXIT0_SAME_PARENT_WAIT4')
    c.require(done['result_sha256']==c.sha(output/'result.json') and not (attempt/'driver_STOP.json').exists()
        and not (output/'STOP.json').exists(),'P_RESULT_HASH_NO_STOP')
    c.require(result['saved_signals_shots_PF_unchanged'] and not result['old_cohorts_pooled']
        and result['additional_sampling']==0 and not result['automatic_retry']
        and not result['next_stage_authorized'] and not result['population_mean_certified'],'P_ONLY_FIXED_N_NO_RETRY')
    reserved={k:0 for k in c.CALLS}
    for p in sorted(output.glob('reservation_*.json')):
        r=c.read(p);c.require(r['nonrefundable'] and type(r['count']) is int and r['count']>0,'NONREFUNDABLE_P_RESERVATION')
        reserved[r['category']]+=r['count']
    c.require(reserved==result['calls_reserved']==done['calls_reserved']==spec['calls']==result['calls_maximum']==c.CALLS,
        'EXACT_P_BUDGET_AND_LEDGER')
    registry=c.read(spec['seed_registry']);c.check_measurements(result['measurements'],registry,spec['endpoints'])
    byid={t['id']:t for t in registry['tasks']};pairs={r['cell_id']+'_'+str(r['replica']):r for r in result['measurements']}
    first={t['id'] for t in registry['tasks'] if t['replica']==0};rest=set(byid)-first
    workers=[];controls=[];trajectories=[];strata={cid:Counter() for cid in c.N_FIXED};receipts={}
    occurrence=0;events_count=0
    for group,expected in [('first_control_workers',first),('remaining_workers',rest)]:
        directory=output/group;receipt=c.read(directory/'accepted_native_receipt.json');receipts[group]=receipt
        c.require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors']
            and receipt['first_error'] is None and not receipt['automatic_retry']
            and len(receipt['workers'])==len(expected) and {w['task_id'] for w in receipt['workers']}==expected,
            'P_COMPLETE_OWNED_WORKER_WAVES')
        for w in receipt['workers']:
            d=c.owned(w['directory']);task=byid[w['task_id']];value=c.read(d/'task_result.json');ident=w['identity']
            c.require(d.parent==directory and w['exit_code']==0 and w['reaped']
                and c.sha(d/'task_result.json')==w['result_sha256'] and value==pairs[task['id']], 'P_WORKER_BYTES_AND_EXIT')
            c.require(ident[0]==w['pid'] and ident[2]==native['pid'] and ident[3]==os.getuid()
                and w['cpu'] in spec['worker_cpus'] and not original_process_alive(ident[0],ident[1]), 'P_OWNED_WORKER_IDENTITY_REAP')
            trajectory=c.read(d/'trajectory.json');events=trajectory['events'];identity=c.read(d/'confirmation_seed_identity.json')
            c.require(trajectory['cell']==task['cell'] and trajectory['seed']==task['seed']
                and trajectory['cohort']==c.COHORT and not trajectory['conditional'] and not trajectory['quantum_weights_changed']
                and c.digest(events)==trajectory['event_digest']==value['event_digest'], 'P_ORIGINAL_UNCONDITIONAL_PAIRED_EVENTS')
            c.require(identity==dict(master_seed=task['seed'],step_seeds=task['step_seeds'],cell_id=task['cell']['id'],
                replica=task['replica'],seed_plan_digest=spec['seed_plan_digest'],cohort=c.COHORT,redraw=False), 'P_EXACT_MASTER_STEP_SEEDS')
            c.require(len(events)==task['cell']['q'] and all(len(e)==task['cell']['r'] for e in events)
                and c.base.stats.event_stratum(events)==value['stratum'],'P_EVENT_LAYOUT_AND_SAVED_STRATUM')
            occurrence+=len(events);events_count+=sum(map(len,events))
            for axis in ('cosine','sine'):
                c.require(c.read(d/('measured_'+axis+'.json'))==value['axes'][axis], 'P_SAVED_COMPILED_AXIS')
                r=c.read(d/('compile_'+axis+'_reserved.json'))
                c.require(r['nonrefundable'] and r['fingerprint']==value['axes'][axis]['fingerprint']
                    and r['event_digest']==value['event_digest'],'P_COMPILE_EVENT_FINGERPRINT')
            if task['replica']==0:
                control=c.read(d/'control.json')
                c.require(control['status']=='PASS' and control['tolerance']==1e-9 and math.isfinite(control['max_error'])
                    and control['max_error']<1e-9 and control['control_probes']==4 and control['independent_event_oracles']==1
                    and control['full_Fock_scalar_once'] and not control['reference_rotated']
                    and control['ordinary_ancilla_branch'],'P_FIRST_REPLICA_CONTROL_ORACLE')
                controls.append(dict(cell_id=task['cell']['id'],**control))
            else:c.require(not (d/'control.json').exists(),'NO_UNREGISTERED_EXTRA_CONTROL')
            workers.append(w);strata[task['cell']['id']][value['stratum']]+=1
            trajectories.append(dict(task_id=task['id'],cohort=c.COHORT,stratum=value['stratum'],
                event_digest=value['event_digest'],trajectory_sha256=c.sha(d/'trajectory.json'),
                seed_identity_sha256=c.sha(d/'confirmation_seed_identity.json')))
    c.require(len(workers)==92 and len(controls)==2 and occurrence==128 and events_count==256
        and {cid:sum(v.values()) for cid,v in strata.items()}==c.N_FIXED,'P_FIXED_TRAJECTORY_EVENT_COUNTS')
    prep=c.read(output/'preparation_receipt.json')
    c.require(prep['prefix']==1 and prep['saved_input_load']==prep['preparation']==1
        and prep['SCF_DF_state_solver_signal_reference']==0 and prep['hamiltonian_hash']==c.base.BINDING['hamiltonian']
        and all(prep['preparation_hash']==e['preparation_hash'] for e in spec['endpoints']),'P_SINGLE_PREFIX_PREPARATION')
    check_saved_accounting(result['accounting'],spec['endpoints'])
    preservation=c.PRIVATE/'P_completion_audit01';preservation.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__),preservation/'auditor.py')
    files=[p for directory in (attempt,activation,preservation) for p in sorted(directory.rglob('*')) if p.is_file()]
    for name in ('P_activation_metadata01.py','P_activation_stdout01.txt','P_activation_stderr01.txt'):
        files.append(c.PRIVATE/name)
    manifest=preservation/'completed_preservation_manifest_PRIVATE.json';c.save(manifest,[row(p,str(p.relative_to(c.PRIVATE))) for p in files])
    destination=c.PUBLIC/'P_completed_run01';destination.mkdir(exist_ok=False)
    projection={k:deepcopy(v) for k,v in result.items() if k!='measurements'}
    projection.update(scientific_result_sha256=c.sha(output/'result.json'),n_fixed=c.N_FIXED,
        statistical_rule='FIXED_N_POINT_DIRECTION_AGAINST_SAVED_D',private_seeds_and_events_published=False)
    c.save(destination/'confirmation_result.json',projection)
    c.save(destination/'paired_costs_projection.json',dict(rows=[public_pair(v) for v in result['measurements']],
        private_seeds_and_events_published=False,trajectories=92))
    c.save(destination/'controls.json',dict(rows=controls,maximum_error=max(r['max_error'] for r in controls)))
    c.save(destination/'worker_receipts.json',receipts)
    c.save(destination/'trajectory_receipts_projection.json',dict(rows=trajectories,stratum_counts={cid:dict(v) for cid,v in strata.items()},
        private_seeds_and_events_published=False,law_probabilities=None,unseen_event_cost=None,unseen_event_variance=None))
    for src,name in [(attempt/'native_exit.json','native_exit.json'),(attempt/'native_started.json','native_started.json'),
        (attempt/'completion_audit.json','driver_completion_audit.json'),(output/'terminal.json','terminal.json'),
        (output/'preparation_receipt.json','preparation_receipt.json'),(Path(grant['approval_receipt']),'approval_receipt.json'),
        (Path(spec['resource_receipt']),'resource_admission.json'),(Path(spec['seed_refresh']),'seed_refresh.json')]:
        shutil.copyfile(src,destination/name)
    c.save(destination/'execution_binding.json',dict(SOURCE=spec['SOURCE'],scope=c.SCOPE,spec_digest=c.digest(spec),
        pending_spec_sha256=spec['pending_spec_sha256'],grant_sha256=c.sha(activation/'P_grant_APPROVED_PRIVATE.json'),
        source_hashes=spec['source_hashes'],inputs=spec['inputs'],call_budget=c.CALLS,binding=c.base.BINDING,
        seed_registry_sha256=c.sha(spec['seed_registry']),seed_plan_digest=spec['seed_plan_digest'],
        runtime_manifest_sha256=spec['runtime_manifest_sha256'],runtime_files=len(c.read(spec['runtime_manifest'])),
        provider_files=len(spec['provider_files']),kernel_files=len(spec['kernel_files']),python_sha256=spec['python_sha256'],
        worker_cpus=spec['worker_cpus'],valid_grant_body_exported=False,private_seed_list_exported=False,
        activation_fixed_fields_unchanged=True,push_authorized=False))
    audit=dict(status='COMPLETE_MANDATORY_STOP',stage='P',SOURCE=spec['SOURCE'],result_sha256=c.sha(output/'result.json'),
        native_exit_code=0,same_parent_reaped=True,wait4_complete=True,worker_processes_reaped=len(workers),
        first_control_workers=2,remaining_workers=90,admitted_worker_CPUs=len(spec['worker_cpus']),
        native_wall_seconds=native['wall_seconds'],native_peak_RSS_bytes=native['peak_RSS_bytes'],
        calls_reserved=reserved,n_fixed=c.N_FIXED,comparisons=dict(Counter(r['status'] for r in result['accounting']['comparisons'])),
        stratum_counts={cid:dict(v) for cid,v in strata.items()},controls_PASS=2,maximum_control_error=max(r['max_error'] for r in controls),
        source_files=len(spec['source_hashes']),input_files=len(spec['inputs']),provider_files=len(spec['provider_files']),
        kernel_files=len(spec['kernel_files']),runtime_files=len(c.read(spec['runtime_manifest'])),
        prior_records_preserved=len(c.read(spec['prior_manifest'])),seal_records_preserved=len(c.read(c.STORE/'seal_preservation_manifest_PRIVATE.json')),
        completed_private_files=len(files),completed_manifest_sha256=c.sha(manifest),
        sampling_compile_accounting_repeated_by_auditor=False,NPZ_decoded=False,new_seed_generation=0,
        additional_sampling=0,population_mean_certified=False,actual_power_certified=False,
        law_probabilities=None,unseen_event_cost=None,unseen_event_variance=None,private_seeds_and_events_published=False,
        independent_scientific_review=False,automatic_retry=False,next_stage_authorized=False,push_performed=False)
    c.save(destination/'native_completion_audit.json',audit)
    c.save(destination/'artifact_manifest.json',dict(files=[dict(path=p.name,sha256=c.sha(p),bytes=p.stat().st_size)
        for p in sorted(destination.iterdir()) if p.is_file()],manifest_self_excluded=True))
    return audit


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(__import__('json').dumps(audit(),indent=2))


if __name__=='__main__':main()
