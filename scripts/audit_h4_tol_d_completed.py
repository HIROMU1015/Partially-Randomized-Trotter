"""Audit a finished DF10 D run and preserve bytes. Never dispatch science."""
import gzip
import json
import math
import os
from pathlib import Path
from fractions import Fraction

import h4_tol_main_contract as c
import seal_h4_tol_d as s

PUBLIC=c.ROOT/'artifacts/resource_applicability/track_a_h4_tol_d_completed/2026-10-19'


def owned_receipt(directory, expected):
    receipt=c.read(directory/'accepted_native_receipt.json')
    c.require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors'],'OWNED_NATIVE_EXIT')
    c.require({x['task_id'] for x in receipt['workers']}==set(expected)
        and len(receipt['workers'])==len(expected),'EXACT_NATIVE_TASK_SET')
    sanitized=[]
    for worker in receipt['workers']:
        task=worker['task_id']; work=directory/task
        c.require(Path(worker['directory'])==work and worker['exit_code']==0 and worker['reaped'],'SAME_PARENT_NATIVE_WORKER')
        c.require(c.sha(work/'task_result.json')==worker['result_sha256'],'NATIVE_RESULT_SHA')
        c.require(not (work/'task_STOP.json').exists(),'NO_WORKER_STOP')
        sanitized.append({k:worker[k] for k in ('task_id','cpu','exit_code','reaped','result_sha256','cpu_seconds','peak_RSS_bytes')})
    return sanitized


def files(directory):
    # The one known run only; disposable runtime caches are not scientific data.
    excluded={'scratch','tmpdir','numba_cache_dir','mplconfigdir','xdg_cache_home'}
    for parent,dirs,names in os.walk(directory):
        dirs[:]=[d for d in dirs if d not in excluded]
        for name in sorted(names): yield Path(parent)/name


def audit():
    contract_path=s.PRIVATE/'D_contract_EXECUTION_PRIVATE.json';grant_path=s.PRIVATE/'D_grant_APPROVED_PRIVATE.json'
    contract=c.read(contract_path);grant=c.read(grant_path);output=Path(contract['output']);run=output.parent
    terminal=c.read(output/'terminal.json');native=c.read(run/'native_exit.json');result=c.read(output/'result.json')
    c.require(contract['status']=='SEALED_FOR_EXECUTION' and not contract['pending'],'ACTIVATED_CONTRACT')
    c.require(contract['SOURCE']==grant['SOURCE']==terminal['SOURCE']==native['SOURCE']==s.SCIENCE_SOURCE,'SOURCE_CLOSURE_IDENTITY')
    c.require(c.digest(contract)==grant['contract_digest']==c.read(run/'ONE_SHOT.json')['contract_digest'],'ONE_SHOT_CONTRACT')
    c.require(grant['approved_by_user'] and grant['runtime_authorization'] and grant['one_shot']
        and not grant['automatic_retry'] and not grant['next_stage_authorized'],'EXPLICIT_D_APPROVAL')
    c.require(c.sha(grant['approval_receipt'])==grant['approval_receipt_sha256'],'APPROVAL_RECEIPT')
    c.require(native['exit_code']==0 and native['same_parent_reaped']
        and native['result_sha256']==c.sha(output/'result.json'),'NATIVE_DRIVER_ACCEPTANCE')
    c.require(terminal['status']=='COMPLETE_MANDATORY_STOP' and terminal['own_workers_reaped']
        and not (output/'STOP.json').exists(),'NO_TECHNICAL_STOP')
    c.require(contract['plan']==c.plan('D') and result['binding']==c.BINDING
        and result['stage']=='D' and result['SOURCE']==s.SCIENCE_SOURCE
        and result['status']=='COMPLETE_MANDATORY_STOP' and result['registered_set_complete'],'COMPLETE_D_SCOPE')
    draft=c.read(s.OLD_PRIVATE/'D_contract_DRAFT_PRIVATE.json')
    c.require(all(contract[k]==draft[k] for k in s.IMMUTABLE),'UNCHANGED_FIXED_SCIENCE')
    counts=s.verify_bytes(contract)
    preserved=c.read(s.OLD_PRIVATE/'preservation_manifest_PRIVATE.json')
    for row in preserved:
        for path in {row['path'],row['copy']}:c.require(c.sha(path)==row['sha256'],'PRIOR_RECORD_PRESERVED')
    signals=c.read(output/'signals.json')['signals'];registered=c.registrations()['D']
    eligible=c.selected(signals,registered);by_id={x['cell']['id']:x for x in signals}
    fp=c.read(contract['FP_evidence']);saved={x['cell']['id']:x for x in fp['signals']}
    for cid in c.FP_SIGNALS_D:c.require(by_id[cid]==saved[cid],'UNCHANGED_REUSED_FP_SIGNAL')
    new_signal={x['id'] for x in registered if x['method']!='B0I' and x['id'] not in saved}
    signal_workers=owned_receipt(output/'signal_workers',new_signal)
    for cid in new_signal:
        c.require(c.read(output/'signal_workers'/cid/'task_result.json')==by_id[cid],'COMPLETE_SIGNAL_VALUE')
        c.require(c.read(output/'signal_workers'/cid/'signal_reserved.json')['nonrefundable'],'SIGNAL_RESERVATION')
    overlay=[x for x in registered if x['method']=='B0I' and x['id'] not in saved]
    c.require(len(new_signal)==96 and len(overlay)==88,'FIXED_NEW_SIGNAL_OVERLAY_COUNTS')
    for cell in overlay:c.require(c.read(output/(cell['id']+'_overlay_reserved.json'))['nonrefundable'],'OVERLAY_RESERVATION')
    primitive=c.read(output/'primitive_validation.json');preparations=c.read(output/'preparations.json')
    c.require(primitive['actual_time_count']*primitive['probe_count']==1095
        and primitive['all_registered_times'] and primitive['includes_negative_yoshida_times'],'COMPLETE_PRIMITIVE_COVERAGE')
    c.require(preparations['reconstructions']==10 and preparations['new_reference']==0
        and preparations['SCF_DF_state_solver']==0,'NO_SAVED_INPUT_REGENERATION')
    expected_compile={x['id'] for x in eligible if x['id'] not in c.FP_DETERMINISTIC}
    compile_workers=owned_receipt(output/'compile_workers',expected_compile) if expected_compile else []
    measured=c.read(output/'paired_measurements.json')['rows']
    c.require({r['cell_id'] for r in measured}==expected_compile and len(measured)==len(expected_compile),'COMPLETE_COMPILED_DOMAIN')
    units={};controls=[]
    for row in measured:
        cid=row['cell_id'];work=output/'compile_workers'/cid
        c.require(row==c.read(work/'task_result.json') and row['replica']==0 and row['control_probes']==4,'COMPILE_TASK_VALUE')
        control=c.read(work/'control.json');c.require(control['status']=='PASS' and control['probes']==4
            and math.isfinite(control['max_error']) and control['max_error']<1e-9,'CONTROL_SCALAR_GATE')
        controls.append(dict(cell_id=cid,**control))
        c.require(c.read(work/'build_reserved.json')['nonrefundable'],'NATIVE_BUILD_RESERVATION')
        c.require({a['axis'] for a in row['rows']}==set(c.old.AXES) and len(row['rows'])==2,'PAIRED_FULL_WRAPPERS')
        units[cid]={}
        for axis in row['rows']:
            a=axis['axis'];reservation=c.read(work/('compile_'+a+'_reserved.json'))
            c.require(reservation['nonrefundable'] and reservation['fingerprint']==axis['fingerprint'],'BEFORE_COMPILE_RESERVATION')
            c.require(c.read(work/('measured_'+a+'.json'))==axis,'PER_AXIS_RAW_VALUE')
            c.require(axis['compiler_hash']==c.BINDING['compiler'] and len(axis['fingerprint'])==64,'FROZEN_COMPILE_IDENTITY')
            c.require(type(axis['metrics']['RZ']) is int and axis['metrics']['RZ']>0,'INTEGER_COMPILED_RZ')
            units[cid][a]=axis['metrics']['RZ']
    for row in fp['measurements']:
        if row['cell_id'] in c.FP_DETERMINISTIC:
            units[row['cell_id']]={a['axis']:a['metrics']['RZ'] for a in row['rows']}
    rows=result['rows'];c.require(len(rows)==950 and len({(x['cell_id'],x['epsilon']) for x in rows})==950,'ALL_190_X_5_ENDPOINTS')
    for row in rows:
        ep=by_id[row['cell_id']]['precision_rows'][str(row['epsilon'])]['1.0']
        c.require(row['eligible']==ep['cost_acquisition_eligible'],'RECORDED_ELIGIBILITY')
        if row['eligible']:
            # Integrity equation on the saved result. No second accounting dispatch.
            shots={a:ep['axes'][b]['shots'] for a,b in zip(c.old.AXES,('real','imag'))}
            c.require(row['n']==1 and row['shots']==shots and Fraction(row['G_RZ_exact'])
                ==sum(shots[a]*units[row['cell_id']][a] for a in c.old.AXES),'SAVED_COST_INTEGRITY')
        else:c.require(row['G_RZ_exact'] is None,'INELIGIBLE_NOT_ZERO_FILLED')
    c.require(len(result['winners'])==5 and {w['epsilon'] for w in result['winners']}==set(c.EPS),'FIVE_FIXED_EPSILONS')
    for winner in result['winners']:
        available=[r for r in rows if r['epsilon']==winner['epsilon'] and r['eligible']]
        best=winner['best']
        c.require((not available and best is None) or best in available,'RECORDED_WINNER_MEMBERSHIP')
        if best is not None:
            c.require(all((Fraction(best['G_RZ_exact']),best['cell_id'])<=(Fraction(r['G_RZ_exact']),r['cell_id'])
                for r in available),'REGISTERED_MINIMUM_AND_FIXED_TIES')
    n=len(measured);c.require(n<=186 and result['actual_new_base_signals']==96
        and result['actual_new_compiled_trajectories']==n and result['actual_new_wrapper_compiles']==2*n
        and result['endpoint_accounting_pass']==1 and result['new_SCF_DF_state_reference']==0,'ACTUAL_SCIENTIFIC_BUDGET')
    c.require(not result['next_stage_authorized'] and not result['population_mean_certified'],'MANDATORY_STOP_LIMITS')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    (PUBLIC/'result.json').write_bytes((output/'result.json').read_bytes())
    for name in ('signals.json','paired_measurements.json'):
        (PUBLIC/(name+'.gz')).write_bytes(gzip.compress((output/name).read_bytes(),mtime=0))
    c.save(PUBLIC/'native_exit.json',native)
    c.save(PUBLIC/'terminal.json',terminal)
    c.save(PUBLIC/'primitive_validation.json',primitive)
    c.save(PUBLIC/'control_validation.json',dict(configurations=n,probes=4*n,all_PASS=True,rows=controls))
    c.save(PUBLIC/'worker_receipts.json',dict(signal_workers=signal_workers,compile_workers=compile_workers,all_exit0_reaped=True))
    c.save(PUBLIC/'approval_receipt.json',c.read(grant['approval_receipt']))
    c.save(PUBLIC/'execution_binding.json',dict(SOURCE=s.SCIENCE_SOURCE,counts=counts,
        contract_digest=c.digest(contract),grant_sha256=c.sha(grant_path),approval_receipt_sha256=grant['approval_receipt_sha256'],
        fresh_resource_receipt_sha256=contract['fresh_resources']['sha256'],worker_cpus=contract['worker_cpus'],
        scientific_conditions_changed=False,one_shot=True,automatic_retry=False))
    new_records=[]
    for path in files(run):new_records.append(dict(path=str(path),sha256=c.sha(path),bytes=path.stat().st_size))
    for path in sorted(s.PRIVATE.glob('*.json')):
        if path.name!='completed_D_preservation_manifest_PRIVATE.json':
            new_records.append(dict(path=str(path),sha256=c.sha(path),bytes=path.stat().st_size))
    c.save(s.PRIVATE/'completed_D_preservation_manifest_PRIVATE.json',new_records)
    receipt=dict(status='COMPLETE_MANDATORY_STOP',SOURCE=s.SCIENCE_SOURCE,result_sha256=c.sha(output/'result.json'),
        registered_candidates=190,endpoint_rows=950,epsilons=c.EPS,new_base_signals=96,new_analytic_overlays=88,
        reused_FP_signals=6,reused_FP_deterministic_configurations=4,reused_FP_axes=8,
        new_compiled_configurations=n,new_wrapper_compile_axes=2*n,control_probes=4*n,
        primitive_probes=1095,preparation_reconstructions=10,new_SCF_integral_DF_saved_state_full_reference=0,
        worker_pool_maximum=len(contract['worker_cpus']),native_worker_processes=len(signal_workers)+len(compile_workers),
        native_exit_code=0,all_workers_reaped=True,technical_interruptions=0,retries=0,
        endpoint_accounting_pass=1,strict_bias_certificate=None,population_mean_certified=False,
        old_preservation_records=len(preserved),new_preservation_records=len(new_records),
        private_preservation_manifest_sha256=c.sha(s.PRIVATE/'completed_D_preservation_manifest_PRIVATE.json'),
        wall_seconds=native['wall_seconds'],A_status='PENDING_FORMAL_D_ACCEPTANCE_AND_NEXT_INSTRUCTION',
        A_seed_assignment=False,A_scientific_execution=False,next_stage_authorized=False,
        author_checks_only=True,mandatory_STOP=True)
    c.save(PUBLIC/'completion_audit.json',receipt)
    c.save(PUBLIC/'artifact_manifest.json',dict(files={p.name:c.sha(p) for p in sorted(PUBLIC.iterdir())
        if p.is_file() and p.name!='artifact_manifest.json'},scientific_dispatches=1,additional_scientific_dispatches_in_audit=0))
    return receipt


if __name__=='__main__':print(json.dumps(audit()))
