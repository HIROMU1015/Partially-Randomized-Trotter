"""H2 D_run02: saved signals, fresh deterministic costs, one authorized attempt."""
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h2_panels_contract as old
from prepare_h2_panels import row, preservation

require,read,save,sha,digest,owned=old.require,old.read,old.save,old.sha,old.digest,old.owned
ROOT,PRIVATE,PUBLIC=old.ROOT,old.PRIVATE,old.PUBLIC
SCOPE='H2_D_RECOVERY_V1_20261019'
OLD_SOURCE='de72e5d4e48177734c0ddd79dffac14222410658'
STORE=PRIVATE/'d_recovery01'
EVIDENCE=PUBLIC/'D_recovery_preparation01'
CALLS=dict(SCF=0,integral_build=0,DF=0,saved_state_solver=0,quantum_shots=0,
    saved_input_load=1,preparation=2,full_reference_pair=0,truncated_reference_pair=0,
    primitive_probe=0,primitive_spectrum=0,base_signal=0,independent_forward_signal=0,
    B0I_overlay=0,new_signed_shot_endpoints=0,trajectory=0,occurrence=0,RTE_event=0,
    native_evolution=28,wrapper_compile=56,control_probe=112,independent_event_oracle=0,
    endpoint_accounting_pass=1)
ENTRY=('scripts/h2_d_recovery_contract.py','scripts/h2_d_recovery_science.py',
       'scripts/run_h2_d_recovery.py','scripts/prepare_h2_d_recovery.py',
       'scripts/audit_h2_d_recovery_completed.py')


def cells():
    return [x for x in old.registrations()['D'] if
        (x['method'] in ('B0','B0I') and x['prefix']==2) or (x['method']=='B1' and x['q']>1)]


def plan():
    return dict(scope=SCOPE,stage='D',original_SOURCE=OLD_SOURCE,
        original_D_plan=old.plan('D'),recompile_cells=cells(),calls=CALLS,
        saved_signal_records=70,saved_FP_deterministic_pairs=6,
        unconfirmed_D_compile_pairs_reused=0,old_compile_reservations_nonrefundable=56,
        new_compile_reservations_maximum=56,cumulative_compile_reservations_maximum=112,
        new_signal_shot_reference_solver_calls=0,
        source_change='H2-only adapter for accepted saved D signals and new attempt; frozen kernels unchanged',
        disconnect_remediation=False,automatic_retry=False,next_stage_authorized=False,mandatory_STOP=True)


def source_closure():
    found={};pending=list(ENTRY)
    while pending:
        rel=pending.pop()
        if rel in found:continue
        path=ROOT/rel;found[rel]=sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[x.name for x in node.names] if isinstance(node,ast.Import) else (
                [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for module in modules:
                candidate='scripts/'+module.split('.')[0]+'.py'
                if (ROOT/candidate).is_file():pending.append(candidate)
    return found


def verify_hashes(rows):
    require(len({r['role'] for r in rows})==len(rows),'UNIQUE_INPUT_ROLES')
    for r in rows:
        p=owned(r['path'])
        require(p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'SAVED_BYTES_CHANGED:'+r['role'])


def verify_saved_signals(signals,fp,expected_preps,signal_receipt):
    records=signals['signals'];grid=old.registrations()['D']
    require([r['cell'] for r in records]==grid,'EXACT_D70_SIGNALS')
    selected={x['id'] for x in old.selected(records,grid)}
    fpids={r['cell_id'] for r in fp['measurements'] if r['cell_id'] in {x['id'] for x in grid}}
    require(len(fpids)==6 and selected-fpids=={x['id'] for x in cells()},'SAVED_ACQUISITION_PARTITION')
    original={r['cell']['id']:r for r in fp['signals']}
    byid={r['cell']['id']:r for r in records}
    require(set(signals['reused_signal_ids'])==fpids and all(byid[k]==original[k] for k in fpids),'FP_SIGNALS_UNCHANGED')
    require(signal_receipt['status']=='PASS' and signal_receipt['all_reaped']
        and not signal_receipt['cleanup_errors'] and len(signal_receipt['workers'])==36,'REAPED_SIGNAL_PROVENANCE')
    for worker in signal_receipt['workers']:
        p=owned(worker['directory'])/'task_result.json';value=read(p)
        require(worker['exit_code']==0 and worker['reaped'] and sha(p)==worker['result_sha256']
            and value==byid[value['cell']['id']],'SAVED_SIGNAL_WORKER_IDENTITY')
    require(expected_preps['binding']==old.BINDING,'OLD_PREPARATION_INPUT_BINDING')
    for x in cells():
        require(x['id'].replace('B0I_','B0_',1) in expected_preps['rows'],'SAVED_PREPARATION_ROW')


def verify_static(spec,*,runtime=True,interpreter=False):
    require(spec['scope']==SCOPE and spec['stage']=='D' and spec['plan']==plan()
        and spec['calls']==CALLS,'FIXED_RECOVERY_SCOPE_GRID_BUDGET')
    require(spec['source_hashes']==source_closure(),'EXACT_RECOVERY_SOURCE_CLOSURE')
    for rel,h in spec['source_hashes'].items():
        require(sha(ROOT/rel)==h,'SOURCE_BYTES_CHANGED:'+rel)
        blob=subprocess.check_output(['git','--no-optional-locks','show',spec['SOURCE']+':'+rel],cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest()==h,'SOURCE_GIT_IDENTITY:'+rel)
    verify_hashes(spec['inputs'])
    paths={r['path'] for r in spec['inputs']}
    require({spec[k] for k in ('base_spec','saved_signals','saved_preparations','signal_receipt','FP_result',
        'old_STOP','old_preservation_manifest','FP_preservation_manifest','input_acceptance')}<=paths,'ACTUALLY_USED_INPUT_BINDING')
    base=read(spec['base_spec']);require(base['SOURCE']==OLD_SOURCE and base['stage']=='D','ORIGINAL_D_SOURCE')
    old.validate_metadata(base)
    require(spec['native_config']==base['native_config'],'SAME_SAVED_F_INPUT_COMPILER')
    stop=read(spec['old_STOP'])
    require(stop['status']=='STOP_NATIVE_COMPLETION_UNCONFIRMED'
        and stop['reserved_calls']['wrapper_compile']==56 and stop['final_accounting_pass']==0,'OLD_STOP_CHARGES_PRESERVED')
    for manifest in ('old_preservation_manifest','FP_preservation_manifest'):
        for r in read(spec[manifest]):require(sha(r['path'])==r['sha256'],'ORIGINAL_ARTIFACT_CHANGED')
    require(preservation()['records_checked']==295,'F_PRESERVATION')
    verify_saved_signals(read(spec['saved_signals']),read(spec['FP_result']),read(spec['saved_preparations']),read(spec['signal_receipt']))
    accepted=read(spec['input_acceptance'])
    require(accepted['accepted_by_user'] is True and accepted['signal_sha256']==sha(spec['saved_signals'])
        and accepted['unconfirmed_compile_pairs_accepted']==0 and accepted['verbatim_user_message'], 'SAVED_SIGNAL_ACCEPTANCE')
    require(owned(spec['attempt'])==PRIVATE/'attempts/D_run02','FRESH_D_RUN02_ONLY')
    if runtime:
        for path,h in read(base['runtime_manifest']).items():require(sha(path)==h,'RUNTIME_CHANGED:'+path)
    require(sha(base['python'])==base['python_sha256'] and spec['python']==base['python'],'FIXED_INTERPRETER_BYTES')
    if interpreter:
        require(Path(sys.executable).absolute()==Path(spec['python']).absolute(),'PINNED_PYTHON_REQUIRED')
        require(sys.pycache_prefix==str(PRIVATE/'panels_runtime_pycache'),'PRIVATE_PYCACHE_REQUIRED')


def authorize(spec,grant,*,runtime=True,interpreter=True):
    require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
        'EXPLICIT_D_RECOVERY_APPROVAL_REQUIRED')
    require(grant.get('scope')==SCOPE and grant.get('stage')=='D','D_RECOVERY_ONLY')
    require(grant.get('one_shot') is True and grant.get('automatic_retry') is False
        and grant.get('next_stage_authorized') is False,'ONE_SHOT_STOP_REQUIRED')
    require(spec.get('status')=='SEALED_FOR_EXECUTION' and spec.get('pending')==[], 'RECOVERY_PENDING')
    require(grant.get('SOURCE')==spec.get('SOURCE') and grant.get('spec_digest')==digest(spec),'GRANT_BINDING')
    approval=owned(grant['approval_receipt']);require(sha(approval)==grant['approval_receipt_sha256'],'USER_APPROVAL_BYTES')
    receipt=read(approval)
    require(receipt.get('role')=='user' and receipt.get('stage')=='D' and receipt.get('SOURCE')==spec['SOURCE']
        and receipt.get('plan_digest')==digest(plan()) and receipt.get('verbatim_user_message'), 'SCOPED_USER_APPROVAL')
    pending=owned(spec['pending_spec']);require(sha(pending)==spec['pending_spec_sha256'],'PENDING_SEAL_PRESERVED')
    original=read(pending)
    extra={'status','pending','pending_spec','pending_spec_sha256','worker_cpus','resource_receipt','resource_receipt_sha256'}
    require({k:v for k,v in spec.items() if k not in extra}=={k:v for k,v in original.items() if k not in extra},'ACTIVATION_MAY_NOT_CHANGE_SEAL')
    verify_static(spec,runtime=runtime,interpreter=interpreter)
    cpus=spec['worker_cpus'];require(cpus and len(set(cpus))==len(cpus) and set(cpus)<=os.sched_getaffinity(0),'FRESH_ALLOWED_CPUS')
    resource=owned(spec['resource_receipt']);require(sha(resource)==spec['resource_receipt_sha256']
        and read(resource)['admitted'] is True,'FRESH_NONINTERFERENCE_ADMISSION')


def invalid_grant():
    return dict(scope=SCOPE,stage='D',approved_by_user=False,runtime_authorization=False,
        one_shot=True,automatic_retry=False,next_stage_authorized=False)
