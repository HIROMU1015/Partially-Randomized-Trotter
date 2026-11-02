"""Saved-cosine acceptance and a separate, disabled-by-default sine contract."""
import math
from pathlib import Path
import subprocess
import h6_m5_recovery_contract as old
from h6_m5_phase_repair_contract import sha, read, save, seal, digest, require

ROOT=old.ROOT
SCOPE='H6_FULL_M5_CT01_SAVED_COSINE_ONE_SINE_V1'
POLICY=old.POLICY
COMPILER_HASH=old.COMPILER_HASH
RECOVERY_SOURCE='f743a88225f7cb2c8d2c91d7e19813a5780f803b'
PROTOCOL_COMMIT='94d18a979cdc1a390fa7e2eb3ae2f18f700b47d5'
PROTOCOL='docs/research/track_a_pr_resource_evaluation_protocol_v1.md'
TOLERANCE=1e-7
STAGES=('RESTORE_SAVED_INPUTS','COMPILE_SINE','REPLAY_COMPILED_SINE')
NEW_SOURCE_PATHS=('scripts/h6_m5_ct01_sine_contract.py','scripts/h6_m5_ct01_sine_native.py',
 'scripts/run_h6_m5_ct01_sine.py','scripts/prepare_h6_m5_ct01_sine.py','tests/test_h6_m5_ct01_sine.py',
 'scripts/run_h6_m5_saved_phase.py','scripts/h6_m5_phase_engine.py','scripts/h6_m5_phase_repair_contract.py')
SOURCE_PATHS=tuple(dict.fromkeys(old.SOURCE_PATHS+NEW_SOURCE_PATHS))


def scope():
    return dict(scope_id=SCOPE,target=old.TASK,protocol_commit=PROTOCOL_COMMIT,
      compiler=dict(qiskit='1.3.0',basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1),
      compiler_policy='ORIGINAL_PRESET_OPT1_UNMODIFIED',native_policy=POLICY,
      compile_vector_strictly_below=TOLERANCE,other_checks_strictly_below=1e-9,
      maximum_new_compile=1,maximum_compiled_state_actions=1,reused_cosine_compile=1,
      new_cosine_compile=0,initial_state_generation=0,SCF_DF=0,full_H_reference_generation=0,
      expected_sine='H_SDG_H_ON_SAVED_UNCOMPILED_COSINE_VECTOR',new_uncompiled_full_replays=0,
      new_signal_or_oracle=0,new_sampling=0,stages=list(STAGES),
      phase_alignment=False,postcompile_phase_correction=False,normalization=False,
      pilot_axis_receipts_only=True,cost_accounting=False,ranking=False,refit=False,retry=False,
      old_STOP_preserved=True,old_58_axes_preserved=True,AD03='DEFERRED',next_phase_authorized=False,mandatory_STOP=True)


def assess_saved_cosine(inputs):
    """Read saved JSON only. Do not restore arrays or execute molecular code."""
    origin=read(inputs['old_config']['path'])
    require(origin['SOURCE']==RECOVERY_SOURCE and origin['plan_sha256']=='ffe5d880ecdebdc6616a0e4a41bd3432115503e2feba16f2414a081a2ac098da','FIXED_ORIGINAL_RECOVERY')
    stop=read(inputs['old_stop']['path']);diag=read(inputs['diagnostics']['path'])
    require(stop['task']==old.TASK['id'] and stop['lane']==old.LANE and stop['status']=='STOP'
       and stop['reason']=='NUMERIC_DISAGREEMENT:compiled_vector_cosine','ORIGINAL_COSINE_STOP')
    rows=stop['checks'];names=['normalization_log_B','native_low','native_leakage','independent_occupation_vector','hadamard_cosine','compiled_vector_cosine']
    require([r['check'] for r in rows]==names,'SIX_SAVED_CHECKS')
    require(all(math.isfinite(r['error']) and 0<=r['error']<1e-9 for r in rows[:-1]),'REUSED_PRIOR_CHECKS_PASS')
    error=rows[-1]['error']
    require(error==diag['raw_error'] and math.isfinite(error) and 0<=error<TOLERANCE,'SAVED_COSINE_CT01_PASS')
    require(diag['raw_gate_PASS'] is False and error>=1e-9 and not diag['acceptance_override'],'OLD_FAIL_NOT_REWRITTEN')
    wr=read(inputs['wrapper_receipt']['path']);cr=read(inputs['compiled_receipt']['path']);vr=read(inputs['vectors_receipt']['path'])
    require(wr['SOURCE']==RECOVERY_SOURCE and wr['plan_sha256']==origin['plan_sha256'] and wr['axis']=='cosine','ORIGINAL_WRAPPER_IDENTITY')
    require(wr['wrapper_fingerprint']==cr['pretranspile_fingerprint']=='da9afa1d776bf6da829119b2f4ff888042b243ba7d4b98a037134516ed9cf83c','ORIGINAL_WRAPPER_FINGERPRINT')
    require(cr['compiler_settings_hash']==COMPILER_HASH and cr['layout']=='None','ORIGINAL_COMPILER_PROFILE')
    for role,record in [('wrapper_qpy',wr['circuit']),('compiled_cosine_qpy',cr['circuit']),('initial_npz',wr['state']),('saved_vectors_npz',vr['vectors'])]:
        require(all(inputs[role][k]==record[k] for k in ('sha256','bytes')),'SAVED_ARTIFACT_JOIN:'+role)
    require(vr['shape']==[8192] and vr['dtype']=='complex128','ORIGINAL_STATE_SHAPE')
    require(all(type(v) is int and v>=0 for v in cr['metrics'].values()) and cr['metrics']['rz_count']==831662,'SAVED_COSINE_INTEGER_METRICS')
    completion=read(inputs['old_completion']['path']);exited=read(inputs['old_exit']['path'])
    require(completion['SOURCE']==RECOVERY_SOURCE and completion['status']=='STOP' and completion['worker_exit']==0 and completion['worker_reaped'],'RECOVERY_STOP_AND_REAP')
    require(exited['native_exit']==1 and exited['reaped'],'RECOVERY_EXTERNAL_EXIT')
    return dict(status='SAVED_COSINE_ACCEPTED_UNDER_CT01',task=old.TASK['id'],axis='cosine',
      original_SOURCE=RECOVERY_SOURCE,protocol_commit=PROTOCOL_COMMIT,raw_error=error,
      old_strictly_below=1e-9,old_PASS=False,new_strictly_below=TOLERANCE,new_PASS=True,
      other_checks=rows[:-1],compiler_settings_hash=COMPILER_HASH,metrics=cr['metrics'],
      wrapper_fingerprint=wr['wrapper_fingerprint'],compiled_sha256=cr['circuit']['sha256'],
      old_STOP_preserved=True,prior_checks_reused=True,new_compile=0,new_state_action=0,
      main_table_adoption=False,cost_accounting=False,scope_id=SCOPE)


def verify_seals(cfg):
    require(cfg['plan']['scope']==scope() and cfg['plan']['SOURCE']==cfg['SOURCE'],'EXACT_CT01_SCOPE')
    require(read(cfg['plan_path'])==cfg['plan'] and sha(cfg['plan_path'])==cfg['plan_sha256'],'CT01_PLAN_BYTES')
    rows=cfg['sealed_files'];by_path={r['path']:r for r in rows}
    require(len(rows)==len(by_path) and digest(rows)==cfg['plan']['closure_digest'],'CT01_CLOSURE')
    for row in rows:
        p=Path(row['path']);require(str(p.resolve())==row['path'] and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'SEALED_BYTES:'+row['role'])
    for rel in SOURCE_PATHS:
        p=ROOT/rel;require(str(p.resolve()) in by_path and p.read_bytes()==subprocess.check_output(['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT),'COMMITTED_CT01_SOURCE:'+rel)
    require(digest(cfg['inputs'])==cfg['plan']['inputs_digest'] and all(by_path.get(r['path'])==r for r in cfg['inputs'].values()),'CT01_INPUTS')
    origin=read(cfg['inputs']['old_config']['path']);old.verify_seals(origin);old.consumed(origin)
    run=Path(origin['run_root']);task=Path(origin['output'])/old.TASK['id'];q=task/'quarantine/cosine'
    expected={'old_stop':task/'STOP.json','old_completion':run/'completion_witness.json',
      'old_exit':ROOT/'artifacts/resource_applicability/track_a_h6_m5_full_recovery_run01/2026-10-19/external_exit_witness.json',
      'saved_signal':task/'signal.json',**{k:q/name for k,name in (
      ('wrapper_qpy','wrapper.qpy'),('compiled_cosine_qpy','compiled.qpy'),('initial_npz','initial.npz'),
      ('saved_vectors_npz','vectors.npz'),('wrapper_receipt','wrapper_receipt.json'),
      ('compiled_receipt','compiled_receipt.json'),('vectors_receipt','vectors_receipt.json'),('diagnostics','diagnostics.json'))}}
    require(set(cfg['inputs'])==set(expected)|{'old_config'},'EXACT_CT01_INPUT_ROLES')
    for role,p in expected.items():require(cfg['inputs'][role]['path']==str(p),'ORIGINAL_INPUT_PATH:'+role)
    require(cfg['runtime']==origin['runtime'] and cfg['provider_root']==origin['provider_root'],'SAME_ORIGINAL_RUNTIME_PROVIDER')
    require(cfg['pipeline_identity']==digest(dict(SOURCE=cfg['SOURCE'],scope=scope(),runtime=cfg['runtime'])),'CT01_PIPELINE_IDENTITY')
    require(cfg['plan']['runtime_digest']==digest(cfg['runtime']),'CT01_RUNTIME_DIGEST')
    require(cfg['pipeline_identity']==cfg['plan']['pipeline_identity'] and cfg['run_root']==cfg['plan']['run_root'],'CT01_PLAN_BINDING')
    protocol=subprocess.check_output(['git','show',PROTOCOL_COMMIT+':'+PROTOCOL],cwd=ROOT)
    import hashlib
    require(hashlib.sha256(protocol).hexdigest()==cfg['plan']['protocol_sha256'] and b'norm(after-before) < 1e-7' in protocol,'ADOPTED_CT01_PROTOCOL_BYTES')
    require(read(cfg['cosine_acceptance_path'])==cfg['cosine_acceptance']==assess_saved_cosine(cfg['inputs'])
      and sha(cfg['cosine_acceptance_path'])==cfg['plan']['cosine_acceptance_sha256'],'SAVED_COSINE_ACCEPTANCE_BINDING')
    require(cfg['resources']==cfg['plan']['resources'] and cfg['resources']['runtime_caps'] is False and cfg['resources']['host_PSI_kill'] is False,'CT01_RESOURCE_POLICY')
    require(cfg['provider_root']==cfg['runtime']['provider_root'],'CT01_PROVIDER_BINDING')


def authorize(cfg):
    g=cfg.get('grant',{})
    require(cfg.get('authorized') is True and g.get('valid') is True,'EXPLICIT_CT01_SINE_EXECUTION_APPROVAL_REQUIRED')
    expected=dict(scope_id=SCOPE,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],
      maximum_new_compile=1,maximum_compiled_state_actions=1,new_cosine_compile=0,new_sampling=0,
      compile_vector_strictly_below=TOLERANCE,other_checks_strictly_below=1e-9,retry=False,next_phase_authorized=False)
    require(all(g.get(k)==v for k,v in expected.items()) and g.get('grant_id'),'EXACT_CT01_SINE_GRANT')
    root=Path(cfg['run_root']).resolve()
    require(str(root)==cfg['run_root']==cfg['plan']['run_root'] and not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'NEW_CT01_RUN_ROOT')
    for k,name in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:require(cfg[k]==g.get(k)==str(root/name),'CT01_PATH:'+k)
    expected_supervisor={k:str(root.parent/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    require(cfg['supervisor']==cfg['plan']['supervisor']==g.get('supervisor')==expected_supervisor,'CT01_SUPERVISOR')
    for r in cfg['inputs'].values():require(not Path(r['path']).is_relative_to(root),'NO_OLD_INPUT_OUTPUT_OVERLAP')
    p=cfg.get('approval_receipt_path');require(p and p in {r['path'] for r in cfg['sealed_files']},'NEW_CT01_APPROVAL_REQUIRED')
    require(sha(p)==cfg['approval_receipt_sha256']==cfg['plan']['approval_receipt_sha256'] and g.get('approval_reference')==p+'#sha256='+cfg['approval_receipt_sha256'],'CT01_APPROVAL_REFERENCE')
    a=read(p);require(a['execution_authorized'] is True and a['SOURCE']==cfg['SOURCE'] and a['scope']==scope(),'USER_APPROVED_CT01_SINE')
    verify_seals(cfg)


def consumed(cfg):require(read(cfg['marker'])==dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']),'CONSUMED_CT01_GRANT')


def reserve(cfg,stage):
    require(stage in STAGES,'REGISTERED_CT01_STAGE');consumed(cfg)
    require({p.stem for p in Path(cfg['ledger']).glob('*.json')}==set(STAGES[:STAGES.index(stage)]),'CT01_STAGE_ORDER_NO_REENTRY')
    save(Path(cfg['ledger'])/(stage+'.json'),dict(stage=stage,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],state='ENTERED_NONREFUNDABLE'))
