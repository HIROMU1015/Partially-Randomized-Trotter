"""Separate one-shot compile of a saved PASS lowering; old STOP is retained."""
from pathlib import Path
import subprocess
import h6_m5_phase_repair_contract as prior

ROOT=prior.ROOT
require,read,save,sha,digest,seal=prior.require,prior.read,prior.save,prior.sha,prior.digest,prior.seal
SCOPE='H6_M5_ALGEBRAIC_ZERO_ROUNDING_COSINE_V1'
POLICY='M5_OPT1_ALGEBRAIC_ONEQ_ZERO_ROUNDING_V1'
STAGES=('LOAD_SAVED_PASS_INPUT','COMPILE_COSINE','REPLAY_COMPILED')
AUDIT='artifacts/resource_applicability/track_a_h6_m5_phase_repair_run01/2026-10-19/end_audit.json'
PASSED_CONTROL='artifacts/resource_applicability/track_a_h6_m5_preserve_oneq_run01/2026-10-19/end_audit.json'
SOURCE_PATHS=tuple(dict.fromkeys(prior.SOURCE_PATHS+(
    AUDIT,'scripts/m5_algebraic_zero_compiler.py','scripts/h6_m5_algebraic_zero_contract.py',
    'scripts/h6_m5_algebraic_zero_native.py','scripts/run_h6_m5_algebraic_zero.py',
    'scripts/prepare_h6_m5_algebraic_zero.py','tests/test_m5_algebraic_zero_compiler.py',
    'tests/test_h6_m5_algebraic_zero.py',PASSED_CONTROL,
    'scripts/m5_preserve_oneq_compiler.py','scripts/m5_algebraic_oneq_prototype.py',
    'scripts/run_m5_algebraic_oneq_artificial.py','tests/test_m5_algebraic_oneq_prototype.py')))


def scope():
    old=prior.scope()
    return dict(scope_id=SCOPE,target=old['target'],nqubits=13,native_policy=old['native_policy'],
        compiler_policy=POLICY,maximum_new_compile=1,maximum_statevector_replays=1,
        reused_lowering_QPY=1,reused_lowered_vector=1,reused_initial_raw_gate=1,
        new_lowering=0,new_lowered_replay=0,stages=list(STAGES),
        compiler=old['compiler'],phase_sum='NO_ONEQ_GENERATED_PHASE',
        oneq_policy='EXACT_RZ_SUMS_SX_X_ONLY_NO_EULER',
        additional_angle_rounding_budget_radians=0.0,
        nonzero_angles_discarded_by_threshold=False,
        ordinary_simulator_error_bound=None,
        other_opt1_tasks='UNCHANGED_PRESET_WITH_ORIGINAL_FLOW_CONDITIONS',
        accelerated_kernel='ONEQ_OPTIMIZER_NOT_CALLED;SHARED_RUNTIME_UNCHANGED',
        small_angle_only_causal_attribution=False,
        raw_strictly_below=1e-9,numerical_checks=old['numerical_checks'],
        first_check='REUSED_PREVIOUS_PASS_SHA_BOUND',replay_precision=old['replay_precision'],
        phase_alignment_for_acceptance=False,postcompile_phase_correction=False,
        sine=0,sampling=0,SCF_DF_state_reference=0,signal=0,accepted_cost_rows=0,
        cost_accounting=False,old_cost_compatibility=False,retry=False,
        old_STOP_remains_STOP=True,old_58_axes_preserved=True,AD03='DEFERRED',
        next_phase_authorized=False,mandatory_STOP=True)


def verify_seals(cfg):
    require(cfg['plan']['SOURCE']==cfg['SOURCE'] and cfg['plan']['scope']==scope(),'EXACT_ALGEBRAIC_ZERO_SCOPE')
    require(read(cfg['plan_path'])==cfg['plan'] and sha(cfg['plan_path'])==cfg['plan_sha256'],'ALGEBRAIC_ZERO_PLAN_BYTES')
    rows=cfg['sealed_files'];sealed={r['path']:r for r in rows}
    require(len(rows)==len(sealed) and digest(rows)==cfg['plan']['closure_digest'],'ALGEBRAIC_ZERO_CLOSURE')
    for row in rows:
        p=Path(row['path']);require(str(p.resolve())==row['path'] and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'SEALED_BYTES:'+row['role'])
    for rel in SOURCE_PATHS:
        p=ROOT/rel
        require(str(p.resolve()) in sealed and p.read_bytes()==subprocess.check_output(['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT),'GIT_SOURCE:'+rel)
    control=read(ROOT/PASSED_CONTROL)
    require(control['SOURCE']=='68ee03a7cef019daa0e6cf917b6104bb645884f7' and control['scientific_status']=='SINGLE_STATE_PASS' and
        all(r['PASS'] and r['raw_error']<1e-9 for r in control['raw_gates']) and control['raw_gates'][1]['raw_error']==0,
        'FIXED_H6_NO_RESYNTHESIS_PASS_BASELINE')
    inputs=cfg['inputs']
    require(set(inputs)=={'origin_repair_config','origin_completion','origin_exit','origin_end_audit','lowered_qpy','lowered_vector','lowering_receipt','lowered_gate','initial_npz','ordinary_vector'},'EXACT_TEN_REUSE_ROLES')
    require(all(sealed.get(r['path'])==r for r in inputs.values()),'INPUTS_SEALED')
    require(digest(inputs)==cfg['plan']['inputs_digest'] and digest(cfg['runtime'])==cfg['plan']['runtime_digest'],'ALGEBRAIC_ZERO_INPUT_RUNTIME_DIGEST')
    old=read(inputs['origin_repair_config']['path']);prior.verify_seals(old);prior.consumed(old)
    require(old['SOURCE']=='02cb8b124302af334d0b02fe53213d3e5a932b78' and old['plan_sha256']=='9c70e90b354537732cc734ad5bf2a9ec5b440d49d900bd17b3006093ed2f8491' and old['runtime']==cfg['runtime'],'FIXED_ORIGIN_REPAIR')
    oldroot=Path(old['run_root']);oldout=Path(old['output'])
    expected={'origin_completion':oldroot/'completion_witness.json','origin_exit':Path(old['supervisor']['exit']),
        'origin_end_audit':ROOT/AUDIT,'lowered_qpy':oldout/'lowered.qpy','lowered_vector':oldout/'lowered_vector.npz',
        'lowering_receipt':oldout/'lowering_receipt.json','lowered_gate':oldout/'lowered_vs_saved_wrapper_vector.json'}
    for role,path in expected.items():require(inputs[role]['path']==str(path),'REUSE_PATH:'+role)
    for role in ('initial_npz','ordinary_vector'):
        require(all(inputs[role][k]==old['inputs'][role][k] for k in ('path','bytes','sha256')),'ORIGINAL_STATE_REFERENCE:'+role)
    completion=read(inputs['origin_completion']['path']);exited=read(inputs['origin_exit']['path'])
    require(completion['status']=='STOP' and completion['reason']=='REPAIR_SCIENTIFIC_OR_TECHNICAL_STOP' and completion['worker_exit']==0 and completion['worker_reaped'] and exited['native_exit']==1 and exited['reaped'],'OLD_STOP_WITH_EXIT_EVIDENCE')
    audit=read(inputs['origin_end_audit']['path'])
    require(audit['status']=='PASS_EVIDENCE_AUDIT_OF_SCIENTIFIC_STOP' and audit['SOURCE']==old['SOURCE'] and audit['plan_sha256']==old['plan_sha256'] and audit['exact_scientific_STOP']=='NUMERIC_DISAGREEMENT:compiled_vs_lowered','INDEPENDENT_ACCEPTANCE_OF_SAVED_FIRST_GATE')
    for role,file in [('lowered_qpy','lowered.qpy'),('lowered_vector','lowered_vector.npz')]:
        row=next(r for r in audit['artifacts'] if r['file']==file)
        require(all(row[k]==inputs[role][k] for k in ('bytes','sha256')),'AUDITED_ARTIFACT:'+role)
    gate=read(inputs['lowered_gate']['path'])
    require(gate=={k:v for k,v in audit['raw_gates'][0].items() if k!='executed'},'FIRST_GATE_EXACT_AUDIT_JOIN')
    require(gate['PASS'] is True and gate['raw_error']<1e-9 and gate['strictly_below']==1e-9 and gate['phase_alignment'] is False,'REUSED_RAW_GATE_PASS')
    require(cfg['pipeline_identity']==digest(dict(SOURCE=cfg['SOURCE'],scope=scope(),runtime=cfg['runtime'])),'NEW_ALGEBRAIC_ZERO_PIPELINE_IDENTITY')
    require(cfg['resources']==cfg['plan']['resources'] and not cfg['resources']['runtime_caps'] and not cfg['resources']['host_PSI_kill'],'ALGEBRAIC_ZERO_RESOURCE_POLICY')


def authorize(cfg):
    g=cfg.get('grant',{})
    require(cfg.get('authorized') is True and g.get('valid') is True,'EXPLICIT_ALGEBRAIC_ZERO_EXECUTION_APPROVAL_REQUIRED')
    expected=dict(scope_id=SCOPE,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],maximum_new_compile=1,maximum_statevector_replays=1,additional_angle_rounding_budget_radians=0.0,sine=0,sampling=0,accepted_cost_rows=0,retry=False,next_phase_authorized=False)
    require(all(g.get(k)==v for k,v in expected.items()) and g.get('grant_id'),'EXACT_NEW_ALGEBRAIC_ZERO_GRANT')
    root=Path(cfg['run_root']).resolve()
    require(str(root)==cfg['run_root']==cfg['plan']['run_root'] and not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'NEW_ALGEBRAIC_ZERO_ROOT')
    for key,name in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:require(cfg[key]==g.get(key)==str(root/name),'FIXED_ALGEBRAIC_ZERO_PATH:'+key)
    supervisor={k:str(root.parent/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    require(cfg['supervisor']==cfg['plan']['supervisor']==g.get('supervisor')==supervisor,'ALGEBRAIC_ZERO_SUPERVISOR_BINDING')
    for row in cfg['inputs'].values():require(not Path(row['path']).resolve().is_relative_to(root),'NO_OLD_INPUT_OUTPUT_OVERLAP')
    p=cfg['approval_receipt_path']
    require(p in {r['path'] for r in cfg['sealed_files']} and sha(p)==cfg['approval_receipt_sha256'] and g.get('approval_reference')==p+'#sha256='+cfg['approval_receipt_sha256'],'SEALED_ALGEBRAIC_ZERO_APPROVAL')
    approval=read(p)
    require(approval['SOURCE']==cfg['SOURCE'] and approval['scope']==scope() and approval['execution_authorized'] is True,'USER_APPROVED_ALGEBRAIC_ZERO_SCOPE')
    verify_seals(cfg)


def consumed(cfg):require(read(cfg['marker'])==dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']),'CONSUMED_ALGEBRAIC_ZERO_GRANT')


def reserve(cfg,stage):
    require(stage in STAGES,'REGISTERED_ALGEBRAIC_ZERO_STAGE');consumed(cfg)
    require({p.stem for p in Path(cfg['ledger']).glob('*.json')}==set(STAGES[:STAGES.index(stage)]),'ALGEBRAIC_ZERO_ORDER_NO_REENTRY')
    save(Path(cfg['ledger'])/(stage+'.json'),dict(stage=stage,state='ENTERED_NONREFUNDABLE',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256']))
