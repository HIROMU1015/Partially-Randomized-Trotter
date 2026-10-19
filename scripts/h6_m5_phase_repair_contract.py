"""Stdlib-only authorization for one isolated native phase-repair pilot."""
from pathlib import Path
import subprocess
import h6_m5_phase_ledger_contract as prior

ROOT = prior.ROOT
require, read, save, sha, digest, seal = prior.require, prior.read, prior.save, prior.sha, prior.digest, prior.seal
SCOPE = 'H6_M5_NATIVE_PHASE_REPAIR_COSINE_PILOT_V1'
POLICY = 'M5_DF_EXPLICIT_BASIS_PHASE_ONCE_V1'
DESIGN = 'artifacts/resource_applicability/track_a_h6_m5_phase_repair_design/2026-10-19/pilot_plan_DRAFT.json'
STAGES = ('LOAD_SAVED_INPUT', 'LOWER_NATIVE', 'REPLAY_LOWERED', 'COMPILE_COSINE', 'REPLAY_COMPILED')
SOURCE_PATHS = tuple(dict.fromkeys(prior.SOURCE_PATHS + (
    DESIGN, 'scripts/m5_phase_repair_prototype.py',
    'scripts/h6_m5_phase_repair_contract.py', 'scripts/h6_m5_phase_repair_native.py',
    'scripts/run_h6_m5_phase_repair.py', 'scripts/prepare_h6_m5_phase_repair.py',
)))


def scope():
    return dict(scope_id=SCOPE, target=read(ROOT/DESIGN)['target'], nqubits=13,
        native_policy=POLICY, maximum_new_compile=1, maximum_statevector_replays=2,
        stages=list(STAGES), compiler=dict(basis_gates=['rz','sx','x','cx'],
            optimization_level=1, seed_transpiler=17, num_processes=1, qiskit='1.3.0'),
        replay_precision='ORIGINAL_BINARY64_STATEVECTOR_GLOBAL_PHASE_ONCE',
        numerical_checks=['lowered_vs_saved_wrapper_vector','compiled_vs_lowered','compiled_vs_saved_wrapper_vector'],
        all_finite=True, raw_strictly_below=1e-9, sine=0, sampling=0,
        state_generation=0, SCF_DF=0, signal=0, cost_accounting=False,
        accepted_cost_rows=0, old_cost_compatibility=False, retry=False,
        phase_alignment_for_acceptance=False, postcompile_phase_correction=False,
        old_STOP_remains_STOP=True, old_58_axes_preserved=True,
        AD03='DEFERRED', next_phase_authorized=False, mandatory_STOP=True)


def verify_seals(cfg):
    require(cfg['plan']['scope']==scope() and cfg['plan']['SOURCE']==cfg['SOURCE'], 'FIXED_REPAIR_SCOPE')
    require(read(cfg['plan_path'])==cfg['plan'] and sha(cfg['plan_path'])==cfg['plan_sha256'], 'REPAIR_PLAN_BYTES')
    rows=cfg['sealed_files']; sealed={r['path']:r for r in rows}
    require(len(sealed)==len(rows) and digest(rows)==cfg['plan']['closure_digest'], 'UNIQUE_REPAIR_CLOSURE')
    for row in rows:
        p=Path(row['path'])
        require(str(p.resolve())==row['path'] and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'], 'SEALED_BYTES:'+row['role'])
    for rel in SOURCE_PATHS:
        p=ROOT/rel
        require(str(p.resolve()) in sealed and p.read_bytes()==subprocess.check_output(['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT), 'GIT_SOURCE:'+rel)
    require(set(cfg['inputs'])=={'origin_ledger_config','wrapper_qpy','initial_npz','saved_vectors_npz','ordinary_result','ordinary_vector','static_metadata','recovery_exit'}, 'EXACT_REPAIR_INPUT_ROLES')
    for role,row in cfg['inputs'].items():
        require(sealed.get(row['path'])==row, 'SEALED_INPUT:'+role)
    require(digest(cfg['inputs'])==cfg['plan']['inputs_digest'] and digest(cfg['runtime'])==cfg['plan']['runtime_digest'], 'REPAIR_BINDINGS')
    origin=read(cfg['inputs']['origin_ledger_config']['path']); prior.verify_seals(origin)
    require(origin['SOURCE']=='f0022cec13be6f959b1a7b9425a9ba55657542fa' and cfg['runtime']==origin['runtime'], 'PRIOR_LEDGER_RUNTIME')
    replay=read(origin['inputs']['origin_config']['path'])
    for new,old in [('wrapper_qpy','wrapper_qpy'),('initial_npz','initial_npz'),('saved_vectors_npz','vectors_npz'),('recovery_exit','saved_exit')]:
        require(all(cfg['inputs'][new][k]==replay['inputs'][old][k] for k in ('path','bytes','sha256')), 'ORIGIN_JOIN:'+new)
    rr=Path(replay['output'])/'WRAPPER_ORDINARY'
    require(Path(cfg['inputs']['ordinary_result']['path'])==rr/'result.json' and Path(cfg['inputs']['ordinary_vector']['path'])==rr/'vector.npz', 'FIXED_ORDINARY_RESULT_PATHS')
    result=read(rr/'result.json')
    require(result['SOURCE']==replay['SOURCE'] and result['ordinary_reproduced'] is True and result['ordinary_saved_vector_error']==0., 'EXACT_SAVED_WRAPPER_REPRODUCTION')
    require(all(result['vector'][k]==cfg['inputs']['ordinary_vector'][k] for k in ('bytes','sha256')), 'SAVED_VECTOR_RECEIPT_JOIN')
    require(cfg['inputs']['static_metadata']['path']==str(prior.STATIC_ROOT/'saved_circuit_metadata.json'), 'FIXED_STATIC_METADATA')
    require(cfg['pipeline_identity']==digest(dict(SOURCE=cfg['SOURCE'],scope=scope(),runtime=cfg['runtime'])), 'NEW_PIPELINE_IDENTITY')


def authorize(cfg):
    g=cfg.get('grant',{})
    require(cfg.get('authorized') is True and g.get('valid') is True, 'EXPLICIT_H6_REPAIR_EXECUTION_APPROVAL_REQUIRED')
    expected=dict(scope_id=SCOPE,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
        pipeline_identity=cfg['pipeline_identity'],maximum_new_compile=1,
        maximum_statevector_replays=2,sine=0,sampling=0,retry=False,
        accepted_cost_rows=0,next_phase_authorized=False)
    require(all(g.get(k)==v for k,v in expected.items()) and g.get('grant_id'), 'EXACT_NEW_REPAIR_GRANT')
    root=Path(cfg['run_root']).resolve()
    require(str(root)==cfg['run_root']==cfg['plan']['run_root'] and not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root), 'NEW_REPAIR_ROOT')
    for k,name in [('output','output'),('ledger','ledger'),('marker','CONSUMED.json')]:
        require(cfg[k]==g.get(k)==str(root/name), 'FIXED_REPAIR_PATH:'+k)
    supervisor={k:str(root.parent/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    require(cfg['supervisor']==cfg['plan']['supervisor']==g.get('supervisor')==supervisor, 'REPAIR_SUPERVISOR_BINDING')
    require(cfg['resources']==cfg['plan']['resources'], 'RESOURCE_PLANNING_BINDING')
    for row in cfg['inputs'].values():
        p=Path(row['path']).resolve()
        require(not p.is_relative_to(root) and not root.is_relative_to(p.parent), 'NO_OLD_INPUT_OUTPUT_OVERLAP')
    p=cfg['approval_receipt_path']; require(p in {r['path'] for r in cfg['sealed_files']}, 'APPROVAL_SEALED')
    require(sha(p)==cfg['approval_receipt_sha256'] and g.get('approval_reference')==p+'#sha256='+cfg['approval_receipt_sha256'], 'EXACT_APPROVAL_REFERENCE')
    approval=read(p)
    require(approval['SOURCE']==cfg['SOURCE'] and approval['scope']==scope() and approval['execution_authorized'] is True, 'USER_APPROVED_REPAIR_SCOPE')
    verify_seals(cfg)


def consumed(cfg):
    require(read(cfg['marker'])==dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']), 'CONSUMED_REPAIR_GRANT')


def reserve(cfg,stage):
    require(stage in STAGES, 'REGISTERED_REPAIR_STAGE'); consumed(cfg)
    entered={p.stem for p in Path(cfg['ledger']).glob('*.json')}
    require(entered==set(STAGES[:STAGES.index(stage)]), 'EXACT_STAGE_ORDER_NO_RETRY')
    save(Path(cfg['ledger'])/(stage+'.json'),dict(stage=stage,state='ENTERED_NONREFUNDABLE',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256']))
