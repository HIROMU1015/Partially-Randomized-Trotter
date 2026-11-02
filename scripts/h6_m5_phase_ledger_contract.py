"""New one-shot scope for saved scalar ledger only; all imports stdlib-only."""
from pathlib import Path
import subprocess
import h6_m5_phase_contract as original

ROOT=original.ROOT
require,read,save,sha,digest,seal=original.require,original.read,original.save,original.sha,original.digest,original.seal
SCOPE='H6_M5_SAVED_DETERMINANT_PHASE_LEDGER_V1'
STATIC_ROOT=ROOT/'artifacts/resource_applicability/track_a_h6_m5_phase_ledger_static/2026-10-19'
SOURCE_PATHS=original.SOURCE_PATHS+(
    'scripts/inspect_h6_m5_phase_structure.py','scripts/run_h6_m5_saved_phase.py',
    'scripts/h6_m5_phase_ledger_core.py','scripts/h6_m5_phase_ledger_contract.py',
    'scripts/run_h6_m5_phase_ledger.py','scripts/prepare_h6_m5_phase_ledger.py',
    str((STATIC_ROOT/'saved_circuit_metadata.json').relative_to(ROOT)),
    str((STATIC_ROOT/'determinant_ledger_plan_DRAFT.json').relative_to(ROOT)),
)


def scope():
    return dict(scope_id=SCOPE,target=original.scope()['target'],nqubits=13,
        saved_QPY_roles=2,scalar_phase_ledgers=1,internal_decimal_precisions=[80,120],
        internal_convergence_atol_rad='1e-60',internal_branch_boundary_minimum_rad='1e-60',
        principal_interval='[-pi/8192,pi/8192)',sum_policy='EXACT_BINARY64_RATIONAL',
        mathematical_pi=True,physical_match_threshold=None,branch_fit=False,
        new_compile=0,new_statevector_replays=0,new_sampling=0,new_states=0,
        accounting=False,refit=False,retry=False,acceptance_override=False,
        original_raw_strictly_below=1e-9,old_STOP_remains_STOP=True,
        AD03='DEFERRED',next_phase_authorized=False,mandatory_STOP=True)


def verify_seals(cfg):
    require(cfg['plan']['scope']==scope() and cfg['plan']['SOURCE']==cfg['SOURCE'],'EXACT_LEDGER_SCOPE')
    require(read(cfg['plan_path'])==cfg['plan'] and sha(cfg['plan_path'])==cfg['plan_sha256'],'LEDGER_PLAN_BYTES')
    sealed={r['path']:r for r in cfg['sealed_files']}
    require(len(sealed)==len(cfg['sealed_files']) and digest(cfg['sealed_files'])==cfg['plan']['closure_digest'],'UNIQUE_CLOSURE')
    for row in cfg['sealed_files']:
        p=Path(row['path']);require(str(p.resolve())==row['path'] and p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'SEALED_BYTES:'+row['role'])
    for rel in SOURCE_PATHS:
        p=ROOT/rel
        require(str(p.resolve()) in sealed and p.read_bytes()==subprocess.check_output(['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT),'COMMITTED_SOURCE:'+rel)
    require(digest(cfg['inputs'])==cfg['plan']['inputs_digest'] and digest(cfg['runtime'])==cfg['plan']['runtime_digest'],'EXACT_USED_BINDING')
    require(cfg['numerical_runtime']==dict(mpmath_version='1.3.0',backend='python') and digest(cfg['numerical_runtime'])==cfg['plan']['numerical_runtime_digest'],'FIXED_NUMERICAL_RUNTIME')
    for role,row in cfg['inputs'].items():require(sealed.get(row['path'])==row,'INPUT_SEALED:'+role)
    origin=read(cfg['inputs']['origin_config']['path']);original.verify_seals(origin)
    require(origin['SOURCE']=='99e8723d402245a9ad923e7dbc0800f47bc575b6' and cfg['runtime']==origin['runtime'],'ORIGIN_SAVED_PHASE_IDENTITY')
    require(set(cfg['inputs'])=={'wrapper_qpy','compiled_qpy','origin_config','static_metadata','saved_report','saved_completion','saved_exit'},'EXACT_SEVEN_ROLES')
    for kind in ('wrapper','compiled'):
        row=cfg['inputs'][kind+'_qpy'];old=origin['inputs'][kind+'_qpy']
        require(all(row[k]==old[k] for k in ('path','bytes','sha256')),'ORIGINAL_QPY_IDENTITY')
    require(Path(cfg['inputs']['static_metadata']['path']).resolve()==(STATIC_ROOT/'saved_circuit_metadata.json').resolve(),'FIXED_STATIC_METADATA')
    completed=read(cfg['inputs']['saved_completion']['path']);exit=read(cfg['inputs']['saved_exit']['path']);report=read(cfg['inputs']['saved_report']['path'])
    require(completed['status']=='DIAGNOSTIC_COMPLETE_MANDATORY_STOP' and completed['all_reaped'] and
        completed['SOURCE']==origin['SOURCE'] and completed['report_sha256']==sha(cfg['inputs']['saved_report']['path']),'COMPLETED_ORIGIN_REPORT')
    require(exit['native_exit']==0 and exit['reaped'],'ORIGIN_NATIVE_EXIT')
    require(report['SOURCE']==origin['SOURCE'] and report['old_STOP_remains_STOP'] and not report['ordinary']['raw_gate_PASS'] and not report['extended']['raw_gate_PASS'],'ORIGINAL_RAW_FAIL_PRESERVED')


def authorize(cfg):
    g=cfg.get('grant',{})
    require(cfg.get('authorized') is True and g.get('valid') is True and isinstance(g.get('approval_reference'),str) and g['approval_reference'].strip(),'EXPLICIT_SCALAR_LEDGER_EXECUTION_APPROVAL_REQUIRED')
    expected=dict(scope_id=SCOPE,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
        saved_QPY_roles=2,scalar_phase_ledgers=1,new_compile=0,new_statevector_replays=0,
        new_sampling=0,retry=False,acceptance_override=False,next_phase_authorized=False)
    require(all(g.get(k)==v for k,v in expected.items()),'GRANT_EXACT_LEDGER_SCOPE')
    require(isinstance(g.get('grant_id'),str) and g['grant_id'],'NEW_GRANT_REQUIRED')
    root=Path(cfg['run_root']).resolve()
    require(str(root)==cfg['run_root']==cfg['plan']['run_root'] and not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root),'NEW_OUTPUT_OUTSIDE_SOURCE')
    for key in ('output','ledger','marker'):
        p=Path(cfg[key]).resolve();require(str(p)==cfg[key]==g.get(key) and p.is_relative_to(root),'NEW_OUTPUT_BINDING:'+key)
    require(len({cfg[k] for k in ('output','ledger','marker')})==3,'DISTINCT_OUTPUT_PATHS')
    expected_supervisor={k:str(root.parent/name) for k,name in [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    require(cfg['supervisor']==cfg['plan']['supervisor']==g.get('supervisor')==expected_supervisor,'SUPERVISOR_BINDING')
    for row in cfg['inputs'].values():require(not root.is_relative_to(Path(row['path']).parent.resolve()) and not Path(row['path']).resolve().is_relative_to(root),'NO_OLD_INPUT_WRITES')
    require(cfg['resources']==cfg['plan']['resources'],'RESOURCE_PLANNING_BINDING')
    require(cfg['approval_receipt_path'] in {r['path'] for r in cfg['sealed_files']},'APPROVAL_RECEIPT_SEALED')
    require(sha(cfg['approval_receipt_path'])==cfg['approval_receipt_sha256'] and g['approval_reference']==cfg['approval_receipt_path']+'#sha256='+cfg['approval_receipt_sha256'],'APPROVAL_BINDING')
    approval=read(cfg['approval_receipt_path'])
    require(approval['scope']==scope() and approval['SOURCE']==cfg['SOURCE'] and approval['one_shot_execution_authorized'] is True,'USER_APPROVED_FIXED_SCOPE')
    verify_seals(cfg)


def consumed(cfg):
    require(read(cfg['marker'])==dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']),'CONSUMED_NEW_GRANT')


def reserve(cfg,name):
    require(name in ('EXTRACT_WRAPPER','EXTRACT_COMPILED','SCALAR_LEDGER'),'REGISTERED_ONE_SHOT_STAGE')
    consumed(cfg)
    save(Path(cfg['ledger'])/(name+'.json'),dict(state='ENTERED_NONREFUNDABLE',stage=name,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],new_compile=0))
