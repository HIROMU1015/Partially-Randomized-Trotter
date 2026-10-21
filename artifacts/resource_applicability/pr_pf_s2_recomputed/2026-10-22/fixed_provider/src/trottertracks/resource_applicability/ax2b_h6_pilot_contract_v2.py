"""Stdlib-only, saved-snapshot H6 technical pilot. A plan never grants execution."""
from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path
from .ax2a_preparation import digest
from .ax2b_h6_contract import h6_cells, wrapper_tasks, primitive_time_schedule, PHASES
from .ax2b_h6_saved_completion_contract_v2 import resources, valid_cpus, install_parallel_limits
from .ax2b_h6_input_generation_contract_v1 import environment
from .ax2b_h6_df_diagnostic_contract_v1 import safe_path, file_hash
from .ax2b_h6_input_generation_audit_v1 import read_json, npz_bytes

KIND = 'H6_SAVED_SNAPSHOT_TECHNICAL_PILOT'
RUNNER = 'scripts/resource_applicability/run_track_a_h6_pilot_v2.py'
AUDITOR = 'scripts/resource_applicability/audit_track_a_h6_pilot_v2.py'
NAMESPACE = 'artifacts/resource_applicability/track_a_h6_technical_pilot_v2/2026-10-10/'
PARENT_SOURCE = 'f37005f01b2be38c5993d6e82df91abe9c643d21'
PARENT_RESULT = '554fc52add39c2c1b45b765a3135df76fda6f15a'
PARENT_EVIDENCE = 'f52787a22542b31bd39fd004a8d3d71325bc56b0'
PARENT = 'artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_v2/2026-10-10/launch_v1/'
PARENT_INVENTORY = 'artifacts/resource_applicability/track_a_h6_saved_df_completion_parallel_execution_v2/2026-10-10/execution_evidence_inventory_v2.json'


def coverage():
    """All physical primitive/time keys, without opening numerical input arrays."""
    pairs = set()
    rows = []
    for c in h6_cells(19):
        s = primitive_time_schedule(c, T=.8)
        # H6 has no H4-E extra microstep probes. Bind every runtime field.
        s['registered_validation_times_v2'] = list(s['unique_primitive_times'])
        rows.append({'cell_id': c['id'], 'schedule': s})
        pairs.update(('one' if i == 0 else str(i-1), t) for i, t in s['unique_primitive_times'])
    return {'cells': rows, 'physical_times': sorted(pairs), 'primitive_probe_count': 3,
            'primitive_actions': 3*len(pairs), 'probes': ['saved_state', 'first_sector_column', 'last_sector_column'],
            'instruction_bounds': 'computed from actual prepared representation before any action, sampling or circuit build'}


def plan():
    return {'schema': 'track_a_h6_technical_pilot_plan_v2', 'kind': KIND,
        'actual_rank': 19, 'T': .8, 'epsilon_signal_diagnostic': .001,
        'target': {'model': 'linear_H6', 'geometry_angstrom': 1., 'basis': 'sto-3g',
                   'n_qubits': 12, 'sector_dimension': 400, 'nelec_alpha': 3, 'nelec_beta': 3},
        'input_policy': 'WEIGHTED_HERMITIAN_PROJECTION_FROM_SAVED_RAW_V1',
        'df_policy': {'df_tol': 1e-8, 'final_rank_supplied': False, 'cutoff': 0.,
                      'signed_generation_order_preserved': True, 'fragment_deletion': False},
        'cells': h6_cells(19), 'wrapper_tasks': wrapper_tasks(h6_cells(19)), 'coverage': coverage(),
        'caps_proposed': {'phase_wall_seconds': dict(zip(PHASES, (1800,1800,3600))),
            'total_wall_seconds': 7200, 'address_space_bytes': 8*2**30, 'output_bytes': 512*2**20,
            'log_bytes': 65536, 'diagnostics': 1024, 'progress_records': 1024,
            'primitive': 2000, 'control_probe': 256, 'compile': 36, 'trajectory': 4, 'occurrence': 8,
            'reference_matvec': 20000, 'reference_matvec_per_action': 20000,
            'deterministic_actions_per_cell': 100000, 'tail_matvecs_corrected_and_raw': {'B2':24,'B3':56},
            'untranspiled_instructions': 1000000, 'transpiled_instructions': 5000000,
            'integral_build': 0, 'df_decomposition': 0, 'state_solver': 0, 'solver_matvec': 0},
        'compiler_proposed': {'basis_gates':['rz','sx','x','cx'], 'optimization_level':1,
            'seed_transpiler':17, 'backend':None, 'coupling_map':None},
        'matrix_free': {'backend':'numba','num_threads':4,'block_chunk_size':1},
        'gates': {'norm_and_leakage':1e-12,'absolute_agreement':1e-9,'relative_agreement':1e-10,
                   'reference_expm_eigh':1e-10},
        'cost_primary':'symmetric_directional', 'ordinary_cost':'paired sensitivity',
        'oracle':'binary64 independent occupation columns and forward Taylor recurrence; sector expm/eigh',
        'prepared_representation_computed_in_preparation':False,
        'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
        'ground_state_certified':False,'retry':False,'resume':False,'gpu':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False}


def preparation():
    return {'schema':'track_a_h6_technical_pilot_preparation_v2','kind':KIND,'plan':plan(),
        'status':'H6_PILOT_NOT_AUTHORIZED','source_commit':None,'source_hashes':None,
        'input_identity':None,'environment':None,'assigned_resources':None,'exclusive_output':None,
        'execution_plan_sealed':False,'science_authorized':False,'launch_allowed':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False}


def source_paths(root):
    root=Path(root)
    return sorted({str(p.relative_to(root)) for d in ('src/trotterlib','src/trottertracks')
        for p in (root/d).rglob('*.py')} | {RUNNER,AUDITOR})


def verify_sources(root, commit, hashes):
    if not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('PILOT_SOURCE_COMMIT')
    if not isinstance(hashes,dict) or set(hashes)!=set(source_paths(root)):
        raise ValueError('PILOT_SOURCE_CLOSURE')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=root,check=True,stdout=subprocess.DEVNULL)
    for p,sha in hashes.items():
        if file_hash(safe_path(root,p))!=sha or hashlib.sha256(subprocess.check_output(['git','show',commit+':'+p],cwd=root)).hexdigest()!=sha:
            raise ValueError('PILOT_SOURCE_CHANGED:'+p)


def verify_parent(root):
    """Git blobs, receipt fields and NPZ headers only; no ndarray decoding."""
    root=Path(root);inv=read_json(root/PARENT_INVENTORY)
    if (inv.get('source_commit')!=PARENT_SOURCE or inv.get('H6_input_accepted') is not True
            or inv.get('one_shot_authorization_consumed') is not True or inv.get('missing_records')!=[]):
        raise ValueError('PILOT_INPUT_PARENT')
    if hashlib.sha256(subprocess.check_output(['git','show',PARENT_EVIDENCE+':'+PARENT_INVENTORY],cwd=root)).hexdigest()!=file_hash(root/PARENT_INVENTORY):
        raise ValueError('PILOT_PARENT_INVENTORY_CHANGED')
    for p,sha in inv['raw_file_hashes'].items():
        if file_hash(safe_path(root,p))!=sha or hashlib.sha256(subprocess.check_output(['git','show',PARENT_RESULT+':'+p],cwd=root)).hexdigest()!=sha:
            raise ValueError('PILOT_PARENT_CHANGED:'+p)
    receipt=read_json(root/PARENT/'snapshot_receipt.json');df=read_json(root/PARENT/'df_receipt.json')
    members=npz_bytes(root/PARENT/'h6_input_snapshot.npz',receipt,cap=16*2**20)
    if len(members)!=8 or df.get('hamiltonian_hash')!=receipt['metadata']['hamiltonian_hash']:
        raise ValueError('PILOT_PARENT_SNAPSHOT')
    return {'parent_source_commit':PARENT_SOURCE,'parent_result_commit':PARENT_RESULT,
        'parent_evidence_commit':PARENT_EVIDENCE,'parent_inventory_path':PARENT_INVENTORY,
        'parent_inventory_sha256':file_hash(root/PARENT_INVENTORY),
        'snapshot_path':PARENT+'h6_input_snapshot.npz','snapshot_sha256':receipt['sha256'],
        'snapshot_receipt_path':PARENT+'snapshot_receipt.json','snapshot_receipt_sha256':file_hash(root/PARENT/'snapshot_receipt.json'),
        'df_receipt_path':PARENT+'df_receipt.json','df_receipt_sha256':file_hash(root/PARENT/'df_receipt.json'),
        'hamiltonian_hash':df['hamiltonian_hash'],'state_hash':receipt['metadata']['state_hash'],
        'parent_raw_hashes':inv['raw_file_hashes'],'parent_grant_consumed':True}


def validate_launch(root,manifest,grant,output,*,requested=False,worker=False):
    if requested is not True or not isinstance(grant,dict) or grant.get('approved_by_user') is not True:
        raise ValueError('NEW_H6_PILOT_GRANT_REQUIRED')
    if grant.get('schema')!='track_a_h6_technical_pilot_authorization_v2' or grant.get('kind')!=KIND:
        raise ValueError('PILOT_GRANT_SCHEMA')
    expected=preparation()
    for k in ('schema','kind','status','science_authorized','launch_allowed','H6_status','contract_status','mandatory_stop','next_stage_authorized'):
        if type(manifest.get(k)) is not type(expected[k]) or manifest[k]!=expected[k]:
            raise ValueError('PILOT_MANIFEST_FLAG:'+k)
    if manifest.get('execution_plan_sealed') is not True or digest(manifest.get('plan'))!=digest(plan()):
        raise ValueError('PILOT_FIXED_PLAN')
    if (grant.get('manifest_digest')!=digest(manifest) or grant.get('source_commit')!=manifest.get('source_commit')
            or grant.get('retry') is not False or grant.get('resume') is not False or grant.get('one_shot') is not True):
        raise ValueError('PILOT_GRANT_BINDING')
    cpus=grant.get('assigned_cpus')
    if not valid_cpus(cpus) or digest(manifest.get('assigned_resources'))!=digest(resources(cpus)):
        raise ValueError('PILOT_CPU')
    output=Path(output).resolve();intended=manifest.get('exclusive_output') or {};name=intended.get('repository_path','')
    if (not name.startswith(NAMESPACE) or safe_path(root,name)!=output or intended.get('absolute_path')!=str(output)
            or grant.get('exclusive_output')!=str(output) or (not worker and output.exists())):
        raise ValueError('PILOT_EXCLUSIVE_OUTPUT')
    if worker and (read_json(output/'launch_binding.json')!={'manifest_digest':digest(manifest),'authorization_digest':digest(grant)} or (output/'worker_claim.json').exists()):
        raise ValueError('PILOT_ONE_SHOT_BINDING')
    verify_sources(root,manifest.get('source_commit'),manifest.get('source_hashes'))
    if manifest.get('input_identity')!=verify_parent(root):raise ValueError('PILOT_INPUT_BINDING')
    if manifest.get('environment')!=environment():raise ValueError('PILOT_ENVIRONMENT')
    return cpus
