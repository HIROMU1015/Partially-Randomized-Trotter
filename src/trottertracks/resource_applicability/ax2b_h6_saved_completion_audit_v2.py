"""Stdlib byte/schema/provenance audit; no reconstruction, solver or new science."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import struct
from .ax2a_preparation import digest
from .ax2b_h6_input_generation_audit_v1 import read_json,npz_bytes,array_hash
from .ax2b_h6_saved_completion_contract_v2 import plan,KIND,resources


def audit_saved(directory):
    directory=Path(directory).resolve();files=sorted(p for p in directory.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in directory.rglob('*')):
        raise ValueError('COMPLETION_AUDIT_SYMLINK')
    manifest=read_json(directory/'frozen_completion.json')
    if digest(manifest['plan'])!=digest(plan()):
        raise ValueError('COMPLETION_AUDIT_PLAN')
    caps=manifest['plan']['caps']
    if sum(p.stat().st_size for p in files)>caps['output_bytes']:
        raise ValueError('COMPLETION_AUDIT_OUTPUT_CAP')
    grant=read_json(directory/'authorization.json')
    hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    binding={'manifest_digest':digest(manifest),'authorization_digest':digest(grant)}
    if (read_json(directory/'authorization_source.json')!=grant
            or grant.get('schema')!='track_a_h6_saved_df_completion_authorization_v2'
            or grant.get('approved_by_user') is not True or grant.get('kind')!=KIND
            or grant.get('manifest_digest')!=digest(manifest)
            or grant.get('retry') is not False or grant.get('resume') is not False
            or read_json(directory/'launch_binding.json')!=binding):
        raise ValueError('COMPLETION_AUDIT_GRANT')
    claim=read_json(directory/'worker_claim.json')
    if claim!=dict(binding,assigned_resources=resources(grant['assigned_cpus']),retry=False,resume=False,authorization_sha256=hashes['authorization_source.json']):
        raise ValueError('COMPLETION_AUDIT_CLAIM')
    if manifest['assigned_resources']!=resources(grant['assigned_cpus']):
        raise ValueError('COMPLETION_AUDIT_RESOURCES')
    parent=read_json(directory/'terminal_status.json',65536)
    worker=read_json(directory/'worker_terminal.json',8192) if (directory/'worker_terminal.json').exists() else None
    if parent.get('worker_terminal')!=worker:
        raise ValueError('COMPLETION_AUDIT_TERMINAL_BINDING')
    required=('accepted_df.npz','accepted_df_receipt.json','independent_coefficients.npz','coefficient_receipt.json',
        'df_receipt.json','state_receipt.json','h6_input_snapshot.npz','snapshot_receipt.json','worker_terminal.json',
        'parallel_resource_receipt.json')
    missing=[p for p in required if not (directory/p).is_file()];checks={}
    if parent['status']=='H6_SAVED_DF_COMPLETION_COMPLETE':
        if missing or worker is None or worker['status']!=parent['status'] or parent['worker_exit_code']!=0 or parent['reason'] is not None:
            raise ValueError('COMPLETION_AUDIT_MISSING_RECORDS')
        for row in (parent,worker):
            for k,v in {'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
                'mandatory_stop':True,'next_stage_authorized':False,'N':None,'G':None,
                'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED'}.items():
                if type(row.get(k)) is not type(v) or row[k]!=v:
                    raise ValueError('COMPLETION_AUDIT_FLAGS:'+k)
        used=worker['calls_attempted'];done=worker['calls_completed']
        parallel=read_json(directory/'parallel_resource_receipt.json')
        if (parallel.get('assigned_cpus')!=grant['assigned_cpus'] or parallel.get('numba_threads')!=4
                or parallel.get('operator_backend')!='numba' or parallel.get('block_chunk_size')!=1
                or parallel.get('science_workers')!=1 or parallel.get('thread_environment')!={
                    'NUMBA_NUM_THREADS':'4','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}):
            raise ValueError('COMPLETION_AUDIT_PARALLEL_RUNTIME')
        if (used['acceptance']!=1 or used['state_solver']!=1 or not 1<=used['solver_matvec']<=caps['solver_matvec']
                or any(used[k]!=0 for k in ('integral_build','df_decomposition','signal','trajectory','occurrence','compile'))
                or any(used[k]!=done[k] for k in done)):
            raise ValueError('COMPLETION_AUDIT_CALLS')
        accepted=read_json(directory/'accepted_df_receipt.json');df=read_json(directory/'df_receipt.json')
        snapshot=read_json(directory/'snapshot_receipt.json');metadata=snapshot['metadata']
        projection=df['projection_receipt']
        if (projection!=accepted['projection_receipt'] or projection.get('status')!='PASS_ENGINEERING'
                or projection['parents']!=manifest['input_identity'] or projection['source_commit']!=manifest['source_commit']
                or projection['authorization_sha256']!=hashes['authorization_source.json']
                or projection['representation_error_certified'] is not False
                or not projection['independent_coefficients_checked'] or not projection['saved_summary_checked']
                or metadata['hamiltonian_metadata']!=df['hamiltonian_metadata']
                or metadata['hamiltonian_metadata']['projection_receipt_digest']!=digest(projection)
                or metadata['hamiltonian_metadata']['df_rank_actual']!=19
                or metadata['saved_df_completion']['source_commit']!=manifest['source_commit']
                or metadata['saved_df_completion']['authorization_sha256']!=hashes['authorization_source.json']
                or metadata['state_policy']!=manifest['plan']['state_policy']):
            raise ValueError('COMPLETION_AUDIT_PROVENANCE')
        adopted=npz_bytes(directory/'accepted_df.npz',accepted,cap=caps['snapshot_expanded_bytes'])
        members=npz_bytes(directory/'h6_input_snapshot.npz',snapshot,cap=caps['snapshot_expanded_bytes'])
        npz_bytes(directory/'independent_coefficients.npz',read_json(directory/'coefficient_receipt.json'),cap=caps['snapshot_expanded_bytes'])
        layouts={'constant':((),'<f8'),'one_body':((12,12),'<c16'),'lambdas':((19,),'<f8'),
            'g_matrices':((19,12,12),'<c16'),'sector_basis_indices':((400,),'<i8'),
            'state_vector':((4096,),'<c16'),'sector_state_vector':((400,),'<c16')}
        if set(adopted)!={'constant','one_body','lambdas','g_matrices'} or set(members)!=set(layouts)|{'metadata_json'}:
            raise ValueError('COMPLETION_AUDIT_LAYOUT')
        for k,(shape,dtype) in layouts.items():
            if members[k][0]!={'shape':shape,'descr':dtype,'fortran_order':False}:
                raise ValueError('COMPLETION_AUDIT_LAYOUT:'+k)
            if k in adopted and adopted[k]!=members[k]:
                raise ValueError('COMPLETION_AUDIT_TARGET_BYTES:'+k)
        mh,mraw=members['metadata_json']
        if mh['shape']!=() or not mh['descr'].startswith('<U') or json.loads(mraw.decode('utf-32-le').rstrip('\0'))!=metadata:
            raise ValueError('COMPLETION_AUDIT_METADATA')
        indices=struct.unpack('<400q',members['sector_basis_indices'][1])
        expected=tuple(i for i in range(4096) if sum((i>>(11-m))&1 for m in range(0,12,2))==3 and sum((i>>(11-m))&1 for m in range(1,12,2))==3)
        if indices!=expected:
            raise ValueError('COMPLETION_AUDIT_SECTOR')
        for k in ('state_vector','sector_state_vector'):
            if array_hash(*members[k])!=metadata[k+'_hash']:
                raise ValueError('COMPLETION_AUDIT_STATE_HASH')
        if snapshot.get('loader_roundtrip_checked') is not True or metadata['ground_state_certified'] is not False:
            raise ValueError('COMPLETION_AUDIT_LOADER_SCOPE')
        checks={'NPZ_data_hashes_checked':True,'adopted_snapshot_target_bytes_equal':True,
            'sector_occupations_checked':True,'actual_rank':19,'scope':'saved bytes/schema/receipts only; numerical quantities not re-evaluated'}
    elif parent['status']!='H6_SAVED_DF_COMPLETION_STOP':
        raise ValueError('COMPLETION_AUDIT_STATUS')
    return {'schema':'track_a_h6_saved_df_completion_saved_audit_v2',
        'status':'SAVED_COMPLETION_BYTES_PASS' if checks else 'SAVED_COMPLETION_STOP_RECORDED',
        'source_commit':manifest['source_commit'],'science_run_status':parent['status'],
        'raw_files':len(files),'file_hashes':hashes,'raw_bytes':sum(p.stat().st_size for p in files),
        'missing_records':missing,'checks':checks,'H6_status':'H6_NOT_AUTHORIZED',
        'contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}
