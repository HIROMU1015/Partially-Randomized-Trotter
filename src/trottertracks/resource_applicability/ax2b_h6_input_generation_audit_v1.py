"""Saved-byte audit only. Parses NPY headers/data using stdlib, never NumPy."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile

from .ax2a_preparation import digest
from .ax2b_h6_input_generation_contract_v1 import plan


def read_json(path, cap=4*2**20):
    if path.stat().st_size > cap:
        raise ValueError('SAVED_JSON_CAP')
    return json.loads(path.read_text())


def npz_bytes(path, receipt, *, cap):
    """Check member identity, expanded size, C-layout, dtype, and exact data hash."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256'] or path.stat().st_size != receipt['bytes']:
        raise ValueError('SAVED_NPZ_BYTES')
    result = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        wanted = {k+'.npy' for k in receipt['arrays']}
        if (len(infos) != len(wanted) or {i.filename for i in infos} != wanted
                or sum(i.file_size for i in infos) > cap):
            raise ValueError('SAVED_NPZ_MEMBERS_OR_EXPANDED_CAP')
        for info in infos:
            if info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED:
                raise ValueError('SAVED_NPZ_STORAGE')
            with archive.open(info) as stream:
                if stream.read(6) != b'\x93NUMPY':
                    raise ValueError('SAVED_NPY_MAGIC')
                version = tuple(stream.read(2))
                if version not in ((1,0),(2,0),(3,0)):
                    raise ValueError('SAVED_NPY_VERSION')
                count = int.from_bytes(stream.read(2 if version == (1,0) else 4), 'little')
                if not 0 < count <= 65536:
                    raise ValueError('SAVED_NPY_HEADER_CAP')
                header = ast.literal_eval(stream.read(count).decode())
                raw = stream.read(cap+1)
            name = info.filename.removesuffix('.npy'); item = receipt['arrays'][name]
            if header != {'descr':item['dtype'],'fortran_order':False,'shape':tuple(item['shape'])}:
                raise ValueError('SAVED_NPY_LAYOUT:'+name)
            dtype = item['dtype']
            if dtype not in ('<f8','<c16','<i8') and not (dtype.startswith('<U') and dtype[2:].isdigit()):
                raise ValueError('SAVED_NPY_DTYPE:'+name)
            width = 4*int(dtype[2:]) if dtype.startswith('<U') else int(dtype[2:])
            if len(raw) != math.prod(item['shape'])*width or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('SAVED_NPY_DATA:'+name)
            result[name] = (header, raw)
    return result


def array_hash(header, raw):
    return hashlib.sha256(header['descr'].encode()+
                          json.dumps(list(header['shape']),sort_keys=True,separators=(',',':')).encode()+raw).hexdigest()


def audit_saved(directory):
    directory = Path(directory).resolve()
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in files):
        raise ValueError('SAVED_SYMLINK')
    manifest = read_json(directory/'frozen_preparation.json')
    if digest(manifest.get('plan')) != digest(plan()):
        raise ValueError('SAVED_INPUT_SCOPE_CHANGED')
    caps = manifest['plan']['caps_proposed']
    if sum(p.stat().st_size for p in files) > caps['output_bytes']:
        raise ValueError('SAVED_OUTPUT_CAP')
    grant = read_json(directory/'authorization.json')
    if read_json(directory/'authorization_source.json') != grant:
        raise ValueError('SAVED_EXACT_AUTHORIZATION_JSON')
    binding = {'manifest_digest':digest(manifest),'authorization_digest':digest(grant)}
    if (grant.get('schema') != 'track_a_ax2b_h6_input_generation_authorization_v1'
            or grant.get('approved_by_user') is not True or grant.get('manifest_digest') != digest(manifest)
            or grant.get('retry') is not False or grant.get('resume') is not False
            or read_json(directory/'launch_binding.json') != binding
            or read_json(directory/'worker_claim.json') != dict(binding,assigned_cpu=grant['assigned_cpu'],retry=False,resume=False)):
        raise ValueError('SAVED_GRANT_BINDING')
    parent = read_json(directory/'terminal_status.json', 65536)
    worker = read_json(directory/'worker_terminal.json', 8192) if (directory/'worker_terminal.json').is_file() else None
    if parent.get('worker_terminal') != worker:
        raise ValueError('SAVED_TERMINAL_BINDING')
    required = ('integral_receipt.json','integrals.npz','df_receipt.json','snapshot_receipt.json','h6_input_snapshot.npz','worker_terminal.json')
    missing = [name for name in required if not (directory/name).is_file()]
    hashes = {str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    checked = {}
    if parent['status'] == 'H6_INPUT_GENERATION_COMPLETE':
        if missing or worker is None or worker['status'] != 'H6_INPUT_GENERATION_COMPLETE' or parent['worker_exit_code'] != 0 or parent['reason'] is not None:
            raise ValueError('SAVED_COMPLETE_MISSING_RECORDS')
        for record in (parent, worker):
            for key, value in [('N',None),('G',None),('numerical_allowance_certified',False),
                               ('accuracy_eligibility','UNDETERMINED'),('H6_status','H6_NOT_AUTHORIZED'),
                               ('contract_status','DRAFT_NOT_AUTHORIZATION'),('mandatory_stop',True),('next_stage_authorized',False)]:
                if type(record.get(key)) is not type(value) or record[key] != value:
                    raise ValueError('SAVED_TERMINAL_FLAG:'+key)
        used = worker['calls_attempted']; complete = worker['calls_completed']
        if (used['integral_build'] != 1 or used['df_decomposition'] != 1 or not 1 <= used['solver_matvec'] <= 10000
                or any(used[k] != 0 for k in ('trajectory','occurrence','compile'))
                or any(used[k] != complete[k] for k in complete)):
            raise ValueError('SAVED_CALL_COUNTS')
        integral = read_json(directory/'integral_receipt.json')
        npz_bytes(directory/'integrals.npz',integral,cap=caps['snapshot_expanded_bytes'])
        receipt = read_json(directory/'snapshot_receipt.json')
        members = npz_bytes(directory/'h6_input_snapshot.npz',receipt,cap=caps['snapshot_expanded_bytes'])
        expected = {'constant','one_body','lambdas','g_matrices','sector_basis_indices','state_vector','sector_state_vector','metadata_json'}
        if set(members) != expected:
            raise ValueError('SAVED_H6_SNAPSHOT_SCHEMA')
        header, raw = members['metadata_json']
        if header['shape'] != () or not header['descr'].startswith('<U'):
            raise ValueError('SAVED_METADATA_ARRAY')
        metadata = json.loads(raw.decode('utf-32-le').rstrip('\x00'))
        if metadata != receipt['metadata']:
            raise ValueError('SAVED_METADATA_RECEIPT')
        df = read_json(directory/'df_receipt.json')
        if df['hamiltonian_metadata'] != metadata['hamiltonian_metadata']:
            raise ValueError('SAVED_DF_METADATA')
        if (metadata['input_generation']['source_commit'] != manifest['source_commit']
                or metadata['input_generation']['authorization_sha256'] != hashes['authorization_source.json']
                or metadata['state_policy'] != manifest['plan']['state_policy']):
            raise ValueError('SAVED_INPUT_PROVENANCE')
        rank = metadata['hamiltonian_metadata']['df_rank_actual']
        if type(rank) is not int or not 2 <= rank <= 144:
            raise ValueError('SAVED_ACTUAL_RANK')
        layouts={'constant':((),'<f8'),'one_body':((12,12),'<c16'),'lambdas':((rank,),'<f8'),
                 'g_matrices':((rank,12,12),'<c16'),'sector_basis_indices':((400,),'<i8'),
                 'state_vector':((4096,),'<c16'),'sector_state_vector':((400,),'<c16')}
        for key,(shape,dtype) in layouts.items():
            h,_ = members[key]
            if h != {'shape':shape,'descr':dtype,'fortran_order':False}:
                raise ValueError('SAVED_H6_LAYOUT:'+key)
        for key in ('state_vector','sector_state_vector'):
            if array_hash(*members[key]) != metadata[key+'_hash']:
                raise ValueError('SAVED_STATE_HASH:'+key)
        indices = struct.unpack('<400q',members['sector_basis_indices'][1])
        if indices != tuple(sorted(set(indices))) or any(not 0 <= i < 4096 for i in indices):
            raise ValueError('SAVED_SECTOR_BASIS')
        def expected_sector(i):
            return sum((i>>(11-mode))&1 for mode in range(0,12,2)) == 3 and sum((i>>(11-mode))&1 for mode in range(1,12,2)) == 3
        if indices != tuple(i for i in range(4096) if expected_sector(i)):
            raise ValueError('SAVED_SECTOR_OCCUPATIONS')
        sector_hash=digest({'n_qubits':12,'basis_indices_hash':array_hash(*members['sector_basis_indices']),
                            'n_electrons':6,'nelec_alpha':3,'nelec_beta':3,'sz_value':0.})
        if sector_hash != metadata['sector_hash'] or sector_hash != df['sector_hash']:
            raise ValueError('SAVED_SECTOR_HASH')
        state_hash=digest({'state_vector_hash':metadata['state_vector_hash'],
                           'sector_state_vector_hash':metadata['sector_state_vector_hash'],
                           'global_phase_policy':'largest_sector_amplitude_real_positive_v1'})
        if state_hash != metadata['state_hash']:
            raise ValueError('SAVED_COMPOSITE_STATE_HASH')
        if receipt.get('loader_roundtrip_checked') is not True:
            raise ValueError('SAVED_LOADER_WITNESS')
        checked={'NPZ_data_hashes_checked':True,'sector_occupation_indices_checked':True,
                 'actual_rank':rank,'solver_residual_saved':metadata['solver_residual'],
                 'solver_matvec_calls_including_residual':metadata['solver_matvec_calls_including_residual'],
                 'scope':'byte/schema/provenance audit; no new residual, ground-state, or signal computation'}
    elif parent['status'] != 'H6_INPUT_GENERATION_STOP':
        raise ValueError('SAVED_INPUT_STATUS')
    return {'schema':'track_a_ax2b_h6_input_saved_audit_v1',
            'status':'SAVED_INPUT_CONSISTENCY_PASS' if checked else 'SAVED_INPUT_STOP_RECORDED',
            'source_commit':manifest['source_commit'],'science_run_status':parent['status'],
            'raw_files':len(files),'raw_bytes':sum(p.stat().st_size for p in files),
            'file_hashes':hashes,'missing_records':missing,'checks':checked,
            'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
            'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
            'mandatory_stop':True,'next_stage_authorized':False}
