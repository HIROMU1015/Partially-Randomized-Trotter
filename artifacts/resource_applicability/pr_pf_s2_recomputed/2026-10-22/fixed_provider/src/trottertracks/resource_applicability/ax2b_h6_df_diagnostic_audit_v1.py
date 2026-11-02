"""Future saved diagnostic bytes audit, stdlib only; no decomposition/statistics replay."""
import hashlib
import struct
from pathlib import Path
from .ax2a_preparation import digest
from .ax2b_h6_input_generation_audit_v1 import read_json,npz_bytes
from .ax2b_h6_df_diagnostic_contract_v1 import plan

def audit_saved(directory):
    d=Path(directory).resolve();files=sorted(p for p in d.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in files):raise ValueError('SAVED_DIAGNOSTIC_SYMLINK')
    m=read_json(d/'frozen_diagnostic.json');grant=read_json(d/'authorization.json')
    if digest(m['plan'])!=digest(plan()):raise ValueError('SAVED_DIAGNOSTIC_PLAN')
    if sum(p.stat().st_size for p in files)>m['plan']['caps']['output_bytes']:raise ValueError('SAVED_DIAGNOSTIC_OUTPUT_CAP')
    if (grant.get('schema')!='track_a_h6_df_diagnostic_authorization_v1' or grant.get('approved_by_user') is not True
            or grant.get('manifest_digest')!=digest(m) or grant.get('retry') is not False or grant.get('resume') is not False
            or read_json(d/'authorization_source.json')!=grant):raise ValueError('SAVED_DIAGNOSTIC_GRANT')
    binding={'manifest_digest':digest(m),'authorization_digest':digest(grant)}
    grant_sha=hashlib.sha256((d/'authorization_source.json').read_bytes()).hexdigest()
    if read_json(d/'launch_binding.json')!=binding or read_json(d/'worker_claim.json')!=dict(binding,assigned_cpu=grant['assigned_cpu'],retry=False,resume=False,authorization_sha256=grant_sha):raise ValueError('SAVED_DIAGNOSTIC_BINDING')
    parent=read_json(d/'terminal_status.json',65536)
    worker=read_json(d/'worker_terminal.json',8192) if (d/'worker_terminal.json').exists() else None
    if parent['worker_terminal']!=worker:raise ValueError('SAVED_DIAGNOSTIC_TERMINAL')
    required=['raw_decomposition.npz','raw_decomposition_receipt.json','hypothetical_hermitization.npz','hypothetical_receipt.json','diagnostic_summary.json','input_decode_receipt.json','decomposition_call.json','runtime_environment.json']
    missing=[x for x in required if not (d/x).exists()];checks={}
    if parent['status']=='H6_DF_DIAGNOSTIC_RECORDED':
        if missing or worker is None or worker['status']!=parent['status'] or parent['worker_exit_code']!=0 or worker['authorization_sha256']!=grant_sha:raise ValueError('SAVED_DIAGNOSTIC_MISSING')
        decoded=read_json(d/'input_decode_receipt.json');called=read_json(d/'decomposition_call.json')
        if (decoded['input_sha256']!=m['input_identity']['input_sha256'] or called['kwargs']!=plan()['kwargs']
                or called['two_body_before_call']!=decoded['two_body']
                or any(decoded[k]!=m['input_identity']['integral_array_records'][k] for k in ('one_body','two_body'))):raise ValueError('SAVED_DIAGNOSTIC_INPUT_CALL_IDENTITY')
        packs={}
        for stem,receipt in [('raw_decomposition','raw_decomposition_receipt'),('hypothetical_hermitization','hypothetical_receipt')]:
            packs[stem]=npz_bytes(d/(stem+'.npz'),read_json(d/(receipt+'.json')),cap=m['plan']['caps']['raw_expanded_bytes'])
            checks[stem]=list(packs[stem])
        if set(checks['raw_decomposition'])!={'lambdas_raw','g_matrices_raw','one_body_correction_raw','truncation_value_raw'}:raise ValueError('SAVED_DIAGNOSTIC_RAW_SCHEMA')
        summary=read_json(d/'diagnostic_summary.json');rank=summary['actual_rank']
        if type(rank) is not int or not 1<=rank<=36 or [r['index'] for r in summary['fragments']]!=list(range(rank)):raise ValueError('SAVED_DIAGNOSTIC_ALL_FRAGMENTS')
        for stem,key,rowkey in [('raw_decomposition','g_matrices_raw','raw'),('hypothetical_hermitization','g_hypothetical_hermitian','hypothetical_post')]:
            h,raw=packs[stem][key];width=16 if h['descr']=='<c16' else 8
            if h['shape']!=(rank,12,12):raise ValueError('SAVED_DIAGNOSTIC_FRAGMENT_LAYOUT')
            for i,row in enumerate(summary['fragments']):
                piece=raw[i*144*width:(i+1)*144*width]
                if row[rowkey]!={'shape':[12,12],'dtype':h['descr'],'sha256':hashlib.sha256(piece).hexdigest()}:raise ValueError('SAVED_DIAGNOSTIC_FRAGMENT_HASH')
        h,raw=packs['raw_decomposition']['lambdas_raw']
        if h['shape']!=(rank,):raise ValueError('SAVED_DIAGNOSTIC_WEIGHTS')
        values=struct.unpack('<'+str(rank if h['descr']=='<f8' else 2*rank)+'d',raw)
        lambdas=values if h['descr']=='<f8' else values[::2]
        if h['descr']=='<c16' and any(values[1::2]):raise ValueError('SAVED_DIAGNOSTIC_COMPLEX_WEIGHT')
        if any(row['lambda']!=v for row,v in zip(summary['fragments'],lambdas)):raise ValueError('SAVED_DIAGNOSTIC_LAMBDA_ROW')
        for row in (summary,worker,parent):
            for k,v in [('H6_status','H6_NOT_AUTHORIZED'),('mandatory_stop',True),('next_stage_authorized',False)]:
                if type(row.get(k)) is not type(v) or row[k]!=v:raise ValueError('SAVED_DIAGNOSTIC_FLAG:'+k)
        if summary['H6_input_accepted'] is not False or summary['historical_raw_bytes_identity_claim'] is not False:raise ValueError('SAVED_DIAGNOSTIC_NO_PROMOTION')
    elif parent['status']!='H6_DF_DIAGNOSTIC_STOP':raise ValueError('SAVED_DIAGNOSTIC_STATUS')
    elif (d/'raw_decomposition_receipt.json').exists() and (d/'raw_decomposition.npz').exists():
        checks['partial_raw_decomposition']=list(npz_bytes(d/'raw_decomposition.npz',read_json(d/'raw_decomposition_receipt.json'),cap=m['plan']['caps']['raw_expanded_bytes']))
    return {'schema':'track_a_h6_df_saved_diagnostic_audit_v1','status':'SAVED_DIAGNOSTIC_BYTES_PASS' if parent['status']=='H6_DF_DIAGNOSTIC_RECORDED' else 'SAVED_DIAGNOSTIC_STOP_RECORDED',
        'source_commit':m['source_commit'],'input_identity':m['input_identity'],'run_status':parent['status'],
        'file_hashes':{str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'missing_records':missing,'checks':checks,'scope':'bytes/schema/provenance only; numerical summary not recalculated',
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}
