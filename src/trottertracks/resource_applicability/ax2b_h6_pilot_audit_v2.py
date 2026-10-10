"""Stdlib saved-byte audit. Does not re-evaluate a signal or construct a circuit."""
from pathlib import Path
import hashlib
from .ax2a_preparation import digest
from .ax2b_h6_input_generation_audit_v1 import read_json
from .ax2b_h6_pilot_contract_v2 import plan


def audit_saved(output):
    out=Path(output);manifest=read_json(out/'frozen_pilot.json');grant=read_json(out/'authorization.json')
    binding=read_json(out/'launch_binding.json');terminal=read_json(out/'terminal_status.json')
    if (digest(manifest['plan'])!=digest(plan()) or grant.get('approved_by_user') is not True
            or grant.get('manifest_digest')!=digest(manifest) or grant.get('schema')!='track_a_h6_technical_pilot_authorization_v2'
            or grant.get('kind')!=manifest.get('kind') or grant.get('source_commit')!=manifest.get('source_commit')
            or grant.get('one_shot') is not True or grant.get('retry') is not False or grant.get('resume') is not False):
        raise ValueError('PILOT_AUDIT_GRANT_OR_PLAN')
    if binding!={'manifest_digest':digest(manifest),'authorization_digest':digest(grant)}:
        raise ValueError('PILOT_AUDIT_BINDING')
    source=(out/'authorization_source.json').read_bytes();sha=hashlib.sha256(source).hexdigest()
    import json
    if json.loads(source)!=grant:raise ValueError('PILOT_AUDIT_EXACT_GRANT')
    claim=read_json(out/'worker_claim.json') if (out/'worker_claim.json').exists() else None
    if claim and claim.get('authorization_sha256')!=sha:raise ValueError('PILOT_AUDIT_CLAIM')
    missing=[]
    expected=['input_reference.json','primitive_validation.json','actual_prepared_representation.json','parallel_resource_receipt.json','actual_bounds_v2.json','actual_coverage.json','coverage_comparison_v3.json','cost_summary.json','worker_terminal.json','worker_claim.json','worker.log']
    expected += ['phase_'+p+'.json' for p in ('input_reference','correctness','wrapper_cost')]
    expected += [c['id']+'_correctness.json' for c in plan()['cells']]
    expected += [c['id']+'_rep%d_trajectory.json'%i for c in plan()['cells'] for i in range(c['replicas'])]
    expected += ['wrapper_%02d.json'%i for i in range(36)]
    for p in expected:
        if not (out/p).exists():missing.append(p)
    success=terminal.get('status')=='H6_TECHNICAL_PILOT_COMPLETE'
    if (terminal.get('mandatory_stop') is not True or terminal.get('next_stage_authorized') is not False or terminal.get('H6_status')!='H6_NOT_AUTHORIZED'
        or terminal.get('contract_status')!='DRAFT_NOT_AUTHORIZATION' or terminal.get('N') is not None or terminal.get('G') is not None
        or terminal.get('numerical_allowance_certified') is not False or terminal.get('accuracy_eligibility')!='UNDETERMINED'):
        raise ValueError('PILOT_AUDIT_STOP_FLAGS')
    if success:
        if missing:raise ValueError('PILOT_AUDIT_MISSING:'+','.join(missing))
        from .ax2b_h6_pilot_coverage_v2 import audit_coverage_records
        audit_coverage_records(out,manifest['plan']['coverage'])
        worker=read_json(out/'worker_terminal.json')
        if worker.get('source_commit')!=manifest['source_commit'] or worker.get('authorization_sha256')!=sha:
            raise ValueError('PILOT_AUDIT_SOURCE')
        if (worker.get('correctness_completed')!=7 or worker.get('compiled_wrappers')!=36
            or any(worker.get('calls_attempted',{}).get(k)!=n for k,n in [('compile',36),('trajectory',4),('occurrence',8)])):
            raise ValueError('PILOT_AUDIT_COUNTS')
        for i,task in enumerate(plan()['wrapper_tasks']):
            row=read_json(out/('wrapper_%02d.json'%i))
            if row['task']!=task:raise ValueError('PILOT_AUDIT_WRAPPER_TASK')
            tr=read_json(out/(task['cell_id']+'_rep%d_trajectory.json'%task['replica']))
            if row['event_digest']!=tr['event_digest'] or digest(tr['events'])!=tr['event_digest']:
                raise ValueError('PILOT_AUDIT_EVENT_PAIR')
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
    return {'schema':'track_a_h6_pilot_saved_audit_v2','status':'SAVED_PILOT_BYTES_PASS' if success else 'SAVED_PILOT_STOP_RECORDS',
        'source_commit':manifest['source_commit'],'manifest_digest':digest(manifest),'authorization_sha256':sha,
        'raw_file_hashes':hashes,'missing_records':missing,'scientific_result_recomputed':False,
        'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}
