"""Byte-bound handover; never repair an old STOP or load a scientific array.

The new monitored driver announces readiness before the operator stops the old
owned launcher. A single exclusive release is issued only after native cleanup.
Validated QPY and completed-prefix measurements retain their original SOURCE.
"""
from fractions import Fraction
import math
import os
from pathlib import Path
import shutil
import time
import h8_key_compile_contract as c
import h8_key_compile_parallel as pool

OLD_SOURCE='352581103b7dbd4fa58e28af7a6ca65570d36e8e'


def header(config,grant):
    return dict(SOURCE=config['SOURCE'],config_sha256=c.digest(config),grant_sha256=c.digest(grant))


def check_header(receipt,config,grant):
    c.require(all(receipt.get(k)==v for k,v in header(config,grant).items()),'HANDOVER_RELEASE_BINDING')


def publish(path,value):
    """Publish a complete JSON atomically without replacing an existing release."""
    pending=path.with_suffix('.pending');c.save(pending,value)
    os.link(pending,path);pending.unlink()


def wait_for_release(config,grant):
    if not config.get('handover'):return
    h=config['handover'];path=c.home(h['release_path'])
    c.require(path.parent==Path(config['control']),'RELEASE_IN_OWN_CONTROL')
    publish(path.parent/'HANDOVER_READY.json',dict(status='READY_NO_SCIENCE_STARTED',
        **header(config,grant),driver_identity=pool.identity(os.getpid()),old_SOURCE=OLD_SOURCE))
    while not path.exists():time.sleep(.2)
    receipt=c.read(path);check_header(receipt,config,grant)
    c.require(receipt.get('status')=='PASS_OLD_NATIVE_EXIT_AND_REUSABLE_PROGRESS',
              'HANDOVER_RELEASE_NOT_PASS')


def audit_old(config,*,require_exit=True):
    h=config['handover'];root=c.home(h['old_root']);out=root/'output';b=c.binding();files={}
    def record(path):
        path=c.home(path)
        c.require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(),'OLD_PROOF_PATH')
        row=dict(path=str(path),bytes=path.stat().st_size,sha256=c.sha(path));files[str(path)]=row
        return path
    def read(path):return c.read(record(path))
    cfgpath=root/'metadata/config_SEALED_PRIVATE.json';grantpath=root/'metadata/grant_APPROVED_PRIVATE.json'
    c.require(c.sha(cfgpath)==h['old_config_file_sha256'] and c.sha(grantpath)==h['old_grant_file_sha256'],
              'OLD_IMMUTABLE_AUTH_BYTES')
    old=read(cfgpath);oldgrant=read(grantpath)
    c.require(old['SOURCE']==OLD_SOURCE==h['old_SOURCE']==oldgrant['SOURCE'] and
        oldgrant['approved']is True and oldgrant['runtime_authorization']is True and
        oldgrant['config_sha256']==c.digest(old) and oldgrant['plan_sha256']==old['plan_sha256'] and
        old['expected_profile_digest']==config['expected_profile_digest']==c.PROFILE and
        old['provider_hashes']==config['provider_hashes'] and old['runtime_file_hashes']==config['runtime_file_hashes'] and
        old['files']==config['files'] and old['input_mapping']==config['input_mapping'] and
        c.sha(root/'source'/c.INPUTS)==c.sha(c.ROOT/c.INPUTS),'OLD_SCIENCE_INPUT_COMPILER_BINDING')
    for name,sha in old['source_hashes'].items():
        c.require(c.sha(record(root/'source'/name))==sha,'OLD_SOURCE_CHANGED:'+name)
    def group(path,ids):
        receipt=read(path/'parallel_receipt.json')
        c.require(receipt['status']=='PASS' and receipt['all_reaped']is True and not receipt['cleanup_errors'] and
            [r['task_id'] for r in receipt['workers']]==ids,'OLD_VALIDATION_NATIVE_CLEANUP')
        result={}
        for worker in receipt['workers']:
            expected=path/worker['task_id'];p=expected/'task_result.json'
            c.require(Path(worker['directory'])==expected and worker['exit_code']==0 and worker['reaped']is True and
                      c.sha(p)==worker['result_sha256'],'OLD_VALIDATION_RESULT_HASH')
            result[worker['task_id']]=read(p)
        return result
    prepared={}
    for case in b['cases']:
        proofpath=out/(case['id']+'_control_preparation.json')
        if not proofpath.exists():continue
        proof=read(proofpath);identity=b['prefix_preparations'][str(case['prefix'])]
        target=next(r for r in b['targets'] if r['case_id']==case['id'])
        scalar=float.fromhex(identity['constant_hex'])
        if case['method']=='B0I':
            saved=target['scalar_receipt']
            c.require(saved['policy']=='EXACT_STORED_BINARY64_TAIL_IDENTITY' and
                float(Fraction(saved['numerator'])/Fraction(saved['denominator']))==saved['C'], 'OLD_SCALAR_POLICY')
            scalar+=saved['C']
        c.require(proof['SOURCE']==OLD_SOURCE and proof['profile_sha256']==c.PROFILE and
            proof['case']==case and proof['preparation_binding']==identity and proof['scalar_hex']==scalar.hex() and
            math.isfinite(proof['max_control_error']) and 0<=proof['max_control_error']<1e-8,
            'OLD_CONTROL_PREPARATION_BINDING')
        directional=group(out/(case['id']+'_directional_controls'),['left','right'])
        errors=directional['left']['errors']+directional['right']['errors']
        c.require(len(errors)==4 and all(math.isfinite(e) and 0<=e<1e-8 for e in errors),'OLD_DIRECTIONAL_ERRORS')
        overlap=complex(**dict(real=directional['right']['overlap']['real'],imag=directional['right']['overlap']['imag']))
        c.require(all(abs(overlap-complex(target[k]['real'],target[k]['imag']))<1e-8
                      for k in ('corrected','independent_forward')),'OLD_UNROTATED_CONTROL_TARGET')
        axes=group(out/(case['id']+'_axis_controls_and_fingerprints'),['cosine','sine'])
        for axis,row in axes.items():
            spool=proof['spools'][axis];p=c.home(spool['path'])
            c.require(p.is_relative_to(out) and row['axis']==axis and row['qpy_spool']==spool and
                row['fingerprint']==proof['fingerprints'][axis] and math.isfinite(row['max_control_error']) and
                0<=row['max_control_error']<1e-8 and p.stat().st_size==spool['bytes'] and c.sha(p)==spool['sha256'],
                'OLD_AXIS_SPOOL_OR_FINGERPRINT')
            errors.append(row['max_control_error']);record(p)
        c.require(max(errors)==proof['max_control_error'],'OLD_MAX_CONTROL_ERROR')
        prepared[case['id']]=dict(case=case,identity=identity,scalar_hex=proof['scalar_hex'],
            max_control_error=proof['max_control_error'],spools=proof['spools'],fingerprints=proof['fingerprints'],
            preparation_SOURCE=OLD_SOURCE,preparation_proof_sha256=c.sha(proofpath))
    rows=[];missing=False
    for i,axis in enumerate(b['axes'],1):
        path=out/('axis_'+str(i).zfill(2)+'_measured.json')
        if not path.exists():missing=True;continue
        c.require(not missing and axis['case_id'] in prepared,'OLD_COMPLETED_AXES_MUST_BE_PREFIX')
        row=read(path);case=next(r for r in b['cases'] if r['id']==axis['case_id']);built=prepared[case['id']]
        check=c.axis_check(axis['exact_prediction'],row['measured_RZ'])
        c.require(check['PASS'],'OLD_SCIENTIFIC_MODEL_FAILURE_NO_RESTART')
        c.require(row['SOURCE']==OLD_SOURCE and row['model_check']==check and row['model_sha256']==c.MODEL and
            row['compiler_hash']==c.COMPILER and row['profile_sha256']==c.PROFILE and row['native_exit_code']==0 and
            row['reaped']is True and (row['case_id'],row['axis'])==(axis['case_id'],axis['axis']) and
            row['preparation_binding']==built['identity'] and row['scalar_hex']==built['scalar_hex'] and
            row['max_control_error']==built['max_control_error'] and
            row['fingerprint_before']==row['fingerprint_after']==built['fingerprints'][axis['axis']],
            'OLD_COMPLETED_AXIS_BINDING')
        measurement=group(out/('compile_'+str(i).zfill(2)),[case['id']+'_'+axis['axis']])
        raw=measurement[case['id']+'_'+axis['axis']]
        c.require(all(row[k]==v for k,v in raw.items()) and
            row['result_sha256']==c.sha(out/('compile_'+str(i).zfill(2))/(case['id']+'_'+axis['axis'])/'task_result.json'),
            'OLD_COMPILE_RAW_RESULT')
        rows.append(row)
    counts={k:0 for k in c.CAPS}
    for key in counts:
        for path in out.glob('reservation_'+key+'_*.json'):
            read(path);counts[key]+=1
    if require_exit:
        native=read(root/'metadata/launcher_native_exit.json');driver=read(root/'control/driver/parallel_receipt.json')
        c.require(native['SOURCE']==OLD_SOURCE and native['status']=='NATIVE_LAUNCHER_REAPED' and native['reaped']is True and
                  driver['all_reaped']is True and not driver['cleanup_errors'],'OLD_NATIVE_EXIT_OR_CLEANUP')
        terminal=read(out/'terminal.json')
        c.require(native['native_exit_code']==(0 if terminal['status']=='COMPLETE_MANDATORY_STOP' else 1),
                  'OLD_NATIVE_EXIT_CODE')
        c.require(terminal['status']=='COMPLETE_MANDATORY_STOP' or
            read(root/'control/STOP.json')['first_stop']=='OWN_LAUNCHER_SIGNAL:15','OLD_STOP_NOT_AUTHORIZED_HANDOVER')
        c.require('MODEL_ERROR' not in terminal.get('first_stop',''),'OLD_SCIENTIFIC_STOP')
        for path in [root/'control/driver/parallel_receipt.json',*out.glob('*/parallel_receipt.json')]:
            receipt=read(path)
            c.require(receipt['all_reaped']is True and not receipt['cleanup_errors'],'OLD_NESTED_CLEANUP')
            for worker in receipt['workers']:
                c.require(worker['reaped']is True,'OLD_WORKER_NOT_REAPED')
                current=pool.identity(worker['identity'][0]);prior=worker['identity']
                c.require(current is None or (current[0],current[1],current[3])!=(prior[0],prior[1],prior[3]),
                          'OLD_OWNED_PROCESS_STILL_PRESENT')
    return dict(prepared=prepared,rows=rows,prior_reserved_calls_nonrefundable=counts,
                files=list(files.values()),old_SOURCE=OLD_SOURCE,old_run_STOP_preserved=True)


def issue_release(config,grant):
    c.authorize(config,grant)
    ready=c.read(Path(config['control'])/'HANDOVER_READY.json');check_header(ready,config,grant)
    receipt=dict(status='PASS_OLD_NATIVE_EXIT_AND_REUSABLE_PROGRESS',**header(config,grant),**audit_old(config))
    publish(c.home(config['handover']['release_path']),receipt)
    return receipt


def import_prepared(config,grant,output):
    if not config.get('handover'):return {},[],{}
    receipt=c.read(config['handover']['release_path']);check_header(receipt,config,grant)
    c.require(receipt['status']=='PASS_OLD_NATIVE_EXIT_AND_REUSABLE_PROGRESS','HANDOVER_NOT_PASS')
    audited=audit_old(config)
    c.require(all(receipt[k]==v for k,v in audited.items()),'OLD_RECEIPT_CHANGED_AFTER_RELEASE')
    root=Path(output)/'imported_preparation';root.mkdir(mode=0o700)
    for case_id,built in audited['prepared'].items():
        directory=root/case_id;directory.mkdir()
        for axis,spool in list(built['spools'].items()):
            target=directory/(axis+'.qpy')
            with Path(spool['path']).open('rb') as source,target.open('xb') as dest:
                shutil.copyfileobj(source,dest,2**20);dest.flush();os.fsync(dest.fileno())
            c.require(target.stat().st_size==spool['bytes'] and c.sha(target)==spool['sha256'],'IMPORTED_QPY_BYTES')
            built['spools'][axis]=dict(spool,path=str(target))
    c.save(Path(output)/'handover_receipt.json',dict(**header(config,grant),
        release_sha256=c.sha(config['handover']['release_path']),old_SOURCE=OLD_SOURCE,
        reused_case_ids=list(audited['prepared']),reused_axis_count=len(audited['rows']),
        prior_reserved_calls_nonrefundable=audited['prior_reserved_calls_nonrefundable'],
        prior_STOP_preserved=True,scientific_tolerances_changed=False))
    return audited['prepared'],audited['rows'],audited['prior_reserved_calls_nonrefundable']
