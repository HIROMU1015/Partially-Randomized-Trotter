#!/usr/bin/env python3
"""Explanation-only one-shot launcher; old core is reused without arithmetic."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/trotterlib'))
import mechanism_explanation as core

SOURCE_FILES = ('src/trotterlib/mechanism_analysis.py', 'src/trotterlib/mechanism_explanation.py', 'scripts/run_mechanism_explanation.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def duplicate_guard(pairs):
    data={}
    for key,value in pairs:
        core.require(key not in data, 'DUPLICATE_JSON_KEY');data[key]=value
    return data

def read(path):
    path=Path(path);core.require(path.is_file() and not path.is_symlink(), 'READONLY_INPUT_FILE')
    return json.loads(path.read_bytes(),object_pairs_hook=duplicate_guard,
                      parse_constant=lambda x: (_ for _ in ()).throw(core.AnalysisError('NONFINITE_JSON')))

def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def no_symlink(path):
    p=Path(path)
    for item in (p,*p.parents): core.require(not item.is_symlink(), 'SYMLINK_OUTPUT_PATH')

def runtime():
    files={}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            if path.is_file() and ROOT not in path.parents:files[str(path)]=sha(path)
    python=Path(sys.executable).resolve()
    return dict(python=str(python),python_sha256=sha(python),version=sys.version,files=files)

def verify_source(contract):
    expected=contract['source_commit']
    core.require(type(expected) is str and len(expected)==40,'FIXED_SOURCE_COMMIT_REQUIRED')
    for name,digest in contract['source_files'].items():
        core.require(name in SOURCE_FILES,'SOURCE_CLOSURE')
        core.require(sha(ROOT/name)==digest,'SOURCE_OR_BINDING_MISMATCH')
        blob=subprocess.check_output(['git','--no-optional-locks','-c','maintenance.auto=false','-C',str(ROOT),'show',expected+':'+name])
        core.require(hashlib.sha256(blob).hexdigest()==digest,'SOURCE_COMMIT_BLOB_MISMATCH')
    core.require(set(contract['source_files'])==set(SOURCE_FILES),'SOURCE_CLOSURE')

def verify_runtime(contract):
    profile=read(contract['runtime_profile'])
    core.require(sha(contract['runtime_profile'])==contract['runtime_profile_sha256'],'RUNTIME_PROFILE_HASH')
    py=Path(sys.executable).resolve()
    core.require(str(py)==profile['python'] and sha(py)==profile['python_sha256'] and sys.version==profile['version'],'PYTHON_RUNTIME_MISMATCH')
    for name,digest in profile['files'].items():core.require(sha(name)==digest,'RUNTIME_CLOSURE_MISMATCH')

def gate(grant_path, contract_path):
    grant=read(grant_path)
    # No contract, payload or scientific source is read on this rejection path.
    core.require(all(grant.get(k) is True for k in ('approved','approved_by_user','runtime_authorization','one_shot')),
                 'NUMERICAL_AUTHORIZATION_REQUIRED')
    core.require(grant.get('next_stage_authorized') is False and grant.get('retry') is False,'NO_RETRY_OR_NEXT_STAGE')
    core.require(grant.get('kind')=='SAVED_REAL_VALUES' and grant.get('scope')=='TRACK_A_MECHANISM_EXPLANATION_PRIMARY32','APPROVAL_SCOPE')
    core.require(grant.get('approval_receipt') and grant.get('approval_receipt_sha256'),'EXPLICIT_USER_APPROVAL_RECEIPT_REQUIRED')
    core.require(sha(grant['approval_receipt'])==grant['approval_receipt_sha256'],'APPROVAL_RECEIPT_HASH')
    core.require(sha(contract_path)==grant.get('contract_sha256'),'CONTRACT_HASH')
    contract=read(contract_path)
    core.require(contract['schema']==core.SCHEMA and contract['scope']==grant['scope'],'CONTRACT_SCOPE')
    core.require(grant.get('source_commit')==contract['source_commit'],'SOURCE_OR_BINDING_MISMATCH')
    core.require(grant.get('input_binding_sha256')==contract['input_binding_sha256'],'SOURCE_OR_BINDING_MISMATCH')
    core.require(grant.get('target_ids')==contract['target_ids'] and len(contract['target_ids'])==32,'EXACT_PRIMARY32')
    core.require(grant.get('output_root')==contract['output_root'],'OUTPUT_BINDING')
    core.require(grant.get('core_results_sha256')==contract['core_results_sha256'],'FIXED_CORE_HASH_BINDING')
    core.require(grant.get('method_side_ids')==contract['method_side_ids'] and len(contract['method_side_ids'])==64,'EXACT_METHOD_SIDES')
    verify_source(contract)
    verify_runtime(contract)
    core.require(sha(contract['input_binding'])==contract['input_binding_sha256'],'INPUT_BINDING_HASH')
    binding=read(contract['input_binding'])
    core.require(binding['target_ids']==contract['target_ids'] and binding['kind']=='SAVED_REAL_VALUES','EXACT_PRIMARY32')
    core.require(binding['core_results_sha256']==contract['core_results_sha256'] and sha(binding['core_results'])==contract['core_results_sha256'],'FIXED_CORE_HASH_BINDING')
    core.require(binding['method_side_ids']==contract['method_side_ids'],'EXACT_METHOD_SIDES')
    for item in binding['input_files']:core.require(sha(item['path'])==item['sha256'],'INPUT_BYTES_MISMATCH')
    return grant,contract,binding

def prepare_output(root, allowed_parent, protected):
    root=Path(root);parent=Path(allowed_parent)
    core.require(root.is_absolute() and parent.is_absolute(),'ABSOLUTE_RUN_PATH')
    no_symlink(root);no_symlink(parent)
    core.require(root.parent.resolve()==parent.resolve() and parent.is_dir(),'NEW_RUN_PARENT_BINDING')
    for path in protected:
        p=Path(path).resolve();r=root.resolve()
        core.require(r!=p and p not in r.parents and r not in p.parents,'PROTECTED_PATH_WRITE_REFUSED')
    core.require(not root.exists(),'ONE_SHOT_OUTPUT_ALREADY_EXISTS')
    root.mkdir(mode=0o700)
    return root

def save_outputs(root, results, saved_core):
    write_new(root/'explanations.json',results)
    write_new(root/'joined_core_reference.json',dict(target_ids=[r['contrast_id'] for r in saved_core],core_recalculated=False,saved_core=saved_core))
    rows=[]
    for row in results:
        for method in ('PR','PF'):
            side=row['explanation']['sides'][method]
            rows.append(dict(contrast_id=row['contrast_id'],system=row['system'],method=method,status=side['status'],reason=side.get('reason'),explanation_json=json.dumps(side,allow_nan=False)))
    with (root/'explanation_sides.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return {name:sha(root/name) for name in ('explanations.json','joined_core_reference.json','explanation_sides.csv')}


def accept_outputs(root, binding, contract):
    results=read(root/'explanations.json')
    joined=read(root/'joined_core_reference.json')
    core.require([r['contrast_id'] for r in results]==contract['target_ids'],'OUTPUT_TARGETS')
    core.require(joined['target_ids']==contract['target_ids'] and joined['core_recalculated'] is False
                 and joined['saved_core']==read(binding['core_results']),'FIXED_CORE_COPY_MISMATCH')
    for row in results:
        core.require(row['core_reused_without_recalculation'] is True and
                     set(row['explanation']['sides'])=={'PR','PF'},'EXPLANATION_OUTPUT_SCHEMA')
        for method in ('PR','PF'):
            state=row['explanation']['sides'][method]['status']
            allowed=row['contrast_id']+'.'+method in contract['allowed_missing_side_ids']
            core.require(state=='EXPLANATION_COMPLETE' or (allowed and state=='EXPLANATORY_INPUT_MISSING'),
                         'EXPLANATION_NUMERIC_MISMATCH_OR_UNEXPECTED_MISSING')


def worker(grant_path,contract_path,root):
    grant,contract,binding=gate(grant_path,contract_path)
    root=Path(root);core.require(str(root)==contract['output_root'],'OUTPUT_BINDING')
    marker=read(root/'ONE_SHOT.json')
    protected_hashes={str(p):sha(p) for p in [Path(grant_path),Path(contract_path),Path(grant['approval_receipt']),root/'ONE_SHOT.json',root/'ledger.json']}
    core.require(marker['parent_pid']==os.getppid() and marker['grant_sha256']==sha(grant_path)
                 and marker['contract_sha256']==sha(contract_path),'OWN_PARENT_ONE_SHOT_REQUIRED')
    write_new(root/'WORKER_CLAIM.json',dict(pid=os.getpid(),parent_pid=os.getppid(),one_shot=True))
    bundle=read(binding['payload'])
    core.require(sha(binding['payload'])==binding['payload_sha256'] and bundle['target_ids']==contract['target_ids'],'PAYLOAD_BINDING')
    saved_core=read(binding['core_results'])
    core.require(sha(binding['core_results'])==contract['core_results_sha256'],'FIXED_CORE_HASH_BINDING')
    results=core.analyze_bundle(bundle,saved_core,core._authorized_production_permit())
    for item in binding['input_files']:core.require(sha(item['path'])==item['sha256'],'POST_INPUT_BYTES_MISMATCH')
    verify_source(contract);verify_runtime(contract)
    for path,digest in protected_hashes.items():core.require(sha(path)==digest,'POST_GRANT_LEDGER_MARKER_MISMATCH')
    hashes=save_outputs(root,results,saved_core)
    accept_outputs(root,binding,contract)
    write_new(root/'worker_result.json',dict(status='COMPLETE_MANDATORY_STOP',target_ids=contract['target_ids'],hashes=hashes,new_statistics=0,next_stage_authorized=False))

def launch(grant_path,contract_path):
    grant,contract,binding=gate(grant_path,contract_path)
    protected=[str(ROOT),str(Path(grant_path).resolve()),str(Path(contract_path).resolve()),contract['input_binding'],contract['runtime_profile']]+[r['path'] for r in binding['input_files']]+binding.get('protected_paths',[])
    root=prepare_output(contract['output_root'],contract['allowed_run_parent'],protected)
    write_new(root/'ONE_SHOT.json',dict(parent_pid=os.getpid(),grant_sha256=sha(grant_path),contract_sha256=sha(contract_path),consumed=True,analysis_pass_reserved=1))
    write_new(root/'ledger.json',dict(explanation_analysis_pass=1,contrasts=32,method_sides=64,core_reconstruction=0,sampling=0,compile=0,signal=0,statistics=0,refund=False))
    protected_hashes={str(p):sha(p) for p in [Path(grant_path),Path(contract_path),Path(grant['approval_receipt']),root/'ONE_SHOT.json',root/'ledger.json']}
    argv=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--worker','--grant',str(grant_path),'--contract',str(contract_path),'--run-root',str(root)]
    try:
        process=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        stdout,stderr=process.communicate()
        write_new(root/'native_wait.json',dict(pid=process.pid,native_exit_code=process.returncode,reaped=process.poll() is not None,worker_descendants=0,stdout=stdout,stderr=stderr))
        core.require(process.returncode==0,'NATIVE_WORKER_FAILURE')
        result=read(root/'worker_result.json');core.require(result['target_ids']==contract['target_ids'],'OUTPUT_TARGETS')
        core.require(result.get('status')=='COMPLETE_MANDATORY_STOP' and set(result['hashes'])=={'explanations.json','joined_core_reference.json','explanation_sides.csv'},'OUTPUT_SCHEMA')
        for name,digest in result['hashes'].items():core.require(sha(root/name)==digest,'RESULT_SHA_MISMATCH')
        accept_outputs(root,binding,contract)
        for item in binding['input_files']:core.require(sha(item['path'])==item['sha256'],'POST_INPUT_BYTES_MISMATCH')
        verify_source(contract);verify_runtime(contract)
        for path,digest in protected_hashes.items():core.require(sha(path)==digest,'POST_GRANT_LEDGER_MARKER_MISMATCH')
        write_new(root/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',native_exit_code=0,reaped=True,result_sha256=sha(root/'worker_result.json'),source_commit=contract['source_commit'],next_stage_authorized=False))
    except BaseException as exc:
        if 'process' in locals() and process.poll() is None:
            process.terminate();process.communicate()
            if not (root/'native_wait.json').exists():
                write_new(root/'native_wait.json',dict(pid=process.pid,native_exit_code=process.returncode,reaped=True,reason='OWN_CHILD_EXCEPTION_CLEANUP'))
        write_new(root/'STOP.json',dict(status='FAILED_OR_STOPPED',reason=str(exc),retry=False,refund=False,next_stage_authorized=False))
        raise

def artificial(path,root):
    bundle=read(path)
    core.require(bundle.get('kind')=='ARTIFICIAL' and all(r.get('artificial') is True for r in bundle.get('comparisons',[])), 'ARTIFICIAL_INPUT_ONLY')
    root=prepare_output(root,Path(root).parent,[str(ROOT),str(Path(path).resolve())])
    saved_core=bundle['saved_core_artificial']
    results=core.analyze_bundle(bundle,saved_core,core.artificial_permit())
    hashes=save_outputs(root,results,saved_core)
    write_new(root/'terminal.json',dict(status='ARTIFICIAL_ONLY',hashes=hashes,science_calls=0))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grant',type=Path);p.add_argument('--contract',type=Path)
    p.add_argument('--worker',action='store_true');p.add_argument('--run-root',type=Path)
    p.add_argument('--artificial-fixture',type=Path)
    args=p.parse_args()
    if args.artificial_fixture:
        core.require(not args.grant and not args.contract and not args.worker and args.run_root is not None,'ARTIFICIAL_CLI_SCOPE')
        artificial(args.artificial_fixture,args.run_root)
    else:
        core.require(args.grant is not None and args.contract is not None,'GRANT_AND_CONTRACT_REQUIRED')
        worker(args.grant,args.contract,args.run_root) if args.worker else launch(args.grant,args.contract)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);raise SystemExit(1)
