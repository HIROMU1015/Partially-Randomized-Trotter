#!/usr/bin/env python3
"""One-shot A82 PR-side launcher. Saved real data require a new bound grant."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src/trotterlib'))
import mechanism_a82 as adapter
import run_mechanism_explanation as io

SOURCE_FILES = ('src/trotterlib/mechanism_analysis.py',
    'src/trotterlib/mechanism_explanation.py','scripts/run_mechanism_explanation.py',
    'src/trotterlib/mechanism_a82.py','scripts/run_mechanism_a82.py')
require = adapter.require


def verify_source(contract):
    require(set(contract['source_files']) == set(SOURCE_FILES), 'A82_SOURCE_CLOSURE')
    commit = contract['source_commit']
    require(type(commit) is str and len(commit) == 40, 'FIXED_SOURCE_COMMIT_REQUIRED')
    for name,digest in contract['source_files'].items():
        require(io.sha(ROOT/name) == digest, 'SOURCE_BYTES_MISMATCH')
        blob = subprocess.check_output(['git','--no-optional-locks','-c','maintenance.auto=false',
            '-C',str(ROOT),'show',commit+':'+name])
        require(hashlib.sha256(blob).hexdigest() == digest, 'SOURCE_COMMIT_BLOB_MISMATCH')


def gate(grant_path, contract_path):
    grant = io.read(grant_path)
    # DRAFT rejects before contract, runtime or saved scientific input access.
    require(all(grant.get(k) is True for k in ('approved','approved_by_user','runtime_authorization','one_shot')),
            'NUMERICAL_AUTHORIZATION_REQUIRED')
    require(grant.get('scope') == adapter.SCOPE and grant.get('kind') == 'SAVED_REAL_VALUES', 'A82_ONLY_SCOPE')
    require(grant.get('method_side_ids') == [adapter.SIDE] and grant.get('axes') == list(adapter.AXES), 'A82_ONLY_SIDE_AXES')
    require(grant.get('retry') is False and grant.get('next_stage_authorized') is False, 'NO_RETRY_OR_NEXT_STAGE')
    require(grant.get('approval_receipt') and grant.get('approval_receipt_sha256'), 'NEW_EXPLICIT_APPROVAL_REQUIRED')
    require(io.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT_SHA')
    require(io.sha(contract_path) == grant['contract_sha256'], 'CONTRACT_SHA')
    contract = io.read(contract_path)
    require(contract['schema'] == adapter.SCHEMA and contract['scope'] == adapter.SCOPE and
            contract['method_side_ids'] == [adapter.SIDE] and contract['axes'] == list(adapter.AXES), 'CONTRACT_A82_ONLY')
    require(contract['calls'] == adapter.CALLS and contract['retry'] is False and
            contract['next_stage_authorized'] is False, 'FIXED_CALLS_AND_STOP')
    for key in ('source_commit','input_binding_sha256','old_core_sha256','old_explanations_sha256','output_root'):
        require(grant.get(key) == contract[key], 'GRANT_CONTRACT_BINDING_'+key)
    verify_source(contract)
    io.verify_runtime(contract)
    require(io.sha(contract['input_binding']) == contract['input_binding_sha256'], 'INPUT_BINDING_SHA')
    binding = io.read(contract['input_binding'])
    require(binding['schema'] == adapter.SCHEMA and binding['method_side_ids'] == [adapter.SIDE], 'BOUND_SIDE_SCOPE')
    for key in ('old_core_sha256','old_explanations_sha256'):
        require(binding[key] == contract[key], 'ARCHIVE_SHA_BINDING')
    for item in binding['input_files']:
        require(io.sha(item['path']) == item['sha256'], 'INPUT_BYTES_MISMATCH')
    for key in ('payload','old_core','old_explanations'):
        require(io.sha(binding[key]) == binding[key+'_sha256'], 'CANONICAL_INPUT_SHA')
    return grant, contract, binding


def accept(root, binding):
    result = io.read(root/'A82_PR_delta.json')
    archived = io.read(binding['old_explanations'])
    old = next(r for r in archived if r['contrast_id'] == adapter.TARGET)
    require(result['schema'] == adapter.SCHEMA and result['target_side_id'] == adapter.SIDE and
            result['new_PR_side']['status'] == 'EXPLANATION_COMPLETE' and
            set(result['new_PR_side']['axes']) == set(adapter.AXES), 'A82_OUTPUT_SCHEMA')
    require(result['inherited_PF_side'] == old['explanation']['sides']['PF'], 'PF_COPY_CHANGED')
    require(result['calls'] == adapter.CALLS and result['old_archives_modified'] is False and
            result['new_comparison_factors_computed'] is False and result['next_stage_authorized'] is False,
            'A82_OUTPUT_SCOPE')
    receipt = io.read(root/'inheritance_receipt.json')
    require(receipt == dict(old_core_sha256=binding['old_core_sha256'],
        old_explanations_sha256=binding['old_explanations_sha256'],
        core_recalculated=False,other_63_sides_recalculated=False,historical_results_overwritten=False),
        'INHERITANCE_RECEIPT')


def protected_hashes(grant_path, contract_path, grant, root):
    return {str(p):io.sha(p) for p in (Path(grant_path),Path(contract_path),Path(grant['approval_receipt']),root/'ONE_SHOT.json',root/'ledger.json')}


def post_gate(grant_path, contract_path, protected):
    gate(grant_path,contract_path)
    for path,digest in protected.items():require(io.sha(path) == digest, 'POST_CONTROL_SHA_MISMATCH')


def worker(grant_path, contract_path, run_root):
    grant,contract,binding = gate(grant_path,contract_path)
    root = Path(run_root)
    require(str(root) == contract['output_root'], 'OUTPUT_BINDING')
    marker = io.read(root/'ONE_SHOT.json')
    require(marker['parent_pid'] == os.getppid() and marker['consumed'] is True and
            marker['grant_sha256'] == io.sha(grant_path) and marker['contract_sha256'] == io.sha(contract_path),
            'OWN_PARENT_ONE_SHOT_REQUIRED')
    protected = protected_hashes(grant_path,contract_path,grant,root)
    io.write_new(root/'WORKER_CLAIM.json',dict(pid=os.getpid(),parent_pid=os.getppid(),one_shot=True))
    result = adapter.analyze_delta(io.read(binding['payload']),io.read(binding['old_core']),
        io.read(binding['old_explanations']),adapter.equations._authorized_production_permit())
    post_gate(grant_path,contract_path,protected)
    io.write_new(root/'A82_PR_delta.json',result)
    io.write_new(root/'inheritance_receipt.json',dict(old_core_sha256=binding['old_core_sha256'],
        old_explanations_sha256=binding['old_explanations_sha256'],core_recalculated=False,
        other_63_sides_recalculated=False,historical_results_overwritten=False))
    accept(root,binding)
    io.write_new(root/'worker_result.json',dict(status='COMPLETE_MANDATORY_STOP',
        target_side_id=adapter.SIDE,hashes={n:io.sha(root/n) for n in ('A82_PR_delta.json','inheritance_receipt.json')},
        calls=adapter.CALLS,next_stage_authorized=False))


def launch(grant_path, contract_path):
    grant,contract,binding = gate(grant_path,contract_path)
    protected_paths = [str(ROOT),str(Path(grant_path).resolve()),str(Path(contract_path).resolve()),
        contract['input_binding'],contract['runtime_profile']]+[r['path'] for r in binding['input_files']]+binding['protected_paths']
    root = io.prepare_output(contract['output_root'],contract['allowed_run_parent'],protected_paths)
    io.write_new(root/'ONE_SHOT.json',dict(parent_pid=os.getpid(),grant_sha256=io.sha(grant_path),
        contract_sha256=io.sha(contract_path),consumed=True,analysis_pass_reserved=1))
    io.write_new(root/'ledger.json',dict(scope=adapter.SCOPE,calls=adapter.CALLS,refund=False))
    protected = protected_hashes(grant_path,contract_path,grant,root)
    try:
        process = subprocess.Popen([sys.executable,'-I','-B',str(Path(__file__).resolve()),
            '--worker','--grant',str(grant_path),'--contract',str(contract_path),'--run-root',str(root)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        stdout,stderr = process.communicate()
        io.write_new(root/'native_wait.json',dict(pid=process.pid,native_exit_code=process.returncode,
            reaped=process.poll() is not None,stdout=stdout,stderr=stderr,worker_descendants=0))
        require(process.returncode == 0, 'NATIVE_WORKER_FAILURE')
        result = io.read(root/'worker_result.json')
        require(result['status'] == 'COMPLETE_MANDATORY_STOP' and result['target_side_id'] == adapter.SIDE and
                result['calls'] == adapter.CALLS and result['next_stage_authorized'] is False and
                set(result['hashes']) == {'A82_PR_delta.json','inheritance_receipt.json'}, 'WORKER_RESULT_SCHEMA')
        for name,digest in result['hashes'].items():require(io.sha(root/name) == digest, 'OUTPUT_SHA_MISMATCH')
        accept(root,binding);post_gate(grant_path,contract_path,protected)
        io.write_new(root/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',native_exit_code=0,reaped=True,
            source_commit=contract['source_commit'],result_sha256=io.sha(root/'worker_result.json'),next_stage_authorized=False))
    except BaseException as exc:
        if 'process' in locals() and process.poll() is None:
            process.terminate();process.communicate()
            if not (root/'native_wait.json').exists():
                io.write_new(root/'native_wait.json',dict(pid=process.pid,native_exit_code=process.returncode,reaped=True))
        io.write_new(root/'STOP.json',dict(status='FAILED_OR_STOPPED',reason=str(exc),retry=False,refund=False,next_stage_authorized=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grant',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--run-root',type=Path)
    args=parser.parse_args()
    require(not args.worker or args.run_root is not None, 'WORKER_ROOT_REQUIRED')
    worker(args.grant,args.contract,args.run_root) if args.worker else launch(args.grant,args.contract)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);raise SystemExit(1)
