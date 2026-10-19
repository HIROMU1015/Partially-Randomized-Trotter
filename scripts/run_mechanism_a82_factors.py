#!/usr/bin/env python3
"""One-shot saved A82 factor delta. Invalid grants reject before input access."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/trotterlib'))
sys.path.insert(0, str(ROOT/'scripts'))
import mechanism_a82_factors as adapter
import run_mechanism_explanation as io

SOURCE_FILES = ('src/trotterlib/mechanism_analysis.py',
    'src/trotterlib/mechanism_explanation.py', 'scripts/run_mechanism_explanation.py',
    'src/trotterlib/mechanism_a82_factors.py', 'scripts/run_mechanism_a82_factors.py')
REFERENCES = ('payload', 'delta', 'old_core', 'old_explanations', 'prior_summary',
    'side_acceptance', 'side_terminal', 'side_worker_result', 'side_native_wait', 'side_launcher_wait')
OUTPUTS = ('A82_factor_delta.json', 'inheritance_receipt.json')
require = adapter.require


def verify_source(contract):
    require(set(contract['source_files']) == set(SOURCE_FILES), 'FACTOR_SOURCE_CLOSURE')
    commit = contract['source_commit']
    require(type(commit) is str and len(commit) == 40, 'FIXED_SOURCE_COMMIT_REQUIRED')
    for name, digest in contract['source_files'].items():
        require(io.sha(ROOT/name) == digest, 'SOURCE_BYTES_MISMATCH')
        blob = subprocess.check_output(['git', '--no-optional-locks', '-c', 'maintenance.auto=false',
            '-C', str(ROOT), 'show', commit+':'+name])
        require(hashlib.sha256(blob).hexdigest() == digest, 'SOURCE_COMMIT_BLOB_MISMATCH')


def verify_inputs(contract):
    """SHA, source provenance and saved JSON joins only, no factor arithmetic."""
    require(io.sha(contract['input_binding']) == contract['input_binding_sha256'], 'INPUT_BINDING_SHA')
    b = io.read(contract['input_binding'])
    require(b['schema'] == adapter.SCHEMA and b['target_ids'] == [adapter.TARGET], 'BOUND_SCOPE')
    for key in ('delta_sha256', 'old_core_sha256', 'old_explanations_sha256'):
        require(b[key] == contract[key], 'RESULT_DIGEST_BINDING')
    for item in b['input_files']:
        require(io.sha(item['path']) == item['sha256'], 'INPUT_BYTES_MISMATCH')
    for key in REFERENCES:
        require(io.sha(b[key]) == b[key+'_sha256'], 'CANONICAL_INPUT_SHA')
    for item in b['fixed_git_inputs']:
        blob = subprocess.check_output(['git', '--no-optional-locks', '-c', 'maintenance.auto=false',
            '-C', str(ROOT), 'show', item['commit']+':'+item['repo_path']])
        require(hashlib.sha256(blob).hexdigest() == item['sha256'], 'PRIOR_RESULT_GIT_BLOB')
    summary = io.read(b['prior_summary']); receipt = io.read(b['side_acceptance'])
    require(summary['source_commit'] == adapter.SIDE_SOURCE and summary['delta_sha256'] == b['delta_sha256'] and
            summary['new_PR_status'] == 'EXPLANATION_COMPLETE' and
            summary['effective_method_side_validation_complete'] == 64 and
            summary['new_comparative_factor_status'] == 'NOT_COMPUTED_NOT_AUTHORIZED', 'PRIOR_SIDE_STATUS')
    require(receipt['status'] == 'PASS_A82_PR_TWO_AXES_COMPLETE_MANDATORY_STOP' and
            receipt['source_commit'] == adapter.SIDE_SOURCE and receipt['new_delta_sha256'] == b['delta_sha256'] and
            receipt['original_core_sha256'] == b['old_core_sha256'] and
            receipt['original_explanations_sha256'] == b['old_explanations_sha256'] and
            receipt['one_shot_invocations'] == 1 and receipt['parent_and_worker_reaped'] is True,
            'SIDE_VALIDATION_RECEIPT')
    terminal = io.read(b['side_terminal']); worker = io.read(b['side_worker_result'])
    require(terminal['status'] == 'COMPLETE_MANDATORY_STOP' and terminal['native_exit_code'] == 0 and
            terminal['reaped'] is True and terminal['result_sha256'] == b['side_worker_result_sha256'] and
            worker['hashes']['A82_PR_delta.json'] == b['delta_sha256'], 'SIDE_NATIVE_RESULT_SHA')
    require(all(io.read(b[key])['native_exit_code'] == 0 and io.read(b[key])['reaped'] is True
                for key in ('side_native_wait', 'side_launcher_wait')), 'SIDE_NATIVE_WAIT')
    bundle = io.read(b['payload'])
    require(bundle['kind'] == 'SAVED_REAL_VALUES' and bundle['artificial'] is False and
            bundle['input_result_digests'] == {k:b[k] for k in
                ('delta_sha256', 'old_core_sha256', 'old_explanations_sha256')}, 'BOUND_PAYLOAD')
    adapter.validate_saved_lineage(bundle, io.read(b['delta']), io.read(b['old_core']), io.read(b['old_explanations']))
    return b


def gate(grant_path, contract_path):
    grant = io.read(grant_path)
    require(all(grant.get(k) is True for k in ('approved', 'approved_by_user', 'runtime_authorization', 'one_shot')),
            'NUMERICAL_AUTHORIZATION_REQUIRED')
    require(grant.get('schema') == adapter.SCHEMA and grant.get('scope') == adapter.SCOPE and
            grant.get('kind') == 'SAVED_REAL_VALUES' and grant.get('target_ids') == [adapter.TARGET] and
            grant.get('axes') == list(adapter.AXES), 'FACTOR_ONLY_SCOPE')
    require(grant.get('retry') is False and grant.get('next_stage_authorized') is False, 'NO_RETRY_OR_NEXT_STAGE')
    require(grant.get('approval_receipt') and grant.get('approval_receipt_sha256'), 'NEW_APPROVAL_RECEIPT_REQUIRED')
    require(io.sha(grant['approval_receipt']) == grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT_SHA')
    approval = io.read(grant['approval_receipt'])
    require(approval.get('approved_by_user') is True and approval.get('factor_computation_authorized') is True and
            approval.get('scope') == adapter.SCOPE and approval.get('target_ids') == [adapter.TARGET] and
            approval.get('axes') == list(adapter.AXES) and approval.get('retry') is False,
            'NEW_EXPLICIT_FACTOR_APPROVAL_REQUIRED')
    require(io.sha(contract_path) == grant['contract_sha256'] == approval['contract_sha256'], 'CONTRACT_SHA')
    c = io.read(contract_path)
    require(c['schema'] == adapter.SCHEMA and c['scope'] == adapter.SCOPE and
            c['target_ids'] == [adapter.TARGET] and c['axes'] == list(adapter.AXES) and
            c['factor_names'] == list(adapter.FACTORS) and c['calls'] == adapter.CALLS and
            c['retry'] is False and c['next_stage_authorized'] is False, 'CONTRACT_FIXED_SCOPE')
    for key in ('source_commit', 'input_binding_sha256', 'delta_sha256',
                'old_core_sha256', 'old_explanations_sha256', 'output_root'):
        require(grant.get(key) == c[key], 'GRANT_CONTRACT_BINDING_'+key)
    require(approval['source_commit'] == c['source_commit'] and
            approval['input_binding_sha256'] == c['input_binding_sha256'], 'APPROVAL_SOURCE_INPUT_BINDING')
    verify_source(c); io.verify_runtime(c)
    return grant, c, verify_inputs(c)


def inheritance(binding):
    return dict(delta_sha256=binding['delta_sha256'], old_core_sha256=binding['old_core_sha256'],
        old_explanations_sha256=binding['old_explanations_sha256'], core_recalculated=False,
        PR_side_recalculated=False, PF_side_recalculated=False, other31_comparisons_recalculated=False,
        historical_results_overwritten=False)


def accept(root, b):
    result = io.read(root/'A82_factor_delta.json'); adapter.validate_result(result)
    require(result['input_result_digests'] == {k:b[k] for k in
            ('delta_sha256', 'old_core_sha256', 'old_explanations_sha256')}, 'OUTPUT_INPUT_RESULT_BINDING')
    require(io.read(root/'inheritance_receipt.json') == inheritance(b), 'INHERITANCE_RECEIPT')


def protected_hashes(gp, cp, grant, root):
    return {str(p):io.sha(p) for p in (Path(gp), Path(cp), Path(grant['approval_receipt']),
        root/'ONE_SHOT.json', root/'ledger.json')}


def post_gate(gp, cp, protected):
    gate(gp, cp)
    for p,d in protected.items():require(io.sha(p) == d, 'POST_CONTROL_SHA_MISMATCH')


def worker(gp, cp, run_root):
    grant,c,b = gate(gp, cp); root=Path(run_root)
    require(str(root) == c['output_root'], 'OUTPUT_BINDING')
    marker=io.read(root/'ONE_SHOT.json')
    require(marker['parent_pid'] == os.getppid() and marker['consumed'] is True and
            marker['grant_sha256'] == io.sha(gp) and marker['contract_sha256'] == io.sha(cp),
            'OWN_PARENT_ONE_SHOT_REQUIRED')
    protected=protected_hashes(gp, cp, grant, root)
    io.write_new(root/'WORKER_CLAIM.json', dict(pid=os.getpid(), parent_pid=os.getppid(), one_shot=True))
    result=adapter.analyze_factors(io.read(b['payload']), io.read(b['delta']), io.read(b['old_core']),
        io.read(b['old_explanations']), adapter.equations._authorized_production_permit())
    post_gate(gp, cp, protected)
    io.write_new(root/'A82_factor_delta.json', result)
    io.write_new(root/'inheritance_receipt.json', inheritance(b)); accept(root,b)
    io.write_new(root/'worker_result.json', dict(status='COMPLETE_MANDATORY_STOP', target_id=adapter.TARGET,
        hashes={n:io.sha(root/n) for n in OUTPUTS}, calls=adapter.CALLS, next_stage_authorized=False))


def launch(gp, cp):
    grant,c,b=gate(gp,cp)
    protected_paths=[str(ROOT),str(Path(gp).resolve()),str(Path(cp).resolve()),c['input_binding'],
        c['runtime_profile']]+[x['path'] for x in b['input_files']]+b['protected_paths']
    root=io.prepare_output(c['output_root'],c['allowed_run_parent'],protected_paths)
    io.write_new(root/'ONE_SHOT.json',dict(parent_pid=os.getpid(),grant_sha256=io.sha(gp),
        contract_sha256=io.sha(cp),consumed=True,analysis_pass_reserved=1))
    io.write_new(root/'ledger.json',dict(scope=adapter.SCOPE,calls=adapter.CALLS,refund=False))
    protected=protected_hashes(gp,cp,grant,root)
    try:
        p=subprocess.Popen([sys.executable,'-I','-B',str(Path(__file__).resolve()),'--worker',
            '--grant',str(gp),'--contract',str(cp),'--run-root',str(root)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        out,err=p.communicate()
        io.write_new(root/'native_wait.json',dict(pid=p.pid,native_exit_code=p.returncode,reaped=p.poll() is not None,
            stdout=out,stderr=err,worker_descendants=0))
        require(p.returncode == 0,'NATIVE_WORKER_FAILURE')
        result=io.read(root/'worker_result.json')
        require(result['status'] == 'COMPLETE_MANDATORY_STOP' and result['target_id'] == adapter.TARGET and
                result['calls'] == adapter.CALLS and result['next_stage_authorized'] is False and
                set(result['hashes']) == set(OUTPUTS),'WORKER_RESULT_SCHEMA')
        for name,digest in result['hashes'].items():require(io.sha(root/name) == digest,'OUTPUT_SHA_MISMATCH')
        accept(root,b);post_gate(gp,cp,protected)
        io.write_new(root/'terminal.json',dict(status='COMPLETE_MANDATORY_STOP',native_exit_code=0,reaped=True,
            source_commit=c['source_commit'],result_sha256=io.sha(root/'worker_result.json'),next_stage_authorized=False))
    except BaseException as exc:
        if 'p' in locals() and p.poll() is None:
            p.terminate();p.communicate()
            if not (root/'native_wait.json').exists():
                io.write_new(root/'native_wait.json',dict(pid=p.pid,native_exit_code=p.returncode,reaped=True))
        io.write_new(root/'STOP.json',dict(status='FAILED_OR_STOPPED',reason=str(exc),retry=False,refund=False,next_stage_authorized=False))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grant',type=Path,required=True);parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--run-root',type=Path)
    args=parser.parse_args();require(not args.worker or args.run_root is not None,'WORKER_ROOT_REQUIRED')
    worker(args.grant,args.contract,args.run_root) if args.worker else launch(args.grant,args.contract)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);raise SystemExit(1)
