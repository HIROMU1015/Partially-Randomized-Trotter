"""Plan an absolute-kappa scope through the ordinary physical schedule CLI."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .rpe_local0_search import (RequestError, digest, immutable, load, locked,
                               request_service, resolve, summarize)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def handle(config, request):
    config = request_service(config, request)
    if request.get('action') != 'planonly' or not config.get('absolute_kappa_schedule'):
        raise RequestError('action', 'Absolute-kappa planonly scope required')
    root = Path(config['code_root'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() != config['code_source']:
        raise RequestError('code_source', 'Fixed server2 implementation changed')
    if subprocess.check_output(['git','diff','HEAD','--name-only','--','scripts','src'],cwd=root,text=True).strip():
        raise RequestError('code_source', 'Tracked server2 implementation changed')
    for path, expected in config['fixed_settings'].items():
        if sha(path) != expected:
            raise RequestError('settings', 'Adopted configuration hash differs: ' + path)
    problem, spec, canonical = resolve(config, request)
    spec.update(science_approved=False,max_new_q3_per_method={m:0 for m in spec['methods']})
    applied = summarize(problem,spec,canonical)
    identity = dict(problem=problem,round={k:v for k,v in spec.items() if k!='round_id'},
                    scope=canonical,code_source=config['code_source'],fixed_settings=config['fixed_settings'])
    signature = digest(identity)
    run_id = 'plan-kappa-' + signature[:20]
    state = Path(config['state_directory']) / 'scope-plan-only'
    with locked(state):
        run = state / 'runs' / run_id
        run.mkdir(parents=True,exist_ok=True)
        immutable(run/'IDENTITY.json',identity)
        immutable(run/'received_requests'/(digest(request)+'.json'),request)
        if not (run/'RECEIVED_REQUEST.json').exists():immutable(run/'RECEIVED_REQUEST.json',request)
        spec['round_id'] = run_id
        immutable(run/'PROBLEM.json',problem)
        immutable(run/'ROUND.plan-only.json',spec)
        immutable(run/'APPLIED_SCOPE.json',applied)
        result_path = run/'RESULT.json'
        if result_path.exists():
            result = load(result_path)
            for path, expected in result['output_hashes'].items():
                if sha(path) != expected:raise RequestError('run','Saved plan output changed: '+path)
            return dict(result,reused_run=True,normal_entry_invocations_this_call=0)
        output = run/'physical-schedule-plan'
        if output.exists():raise RequestError('run','Incomplete plan preserved: '+str(run/'NORMAL_ENTRY.log'))
        command = [config['python'],'-B',str(root/'scripts/run_rpe_schedule_search.py'),
            '--problem',str(run/'PROBLEM.json'),'--round',str(run/'ROUND.plan-only.json'),
            '--run-dir',str(output),'--plan-only']
        immutable(run/'COMMAND.json',command)
        current = Path(config['state_directory'])/'CURRENT_TARGET.json'
        current_before = current.read_bytes() if current.exists() else None
        started = time.monotonic()
        with (run/'NORMAL_ENTRY.log').open('w') as log:
            process = subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:
            raise RequestError('normal_entry','plan-only exit=%d; see %s'%(process.returncode,run/'NORMAL_ENTRY.log'))
        actual = load(output/'KAPPA_STATE.json')
        actual_fixed = load(output/'FIXED_INPUTS.json')
        from fractions import Fraction
        lo,hi=(Fraction(v) for v in applied['search']['schedule_wide']['kappa_bounds'])
        allowed=applied['method_settings']['PR_PAULI']['L_D_values']
        assert actual['scope_plan']['kappa_bounds']==[str(lo),str(hi)]
        assert actual['scope_plan']['requested_L_D']=={'PR_PAULI':allowed}
        assert actual_fixed['round_scope']['search_overrides']==spec['search_overrides']
        assert all(v['L_D'] in allowed for v in actual['families'].values())
        assert set(v['L_D'] for v in actual['families'].values())==set(allowed)
        proposals=actual['history'][-1]['proposals']
        assert all(lo<=Fraction(v['kappa'])<=hi for v in proposals)
        assert actual['scope_plan']['scientific_calls']==actual['scope_plan']['queue_operations']==actual['scope_plan']['dry_check_calls']==0
        assert not (output/'jobs').exists() and not (output/'distributed').exists()
        assert (current.read_bytes() if current.exists() else None)==current_before
        distributed=deepcopy(spec['executor'])
        assert distributed['scheduler']=='concurrent' and distributed['max_concurrent_jobs']=='auto'
        immutable(run/'DISTRIBUTED_OPTIONS.json',dict(executor=distributed))
        paths=[run/'PROBLEM.json',run/'ROUND.plan-only.json',run/'APPLIED_SCOPE.json',
               run/'DISTRIBUTED_OPTIONS.json',output/'KAPPA_STATE.json',output/'FIXED_INPUTS.json']
        result=dict(status='PLAN_ONLY_READY',started=False,run_id=run_id,run_directory=str(run),
            applied=applied,computation_source=config['computation_source'],implementation=config['code_source'],
            normal_entry='run_rpe_schedule_search.py',selection_rule=actual['scope_plan']['rule'],
            requested_kappa_bounds=[str(lo),str(hi)],bounds_inclusive=[True,True],
            distributed=dict(scheduler=distributed['scheduler'],max_concurrent_jobs=distributed['max_concurrent_jobs'],
                priority_groups=distributed['distributed']['auto_allocation']['priority_groups'],
                required_hosts=distributed['distributed']['required_hosts']),
            physical_families=actual['scope_plan']['family_status'],
            preparation_pending=actual['scope_plan']['preparation_pending'],proposals=proposals,
            command=command,output_hashes={str(p):sha(p) for p in paths},
            reused_run=False,normal_entry_invocations_this_call=1,elapsed_seconds=time.monotonic()-started,
            current_target_unchanged=True,scientific_calls=0,dry_check_calls=0,queue_operations=0,relay_starts=0)
        immutable(result_path,result)
        return result
