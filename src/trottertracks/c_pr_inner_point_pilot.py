"""Dormant PR-only one-inner-solve pilot at saved trial2 U. No science on import."""
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import time
from . import c_pr_inner_point_solver as algorithm
from . import c_pr_coefficient_diagnostic as previous

io=previous.prior.io;save=previous.prior.save;existing=previous.prior.existing
BASE='0e120c13672dce34d95352f10403939caa04c277'
SCOPE='C_PR_FIXED_TRIAL2_U_ONE_INNER_SOLVE_V1'
FAMILY='artifacts/research_c_pr_inner_point/2026-11-02'
FRESH=FAMILY+'/run01'
LIMITS={k:1 for k in ('saved_load','saved_point_identity','objective_setup','inner_solver',
    'independent_state','tensor_metrics','classification','export')}
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    wanted=dict(base_commit=BASE,scope=SCOPE,entry='scripts/run_research_c_pr_inner_point.py',objective=previous.OBJECTIVE,
        policy=asdict(algorithm.DEFAULT),fresh_output=FRESH,call_limits=LIMITS,maximum_logical_units=8,
        new_inner_solver_invocations=1,new_refinements=0,new_outer_optimizations=0,new_U_updates=0,retry=0,resume=0,mandatory_stop=True)
    for k,v in wanted.items():
        if c[k]!=v:raise ValueError('FIXED_PR_INNER_PILOT:'+k)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            target=(root/p).resolve()
            if not target.is_relative_to(root) or io.sha(target)!=h:raise ValueError('SOURCE_OR_INPUT:'+p)
    if set(c['input_paths'])!={'input','initial','diagnostic','saved_result','saved_identity'}:raise ValueError('INPUT_ROLES')
    if not set(c['input_paths'].values())<=c['saved_inputs'].keys():raise ValueError('UNPINNED_INPUT')
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('ENVIRONMENT:'+p)
    dist=importlib.metadata.distribution('ffsim')
    for p,h in c['ffsim_sources'].items():
        if io.sha(dist.locate_file('ffsim/'+p))!=h:raise ValueError('FFSIM_SOURCE:'+p)
    algorithm.stable.precision.require_extended();return c


def load(root,c):
    import numpy as np
    paths=c['input_paths']
    with np.load(root/paths['input'],allow_pickle=False) as a:g=a['g'].copy()
    with np.load(root/paths['initial'],allow_pickle=False) as a:J=a['J'].copy()
    with np.load(root/paths['diagnostic'],allow_pickle=False) as a:U=a['U'].copy()
    return dict(g=g,J=J,U=U,result=io.read(root/paths['saved_result']),identity=io.read(root/paths['saved_identity']))


def run(root,path,*,loader=load):
    import numpy as np
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH
    if out.exists():raise ValueError('FRESH_PR_INNER_RUN_NO_RETRY_RESUME')
    out.mkdir(parents=True);calls=previous.prior.common.old.Calls(out,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        save(out/'SOURCE.json',io.read(root/c['source_manifest']));save(out/'contract.json',c)
        d=calls('saved_load','same original g/J and validated saved U2',lambda:loader(root,c))
        def identity():
            if d['result']['status']!='NEW_SAVED_POINT_NUMERICAL_CONSISTENCY_PASS' or d['identity']['status']!='SAVED_TRACE_AND_FAILED_POINT_IDENTITIES_PASS':raise ValueError('SAVED_TRIAL2_IDENTITY_REQUIRED')
            if d['identity']['new_scientific_objective_queries_in_replay']!=0 or d['identity']['accepted_replay_steps']!=0:raise ValueError('SAVED_REPLAY_SCOPE')
            if d['J'].shape!=d['U'].shape or d['J'].shape!=(c['objective']['rank'],c['objective']['norb'],c['objective']['norb']):raise ValueError('FIXED_J_U_SHAPE')
            algorithm.stable.linear.pack(d['J']);algorithm.stable.linear.FixedU(d['U'])
            return dict(status='PASS',same_original_J=True,U='saved and identity-qualified trial2 U',new_U_update=False)
        save(out/'initial_identity.json',calls('saved_point_identity','saved qualification and schema',identity))
        def setup():
            spec=algorithm.stable.pr.Objective(**c['objective'])
            return spec,algorithm.stable.Coefficients(d['g'],d['U'],spec)
        spec,p=calls('objective_setup','unchanged PR and isometric coordinates',setup)
        def record(data,arrays):
            import json
            with (out/'inner_trace.jsonl').open('a') as f:f.write(json.dumps(data,allow_nan=False)+'\n')
            if arrays is not None:existing.arrays(out/'inner_raw.npz',**arrays)
        J,info=calls('inner_solver','one Newton-CG,no refinement',lambda:algorithm.solve_j(p,d['J'],record=record))
        save(out/'optimizer.json',info)
        state,gj,gu,chart,_=calls('independent_state','inherited same-array J/U diagnostics',lambda:algorithm.stable.state(d['g'],J,d['U'],spec))
        save(out/'state.json',state)
        tensor=calls('tensor_metrics','existing tensor residual only',lambda:previous.prior.common.old.tensor_metrics(d['g'],J,d['U']))
        save(out/'tensor.json',tensor)
        def classify():
            met=info['stationarity']['status']==state['J_stationarity']['status']=='MET'
            return dict(status='INNER_CORRECTNESS_PASS_J_MET' if met else 'INNER_CORRECTNESS_PASS_J_UNQUALIFIED',
                optimizer_success=info['success'],optimizer_status=info['status'],J_stationarity=state['J_stationarity'],U_partial_gradient=state['U_stationarity'],
                objective_before=info['initial_objective'],objective_after=info['independent_objective'],tensor=tensor,
                new_inner_solver_invocations=1,new_refinements=0,new_U_updates=0,new_outer_optimizations=0,
                reduced_gradient_qualification=None,DF_budget=None,PR_performance=None,rigorous_gradient_error_bound=None,
                retry=0,resume=0,paired_execution_authorized=False,mandatory_stop=True)
        result=calls('classification','correctness,J stationarity and raw flag separate',classify)
        def export():
            existing.arrays(out/'diagnostic_endpoint.npz',J=J,U=d['U'],gradient_J=gj,partial_gradient_U=gu,chart_gradient_U=chart)
            save(out/'result.json',result)
        calls('export','diagnostic endpoint only,no DF qualification',export)
    except BaseException as e:
        calls.stop.set();save(out/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:existing.finish(out,calls,start)
