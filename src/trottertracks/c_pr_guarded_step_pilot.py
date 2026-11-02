"""Dormant qualified-only step proposal. No molecular array decode on preflight."""
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import json,time
from . import c_pr_guarded_step as algorithm
from . import c_pr_reduced_limited_pilot as previous
from .c1_stage3_integrated import Calls

io,save,existing=previous.io,previous.save,previous.existing
BASE='fc4ee5712d475812c03e647472f41901bdfa0165'
SCOPE='C_PR_QUALIFIED_ONLY_ONE_ARMIJO_STEP_V1'
ENTRY='scripts/run_research_c_pr_guarded_step.py'
OBJECTIVE=dict(previous.OBJECTIVE)
FAMILY='artifacts/research_c_pr_guarded_step/2026-11-02'
FRESH=FAMILY+'/run01'
LIMITS=dict(saved_load=1,initial_identity=1,objective_setup=1,outer_driver=1,
    inner_J_solve=3*algorithm.DEFAULT.max_points,precision_state=2*algorithm.DEFAULT.max_points,
    reduced_admission=algorithm.DEFAULT.max_points,classification=1,export=1)
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    for k,v in dict(base_commit=BASE,scope=SCOPE,entry=ENTRY,objective=OBJECTIVE,
        policy=asdict(algorithm.DEFAULT),search_policy=asdict(algorithm.SEARCH),
        fresh_output=FRESH,call_limits=LIMITS,maximum_logical_units=MAX_CALLS,
        retry=0,resume=0,mandatory_stop=True).items():
        if c[k]!=v:raise ValueError('FIXED_GUARDED_PR_STEP:'+k)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    files=io.read(root/c['source_manifest'])['files']
    if ENTRY not in files:raise ValueError('UNPINNED_ENTRY')
    for group in (files,c['saved_inputs']):
        for name,h in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=h:raise ValueError('SOURCE_OR_INPUT:'+name)
    if set(c['input_paths'])!={'input','initial','scales'} or not set(c['input_paths'].values())<=c['saved_inputs'].keys():raise ValueError('UNPINNED_INPUT_ROLES')
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('ENVIRONMENT:'+p)
    dist=importlib.metadata.distribution('ffsim')
    for p,h in c['ffsim_sources'].items():
        if io.sha(dist.locate_file('ffsim/'+p))!=h:raise ValueError('FFSIM_SOURCE:'+p)
    algorithm.stable.precision.require_extended()
    if (root/FRESH).exists():raise ValueError('FRESH_GUARDED_PR_STEP_NO_RETRY_RESUME')
    return c


def run(root,path,*,loader=previous.load):
    import numpy as np
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH
    out.mkdir(parents=True);calls=Calls(out,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        save(out/'SOURCE.json',io.read(root/c['source_manifest']));save(out/'contract.json',c)
        g,J,U,scales=calls('saved_load','original saved g/J/U and scales only',lambda:loader(root,c))
        def identity():
            shape=(c['objective']['rank'],c['objective']['norb'],c['objective']['norb'])
            if J.shape!=shape or U.shape!=shape or g.shape!=(shape[1],)*4:raise ValueError('INITIAL_SHAPE')
            if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in (g,J,U)):raise ValueError('FINITE_BINARY64_INITIAL_REQUIRED')
            algorithm.stable.linear.pack(J);algorithm.stable.linear.FixedU(U)
            if scales['initial_tensor_objective']!=c['objective']['loss_scale'] or scales['initial_smooth_tail_mass']!=c['objective']['mass_scale']:raise ValueError('ORIGINAL_SCALES')
            return dict(status='PASS',same_original_J_U=True,warmstart_or_continuation=False)
        save(out/'initial_identity.json',calls('initial_identity','original qualified point schema',identity))
        spec=calls('objective_setup','unchanged PR objective',lambda:algorithm.stable.pr.Objective(**c['objective']))
        def record(kind,label,data,arrays):
            if arrays is not None:existing.arrays(out/(kind+'_'+label+'.npz'),**arrays)
            with (out/'optimization_trace.jsonl').open('a') as f:
                f.write(json.dumps(dict(kind=kind,label=label,**data),allow_nan=False)+'\n')
        j,u,result=calls('outer_driver','one bounded qualified Armijo step,no BFGS invocation',lambda:algorithm.optimize(g,J,U,spec,call=calls,record=record))
        save(out/'optimization.json',result)
        def classify():
            return dict(status=result['status'],accepted_U_updates=result['accepted_U_updates'],trials=result['trials'],
                distinct_U_points=result['distinct_U_points'],inner_solves=result['inner_solves'],refinement_solves=result['refinement_solves'],
                terminal_label=result['terminal_label'],terminal_kind=result['terminal_kind'],terminal_qualification=result['terminal_qualification'],
                last_trial_label=result['last_trial_label'],last_trial_qualification=result['last_trial_qualification'],
                outer_BFGS_invocations=0,ffsim_joint_fitter_invocations=0,RC2_continuation=0,
                DF_budget=None,PR_performance=None,rigorous_reduced_gradient=False,retry=0,resume=0,mandatory_stop=True)
        decision=calls('classification','qualified step is numerical pilot,not final convergence',classify)
        def export():
            existing.arrays(out/'diagnostic_terminal.npz',J=j,U=u)
            save(out/'result.json',decision)
        calls('export','already computed terminal;no new DF/operator assessment',export)
        return decision
    except BaseException as e:
        calls.stop.set();save(out/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:existing.finish(out,calls,start)
