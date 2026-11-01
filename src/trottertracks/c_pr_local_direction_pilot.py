"""Dormant H4 symmetric directional diagnosis; passive checks never decode NPZ."""
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import json,time
from . import c_pr_local_direction as algorithm
from . import c_pr_guarded_step_pilot as previous
from .c1_stage3_integrated import Calls

io,save,existing=previous.io,previous.save,previous.existing
BASE='0bb25e3c0cd50b6794a78513925043d216607827'
SCOPE='C_PR_LOCAL_DIRECTION_SYMMETRIC_DIAGNOSIS_V1'
ENTRY='scripts/run_research_c_pr_local_direction.py'
OBJECTIVE=dict(previous.OBJECTIVE)
FAMILY='artifacts/research_c_pr_local_step/2026-11-02'
FRESH=FAMILY+'/run01'
LIMITS=dict(saved_load=1,initial_identity=1,objective_setup=1,local_probe_driver=1,
    inner_J_solve=21,precision_state=14,reduced_admission=7,directional_diagnostics=1,classification=1,export=1)
MAX_CALLS=sum(LIMITS.values())


def preflight(root,path):
    root=Path(root).resolve();c=io.read(path)
    for k,v in dict(base_commit=BASE,scope=SCOPE,entry=ENTRY,objective=OBJECTIVE,policy=asdict(algorithm.DEFAULT),
        fractions=list(algorithm.FRACTIONS),fresh_output=FRESH,call_limits=LIMITS,maximum_logical_units=MAX_CALLS,
        retry=0,resume=0,mandatory_stop=True).items():
        if c[k]!=v:raise ValueError('FIXED_LOCAL_DIRECTION:'+k)
    if io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    files=io.read(root/c['source_manifest'])['files']
    if ENTRY not in files:raise ValueError('UNPINNED_ENTRY')
    for group in (files,c['saved_inputs']):
        for name,h in group.items():
            p=(root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p)!=h:raise ValueError('SOURCE_OR_INPUT:'+name)
    if set(c['input_paths'])!={'input','initial','scales'} or not set(c['input_paths'].values())<=c['saved_inputs'].keys():raise ValueError('UNPINNED_INPUT_ROLES')
    for name,v in c['packages'].items():
        if importlib.metadata.version(name)!=v:raise ValueError('ENVIRONMENT:'+name)
    dist=importlib.metadata.distribution('ffsim')
    for p,h in c['ffsim_sources'].items():
        if io.sha(dist.locate_file('ffsim/'+p))!=h:raise ValueError('FFSIM_SOURCE:'+p)
    algorithm.stable.precision.require_extended()
    if (root/FRESH).exists():raise ValueError('FRESH_LOCAL_DIRECTION_NO_RETRY_RESUME')
    return c


def run(root,path,*,loader=previous.previous.load):
    import numpy as np
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH
    out.mkdir(parents=True);calls=Calls(out,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        save(out/'SOURCE.json',io.read(root/c['source_manifest']));save(out/'contract.json',c)
        g,J,U,scales=calls('saved_load','original g/J/U and scale metadata',lambda:loader(root,c))
        def identity():
            shape=(c['objective']['rank'],c['objective']['norb'],c['objective']['norb'])
            if J.shape!=shape or U.shape!=shape or g.shape!=(shape[1],)*4:raise ValueError('INITIAL_SHAPE')
            if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in (g,J,U)):raise ValueError('FINITE_BINARY64_INITIAL_REQUIRED')
            algorithm.stable.linear.pack(J);algorithm.stable.linear.FixedU(U)
            if scales['initial_tensor_objective']!=c['objective']['loss_scale'] or scales['initial_smooth_tail_mass']!=c['objective']['mass_scale']:raise ValueError('ORIGINAL_SCALES')
            return dict(status='PASS',same_original_J_U=True,warmstart_or_continuation=False)
        save(out/'initial_identity.json',calls('initial_identity','original qualified schema',identity))
        spec=calls('objective_setup','unchanged PR objective',lambda:algorithm.stable.pr.Objective(**c['objective']))
        def record(kind,label,data,arrays):
            if arrays is not None:existing.arrays(out/(kind+'_'+label+'.npz'),**arrays)
            with (out/'diagnostic_trace.jsonl').open('a') as f:f.write(json.dumps(dict(kind=kind,label=label,**data),allow_nan=False)+'\n')
        oracle=algorithm.previous.Oracle(g,J,U,spec,algorithm.DEFAULT,call=calls,record=record)
        zero=np.zeros(spec.rank*spec.norb*(spec.norb-1)//2)
        experiment=calls('local_probe_driver','initial plus3 fixed signed probe pairs,no optimizer',lambda:algorithm.collect(oracle.evaluate,zero,record=record))
        diagnostics=calls('directional_diagnostics','saved-in-memory values only,no new gradient queries',lambda:algorithm.analyse(experiment))
        save(out/'directional_diagnostics.json',diagnostics)
        def classify():
            return dict(status=diagnostics['status'],probe_status=experiment['status'],distinct_U_points=len(oracle.points),
                inner_solves=3*len(oracle.points),refinement_solves=2*len(oracle.points),accepted_U_updates=0,
                point_qualifications=[dict(label=p['label'],**p['qualification']) for p in oracle.points],
                true_reduced_gradient_qualification=None,certified_descent=None,DF_budget=None,PR_performance=None,
                outer_BFGS_invocations=0,ffsim_joint_fitter_invocations=0,RC2_continuation=0,retry=0,resume=0,mandatory_stop=True)
        result=calls('classification','diagnostic availability,not derivative truth or U acceptance',classify)
        def export():
            initial=experiment['initial'] if experiment['initial'] is not None else oracle.points[0]
            existing.arrays(out/'diagnostic_initial.npz',J=initial['J'],U=initial['U'])
            save(out/'result.json',result)
        calls('export','already computed diagnostic initial;no accepted candidate',export)
        return result
    except BaseException as e:
        calls.stop.set();save(out/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:existing.finish(out,calls,start)
