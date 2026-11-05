"""Paired finite-recipe preparation/runner; byte preflight is passive.

Science is dormant unless explicitly requested after chat permission. No grant
or permission record. Existing Calls, optimizer, CAR worker and codec are reused.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict,replace
from functools import partial
from pathlib import Path
import importlib.metadata
import json
import resource
import subprocess
import threading
import time
from . import c_near_stable as algorithm
from . import c_u_step_control_pilot as prior
from .c1_stage3_integrated import Calls,resource_plan

io=prior.io
BASE='2459e018ea2276c359d6fe96253fbcad0bfd65bb'
SCOPE='C_NEAR_OBJECTIVE_PRACTICAL_COMPARISON_II_O_V1'
ENTRY='scripts/run_research_c_near_practical.py'
FAMILY='artifacts/research_c_near_practical/2026-11-03'
FRESH=FAMILY+'/run01'
STEP=replace(algorithm.control.STEP,max_updates=16)
POLICY=replace(algorithm.control.DEFAULT,outer_maxiter=16)
MAX_TRIALS=128
MAX_POINTS=1+MAX_TRIALS
ARMS=('PR','ENTRY')
COMMON_LIMITS={k:1 for k in ('saved_load','initial_identity','physical_context',
    'common_physical_assessment','objective_setup','comparison','export')}
ARM_LIMITS=dict(optimization_driver=1,inner_J_solve=3*MAX_POINTS,
    precision_state=2*MAX_POINTS,reduced_admission=MAX_POINTS,
    independent_terminal_assessment=STEP.max_updates+1,
    physical_assessment=STEP.max_updates+1,endpoint_diagnostics=1,arm_export=1)
LIMITS={**COMMON_LIMITS,**{k:2*v for k,v in ARM_LIMITS.items()}}
MAX_CALLS=sum(LIMITS.values())
OBJECTIVE=dict(rank=8,norb=4,ld=2,loss_scale=2.1839632053132894e-9,
    mass_scale=2.2093719370722074,gamma=.1,epsilon=1e-8)


def fixed():
    return dict(base_commit=BASE,scope=SCOPE,entry=ENTRY,
        comparison='II_practical_finite_recipe',arms=list(ARMS),PR_objective=OBJECTIVE,
        initial_penalty_contribution=.1,policy=asdict(POLICY),step_policy=asdict(STEP),
        maximum_total_trials_per_arm=MAX_TRIALS,maximum_U_points_per_arm=MAX_POINTS,
        DF_budget_Ha=.00019875,fresh_output=FRESH,call_limits=LIMITS,
        per_arm_call_limits=ARM_LIMITS,maximum_logical_units=MAX_CALLS,
        included_stage='O_optimization_physical_tail_only',retry=0,resume=0,mandatory_stop=True)


def preflight(root,path,*,environment=False,evaluation=False):
    """Read bytes/JSON/package metadata only: never decode molecular arrays."""
    root=Path(root).resolve();c=io.read(path)
    for k,v in fixed().items():
        if c[k]!=v:raise ValueError('FIXED_NEAR_PRACTICAL:'+k)
    source=root/c['source_manifest']
    if io.sha(source)!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    manifest=io.read(source)
    if ENTRY not in manifest['files']:raise ValueError('UNPINNED_ENTRY')
    for group in (manifest['files'],c['saved_inputs']):
        for name,want in group.items():
            target=(root/name).resolve()
            if not target.is_relative_to(root) or io.sha(target)!=want:
                raise ValueError('SOURCE_INPUT:'+name)
    if set(c['input_paths'])!={'input','initial','scales'} or not set(c['input_paths'].values())<=c['saved_inputs'].keys():
        raise ValueError('PINNED_NEUTRAL_INPUT_ROLES')
    if not evaluation and (root/FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
    if environment:
        for name,want in c['evaluation_packages' if evaluation else 'fit_packages'].items():
            if importlib.metadata.version(name)!=want:raise ValueError('ENVIRONMENT:'+name)
        if not evaluation:
            dist=importlib.metadata.distribution('ffsim')
            for name,want in c['ffsim_sources'].items():
                if io.sha(dist.locate_file('ffsim/'+name))!=want:raise ValueError('FFSIM_SOURCE:'+name)
            algorithm.stable.precision.require_extended()
    return c


class Service(prior.Service):
    """One persistent original-H CAR worker; serialize its short JSON pipe."""
    def __init__(self,*args):
        root,path,c,out=args
        self.lock=threading.Lock()
        self.log=(out/'evaluation_worker.log').open('x')
        self.process=subprocess.Popen([c['evaluation_python'],'-B',str(root/ENTRY),
            '--contract',str(path),'--evaluation-worker','--science-permitted'],cwd=root,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)

    def request(self,*args,**kwargs):
        with self.lock:return super().request(*args,**kwargs)


def load(root,c):
    from .c_pr_aware_df_stage_b_runner import load
    p=c['input_paths']
    return (load(root/p['input'],('g',)),load(root/p['initial'],('J','U','x0')),
            io.read(root/p['scales']))


def initial_identity(packet,cold,scales,c):
    import numpy as np
    from ffsim.linalg.util import df_tensors_from_params
    from .c1_real_orthogonal_codec import ATOL
    g,J,U,x=packet['g'],cold['J'],cold['U'],cold['x0']
    if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in (g,J,U,x)):
        raise ValueError('INITIAL_FINITE_BINARY64')
    if g.shape!=(4,)*4 or J.shape!=(8,4,4) or U.shape!=J.shape or x.shape!=(128,):
        raise ValueError('NEUTRAL_INPUT_SHAPE')
    j,u=df_tensors_from_params(x,8,4,None,real=True)
    if (not np.array_equal(j,J) or not np.array_equal(J,J.transpose(0,2,1))
        or np.max(abs(u-U))>ATOL or np.max(abs(U.transpose(0,2,1)@U-np.eye(4)))>ATOL):
        raise ValueError('QUALIFIED_INITIAL_X0_MISMATCH')
    if (scales['initial_tensor_objective']!=OBJECTIVE['loss_scale'] or
        scales['initial_smooth_tail_mass']!=OBJECTIVE['mass_scale']):raise ValueError('FIXED_NEUTRAL_SCALES')
    return dict(status='COMMON_INITIAL_X0_PASS',parameterization='saved faithful real_schur x0;no repack/logm',
        neutral_start='unregularized tensor-only CDF8 export;not update005/064',
        initial_file_sha256=c['saved_inputs'][c['input_paths']['initial']],
        optimizer_coordinates='same isometric pack(J0),zero local eta;ffsim global x0 is provenance,not a new fitter invocation',
        saved_x0_identity_verified_for_both_arms=True)


def specs(g,J,U):
    """Scales fixed once at neutral start; no strength grid or result tuning."""
    import numpy as np
    s=algorithm.near.pr.Objective(**OBJECTIVE)
    entry=algorithm.near.initial_spec(J,ld=s.ld,loss_scale=s.loss_scale,
        strength=.1,epsilon=s.epsilon)
    mass=float(algorithm.near.pr.smooth_mass(J,s,np))
    if not np.isclose(mass,s.mass_scale,atol=1e-14,rtol=1e-13):raise ValueError('INITIAL_PR_MASS_SCALE')
    return dict(PR=s,ENTRY=entry)


def run_arm(name,g,cold,spec,physical,call,record,*,trajectory=algorithm.trajectory,
            step=STEP,policy=POLICY,maximum_points=MAX_POINTS,arm_limits=ARM_LIMITS):
    """Same policy; independent endpoint, counts and raw flags for each arm."""
    import numpy as np
    expected=algorithm.near.pr.Objective if name=='PR' else algorithm.near.Objective if name=='ENTRY' else None
    if type(spec) is not expected:raise ValueError('OBJECTIVE_ARM_IDENTITY')
    tick,cpu=time.monotonic(),time.thread_time()
    metrics=Counter();raw_flags=Counter();local_counts=Counter();physics={}
    def local_call(kind,label,fn):
        local_counts[kind]+=1
        if kind not in arm_limits or local_counts[kind]>arm_limits[kind]:raise ValueError('ARM_CALL_LIMIT:'+name+':'+kind)
        return call(kind,name+'_'+label,fn)
    local_call.counts=local_counts
    def evidence(kind,label,data,arrays):
        if kind=='inner_raw' and data.get('inner_summary_record'):
            for k in ('nfev','njev','nhev','nit','value_gradient_calls_in_block','hessian_calls_in_block'):
                metrics[k]+=data.get(k,0)
            raw_flags[str(data['status'])]+=1
        record(kind,label,data,arrays)
    def key(J,U):return (J.tobytes(),U.tobytes())
    def assess(label,J,U):
        row=physical(name+'_'+label,J,U);physics[key(J,U)]=row;return row
    result=local_call('optimization_driver','paired finite recipe',lambda:trajectory(
        g,cold['J'].copy(),cold['U'].copy(),spec,assess,step=step,policy=policy,
        maximum_total_points=maximum_points,call=local_call,record=evidence))
    point=result['terminal'];post=None;row=None
    if point is not None:
        assessor=algorithm.control.small.assess if name=='PR' else partial(
            algorithm.control.small.assess,state_evaluator=algorithm.state)
        post=local_call('independent_terminal_assessment','final_endpoint',lambda:assessor(
            g,point,point['U'],spec,policy,record=evidence))
        row=physics.get(key(point['J'],point['U']))
        if row is None:raise ValueError('TERMINAL_PHYSICAL_IDENTITY')
    summary=dict(arm=name,objective_version=algorithm.near.pr.VERSION if name=='PR' else algorithm.near.VERSION,
        spec=asdict(spec),status=result['status'],trajectory={k:v for k,v in result.items() if k!='terminal'},
        independent_terminal=post,physical=row,endpoint_eligible=bool(row and row['budget_PASS'] and post['J_numerically_qualified']),
        solver_metrics=dict(metrics),raw_optimizer_status_counts=dict(raw_flags),
        CPU_thread_seconds=time.thread_time()-cpu,wall_seconds=time.monotonic()-tick,
        actual_CA_achieved=None,PF_RTE=None,shots=None,native_price=None,total_RZ=None)
    if point is not None:
        diagnostic_spec=(spec if name=='ENTRY' else algorithm.near.Objective(spec.rank,spec.norb,spec.ld,spec.loss_scale,1.,spec.gamma,spec.epsilon))
        summary['mass_diagnostics']=local_call('endpoint_diagnostics','exact lambda / entry mass / cancellation',lambda:algorithm.near.mass_diagnostics(point['J'],diagnostic_spec))
    return summary,point,local_call


def compare(results):
    """Common metrics only; self-F is never ranked between objectives."""
    return dict(status='PAIRED_FINITE_RECIPE_ENDPOINTS_RECORDED',
        both_physically_eligible=all(x['endpoint_eligible'] for x in results.values()),
        arms={a:{k:r[k] for k in ('status','endpoint_eligible','physical','solver_metrics','raw_optimizer_status_counts')} for a,r in results.items()},
        own_objective_winner=None,objective_capability_winner=None,PR_resource_winner=None,
        comparison='II_practical_finite_recipe',true_gradient_error_bound=None,actual_CA_achieved=None,
        downstream_authorized=False,mandatory_stop=True)


def run(root,path,*,scientific_permission=False,service_factory=Service,loader=load,arm_runner=run_arm):
    """One fresh paired run. Permission is the user's chat, never a certificate."""
    if scientific_permission is not True:raise PermissionError('SEPARATE_EXPLICIT_SCIENTIFIC_PERMISSION_REQUIRED')
    root=Path(root).resolve();path=Path(path).resolve();c=preflight(root,path,environment=True)
    import numpy as np
    from .c_pr_aware_df_stage_b_runner import save_arrays
    out=root/FRESH;out.mkdir(parents=True)
    call=Calls(out,LIMITS,MAX_CALLS);service=None;result=None;tick=time.monotonic();cpu=time.process_time()
    try:
        io.save(out/'contract.json',c);io.save(out/'SOURCE.json',io.read(root/c['source_manifest']))
        packet,cold,scales=call('saved_load','neutral H4 CDF8 saved J/U/x0',lambda:loader(root,c))
        io.save(out/'initial_identity.json',call('initial_identity','both arms same neutral arrays',lambda:initial_identity(packet,cold,scales,c)))
        service=service_factory(root,path,c,out)
        call('physical_context','one shared original Hamiltonian/sector36',lambda:service.request('initialize'))
        def physical(label,J,U):
            target=out/(label+'_candidate.npz');save_arrays(target,J=J,U=U)
            row=service.request('assess',label=label,path=str(target))
            row['candidate_path']=str(target.relative_to(root))
            return row
        initial=call('common_physical_assessment','neutral shared initial',lambda:physical('common_initial',cold['J'],cold['U']))
        if not initial['budget_PASS']:
            result=dict(status='COMMON_INITIAL_DF_BUDGET_NOT_MET',arms={},downstream_authorized=False)
        else:
            objectives=call('objective_setup','fixed normalized starting contributions',lambda:specs(packet['g'],cold['J'],cold['U']))
            io.save(out/'objective_specs.json',{a:asdict(s) for a,s in objectives.items()})
            plan=resource_plan(2,256);io.save(out/'resource_plan.json',plan)
            results={}
            def worker(name):
                folder=out/name;folder.mkdir()
                def record(kind,label,data,arrays):
                    if arrays is not None:save_arrays(folder/(kind+'_'+label+'.npz'),**arrays)
                    with (folder/'trajectory.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(kind=kind,label=label,**data),allow_nan=False)+'\n')
                summary,point,local_call=arm_runner(name,packet['g'],cold,objectives[name],physical,call,record)
                def export():
                    if point is not None:
                        target=folder/'endpoint.npz';save_arrays(target,J=point['J'],U=point['U'])
                        summary['endpoint_sha256']=io.sha(target)
                        summary['candidate_identity']=dict(arm=name,objective_version=summary['objective_version'],
                            spec=summary['spec'],initial_sha256=c['saved_inputs'][c['input_paths']['initial']],
                            source_manifest_sha256=c['source_manifest_sha256'],endpoint_sha256=summary['endpoint_sha256'])
                    summary['call_counts']=dict(local_call.counts)
                    io.save(folder/'result.json',summary)
                local_call('arm_export','finite endpoint',export)
                return name,summary
            # Independent arms share immutable input/JAX shape cache. Scientific
            # state evolution remains sequential within each arm. Calls is locked.
            with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
                futures=[pool.submit(worker,a) for a in ARMS]
                for f in as_completed(futures):
                    name,summary=f.result();results[name]=summary
            result=call('comparison','common physical/attainment metrics;no self-F ranking',lambda:compare(results))
            result['arms']=results
        result.update(retry=0,resume=0,mandatory_stop=True)
        call('export','paired result',lambda:io.save(out/'result.json',result))
        return result
    except BaseException as exc:
        call.stop.set();io.save(out/'failure.json',dict(error=repr(exc),first_failure=call.first_failure));raise
    finally:
        try:
            if service is not None:service.close()
        except BaseException:
            call.stop.set();raise
        finally:
            io.save(out/'STOP.json',dict(failed=call.stop.is_set(),call_counts=dict(call.counts),scientific_units=sum(call.counts.values()),
                maximum_logical_units=MAX_CALLS,retry=0,resume=0,mandatory_stop=True,
                reason='CORRECTNESS_OR_EXECUTION_FAILURE' if call.stop.is_set() else result['status']))
            io.save(out/'resources.json',dict(wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
                peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                evaluation_child_max_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
