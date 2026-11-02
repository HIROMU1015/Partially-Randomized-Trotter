"""Dormant single-saved-point diagnostic. No fitting, sector or outer search.

The old failing trial's U was NOT exported. A bounded replay of the original
SciPy BFGS using ONLY two saved value/gradient pairs recovers the third query;
it stops before returning any newly evaluated objective or accepting a step.
"""
from pathlib import Path
import importlib.metadata
import time
from . import c_pr_coefficient_point as model
from . import c_j_u_stable_pilot as prior

BASE='9c9eafe74493a5bd4bfa59e41db0e8bf80785e6f'
SCOPE='C_PR_SAVED_TRIAL2_COEFFICIENT_POINT_DIAGNOSTIC_V1'
FAMILY='artifacts/research_c_pr_coefficient_point/2026-11-02'
FRESH=FAMILY+'/run01'
LIMITS={k:1 for k in ('saved_load','outer_trace_replay','trial_geometry','coefficient_setup',
    'point_identity','legacy_evaluation','stable_setup','stable_evaluation','classification','export')}
MAX_CALLS=sum(LIMITS.values())
OBJECTIVE=dict(rank=8,norb=4,ld=2,loss_scale=2.1839632053132894e-9,
    mass_scale=2.2093719370722074,gamma=.1,epsilon=1e-8)


def preflight(root,path):
    root=Path(root).resolve();c=prior.io.read(path)
    for k,v in dict(base_commit=BASE,scope=SCOPE,entry='scripts/run_research_c_pr_coefficient_point.py',
        maximum_logical_units=MAX_CALLS,call_limits=LIMITS,fresh_output=FRESH,retry=0,resume=0,
        new_inner_solves=0,new_optimization_trajectories=0,objective=OBJECTIVE,mandatory_stop=True).items():
        if c[k]!=v:raise ValueError('FIXED_POINT_CONTRACT:'+k)
    if prior.io.sha(root/c['source_manifest'])!=c['source_manifest_sha256']:raise ValueError('SOURCE_MANIFEST')
    for group in (prior.io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for p,h in group.items():
            target=(root/p).resolve()
            if not target.is_relative_to(root) or prior.io.sha(target)!=h:raise ValueError('SOURCE_OR_INPUT:'+p)
    if not set(c['input_paths'].values())<=c['saved_inputs'].keys():raise ValueError('UNPINNED_POINT_INPUT')
    if set(c['input_paths'])!={'input','initial','envelope0','envelope1','state1','failing','trace'}:raise ValueError('FIXED_INPUT_ROLES')
    for p,v in c['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('ENVIRONMENT:'+p)
    d=importlib.metadata.distribution('ffsim')
    for p,h in c['ffsim_sources'].items():
        if prior.io.sha(d.locate_file('ffsim/'+p))!=h:raise ValueError('FFSIM_SOURCE:'+p)
    model.precision.require_extended()
    return c


def capture_third_query(samples,*,driver=None):
    """Replay saved f/g, not evaluate a physical objective or complete BFGS."""
    import numpy as np
    from scipy.optimize import minimize
    if len(samples)!=2:raise ValueError('EXACTLY_TWO_SAVED_REPLAY_POINTS')
    seen=[]
    class Captured(Exception):pass
    def query(x):
        x=np.asarray(x)
        for previous in seen:
            if np.array_equal(x,previous):raise ValueError('UNEXPECTED_REPLAY_REPEAT')
        if len(seen)==2:
            seen.append(x.copy());raise Captured()
        row=samples[len(seen)]
        if not np.array_equal(x,row['eta']):raise ValueError('SAVED_REPLAY_QUERY_MISMATCH')
        seen.append(x.copy());return row['value'],row['gradient'].copy()
    def accepted(intermediate_result):raise ValueError('REPLAY_ACCEPTED_BEFORE_CAPTURE')
    try:
        (driver or minimize)(query,np.zeros_like(samples[0]['eta']),jac=True,method='BFGS',callback=accepted,
            options=dict(maxiter=64,gtol=1e-7,norm=2,xrtol=0.,c1=1e-4,c2=.9))
    except Captured:return seen[2]
    raise ValueError('REPLAY_DID_NOT_REACH_THIRD_QUERY')


def load(root,c):
    """Only invoked inside the separately permitted saved_load call."""
    import numpy as np
    data={}
    for key,p in c['input_paths'].items():
        if p.endswith('.npz'):
            with np.load(root/p,allow_pickle=False) as z:data[key]={k:z[k].copy() for k in z.files}
        elif p.endswith('.jsonl'):
            import json
            data[key]=[json.loads(s) for s in (root/p).read_text().splitlines()]
        else:data[key]=prior.io.read(root/p)
    return data


def run(root,path,*,loader=load,replay=capture_third_query):
    import numpy as np
    root=Path(root).resolve();c=preflight(root,path);out=root/FRESH
    if out.exists():raise ValueError('FRESH_POINT_RUN_NO_RETRY_RESUME')
    out.mkdir(parents=True);calls=prior.common.old.Calls(out,LIMITS,MAX_CALLS);start=time.monotonic()
    save=prior.save;arrays=prior.existing.arrays
    try:
        save(out/'SOURCE.json',prior.io.read(root/c['source_manifest']));save(out/'contract.json',c)
        d=calls('saved_load','original point evidence',lambda:loader(root,c))
        samples=[]
        for i in range(2):
            r=next(r for r in d['trace'] if r['kind']=='precision_state' and r['label']==f'{i}_refine2')
            samples.append(dict(eta=d['envelope'+str(i)]['eta'],gradient=d['envelope'+str(i)]['gradient'],value=r['components']['objective']))
        eta=calls('outer_trace_replay','capture query2 before evaluation',lambda:replay(samples))
        U=calls('trial_geometry','fixed original U exponential',lambda:model.precision.geometry(d['initial']['U'],eta))
        def setup():
            s=model.legacy.pr.Objective(**c['objective']);p=model.legacy.Coefficients(d['input']['g'],U,s)
            return p,model.legacy.CoefficientChart(p,d['initial']['J'])
        p,old=calls('coefficient_setup','existing original metric',setup)
        raw=d['failing'];z=raw['z']
        def identity():
            if not np.array_equal(old.x0,raw['x0']):raise ValueError('INITIAL_COEFFICIENT_POINT')
            # Replaying query0/1 alone is insufficient: verify the failed
            # point's metric AND decoded x against the independently saved raw.
            if not np.array_equal(old.W,raw['coordinate_transform']):raise ValueError('FAILED_TRIAL_METRIC_NOT_IDENTICAL')
            if not np.array_equal(old.x0+old.W@z,raw['x']):raise ValueError('FAILED_ENDPOINT_NOT_IDENTICAL')
            if not np.array_equal(model.precision.geometry(d['initial']['U'],samples[1]['eta']),d['state1']['U']):raise ValueError('SAVED_QUERY1_GEOMETRY_NOT_IDENTICAL')
            return dict(status='SAVED_TRACE_AND_FAILED_POINT_IDENTITIES_PASS',new_scientific_objective_queries_in_replay=0,accepted_replay_steps=0)
        save(out/'point_identity.json',calls('point_identity','strict saved-coordinate identity',identity))
        def legacy_eval():
            x=old.x0+old.W@z;value,g=p.vg(x,independent=True);f,gz=old.vg(z)
            check=model.compare(old.W.T@g,gz,atol=1e-8,rtol=1e-7)
            row=dict(value=f,physical_value=value,chart_vs_physical_gradient=check,
                status='LEGACY_MISMATCH_REPRODUCED' if not check['PASS'] else 'LEGACY_MISMATCH_NOT_REPRODUCED')
            save(out/'legacy.json',row);return row
        legacy=calls('legacy_evaluation','known failure is a diagnostic observation',legacy_eval)
        if legacy['status']!='LEGACY_MISMATCH_REPRODUCED':
            save(out/'result.json',dict(status='INCONCLUSIVE_LEGACY_MISMATCH_NOT_REPRODUCED',rigorous_gradient_error_bound=None,mandatory_stop=True));return
        new=calls('stable_setup','same metric and new point evaluator',lambda:model.CoefficientChart(p,d['initial']['J'],original=old))
        def stable_eval():
            row,vectors=new.assess(z,vectors=True)
            check=model.compare(row['objective'],legacy['physical_value'],atol=1e-10,rtol=1e-8)
            if not check['PASS']:raise ValueError('UNCHANGED_OBJECTIVE_GATE')
            row['legacy_physical_objective_agreement']=check
            return row,vectors
        assessed,vectors=calls('stable_evaluation','independent adjoint same-array chain',stable_eval);save(out/'stable.json',assessed)
        result=calls('classification','numerical consistency only',lambda:dict(status='NEW_SAVED_POINT_NUMERICAL_CONSISTENCY_PASS',legacy=legacy,stable=assessed,
            optimizer_success_not_assessed=True,DF_budget=None,PR_performance=None,rigorous_gradient_error_bound=None,paired_execution_authorized=False,mandatory_stop=True))
        def export():
            arrays(out/'diagnostic.npz',eta=eta,U=U,x=raw['x'],z=z,**vectors)
            save(out/'result.json',result)
        calls('export','diagnostic only,no fitted J/U',export)
    except BaseException as e:
        calls.stop.set();save(out/'failure.json',dict(error=str(e),first_failure=calls.first_failure));raise
    finally:prior.existing.finish(out,calls,start)
