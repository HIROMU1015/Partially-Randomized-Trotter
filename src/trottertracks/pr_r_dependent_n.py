"""PR N(T,R) adapter. Metadata planning is passive; no saved numerics on import.

Scientific provider calls require a separately fixed/approved menu and context.
No optimizer, price provider, circuit builder or historical runner is invoked.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
from pathlib import Path

from . import pf_pr_rpe_pipeline as pipeline
from . import h4_df10_rpe_connection as connection
from . import h4_df10_rpe_d1 as scalar

BASELINE='89c91c692e1d35de94d2e7afbe890aabe6d80b7e'
TASK_CARRIER='3b742fff26381514e12c07b51701276b7d1b26ec'


def saved_task(root,system):
    if system not in ('H4','H6'):raise ValueError('H4 DF10 / H6 DF19 only')
    path='docs/research/pf_pr_rpe_cost_integration_20261110/static_plan/'+system+'_PLAN.json'
    return pipeline.SavedArtifacts(root).read(dict(commit=TASK_CARRIER,path=path))['task']


def with_round_R(base,R_by_round,source):
    """Change only explicitly supplied PR R values; no B/signal/N arithmetic."""
    pipeline.validate_task(base);system=base['binding']['system']
    if system not in ('H4','H6') or not pipeline.same(base,saved_task(Path(__file__).resolve().parents[2],system)):raise ValueError('Pinned base SOURCE required')
    if system not in ('H4','H6') or not R_by_round:raise ValueError('Explicit H4/H6 round-R map required')
    task=deepcopy(base);changed=[]
    for m,R in R_by_round.items():
        if type(m) is not int or m not in range(13) or type(R) is not int or R<(1 if system=='H4' else 11):raise ValueError('Integer m and sufficient whole-outer R required')
        if system=='H6' and m<4:raise ValueError('H6 delta1.6/prefix7 menu starts at m4; preserve early rounds')
        row=next(r for r in task['candidates'] if r['m']==m and r['candidate']['method']=='PR');c=row['candidate']
        expected=('2nd',.1,1,2) if system=='H4' else ('4th(m5_best)',1.6,7,2)
        if tuple(c[k] for k in ('formula','delta','prefix','K'))!=expected:raise ValueError('Frozen PR formula/delta/prefix/K required')
        old=c['id'];c['R_outer']=R;c['id']=pipeline.candidate_id(c);row['structure']=pipeline.structure(c)
        if c['id']!=old:row['adoption']='R_VARIANT_UNEVALUATED';changed.append(dict(m=m,old_candidate_id=old,new_candidate_id=c['id']))
    task['source']=dict(adapter_source=deepcopy(source),base_source=deepcopy(base['source']),R_by_round={str(k):v for k,v in R_by_round.items()})
    pipeline.validate_task(task)
    return dict(task=task,changed_candidates=changed,science_calls=0,full_RPE_qualified=False,mandatory_stop=True)


def validate_task(task,base):
    pipeline.validate_task(task)
    declared=task['source'].get('R_by_round')
    if not isinstance(declared,dict) or not declared or any(not isinstance(k,str) or str(int(k))!=k for k in declared):raise ValueError('Explicit canonical round-R scope required')
    canonical=with_round_R(base,{int(k):v for k,v in declared.items()},dict(path='VALIDATION_ONLY'))['task']
    for key in canonical:
        if key!='source' and not pipeline.same(task[key],canonical[key]):raise ValueError('Only PR R may change; fixed scientific task differs')


def partition_saved(task,records):
    """Exact candidate AND common identity reuse only. FAIL remains a FAIL."""
    menu={pipeline.key(r):r for r in task['candidates']};reuse=[];other=[]
    for n in records:
        row=menu.get(pipeline.key(n['identity']))
        if row is None:other.append(deepcopy(n));continue
        if not pipeline.same(n['identity'],pipeline.identity(task,row)):raise ValueError('Saved N SOURCE/input/policy identity differs')
        reuse.append(deepcopy(n))
    pipeline.plan(task,reuse,[])
    return reuse,other


def eligible(task,row):
    c=row['candidate'];system=task['binding']['system']
    return c['method']=='PR' and (system=='H4' or row['m']>=4)


def check_normalizers(ctx,row,backend):
    """Future numeric precision gate, not a new normalizer or R optimizer."""
    c=row['candidate'];lam=ctx.tails[c['prefix']]['lambda_R']
    if not math.isfinite(lam) or lam<=0:raise ValueError('Saved positive tail lambda required')
    for w,r in zip(row['structure']['signed_weights'],row['structure']['allocation']):
        tau=lam*(c['T']*w/c['q'])/r
        b=backend.finite_rte_distribution(tau,c['K']).exact_finite_distribution
        if not math.isfinite(tau) or tau==0 or not math.isfinite(b) or b<=1:
            raise ValueError('Unresolved finite-normalizer precision; do not set B=1 or retry R')


def analyze(task,row,record,reference,metadata,source):
    """Existing disk/geometry/Hoeffding, with R-aware empirical operation count."""
    c=row['candidate'];sig=record['signal'];ev=sig['allowance_evidence']
    if record['candidate']!=c or any(record[k]!=row['structure'][k] for k in ('signed_weights','allocation','event_count')):raise ValueError('Finite-mean recipe differs')
    if not math.isfinite(ev['maximum_intermediate_norm']) or ev['maximum_intermediate_norm']<0:raise ValueError('Finite maximum propagation norm required')
    pair=reference.get('signal_difference')
    if pair is None or not math.isfinite(pair) or not 0<=pair<=1e-10:raise ValueError('Two-path reference numerical gate required')
    if any(not math.isfinite(ev[k]) or not 0<=ev[k]<=1e-9 for k in ('exact_tail_norm_error','raw_corrected_vector_closure')):raise ValueError('Norm/independent closure gate')
    pieces=len(row['structure']['signed_weights']);events=row['structure']['event_count']
    # Head multiplications, exact-tail applications, finite updates, raw divides.
    stages=c['q']*(2*(c['prefix']+1)*pieces+pieces)+2*events
    roundoff=64*2**-52*(1+ev['maximum_intermediate_norm'])*(1+stages)
    u=max(1e-9,10*ev['raw_corrected_vector_closure']+roundoff)+pair
    margin=max(1e-12,64*2**-52*(1+abs(sig['log_B']))*(1+stages))
    point=dict(candidate=deepcopy(c),binding=deepcopy(task['binding']),compiler=task['profile']['compiler'],prices=[],saved_signal=deepcopy(sig),
        saved_allowance=dict(u=u,log_B_margin=margin),saved_finite_time_endpoint=dict(axes={a:dict(allowance=u) for a in ('real','imag')}))
    bound=connection.bind_saved_round(point,target_id=task['task_id'],m=row['m'],base_time=task['configuration']['base_time'],
        expected_system=c['system'],expected_rank=task['binding']['rank'],expected_binding=task['binding'])
    d=scalar.analyze_bound_point(point,bound,task['configuration'],state_energy=metadata['state_energy'],state_residual=metadata['state_residual'])
    d.update(signal=deepcopy(sig),numerical_reference=deepcopy(reference),signed_weights=record['signed_weights'],allocation=record['allocation'],event_count=events,
        allowance=dict(u=u,log_B_margin=margin,stages=stages,roundoff=roundoff,certified=False,policy='R_AWARE_EMPIRICAL_ACTION_COUNT_V1'),G=None)
    n=pipeline.shot_record(task,row,d,source);n['origin']='EXPLICIT_PR_R_VARIANT_N'
    if n['plan']['status']=='CONDITIONAL_ROUND_PASS':
        from verify_h4_df10_rpe_shot_integers import integer_check
        checks={}
        for axis in pipeline.AXES:
            a=n['plan']['axes'][axis];check=integer_check(a['epsilon_coordinate'],a['alpha_axis'],a['shots']);checks[axis]=check
            n['plan']['N_'+axis]=int(check['N_handoff_decimal']);a['shots']=n['plan']['N_'+axis]
        n['diagnostics'].update(integer_checks=checks,N_X_decimal=str(n['plan']['N_X']),N_Y_decimal=str(n['plan']['N_Y']))
        n['diagnostics']['ideal_numerical_pilot']=deepcopy(n['plan'])
    return n


def failure(task,row,source,reason,status='UNQUALIFIED_R_INPUT',evidence=None):
    from .h4_df10_rpe_batch_provider import failed_record
    n=failed_record(task,row,source,reason,status);n['origin']='PR_R_VARIANT_FAILURE';n['diagnostics']['evidence']=evidence
    return n


class RBatch:
    """One declared schedule, one PR batch. B supplies bound context/references.

    context_factory(task) -> (context, {state_energy,state_residual})
    reference_factory(ctx,T) -> {binding,T,source,evidence:{reference,...}}
    Omitting reference_factory requests the existing two-exponential backend.
    """
    def __init__(self,task,base,source,context_factory,reference_factory=None):
        validate_task(task,base)
        if not source:raise ValueError('Explicit scientific SOURCE metadata required')
        self.task=deepcopy(task);self.source=deepcopy(source);self.context_factory=context_factory
        self.reference_factory=reference_factory;self.last_run=None

    def __call__(self,task,rows):
        if not pipeline.same(task,self.task):raise ValueError('Published R task differs')
        if not rows or len({pipeline.key(r) for r in rows})!=len(rows) or any(r not in task['candidates'] for r in rows):raise ValueError('Explicit unique menu required')
        if any(not eligible(task,r) or str(r['m']) not in task['source']['R_by_round'] for r in rows):raise ValueError('Provider accepts only explicitly declared PR R rounds; supply saved other N')
        import run_h6_df19_rpe_d2 as io
        from . import h6_df19_rpe_d2 as backend
        self.last_run=dict(status='RUNNING',reference_times=[],candidate_count=len(rows),price_calls=0,G_calls=0,automatic_retry=False)
        try:
            ctx,metadata=self.context_factory(task)
            if not pipeline.same(ctx.scientific_binding,task['binding']):raise ValueError('Restored H/state/snapshot binding differs')
            refs={};origins={}
            self.last_run.update(references=refs,reference_origins=origins)
            for T in sorted({r['candidate']['T'] for r in rows}):
                if self.reference_factory is None:
                    ref=backend.reference_pair(ctx,T);origin=dict(kind='NEW_TWO_PATH_REFERENCE',source=self.source)
                else:
                    item=self.reference_factory(ctx,T)
                    if not pipeline.same(item['binding'],task['binding']) or item['T']!=T or not item.get('source'):raise ValueError('Supplied reference binding/time/SOURCE differs')
                    ref=item['evidence'];origin=dict(kind='SUPPLIED_BOUND_REFERENCE',source=item['source'])
                refs[T]=deepcopy(ref);origins[T]=origin
                if any(not isinstance(ref.get(k),dict) or any(not isinstance(ref[k].get(a),(int,float)) or not math.isfinite(ref[k][a]) for a in ('real','imag')) for k in ('reference','independent_reference')):raise ValueError('Two saved reference paths required')
                if not isinstance(ref.get('signal_difference'),(int,float)) or not math.isfinite(ref['signal_difference']) or not 0<=ref['signal_difference']<=1e-10:raise ValueError('Reference pair gate')
                observed=abs(complex(ref['reference']['real'],ref['reference']['imag'])-complex(ref['independent_reference']['real'],ref['independent_reference']['imag']))
                if observed>1e-10:raise ValueError('Observed reference pair gate')
                ref=deepcopy(ref);ref['signal_difference']=max(ref['signal_difference'],observed)
                refs[T]=ref;self.last_run['reference_times'].append(T)
            keys=backend.half_step_keys(rows);resources=io.worker_resources(ctx,len(keys))
            with ThreadPoolExecutor(max_workers=resources['half_step_workers']) as pool:
                for key,value in pool.map(lambda k:backend.half_step(ctx,k),keys):ctx.half_steps[key]=value
        except Exception as exc:self.last_run.update(status='STOP_CONTEXT_OR_REFERENCE',reason=str(exc));raise
        jobs=io.worker_resources(ctx,len(rows))
        def job(row):
            record=None;attempted=False
            try:
                check_normalizers(ctx,row,backend)
                ref=refs[row['candidate']['T']];attempted=True;record=backend.evolve(ctx,row['candidate'],ref['reference'])
                n=analyze(task,row,record,ref,metadata,self.source)
                n['diagnostics']['reference_origin']=origins[row['candidate']['T']]
                return n,attempted,True
            except Exception as exc:return failure(task,row,self.source,str(exc),'NUMERICAL_GATE_OR_RANGE_UNQUALIFIED',record),attempted,record is not None
        with ThreadPoolExecutor(max_workers=jobs['half_step_workers']) as pool:outcomes=list(pool.map(job,rows))
        result=[v[0] for v in outcomes]
        self.last_run.update(status='BATCH_EVALUATED' if all(n['plan']['status']=='CONDITIONAL_ROUND_PASS' for n in result) else 'STOP_UNQUALIFIED_N',
            half_step_resources=resources,candidate_resources=jobs,finite_mean_attempts=sum(v[1] for v in outcomes),PR_exact_tail_controls_completed=sum(v[2] for v in outcomes))
        return result
