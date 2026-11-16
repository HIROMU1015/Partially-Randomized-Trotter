"""H4 DF10 S2/B0/PR N batch. Import/planning perform no scientific evaluation.

The C interface is unchanged. Real signal and saved-signal scalar evaluation
are explicit provider calls requiring their separately approved science scope.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

from . import pf_pr_rpe_pipeline as pipeline
from . import h4_df10_rpe_connection as connection
from . import h4_df10_rpe_d1 as scalar

BASELINE='cb2b1d6b4309ca7d705f397e25e27d14e8ba125a'
N_CARRIER='cefc01622a3cd75dd16d30de7b9fb68274f3eddd'
SIGNAL_CARRIER='155529e20a8fac39e00786c40b57830b71203c4f'
SIGNAL_DIR='artifacts/resource_applicability/track_a_h46_long_time/2026-10-25/'
INDEX=dict(commit='7a8bc7f132e166390c943e6b32c92cc6abe1baf4',path='artifacts/resource_applicability/df_full_h_pure_rte_t20_opt/2026-11-09/SOURCE_INDEX.json')
M4_IDS=('8e50c5560bb8f47f026f','64e4f7e9ec7fba28815e')
BINDING=dict(system='H4',rank=10,rank_policy='tol_only_1e-8',
 hamiltonian='c1a6c4281f11bba90df2fd18a998bdec96bed00379116537d22faf548c525ea0',
 snapshot='12219b7d14dc2a275853314269e5e8628005ed72b42d16bc4d3831c8ea515c96',
 state='9bcc0b0ecf44440c4411bcb76d652d78ff0595f036144580660c56c47c78e80b')


def blob(root,commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_task(task, expected):
    pipeline.validate_task(task)
    for field in ('configuration','binding','contract','profile','energy_target','task_id','source'):
        if not pipeline.same(task[field],expected[field]):raise ValueError('H4 fixed scientific task/profile differs')
    if any(task['binding'].get(k)!=v for k,v in BINDING.items()):raise ValueError('H4 DF10 binding required')
    if len(task['candidates'])!=26:raise ValueError('Fixed13-round PF/PR menu required')
    for index,row in enumerate(task['candidates']):
        m=index//2;c=row['candidate'];pr=index%2==1
        if (row['m']!=m or row['q_RPE']!=1<<m or row['q_PF']!=1<<m
                or c['T']!=math.ldexp(.1,m) or c['q']!=1<<m or c['delta']!=.1
                or c['method']!=('PR' if pr else 'B0') or c['formula']!='2nd'
                or (c['prefix'],c['R_outer'],c['K'])!=((1,4,2) if pr else (7,None,None))):
            raise ValueError('Fixed H4 S2/B0/p7 versus PR/p1/R4/K2 required')


def validate_saved_point(task,row,point):
    """Schema/binding checks only: no B, phase, radius or N is computed."""
    if (point['candidate']!=row['candidate'] or not pipeline.same(point['binding'],task['binding'])
            or point['compiler']!=task['profile']['compiler'] or point.get('prices')!=[]):
        raise ValueError('Saved signal identity/source/profile differs')
    sig=point['saved_signal'];allow=point['saved_allowance']
    for field in ('reference','corrected','raw','total_signed_bias'):
        if not isinstance(sig.get(field),dict) or any(not isinstance(sig[field].get(k),(int,float)) or not math.isfinite(sig[field][k]) for k in ('real','imag')):
            raise ValueError('Missing/nonfinite saved complex signal')
    for value in [sig.get('log_B'),allow.get('u'),allow.get('log_B_margin')]+[point['saved_finite_time_endpoint']['axes'][a].get('allowance') for a in ('real','imag')]:
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Missing numerical allowance/normalization')
    if sig['log_B']!=allow['log_B']:raise ValueError('Saved log B differs')


def prepare(root):
    """Static saved JSON/gzip/receipt selection. Never loads NPZ arrays."""
    store=pipeline.SavedArtifacts(root)
    task=store.read(dict(commit='3b742fff26381514e12c07b51701276b7d1b26ec',path='docs/research/pf_pr_rpe_cost_integration_20261110/static_plan/H4_PLAN.json'))['task']
    validate_task(task,task)
    d1ref=dict(commit=N_CARRIER,path='docs/research/h4_df10_rpe_d1_20261110/result.json');d1=store.read(d1ref)
    if d1['configuration']!=task['configuration']:raise ValueError('Saved D1 policy differs')
    cached=[]
    for point in d1['rows']:
        row=next(r for r in task['candidates'] if pipeline.key(r)==pipeline.key(point))
        n=pipeline.shot_record(task,row,point,dict(d1ref,scientific_source=d1['scientific_source']))
        n['diagnostics']={k:v for k,v in n['diagnostics'].items() if k not in ('compiled_prices','G')}
        cached.append(n)
    if len(cached)!=2 or any(n['identity']['m']!=3 for n in cached):raise ValueError('Preserve the two m3 N records')
    index=store.read(INDEX);inputs=index['saved_inputs']['H4']['inputs'];context_ref=index['T20_reference_files']['H4']
    for spec in list(inputs.values())+[context_ref]:
        if sha(spec['path'])!=spec['sha256']:raise ValueError('Saved H4 input/context bytes differ')
    receipt=json.loads(Path(inputs['H4_tol_receipt.json']['path']).read_bytes())
    context=json.loads(Path(context_ref['path']).read_bytes())
    if (not pipeline.same(context['binding'],task['binding']) or context['sector_dimension']!=36
            or receipt['hamiltonian_hash']!=BINDING['hamiltonian'] or receipt['state_hash']!=BINDING['state']
            or receipt['rank']!=10 or receipt['ground_state_certified'] is not False
            or receipt['sha256']!=inputs['H4_tol_snapshot.npz']['sha256']):raise ValueError('H4 receipt/context binding differs')
    if any(x['diagnostics']['reference_solver_energy_ha']!=receipt['state_energy'] or x['diagnostics']['state_residual_ha']!=receipt['state_residual'] for x in cached):
        raise ValueError('Saved numerical state diagnostic provenance differs')
    raw_sources={}
    def read_gzip(name):
        path=SIGNAL_DIR+name;data=blob(root,SIGNAL_CARRIER,path)
        raw_sources[SIGNAL_CARRIER+':'+path]=hashlib.sha256(data).hexdigest()
        return json.loads(gzip.decompress(data))
    signals=read_gzip('signal_records.json.gz');endpoints=read_gzip('signal_endpoints.json.gz')
    points={};origins={}
    for cid in M4_IDS:
        row=next(r for r in task['candidates'] if r['candidate']['id']==cid);record=signals[cid]
        matches=[e for e in endpoints if e['candidate']==row['candidate'] and e['epsilon']==.0005]
        if len(matches)!=1 or record['candidate']!=row['candidate']:raise ValueError('Saved m4 source selector differs')
        ep=matches[0]
        point=dict(candidate=deepcopy(row['candidate']),binding=deepcopy(task['binding']),compiler=task['profile']['compiler'],prices=[],
            saved_signal=deepcopy(record['signal']),saved_allowance={k:ep[k] for k in ('u','log_B','log_B_margin','empirical_not_certified')},
            saved_finite_time_endpoint=dict(axes={a:dict(allowance=ep['axes'][a]['allowance']) for a in ('real','imag')}))
        validate_saved_point(task,row,point)
        if point['saved_signal']['reference']!=context['reference']['1.6']:raise ValueError('Saved m4 reference differs')
        points[cid]=point;origins[cid]=dict(commit=SIGNAL_CARRIER,path=SIGNAL_DIR+'signal_records.json.gz',selector=cid,
            allowance_path=SIGNAL_DIR+'signal_endpoints.json.gz',allowance_epsilon=.0005,finite_time_status=ep['status'],
            reference_provenance='Saved full-H spectrum path; independent expm_multiply check at T2, not a new T1.6 pair')
    return dict(task=task,saved_N=cached,saved_points=points,saved_origins=origins,
        input_specs=inputs,context_spec=context_ref,context_metadata=context,
        state_metadata=dict(state_energy=receipt['state_energy'],state_residual=receipt['state_residual']),
        saved_sources=dict(store.sources,**raw_sources))


def load_saved_context(root,packet):
    """Future science only: unchanged frozen H4 loader bodies, no old launcher."""
    import run_h6_df19_rpe_d2 as io  # existing process-local BLAS/resource policy
    from . import h6_df19_rpe_d2 as action
    inputs=packet['input_specs'];ctx_spec=packet['context_spec']
    if any(sha(v['path'])!=v['sha256'] for v in list(inputs.values())+[ctx_spec]):raise ValueError('H4 original inputs changed')
    src=blob(root,SIGNAL_CARRIER,'scripts/h46_m5_tol_saved_input.py').decode()
    nodes=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name in ('array_record','load')]
    if len(nodes)!=2:raise ValueError('Frozen H4 loader source missing')
    def require(value,message):
        if not value:raise ValueError(message)
    scope=dict(Path=Path,c=SimpleNamespace(sha=sha,read=lambda path:json.loads(Path(path).read_bytes()),require=require))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_h4_saved_loader','exec'),scope)
    ham,sector,full,state,metadata=scope['load'](dict(mapping=dict(tol_snapshot=inputs['H4_tol_snapshot.npz']['path'],tol_receipt=inputs['H4_tol_receipt.json']['path'])))
    if (metadata['hamiltonian_hash']!=BINDING['hamiltonian'] or metadata['state_hash']!=BINDING['state']
            or ham.n_blocks!=10 or ham.n_qubits!=8 or len(state)!=36):raise ValueError('H4 saved layout differs')
    ctx=action.make_context(ham,sector,state,{'1':packet['context_metadata']['prefixes']['1']})
    return ctx,deepcopy(packet['state_metadata'])


def evolve_h4(ctx,candidate,reference):
    """Reuse generic action branch. B0 drops tail and applies only c, not c+s."""
    from . import h6_df19_rpe_d2 as action
    if candidate['formula']!='2nd' or candidate['method'] not in ('B0','PR'):raise ValueError('H4 S2 B0/PR only')
    internal=deepcopy(candidate)
    # In this kernel FULL selects deterministic prefix loops, with no assertion
    # that prefix equals rank. Restore B0 metadata before any binding/analysis.
    if internal['method']=='B0':internal['method']='FULL'
    record=action.evolve(ctx,internal,reference);record['candidate']=deepcopy(candidate)
    record['origin']='NEW_H4_S2_PREFIX_ACTION'
    return record


def analyze_point(task,row,point,state_metadata,source,*,extra=None):
    validate_saved_point(task,row,point)
    bound=connection.bind_saved_round(point,target_id=task['task_id'],m=row['m'],base_time=task['configuration']['base_time'],expected_system='H4',expected_rank=10,expected_binding=BINDING)
    d=scalar.analyze_bound_point(point,bound,task['configuration'],state_energy=state_metadata['state_energy'],state_residual=state_metadata['state_residual'])
    d.update(stage='H4_S2_BATCH',signal=deepcopy(point['saved_signal']),compiled_price=None,G=None,price_coverage='NOT_EVALUATED_N_ONLY',signed_weights=row['structure']['signed_weights'],allocation=row['structure']['allocation'],event_count=row['structure']['event_count'],allowance=dict(point['saved_allowance'],certified=False))
    if extra:d.update(extra)
    n=pipeline.shot_record(task,row,d,source)
    n['origin']='SAVED_SIGNAL_REANALYZED' if extra and 'saved_signal_source' in extra else 'NEW_H4_FINITE_MEAN_N'
    return n


def analyze_fresh(task,row,record,reference,state_metadata,source):
    if record.get('candidate')!=row['candidate'] or any(record.get(k)!=row['structure'][k] for k in ('signed_weights','allocation','event_count')):
        raise ValueError('Finite mean candidate/signed recipe differs')
    sig=record['signal'];ev=sig['allowance_evidence'];pair=reference['signal_difference']
    if not math.isfinite(pair) or not 0<=pair<=1e-10:raise ValueError('H4 reference pair gate')
    stages=row['q_PF']*2*(row['candidate']['prefix']+1)
    roundoff=64*2**-52*(1+ev['maximum_intermediate_norm'])*(1+stages)
    u=max(1e-9,10*ev['raw_corrected_vector_closure']+roundoff)+pair
    margin=max(1e-12,64*2**-52*(1+abs(sig['log_B']))*(1+stages))
    point=dict(candidate=deepcopy(row['candidate']),binding=deepcopy(task['binding']),compiler=task['profile']['compiler'],prices=[],saved_signal=sig,
        saved_allowance=dict(u=u,log_B=sig['log_B'],log_B_margin=margin,empirical_not_certified=True),
        saved_finite_time_endpoint=dict(axes={a:dict(allowance=u) for a in ('real','imag')}))
    return analyze_point(task,row,point,state_metadata,source,extra=dict(numerical_reference=deepcopy(reference),roundoff=roundoff))


def failed_record(task,row,source,reason,status='UNQUALIFIED_N_INPUT'):
    lane=dict(m=row['m'],candidate=deepcopy(row['candidate']),candidate_id=row['candidate']['id'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],target_id=task['task_id'],binding=deepcopy(task['binding']),base_time=task['configuration']['base_time'],alpha_X=task['configuration']['alpha_axis'],alpha_Y=task['configuration']['alpha_axis'],N_X=None,N_Y=None,status=status)
    return dict(identity=pipeline.identity(task,row),plan=lane,actual=dict(status='UNRESOLVED_REFERENCE_OR_ERROR',N_X=None,N_Y=None),diagnostics=dict(reason=reason),source=source,origin='H4_BATCH_UNQUALIFIED')


class H4NBatch:
    def __init__(self,packet,source,context_factory):
        self.packet=deepcopy(packet);self.source=deepcopy(source);self.context_factory=context_factory;self.last_run=None

    def __call__(self,task,rows):
        validate_task(task,self.packet['task'])
        if not rows or len({pipeline.key(r) for r in rows})!=len(rows) or any(r not in task['candidates'] for r in rows):raise ValueError('Explicit unique H4 menu required')
        cache={pipeline.key(n['identity']):n for n in self.packet['saved_N']}
        result={};fresh=[];saved=[]
        for row in rows:
            if pipeline.key(row) in cache:result[pipeline.key(row)]=deepcopy(cache[pipeline.key(row)])
            elif row['candidate']['id'] in M4_IDS:saved.append(row)
            else:fresh.append(row)
        self.last_run=dict(status='RUNNING',saved_N_copied=len(result),saved_signal_scalar=0,new_reference_times=[],reference_state_actions=0,finite_mean_candidates=0,PR_exact_tail_controls=0,automatic_retry=False,price_calls=0,G_calls=0)
        # Preflight every saved point before scalar or fresh science. No fallback.
        try:
            for row in saved:validate_saved_point(task,row,self.packet['saved_points'][row['candidate']['id']])
        except (KeyError,ValueError,TypeError) as exc:
            self.last_run['status']='STOP_SAVED_INPUT'
            return [result.get(pipeline.key(r)) or failed_record(task,r,self.source,str(exc)) for r in rows]
        for row in saved:
            try:
                n=analyze_point(task,row,self.packet['saved_points'][row['candidate']['id']],self.packet['state_metadata'],self.source,extra=dict(saved_signal_source=self.packet['saved_origins'][row['candidate']['id']]))
                self.last_run['saved_signal_scalar']+=1;result[pipeline.key(row)]=n
            except Exception as exc:result[pipeline.key(row)]=failed_record(task,row,self.source,str(exc))
        if any(n['plan']['status']!='CONDITIONAL_ROUND_PASS' for n in result.values()):
            self.last_run['status']='STOP_UNQUALIFIED_SAVED_N'
            return [result.get(pipeline.key(r)) or failed_record(task,r,self.source,'Saved scalar unqualified; fresh science not started','NOT_EVALUATED_AFTER_STOP') for r in rows]
        if fresh:
            import run_h6_df19_rpe_d2 as io
            from . import h6_df19_rpe_d2 as action
            try:ctx,metadata=self.context_factory(task)
            except Exception as exc:self.last_run.update(status='STOP_CONTEXT_INPUT',reason=str(exc));raise
            refs={}
            for T in sorted({r['candidate']['T'] for r in fresh}):
                self.last_run['new_reference_times'].append(T)
                try:refs[T]=action.reference_pair(ctx,T);self.last_run['reference_state_actions']+=2
                except Exception as exc:self.last_run.update(status='STOP_REFERENCE_GATE',reason=str(exc));raise
            keys=action.half_step_keys(fresh);resources=io.worker_resources(ctx,len(keys))
            with ThreadPoolExecutor(max_workers=resources['half_step_workers']) as pool:
                for key,value in pool.map(lambda key:action.half_step(ctx,key),keys):ctx.half_steps[key]=value
            jobs=io.worker_resources(ctx,len(fresh));self.last_run.update(half_step_resources=resources,candidate_resources=jobs,references=refs)
            def job(row):
                rec=None
                try:
                    ref=refs[row['candidate']['T']];rec=evolve_h4(ctx,row['candidate'],ref['reference'])
                    return analyze_fresh(task,row,rec,ref,metadata,self.source)
                except Exception as exc:
                    failed=failed_record(task,row,self.source,str(exc),'NUMERICAL_GATE_FAILURE')
                    failed['diagnostics'].update(finite_mean=rec,numerical_reference=refs[row['candidate']['T']])
                    return failed
            with ThreadPoolExecutor(max_workers=jobs['half_step_workers']) as pool:
                for row,n in zip(fresh,pool.map(job,fresh)):result[pipeline.key(row)]=n
            self.last_run.update(finite_mean_candidates=len(fresh),PR_exact_tail_controls=sum(r['candidate']['method']=='PR' for r in fresh))
        out=[result[pipeline.key(r)] for r in rows]
        self.last_run['status']='BATCH_EVALUATED' if all(n['plan']['status']=='CONDITIONAL_ROUND_PASS' for n in out) else 'STOP_UNQUALIFIED_N'
        return out


def make_provider(root,source):
    packet=prepare(root)
    return H4NBatch(packet,source,lambda task:load_saved_context(root,packet))
