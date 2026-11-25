from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import math
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import patch

from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_scaled_shots as S
from trottertracks import rpe_scope_map as scope
from trottertracks.pr_candidate_completion import physical_evaluation_contract, preparation_failure_class
from trottertracks.rpe_optimizer_concurrent import ConcurrentExecutor
from trottertracks.pf_candidate_selection import metadata_tool_failure
from trottertracks.rpe_pbs import known_independent_search_jobs


def test_max_width_mapper_exact_requested_and_realized_width():
    h=F(.1)
    for n in range(1,11):
        width=n*h
        result=scope.max_width_schedule(h,width,12,lambda_R='1/10',kappa=7)
        assert result['requested_delta_pf']==scope.encode(width)
        for row in result['rows']:
            T=h*2**row['m'];actual=scope.exact(row['actual_delta_pf'])
            assert row['q_PF']>=1 and T==actual*row['q_PF'] and actual<=width
            assert row['R_outer']>=1 and row['event_count']==row['q_PF']*row['R_outer']
    assert [r['q_PF'] for r in scope.max_width_schedule(h,3*h,5)['rows']]==[1,1,2,3,6,11]
    assert [r['q_PF'] for r in scope.max_width_schedule(h,h,5)['rows']]==[1,2,4,8,16,32]


def test_next_mapper_policy_is_explicit_and_matches_old_grid_at_h():
    from trottertracks import pr_rpe_rmse as rmse
    binding=dict(system='ARTIFICIAL',rank=3)
    design=dict(system='ARTIFICIAL',binding=binding,base_time=.125,delta=.125,prefix=1,
                K=2,formula='2nd',PF=dict(method='FULL',prefix=3),profile={},contract={},
                energy_prior_ha=[-1,1],decoder='FIXTURE',source='FIXTURE',state_assumption='FIXTURE')
    tail=dict(binding=binding,prefix=1,lambda_R=.5,source='FIXTURE')
    policy=dict(policy_id=rmse.POLICY)
    args=dict(M=4,kappa_paper=1,tail_norm=tail,policy=policy)
    original=rmse.schedule(design=design,**args)
    explicit=dict(design,pf_grid_policy=scope.MAX_WIDTH_MAPPER,requested_delta_pf=scope.encode(F(1,8)))
    same=rmse.schedule(design=explicit,**args)
    assert [r['identity']['candidate'] for r in same['rows']]==[r['identity']['candidate'] for r in original['rows']]
    explicit['requested_delta_pf']=scope.encode(F(3,8))
    wide=rmse.schedule(design=explicit,**args)
    assert [r['identity']['q_PF'] for r in wide['rows'][1::2]]==[1,1,2,3,6]
    assert all(r['mapping']['pf_grid_policy']==scope.MAX_WIDTH_MAPPER for r in wide['rows'])


def test_complete_scope_counts_aliases_and_unsupported_families():
    h=F(.1); width=scope.encode(h); wide=scope.encode(2*h)
    methods={'PR_DF':dict(L_D_values=[0,1],requested_delta_pf_values=[width,wide],
                         absolute_kappa_values=['1','2'],generation={'formulas':['2nd']})}
    base=dict(spec=dict(h=width,split=dict(lambda_R='1/10')),
              rounds=[dict(structure=dict(signed_weights=[1.]))])
    families={}
    for ld in (0,1):
        for delta in (width,wide):
            key=f'{ld}:{scope.exact(delta)}';ready=ld==1 and delta==width
            families[key]=dict(method='PR_DF',L_D=ld,formula='2nd',delta_pf=delta,requested_delta_pf=delta,
                               status='READY_METADATA' if ready else 'UNSUPPORTED_MAPPER',
                               configuration=base if ready else None,
                               reason='NONEMPTY_DF_HEAD_TAIL_REQUIRED' if ld==0 else 'Nonintegral PF grid')
    key=f'1:{h}'
    points=[dict(method='PR_DF',family=key,kappa=str(k),status='EVALUATED',configuration_id=f'decl{k}',
                 evaluation_configuration_id='canonical') for k in (1,2)]
    row=dict(configuration_id='canonical',M=0,configuration=dict(L_D=1,formula='2nd',delta_pf=width,absolute_kappa='1'),
             allocations=[dict(status=O.PASS)],mse_feasible={'status':O.PASS},cost_status='UNAVAILABLE')
    document=scope.build_scope_map(methods,[0],families=families,points=points,results={'PR_DF':dict(by_configuration=[row])})
    result=document['methods']['PR_DF']
    assert result['requested']==8
    assert result['counts']=={'MAPPER_UNSUPPORTED':6,'MSE_PASS_PRICE_REQUIRED':2}
    assert sum('proposed_mapper' in r for r in result['rows'])==6
    assert all(r['configuration_id']=='canonical' for r in result['rows'] if r['status']=='MSE_PASS_PRICE_REQUIRED')


def test_physical_alias_keeps_law_semantics_but_drops_declaration_ids():
    base=dict(spec={'kappa':{'value':1}},rounds=[dict(m=0,T='1/10',q_PF=1,R_outer=3,actual_delta_pf='1/10',
        structure={'allocation':[3]},physical_candidate={'id':'physical','K':2,'q':1},
        law_binding={'configuration_id':'a','candidate_id':'a0','kappa':{'kind':'PAPER','value':1},
                     'definition':{'model':'ideal','source':{'commit':'fixed'},'A_declaration':{'candidate_id':'physical','R_outer':3}}})])
    alias=deepcopy(base);alias['spec']['kappa']['value']=2;alias['rounds'][0]['law_binding'].update(
        configuration_id='b',candidate_id='b0',kappa={'kind':'PAPER','value':2})
    assert physical_evaluation_contract(base)==physical_evaluation_contract(alias)
    alias['rounds'][0]['law_binding']['definition']['source']['commit']='different'
    assert physical_evaluation_contract(base)!=physical_evaluation_contract(alias)


def test_feasibility_probes_change_order_not_nonmonotone_result():
    policy=S.settings(dict(M=[0],grid=dict(min_k=-2,max_k=2,divisor=2),compare_paper=False))
    candidate=dict(configuration_id='candidate',name='candidate',method='PR_PAULI',key={},M=0,
                   prices=[F(1),F(1)],ready=True,needs=[])
    tau=F(S.EPSILON)**2
    outputs=[]
    for enabled in (False,True):
        seen=[]
        def answer(cid,vector):
            seen.append(vector[0]); value=tau/2 if vector[0]==8 else tau*2
            return value,value
        result=S.run('PR_PAULI',[candidate],{},O.FixtureBackend(answer,tau),
            dict(scaled_shots=policy,exploratory_cost=True,feasibility_first=enabled),tau,None)
        outputs.append((seen,result['by_configuration'][0]['best']['N_X']))
    assert outputs[0][1]==outputs[1][1]==[8]
    assert outputs[1][0][0]==22 and outputs[1][0][1]!=22


def test_preparation_failure_class_distinguishes_numeric_and_owner_hold():
    assert preparation_failure_class('NUMERICAL_GATE')=='SCIENTIFIC_NUMERICAL_FAILURE'
    assert preparation_failure_class('STAGE_FAILED:DF_CALIBRATION:...NO_AUTOMATIC_RETRY')!='SCIENTIFIC_NUMERICAL_FAILURE'
    assert preparation_failure_class('TECHNICAL',{'action':'RECOVER_ORIGINAL_ATTEMPT'})=='EXECUTION_OWNERSHIP_UNRESOLVED'


def test_tail_lambda_diagnostic_does_not_relax_exact_gate():
    import pytest
    from run_pr_df_candidate_prices import require_tail_lambda
    require_tail_lambda(.5,.5,reason='EXACT')
    with pytest.raises(ValueError,match='positive_ulp_distance=1, policy=EXACT_BINARY64'):
        require_tail_lambda(.5,math.nextafter(.5,1),reason='EXACT')


def test_pf_prior_reuse_contents_not_publication_id(tmp_path):
    job=tmp_path/'job';job.mkdir()
    (job/'PROCESS_RETURN.json').write_text(json.dumps(dict(returncode=0,signal=None)))
    (job/'LAUNCH.json').write_text('{}')
    adapter=tmp_path/'adapter.py';adapter.write_text("DROP={'label','allocation'}\nCONTRACT=('options',)\n")
    cid='cfg'; documents={}
    def location(commit,path):return dict(commit=commit,path=path)
    def request(commit):
        signal=location(commit,'signal');cfg=location(commit,'configuration');template=location(commit,'template')
        entry=dict(binding_id=commit,configuration=cfg,signal_packet=signal,total_configuration_id='total')
        documents[commit,'signal']={'data':[.125]};documents[commit,'configuration']={'q':3}
        documents[commit,'template']={'cases':[dict(configuration_id=cid,**entry)],'options':{'model':'same'}}
        route=dict(kind='pf_b_pf6',configurations={cid:entry},templates=[template])
        return dict(request_id=job.name,configuration_id=cid,optimizer_source='fixed',route=route,
            expected=dict(configuration_id=cid,binding_id=commit,signal_source=signal,total_configuration_id='total'))
    old,new=request('before'),request('after')
    (job/'REQUEST.json').write_text(json.dumps(old))
    executor=ConcurrentExecutor.__new__(ConcurrentExecutor)
    executor.root=tmp_path;executor.adapter=adapter;executor.exploratory_cost=True;executor.optimizer_source='fixed'
    executor.route=lambda _:new['route'];executor.expected={cid:new['expected']}
    executor.pf_template_source=lambda route,_:route['templates'][0]
    executor.prior_history_inventory=SimpleNamespace(lock=threading.Lock(),route_equivalence={})
    reader=SimpleNamespace(read=lambda loc:deepcopy(documents[loc['commit'],loc['path']]))
    with patch.object(O,'Reader',return_value=reader):
        assert executor.prior_same_route(job,cid)
        documents['after','signal']['data']=[.25]
        executor.prior_history_inventory.route_equivalence.clear()
        assert not executor.prior_same_route(job,cid)


def test_metadata_failure_cannot_reopen_owned_science(tmp_path):
    script=tmp_path/'check_pf_preparation_owner.py';script.write_text('pass')
    problem=subprocess.CalledProcessError(1,['python',str(script)])
    assert metadata_tool_failure(problem,tmp_path,{})['pre_dispatch']
    (tmp_path/'PRICE.DISTRIBUTED.json').write_text('{}')
    assert not metadata_tool_failure(problem,tmp_path,{})['pre_dispatch']


def test_independent_pbs_job_must_be_registered_and_named(tmp_path):
    central=tmp_path/'.rpe-central';root=central/'state';root.mkdir(parents=True)
    a=root/'search-aaaaa123';b=root/'search-bbbbb123';a.mkdir();b.mkdir()
    (root/'RUNS.json').write_text(json.dumps({p.name:dict(run_directory=str(p)) for p in (a,b)}))
    state=b/'distributed/pbs/STATE.json';state.parent.mkdir(parents=True)
    record=dict(status='R',spec=dict(run_id=str(b)),observed=dict(job_id='one',scheduler=dict(Job_Name='RPEsearch-bbbbb',job_state='R')))
    state.write_text(json.dumps(dict(allocations={'one':record})))
    assert known_independent_search_jobs(a)==['one']
    record['observed']['pilot_token']='not-independent'
    state.write_text(json.dumps(dict(allocations={'one':record})))
    assert known_independent_search_jobs(a)==[]


def test_technical_metadata_failure_is_rechecked_only_after_change(tmp_path):
    from trottertracks import pr_candidate_completion as C
    directory = tmp_path/'candidate'
    directory.mkdir()
    started = directory/'plan.STARTED.json'
    receipt = directory/'plan.RETURN.json'
    started.write_text(json.dumps(dict(science=False, source='provider')))
    receipt.write_text(json.dumps(dict(returncode=1, signal=None, source='provider')))
    assert C.metadata_only_failure(directory)
    completion = C.Completion.__new__(C.Completion)
    completion.directory = tmp_path
    record = dict(key='candidate', directory=str(directory),
                  failure_class='TECHNICAL_INPUT_OR_TOOL_FAILURE',
                  scientific_failure=False, retry_protected=True, metadata_revision='old')
    assert C.preparation_retry_protected(record)
    assert C.preparation_retry_protected(dict(scientific_failure=True))
    with patch.object(C, 'ROOT', tmp_path), patch.object(completion, 'metadata_revision', return_value='old'):
        assert completion.metadata_retry_path(record) is None
    with patch.object(C, 'ROOT', tmp_path), patch.object(completion, 'metadata_revision', return_value='new'):
        assert completion.metadata_retry_path(record).name == 'metadata-revision-new'
        (directory/'science.DISTRIBUTED.json').write_text('{}')
        assert completion.metadata_retry_path(record) is None
    (directory/'science.DISTRIBUTED.json').unlink()
    receipt.unlink()
    assert not C.metadata_only_failure(directory)


def test_scope_status_does_not_turn_one_completed_q3_into_completed_configuration():
    from trottertracks.rpe_scope_progress import scope_progress
    document = dict(methods={'FULL_PF': dict(requested=2, axes=['M'],
        counts={'MSE_PASS_PRICE_REQUIRED':1,'EVALUATION_HELD':1}, rows=[
            dict(coordinates={'M':12}, configuration_id='a', status='MSE_PASS_PRICE_REQUIRED',
                 price_required=True, allocations={'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD':1}),
            dict(coordinates={'M':13}, configuration_id='a', status='EVALUATION_HELD',
                 allocations={'UNKNOWN':1})])})
    observations = {'a':dict(candidate_id='a', method='FULL_PF',state='EVALUATION_COMPLETED')}
    records = [dict(method='FULL_PF',configuration_id='a',stage_progress=dict(price='AVAILABLE',price_requested=True))]
    summary = scope_progress(document,records,observations)
    assert summary['FULL_PF']['live_counts']=={'PRICE_AVAILABLE_PENDING_REPRICE':1,'EVALUATION_HELD':1}
    assert observations['a']['state']=='MIXED_SCOPE_PROGRESS'
    assert len(observations['a']['scope_rows'])==2


def test_preparation_resource_measurements_use_compatible_provider_and_legacy_worker():
    from trottertracks.rpe_q3_resources import measured_resource_class
    from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
    request = dict(preparation=dict(stage='compile',provider='TEST',source={'commit':'a'}))
    profile = dict(task_cpu_count=2,task_memory_bytes=1024,dp_workers=1)
    old = measured_resource_class(request,runtime=profile)
    detailed = measured_resource_class(request,runtime=profile,preparation_detail=True)
    other = deepcopy(request);other['preparation']['source']['commit']='b'
    assert old=='preparation:compile:TEST'
    assert detailed!=measured_resource_class(other,runtime=profile,preparation_detail=True)
    assert detailed!=measured_resource_class(request,runtime=dict(profile,task_memory_bytes=2048),preparation_detail=True)
    row=dict(request=request,eligibility={'execution_profiles':{'old':profile,'new':profile}})
    fake=SimpleNamespace(hosts={'old':{'resource_report':{}},'new':{'resource_report':{'preparation_class_version':2}}})
    assert DistributedExecutor._task_resource_class(fake,row,'old')==old
    assert DistributedExecutor._task_resource_class(fake,row,'new')==detailed


def test_pf_metadata_uses_explicit_non_h4_base_without_borrowing_h4_priority():
    from trottertracks import pf_candidate_space as P
    locator=dict(commit='artificial',path='fixture',selector=['spec'])
    base=dict(binding=dict(rank=3,system='ARTIFICIAL'),h=P.encoded(F(1,8)),M=4)
    def specification(_base,*,formula,delta_pf,M,rank):
        assert rank==3 and M==4 and P.exact(delta_pf)==F(3,8)
        return dict(split={'retained_hamiltonian':{}},formula=formula,delta_pf=delta_pf,M=M)
    with patch.object(P,'base_configuration',return_value=base), \
            patch.object(P.C,'published_api',return_value=SimpleNamespace(D=SimpleNamespace(full_pf_spec=specification))), \
            patch.object(P.C,'configure',side_effect=lambda root,spec,api:dict(id=str(spec),spec=spec)):
        rows=P.candidates('fixture',formulas=['2nd'],widths=['3'],base_spec=locator)
    assert len(rows)==4
    assert rows[-1]['method']=='FULL_PF' and rows[-1]['p']==3
    assert all(r['priority_evidence']['T1'] is None for r in rows)


def test_supervisor_failure_proof_requires_ended_original_attempt_and_no_science(tmp_path):
    import fcntl
    from trottertracks.rpe_dry_check_recovery import dry_check_rejection
    assignment = dict(job_id='job', attempt_id='attempt', request_sha256='fixed')
    detail = dict(reason='OSError: fixed input unavailable', automatic_retry=False)
    def save(name,value): (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'WORKER_PROCESS.lock').touch()
    save('WORKER_STATE.json',dict(state='UNKNOWN',detail=detail))
    save('SUPERVISOR_STARTED.json',dict(pid=30));save('SUPERVISOR_LAUNCH.json',dict(pid=30))
    assert dry_check_rejection(tmp_path,assignment) is None
    save('SUPERVISOR_PRELAUNCH_FAILURE.json',dict(assignment, reason=detail['reason'],
        returncode=2, supervisor_pid=30, scientific_execution_started=False))
    assert dry_check_rejection(tmp_path,assignment)['kind']=='SUPERVISOR_FAILED_BEFORE_EXECUTION'
    assert dry_check_rejection(tmp_path,dict(assignment,attempt_id='other')) is None
    with (tmp_path/'WORKER_PROCESS.lock').open('r+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert dry_check_rejection(tmp_path,assignment) is None
    (tmp_path/'SCIENTIFIC_STARTING.json').touch()
    assert dry_check_rejection(tmp_path,assignment) is None
