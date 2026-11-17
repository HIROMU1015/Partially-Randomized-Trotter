"""Explicitly artificial server2 fixture for the local0 request transport tests."""
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch
import json,sys,os
from trottertracks import rpe_method_optimizer as O
from trottertracks.rpe_local0_search import Gateway,load,write,digest
import run_rpe_method_optimizer as CLI

def generate(caps,prices,budget,rule):
    for n in range(max(caps),0,-1):
        vector=[min(n,c) for c in caps]
        if sum(a*b for a,b in zip(vector,prices))<=budget:return dict(shots=vector)
    return None

def context(problem):
    values=problem['methods']['PR_PAULI'].get('L_D_values',[4,5,6])
    candidates=[]
    for ld in values:
        cid='ARTIFICIAL_LD'+str(ld)
        price=Fraction(100) if ld==5 else Fraction(1) if ld==6 else Fraction(5,4) if ld==7 else Fraction(3)
        candidates.append(dict(method='PR_PAULI',family='PR',key=dict(L_D=ld),configuration_id=cid,name=cid,
            prices=[price]*26,caps=problem['methods']['PR_PAULI']['caps'],ready=True,needs=[],labels=[],
            price_status='ARTIFICIAL',signal_source=None,saved_points=[]))
    return dict(candidates=candidates,expected={c['configuration_id']:dict(total_configuration_id='TOTAL_'+c['configuration_id'],binding_id='BIND_'+c['configuration_id']) for c in candidates},
        tau=Fraction(1,100),saved=O.SavedBackend([],{}),generate=generate,generation={},facts=dict(fragment_count=10),catalog=dict(candidates=[]))

def outcome(request):
    cid=request['configuration_id'];expected=request['expected']
    return dict(artificial=True,status=O.PASS,MSE_lower='1/10000',MSE_upper='1/10000',
        identity=dict(configuration_id=cid,total_configuration_id=expected['total_configuration_id'],binding_id=expected['binding_id'],contract_matches=True,
            allocation=[dict(m=i,N_X=x,N_Y=y) for i,(x,y) in enumerate(zip(request['N_X'],request['N_Y']))]))

def build(config):
    assert load(config['problem'])['problem']['system']=='ARTIFICIAL_LOCAL0_SCOPE_ACCEPTANCE_ONLY'
    gateway=Gateway(config)
    def preflight(run,problem,spec):
        scoped=CLI.scoped(problem,spec.get('method_overrides'))
        with patch.object(O,'prepare',return_value=context(scoped)):
            return gateway.normal_preflight(run,problem,spec)
    def launch(run,problem,spec):
        root=Path(config['code_root']);scoped=CLI.scoped(problem,spec.get('method_overrides'));ctx=context(scoped)
        calls=[];old_events=load(Path(config['state_directory'])/'ARTIFICIAL_CALLS.json') if (Path(config['state_directory'])/'ARTIFICIAL_CALLS.json').exists() else []
        def fake(self,job,rule,mode):
            assert mode=='execute' and rule['kind']=='artificial'
            request=load(job/'REQUEST.json');assert request['configuration_id'].startswith('ARTIFICIAL_')
            calls.append(dict(configuration_id=request['configuration_id'],run=str(run),mode=mode))
            write(job/'OUTCOME.json',outcome(request))
            return SimpleNamespace(pid=os.getpid(),wait=lambda:0)
        invocation='artificial-'+str(len(old_events))+'-'+run.name
        write(run/'LAUNCH.json',dict(pid=os.getpid(),invocation_id=invocation,artificial_only=True))
        a=SimpleNamespace(round=run/'ROUND.json',run_dir=run,plan_only=False,resume_from=[],ingest=None)
        with patch.object(O.Executor,'run_adapter',fake):
            CLI.round_mode(a,problem,[],prepared_context=ctx)
        write(Path(config['state_directory'])/'ARTIFICIAL_CALLS.json',old_events+calls)
        exit=dict(returncode=0,invocation_id=invocation,artificial_only=True)
        write(run/'EXIT.json',exit);write(run/'invocations'/invocation/'EXIT.json',exit)
    gateway.preflight=preflight;gateway.launch=launch
    return gateway

def seed(run,config):
    run=Path(run);problem=load(run/'PROBLEM.json');spec=load(run/'ROUND.json');scoped=CLI.scoped(problem,spec.get('method_overrides'));ctx=context(scoped)
    routes=CLI.merged_routes(problem,spec);ex=O.Executor(config['code_root'],run,routes,ctx['expected'],problem).for_method('PR_PAULI')
    proposal=dict(configuration_id='ARTIFICIAL_LD5',vector=(4,)*26,cost=Fraction(10400),origin=dict(kind='ARTIFICIAL_EXISTING_SAVED'))
    job=ex.job_dir(proposal);job.mkdir(parents=True,exist_ok=False)
    request=ex.request(proposal,job,ex.route(proposal['configuration_id']))
    write(job/'REQUEST.json',request);write(job/'OUTCOME.json',outcome(request))
    write(job/'FIXTURE_RECEIPT.json',dict(artificial_only=True,real_scientific_calls=0,fixture='saved exact Q3 identity for overlap acceptance'))
    return dict(job=str(job),outcome_sha256=digest(load(job/'OUTCOME.json')))
