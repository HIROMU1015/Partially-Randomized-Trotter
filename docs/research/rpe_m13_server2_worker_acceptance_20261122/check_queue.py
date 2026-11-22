"""Start and resume isolated artificial M=13 runs using the fleet worker."""

from collections import Counter
from contextlib import redirect_stdout
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

from check_saved import FAMILIES, FLEET, HERE, PROJECTS, git, get, put

sys.path[:0] = [str(FLEET/'scripts'), str(FLEET/'tests')]
from test_rpe_scaled_shots import Gateway, O, S, entry, fixture

FIXED = 'cbc471c913072de45c25a0ed22d51dff8a9b7034'
DECODER_VIEW = PROJECTS/'partially-randomized-trotter-worktrees/b2-route-view-b-cbc471c91'
TAU = Fraction(S.EPSILON)**2
SOURCE = git('rev-parse','HEAD').decode().strip()


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def snapshot(run, worker):
    db = run/'distributed/QUEUE.sqlite3'
    with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as connection:
        queue=connection.execute('select job_id,method,attempt_id,state,host_id,request_sha256 from jobs order by job_id').fetchall()
        adopted=connection.execute("select job_id,count(*) from events where event='ADOPT' group by job_id").fetchall()
    outputs={p.parent.name:checksum(p) for p in run.glob('jobs/*/*/OUTCOME.json')}
    launches=[(r['job_id'],r['attempt_id']) for r in events(worker/'WORKER_EVENTS.jsonl') if r['event']=='LAUNCH']
    adoption=Counter(r['job'] for r in events(run/'PROGRESS.jsonl') if r['event']=='ADOPT')
    return queue,outputs,launches,adoption,adopted


def check(run_root, *, large):
    existing=(run_root/'ROUND.json').exists()
    run_root.mkdir(parents=True, exist_ok=existing)
    problem,context,routes=fixture(FAMILIES,configurations=1,steps=100)
    problem.update(problem=dict(system='ARTIFICIAL_SCALED_SHOTS',
                   hamiltonian_data=dict(expected=dict(fragment_count=10))),
                   rpe=dict(M=13),execution_routes=routes)
    problem['accuracy']['epsilon_E_decimal']=S.EPSILON
    context['saved'].table.clear()
    context['tau']=TAU
    for row in context['candidates']:
        row.update(M=13,prices=[1]*28)
    for method in FAMILIES:
        spec=routes[method][0]['artificial']
        spec.update(tau=str(TAU),weights=[m+1 for m in range(14)],scale='1/100000000')
        if large:
            spec['finite_support_fixture']=dict(view=str(DECODER_VIEW),commit=FIXED,target=0.0)
    spool=run_root/'server2-spool'
    profile=dict(host_id='server2',root=str(FLEET),interpreter=sys.executable,
                 slots=len(FAMILIES),route_views={},work_dir=str(spool))
    spec=dict(round_id='TEMPLATE',methods=list(FAMILIES),science_approved=False,
              max_new_q3_per_method={method:2 for method in FAMILIES},
              executor=dict(scheduler='concurrent',max_concurrent_jobs=len(FAMILIES),
                            large_shot_bounds=large,distributed=dict(port=0,local_worker=profile)))
    for name,data in [('PROBLEM.json',problem),('ROUND.json',spec)]:
        path=run_root/name
        if existing:
            assert get(path)==data
        else:
            put(path,data)
    config=dict(problem=str(run_root/'PROBLEM.json'),round=str(run_root/'ROUND.json'),
                entry='schedule',state_directory=str(run_root/'state'),code_root=str(FLEET),
                code_source=SOURCE,computation_source=SOURCE,python=sys.executable)
    invocations=[]

    def launch(run, _, selected):
        record=dict(invocation_id=f'artificial-local-{len(invocations)+1+int(existing)}')
        invocations.append(record)
        (run/'LAUNCH.json').write_text(json.dumps(record))
        with (run_root/'CONTROLLER.log').open('a') as log, redirect_stdout(log):
            entry.main(['--problem',str(run/'PROBLEM.json'),'--round',str(run/'ROUND.json'),'--run-dir',str(run)])
        (run/'EXIT.json').write_text(json.dumps(dict(record,returncode=0)))

    gateway=Gateway(config,preflight=lambda *a:dict(status='PASS',artificial=True),launch=launch)
    k=16 if large else -12
    scope=dict(methods=list(FAMILIES),scaled_shots=dict(M=[13],grid=dict(min_k=k,max_k=k,divisor=4),
                                                      compare_paper=False))
    with patch.object(O,'prepare',return_value=context):
        if existing:
            run,=list((run_root/'state/runs').iterdir())
        else:
            started=gateway.handle(dict(action='start',instruction='Artificial server2 only',scope=scope))
            run=Path(started['run_directory'])
        before=snapshot(run,spool)
        queue,outputs,launches,adoption,adopted=before
        # The grid point and the mandatory s=1 reference both enter the test queue.
        assert Counter(r[1] for r in queue)==Counter({m:2 for m in FAMILIES}), queue
        assert all(state=='ACCEPTED' and host=='server2' for _,_,_,state,host,_ in queue), queue
        assert Counter(launches)==Counter((j,a) for j,_,a,_,_,_ in queue), launches
        assert adoption==Counter({j:1 for j,_,_,_,_,_ in queue}), adoption
        assert dict(adopted)==dict(adoption)
        resumed=gateway.handle(dict(action='resume',run_id=run.name,instruction='Artificial server2 resume'))
        assert Path(resumed['run_directory'])==run
        assert resumed['status']=='REUSED_COMPLETED' and len(invocations)==int(not existing), resumed
        assert snapshot(run,spool)==before
        controller_args=['--problem',str(run/'PROBLEM.json'),'--round',str(run/'ROUND.json'),'--run-dir',str(run)]
        original_stop={key:row['stop_reason'] for key,row in get(run/'ROUND_RESULT.json')['methods'].items()}
        with (run_root/'CONTROLLER_RESUME.log').open('x') as log, redirect_stdout(log):
            entry.main(controller_args)
        resumed_stop={key:row['stop_reason'] for key,row in get(run/'ROUND_RESULT.json')['methods'].items()}
        assert resumed_stop==original_stop, resumed_stop
        assert snapshot(run,spool)==before
    rows=[]
    for job,method,attempt,state,host,request_sha in queue:
        worker=spool/job/attempt
        path=run/'jobs'/method/job/'OUTCOME.json'
        outcome=get(path)
        assert outcome['artificial'] and outcome['identity']['M']==13
        nx=[r['N_X'] for r in outcome['identity']['allocation']]
        assert nx in (S.base_shots(13),[S.dyadic_count(x,k,4) for x in S.base_shots(13)])
        lo,hi=(Fraction(**outcome[p]) for p in ('MSE_lower','MSE_upper'))
        assert lo<=hi<=TAU and outcome['status']==O.PASS
        if large:
            assert outcome['evaluator']=='RPE_FINITE_SUPPORT_MSE_V1' and outcome['scientific_evaluation_calls']==0
        else:
            reference=sum(Fraction(m+1,100000000*n) for m,n in enumerate(nx))
            assert lo==reference==hi
        rows.append(dict(method=method,host=host,job_id=job,attempt_id=attempt,state=state,
                        request_sha256=request_sha,outcome_sha256=outputs[job],outcome=str(path),
                        worker_assignment=str(worker/'ASSIGNMENT.json'), resume_additional_launches=0,
                        resume_duplicate_adoptions=0, M=13, shots_each_axis=nx,
                        MSE_lower=str(lo), MSE_upper=str(hi)))
    contract='FIXED_FINITE_SUPPORT_KERNEL' if large else 'TRANSPORT_COST_FREE_SYNTHETIC_MSE_FIXTURE'
    put(run_root/'RESUME_ACCEPTANCE.json',dict(source=SOURCE,contract=contract,rows=rows,
        gateway_resume=resumed, controller_reentry=True, controller_args=controller_args,
        stop_reasons=resumed_stop, central_host='SERVER2_ISOLATED_LOOPBACK',
        live_central_queue=False, true_h4_q3_calls=0))
    return len(rows)


if __name__=='__main__':
    kind=sys.argv[1]
    assert kind in ('standard','large')
    target=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else HERE/'queue01'/kind
    print(kind,check(target,large=kind=='large'),flush=True)
