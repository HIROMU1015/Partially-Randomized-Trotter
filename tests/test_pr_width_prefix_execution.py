"""Normal multi-width CLI + queue + actual fixed DP, on artificial tables only.

The custom interpreter is a test fixture.  It calls the normal Q3 job adapter
and replaces only its labelled artificial numerical leaf.  The prefix context,
disk cache, guarded original DP, REQUEST/OUTCOME checks and queue are real.
No molecular input, saved H4 signal or circuit is loaded by this fixture.
"""
from contextlib import redirect_stdout, nullcontext
from copy import deepcopy
from fractions import Fraction
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
PYTHON=Path('/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python')
VIEW=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/b2-route-view-b-bd42dd1e9')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def fixed_artificial_dp_leaf(adapter,request,out,mode):
    """An artificial table evaluation, not an H4 scientific producer."""
    sys.path[:0]=[str(VIEW/'src'),str(VIEW/'tests')]
    from trottertracks import rpe_total_dp_parallel as P
    from check_rpe_total_decoder_risk import make
    from check_rpe_total_dp_parallel import tables
    artifact=out/'ARTIFICIAL_RESULT.json'
    ident=dict(request['expected'],allocation=[dict(m=m,N_X=x,N_Y=y)
        for m,(x,y)in enumerate(zip(request['N_X'],request['N_Y']))],contract_matches=True)
    if mode=='collect':
        return dict(json.loads(artifact.read_text()),identity=ident,artificial=True,collected=True)
    if mode=='dry-check':return dict(status='DRY_CHECK_PASS',artificial=True)
    signal=request['route']['artificial'].get('signal_token','SAME_ARTIFICIAL_SIGNAL')
    raws=[(.25,-.5)]*13 if signal=='SAME_ARTIFICIAL_SIGNAL' else [(.5,-.5)]*13
    args=make(vector=request['N_X'],raws=raws,total=None,partial=None)
    for row,y in zip(args['allocations'][0]['shots'],request['N_Y']):row['N_Y']=y
    tt=tables(args)
    scope=dict(source={'fixture':'ARTIFICIAL_MULTI_WIDTH_DP'},maximum_frontier_states=8,maximum_transition_pairs=5000)
    case=dict(configuration={'fixture':request['configuration_id']},signal_packet={'fixture':signal},
        evaluation_target={'fixture':0},binding_id=request['expected'].get('binding_id'),
        total_configuration_id=request['expected'].get('total_configuration_id'))
    inputs={k:{'fixture':'FIXED_ARTIFICIAL_EVALUATION_CONTRACT'}for k in adapter.CONTRACT}
    # This is the very context used by existing scientific route workers.
    with adapter.prefix_cache_context(request,case,inputs,out):
        engine=P.Accelerator(workers=request.get('execution_profile',{}).get('dp_workers') or 1)
        dp=engine(tt,h=10.,target=0.,scope=scope)
    write(out/'ARTIFICIAL_DP_RESULT.json',dp)
    tau=Fraction(request['route']['artificial']['tau'])
    bound=Fraction(dp['MSE_upper']['numerator'],dp['MSE_upper']['denominator'])
    value=dict(status=adapter.PASS if bound<=tau else adapter.FAIL,MSE_lower=dp['MSE_lower'],
        MSE_upper=dp['MSE_upper'],scientific_evaluation_calls=1,
        artificial_numerical_scope='ORIGINAL_GUARDED_DP_ON_ARTIFICIAL_TABLES_ONLY')
    write(artifact,value)
    return dict(value,identity=ident,artificial=True)


def worker():
    # Executor command is interpreter -B normal_adapter.py --request ... .
    args=sys.argv[2:]
    assert args[0]=='-B' and Path(args[1]).name=='run_rpe_optimizer_q3_job.py'
    spec=importlib.util.spec_from_file_location('_artificial_prefix_job',args[1])
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    adapter.artificial=lambda request,out,mode:fixed_artificial_dp_leaf(adapter,request,out,mode)
    sys.argv=[args[1]]+args[2:]
    return adapter.main()


def interpreter(path):
    path=Path(path)
    path.write_text('#!/bin/sh\nexec '+shlex.quote(str(PYTHON))+' -B '+
        shlex.quote(str(Path(__file__).resolve()))+' --artificial-dp-worker "$@"\n')
    path.chmod(0o755);return str(path)


def fixture_classes():
    import check_pr_width_search as F
    from trottertracks import rpe_method_optimizer as O
    class Completion(F.ArtificialCompletion):
        def install(self,context,problem,records):
            routes=super().install(context,problem,records)
            for c in context['candidates']:c['caps']=[1]*12+[2]
            for rules in routes.values():
                for rule in rules:
                    rule['interpreter']=str(self.root/'ARTIFICIAL_DP_INTERPRETER')
                    rule['view_commit']='ARTIFICIAL_FIXED_SOURCE'
                    rule['artificial'].update(tau='100',signal_token='SAME_ARTIFICIAL_SIGNAL')
            return routes
    class Driver(F.ArtificialDriver):
        def context(self,problem):
            return dict(candidates=[],expected={},tau=Fraction(100),saved=O.SavedBackend([],{}),generate=lambda *a:None)
        def completion_api(self):
            if self.completion is None:self.completion=Completion(self.run)
            return self.completion
    return F,Driver


def setup_run(root,enabled):
    F,Driver=fixture_classes();root=Path(root)
    args=F.WidthSearchTests().files(root)
    run=root/'run';run.mkdir(parents=True,exist_ok=True)
    interpreter(run/'ARTIFICIAL_DP_INTERPRETER')
    write(run/'ARTIFICIAL_PRICE_EVIDENCE.json',dict(ARTIFICIAL=True,accepted=True))
    spec=json.loads((root/'ROUND.json').read_text())
    # Two physical width families per method are prepared; a third is held.
    spec['search_overrides']['requested_widths']={'PR_PAULI':[.1,.2,'1/5'],'PR_DF':[.1,.05,.2]}
    spec['search_overrides']['lookahead']=0
    spec['executor']=dict(scheduler='concurrent' if enabled else 'legacy',
        max_concurrent_jobs=2 if enabled else 1,dp_workers=1)
    if enabled:spec['executor'].update(prefix_cache_directory='prefix-cache',prefix_checkpoint_rounds=[11])
    write(root/'ROUND.json',spec)
    return args,Driver


def call(args,Driver):
    import run_rpe_schedule_search as E
    with redirect_stdout(io.StringIO()):return E.main(args,driver_factory=Driver)


def outcomes(run):
    result={}
    for path in Path(run).glob('jobs/*/*/OUTCOME.json'):
        request=json.loads((path.parent/'REQUEST.json').read_text());value=json.loads(path.read_text())
        result[(request['method'],request['configuration_id'],tuple(request['N_X']),tuple(request['N_Y']))]=(
            value['status'],value['MSE_lower'],value['MSE_upper'])
    return result


def events(run):
    return [dict(json.loads(line),job=str(path.parent))for path in Path(run).glob('jobs/*/*/DP_PREFIX_EVENTS.jsonl')
            for line in path.read_text().splitlines()]


class IntegratedPrefix(unittest.TestCase):
    artifact_root=None
    def directory(self,name):
        if self.artifact_root is None:return tempfile.TemporaryDirectory(prefix='pr-width-prefix-'+name+'-')
        path=Path(self.artifact_root)/name
        path.mkdir(parents=True,exist_ok=False)
        return nullcontext(path)

    def test_normal_multiwidth_cli_real_dp_hit_disabled_parity_and_resume(self):
        with self.directory('cli')as tmp:
            tmp=Path(tmp);args,Driver=setup_run(tmp/'enabled',True);first=call(args,Driver)
            got=outcomes(first.run);hit=events(first.run)
            self.assertEqual(len(got),8)
            self.assertEqual(sum(e['event']=='HIT'for e in hit),4)
            self.assertTrue(all(e['reused_rounds']==12 for e in hit if e['event']=='HIT'))
            self.assertEqual({json.loads((Path(e['job'])/'REQUEST.json').read_text())['method']for e in hit
                if e['event']=='HIT'},{'PR_PAULI','PR_DF'})
            for e in hit:
                if e['event']=='HIT':
                    request=json.loads((Path(e['job'])/'REQUEST.json').read_text())
                    self.assertEqual(request['N_X'],[1]*13)
                    self.assertEqual(request['execution_profile']['prefix_checkpoint_rounds'],[11])
            queue=[json.loads(line)for line in(first.run/'PROGRESS.jsonl').read_text().splitlines()]
            launches={e['job']:e for e in queue if e['event']=='LAUNCH'}
            exits={e['job']:e for e in queue if e['event']=='EXIT'}
            order=[]
            for e in hit:
                if e['event']!='HIT':continue
                current=launches[Path(e['job']).name]
                donors=[v for v in launches.values()if v['configuration']==current['configuration']and v['job']!=current['job']]
                self.assertEqual(len(donors),1)
                donor=donors[0]
                self.assertLess(exits[donor['job']]['monotonic'],current['monotonic'])
                order.append(dict(configuration_id=current['configuration'],donor_job=donor['job'],
                    successor_job=current['job'],donor_exit_monotonic=exits[donor['job']]['monotonic'],
                    successor_launch_monotonic=current['monotonic'],HIT_reused_rounds=12))
            disabled_args,disabled_driver=setup_run(tmp/'disabled',False);off=call(disabled_args,disabled_driver)
            self.assertEqual(got,outcomes(off.run))
            self.assertFalse(events(off.run))
            reference=json.loads((tmp/'disabled'/'ROUND.json').read_text())['executor']
            self.assertEqual(reference['scheduler'],'legacy')
            self.assertEqual(reference['max_concurrent_jobs'],1)
            self.assertNotIn('prefix_cache_directory',reference)
            markers={str(p):p.read_bytes()for p in first.run.glob('jobs/*/*/LAUNCH.json')}
            service=(first.run/'completion'/'ARTIFICIAL_SERVICE_STATE.json').read_bytes()
            call(args,Driver)
            self.assertEqual(markers,{str(p):p.read_bytes()for p in first.run.glob('jobs/*/*/LAUNCH.json')})
            self.assertEqual(service,(first.run/'completion'/'ARTIFICIAL_SERVICE_STATE.json').read_bytes())
            self.assertEqual(got,outcomes(first.run))
            if self.artifact_root is not None:
                keys=sorted(got)
                self.cli_summary=dict(status='PASS',fixture='ARTIFICIAL_ONLY',
                    methods=sorted({k[0]for k in keys}),ready_width_configurations=4,declared_width_families=6,
                    first_invocation_artificial_Q3_launches=8,disabled_artificial_Q3_launches=8,
                    reference_scheduler='legacy',reference_max_concurrent_jobs=1,reference_prefix_enabled=False,
                    resume_new_launches=0,resume_new_completed_stage_calls=0,
                    cache_HITs=sum(e['event']=='HIT'for e in hit),reused_rounds_per_HIT=12,
                    all_MSE_lower_upper_and_status_equal_when_disabled=True,
                    donor_completed_before_successor_launch=order,
                    point_checks=[dict(method=k[0],configuration_id=k[1],N_X=list(k[2]),N_Y=list(k[3]),
                        status=got[k][0],enabled_and_disabled_MSE_pair_sha256=hashlib.sha256(
                            json.dumps(got[k][1:],sort_keys=True).encode()).hexdigest())for k in keys],
                    scientific_H4_calls=0)
                root=Path(self.artifact_root)
                write(root/'CLI_SUMMARY.json',self.cli_summary)
                (root/'DP_PREFIX_EVENTS.jsonl').write_text(''.join(json.dumps(e)+'\n'for e in hit))
                progress=first.run/'PROGRESS.jsonl'
                (root/'QUEUE_EVENTS.jsonl').write_bytes(progress.read_bytes())

    def test_queue_misses_different_configuration_or_signal(self):
        from trottertracks import rpe_optimizer_concurrent as C,rpe_method_optimizer as O
        import test_rpe_optimizer_concurrent as F
        with self.directory('identity')as tmp:
            root=Path(tmp);exe=interpreter(root/'DP_INTERPRETER')
            problem,context,routes=F.fixture(methods=['PR_PAULI'],configurations=2)
            routes['PR_PAULI'][0].update(interpreter=exe,view_commit='ARTIFICIAL_FIXED_SOURCE')
            routes['PR_PAULI'][0]['artificial'].update(tau='100',signal_token='SAME_ARTIFICIAL_SIGNAL')
            ids=[c['configuration_id']for c in context['candidates']]
            def run(folder,cid,vector,rules):
                ex=C.ConcurrentExecutor(ROOT,folder,rules,context['expected'],problem,max_concurrent=1,
                    budgets={'PR_PAULI':1},dp_workers=1,prefix_cache_directory=root/'shared-cache',prefix_checkpoint_rounds=[11])
                try:
                    answer=ex.for_method('PR_PAULI').answer(dict(configuration_id=cid,vector=tuple(vector),cost=Fraction(1),origin={}))
                    self.assertEqual(answer['calls'],1)
                    self.assertIn(answer['status'],(O.PASS,O.FAIL))
                finally:ex.finish()
                return events(folder)
            run(root/'donor',ids[0],[1]*24+[2,2],routes)
            changed=run(root/'different-config',ids[1],[1]*26,routes)
            self.assertFalse(any(e['event']=='HIT'for e in changed))
            other=deepcopy(routes);other['PR_PAULI'][0]['artificial']['signal_token']='DIFFERENT_ARTIFICIAL_SIGNAL'
            changed=run(root/'different-signal',ids[0],[1]*26,other)
            self.assertFalse(any(e['event']=='HIT'for e in changed))
            if self.artifact_root is not None:
                write(Path(self.artifact_root)/'IDENTITY_SUMMARY.json',dict(status='PASS',
                    different_configuration='MISS',different_signal='MISS',
                    artificial_Q3_calls=3,scientific_H4_calls=0))

    def test_selective_m8_profile_through_normal_executor_and_adapter(self):
        import run_rpe_method_optimizer as CLI
        import test_rpe_optimizer_concurrent as F
        from trottertracks import rpe_method_optimizer as O
        with self.directory('selective') as tmp:
            root=Path(tmp)
            problem,context,routes=F.fixture(methods=['PR_PAULI'],configurations=1)
            routes['PR_PAULI'][0].update(interpreter=interpreter(root/'DP_INTERPRETER'),
                                        view_commit='ARTIFICIAL_FIXED_SOURCE')
            routes['PR_PAULI'][0]['artificial'].update(tau='100',signal_token='SAME_ARTIFICIAL_SIGNAL')
            opts=dict(scheduler='concurrent',max_concurrent_jobs=1,dp_workers=1,
                prefix_cache_directory='cache',prefix_checkpoint_rounds=[5,8,11],
                prefix_selective_checkpoint_rounds=[8,11],prefix_selective_window_seconds=600,
                parallel_dp_residual=True,profile_dp_residual=True)
            runner,_=CLI.make_executor(ROOT,root,routes,context['expected'],problem,
                                      dict(executor=opts,max_new_q3_per_method={'PR_PAULI':3}))
            cid=context['candidates'][0]['configuration_id']
            try:
                for shots in (1,2,3):
                    proposal=dict(configuration_id=cid,vector=tuple([1]*18+[shots,shots]+[1]*6),
                                  cost=Fraction(1),origin={})
                    answer=runner.for_method('PR_PAULI').answer(proposal)
                    self.assertEqual(answer['status'],O.PASS)
                    self.assertEqual(answer['calls'],1)
            finally:runner.finish()
            logs=events(root)
            self.assertCountEqual([x['completed'] for x in logs if x['event']=='HIT'],[5,8])
            self.assertCountEqual([x['completed'] for x in logs if x['event']=='SAVE'],[5,8])
            for p in (root/'jobs').glob('*/*/REQUEST.json'):
                execution=json.loads(p.read_text())['execution_profile']
                self.assertEqual(execution['prefix_selective_checkpoint_rounds'],[8,11])
                self.assertTrue(execution['profile_dp_residual'])


def acceptance(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    checks=IntegratedPrefix();checks.artifact_root=output
    checks.test_normal_multiwidth_cli_real_dp_hit_disabled_parity_and_resume()
    checks.test_queue_misses_different_configuration_or_signal()
    value=dict(schema='PR_WIDTH_THROUGHPUT_ACTUAL_DP_ARTIFICIAL_ACCEPTANCE_V1',status='PASS',
        numerical_scope='ORIGINAL_GUARDED_DP_ARTIFICIAL_TABLES_ONLY',
        normal_multiwidth_CLI=True,common_concurrent_queue=True,
        cache_HITs=checks.cli_summary['cache_HITs'],reused_rounds_per_HIT=12,
        cache_disabled_MSE_and_status_exact_parity=True,
        reference_scheduler='legacy',reference_max_concurrent_jobs=1,reference_prefix_enabled=False,
        different_configuration_or_signal_HITs=0,resume_new_launches=0,
        artificial_DP_Q3_calls=19,new_H4_signal_calibration_compile_Q3_calls=0,
        kernel_view=str(VIEW),runtime=str(PYTHON),
        fixture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        records=['CLI_SUMMARY.json','IDENTITY_SUMMARY.json','QUEUE_EVENTS.jsonl','DP_PREFIX_EVENTS.jsonl'])
    write(output/'SUMMARY.json',value);return value


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--artificial-dp-worker':raise SystemExit(worker())
    unittest.main(verbosity=2)
