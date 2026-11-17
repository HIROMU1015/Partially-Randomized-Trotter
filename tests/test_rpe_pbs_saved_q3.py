"""Synthetic-only one-request caller checks; no adapter, SSH, PBS or H4 call."""
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_rpe_schedule_search as Entry
spec=importlib.util.spec_from_file_location('one_saved_q3_caller',ROOT/'scripts/check_rpe_pbs_saved_q3.py')
Caller=importlib.util.module_from_spec(spec);spec.loader.exec_module(Caller)


class FakeStore:
    """Only metadata changes: no scientific function or process exists here."""
    def __init__(self):self.record=None;self.reservations=[];self.history=[]
    def rows(self):return []if self.record is None else[deepcopy(self.record)]
    def row(self,job):return deepcopy(self.record)
    def eligible(self,request,decision):return decision['request_sha256']==Caller.digest(request)
    def reserve(self,method,job,request,budget,approved,**kwargs):
        self.reservations.append(dict(method=method,job=job,request=deepcopy(request),budget=budget,approved=approved))
        self.record=dict(job_id=job,request=deepcopy(request),request_sha256=Caller.digest(request),state='QUEUED')
        return True
    def counts(self,method,started_only=False):return int(self.record is not None and self.record['state']=='ACCEPTED')
    def events(self):return deepcopy(self.history)


class CallerTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.fixed=self.root/'fixed';self.fixed.mkdir()
        self.output=self.root/'central/runs/one/execution'
        self.request=dict(request_id='SAVED_JOB',method='FULL_PF',configuration_id='SAVED_CONFIGURATION',
            N_X=[2,1],N_Y=[2,1],cost=dict(numerator=7,denominator=1),origin=dict(kind='ORIGINAL'),
            expected=dict(configuration_id='SAVED_CONFIGURATION',binding_id='ORIGINAL_BINDING'),
            route=dict(kind='pf_b_pf6',view_commit='ORIGINAL_ROUTE',configuration_ids=['SAVED_CONFIGURATION']),
            optimizer_source='ORIGINAL_SOURCE')
        decision=dict(request_sha256=Caller.digest(self.request),eligible_host_ids=['pbs-0'],
            source_manifest=dict(commit=Caller.COMPUTATION_SOURCE),hosts={'pbs-0':dict(status='PASS')})
        decision['decision_sha256']=Caller.digest(decision)
        self.decision=decision
        self.problem=dict(accuracy=dict(fixture='ARTIFICIAL_UNIT_ONLY'),target=dict(energy_ha=0),rpe=dict(M=1))
        self.round=dict(methods=['FULL_PF'],science_approved=True,max_new_q3_per_method={'FULL_PF':1},
            price_validation=dict(store=str(self.output/'private-models.sqlite3'),compile_approved=False),
            executor=dict(scheduler='concurrent',max_concurrent_jobs=1,dp_workers=2,prior_run_dirs=[],
                distributed=dict(computation_source=Caller.COMPUTATION_SOURCE,computation_root='/FROZEN',
                    preparation=dict(budgets=dict(signal=0,compile=0,audit=0)),
                    host_eligibility=dict(schema='RPE_Q3_HOST_ELIGIBILITY_V1',jobs={'SAVED_JOB':decision}),
                    pbs=dict(transport_source=Caller.CONTROLLER_SOURCE,workers=[dict(host_id='pbs-0')],
                        resources=dict(ncpus=2,mem='2gb',walltime='00:03:00'),task_seconds=dict(q3=30)))))
        self.service=dict(code_root=str(self.fixed),code_source=Caller.CONTROLLER_SOURCE,
            computation_source=Caller.COMPUTATION_SOURCE,state_directory=str(self.root/'central'),
            problem=str(self.root/'PROBLEM.json'),round=str(self.root/'ROUND.json'),python=sys.executable)
        self.service_path=self.root/'SERVICE.json';self.request_path=self.root/'SAVED_REQUEST.json'
        self.store=FakeStore();self.executed=0;self.fail_unknown=False
        test=self
        class View:
            contract='ORIGINAL_EVALUATION_CONTRACT'
            def job_dir(self,proposal):return test.output/'jobs/FULL_PF/SAVED_JOB'
            def _eligibility_for(self,request,job):return test.decision
            def log(self,**values):pass
            def answer(self,proposal):
                test.executed+=1
                row=test.store.record
                row.update(state='UNKNOWN'if test.fail_unknown else'ACCEPTED',host_id='pbs-0',
                           attempt_id='ARTIFICIAL_ATTEMPT',adopted_epoch=None if test.fail_unknown else 1)
                if test.fail_unknown:return dict(status='UNKNOWN')
                test.store.history=[dict(event=name)for name in('ASSIGN','ACCEPT','ADOPT')]
                Caller.put(self.job_dir(proposal)/'OUTCOME.json',dict(scientific_evaluation_calls=1,
                    artificial_unit_only=True,status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'))
                return dict(status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',calls=1,artificial_unit_only=True)
        view=View()
        executor=SimpleNamespace(store=self.store,for_method=lambda method:view,finish=Mock(),metrics=lambda:dict(
            artificial_unit_only=True,actual_scientific_calls=0))
        def factory(*args,**kwargs):
            path=self.output/'distributed/QUEUE.sqlite3';path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            return executor,None
        self.factory=Mock(side_effect=factory);self.executor=executor
        self.core=SimpleNamespace(root=self.fixed,entry=Entry,cli=SimpleNamespace(make_executor=self.factory),
            optimizer=SimpleNamespace(interleave=lambda x,y:tuple(v for pair in zip(x,y)for v in pair),
                frac=lambda q:Fraction(q['numerator'],q['denominator']),jsonable=lambda value:value),
            coordinator=SimpleNamespace(prefix_key=lambda request:'ORIGINAL_PREFIX'),
            JobStore=lambda path:self.store,stamp=lambda:dict(clock_id='ARTIFICIAL_UNIT',monotonic=1))

    def files(self):
        for path,value in((self.root/'PROBLEM.json',self.problem),(self.root/'ROUND.json',self.round),
                          (self.service_path,self.service),(self.request_path,self.request)):
            Caller.put(path,value)

    def execute(self):
        self.files()
        with patch.object(Caller,'load_fixed',return_value=self.core),redirect_stdout(io.StringIO()):
            return Caller.execute(self.service_path,self.request_path,self.output,self.fixed)

    def test_exact_request_normal_entry_reservation_and_completed_resume_zero(self):
        result=self.execute();raw=self.request_path.read_bytes()
        self.assertEqual(self.executed,1)
        self.assertEqual(self.store.reservations,[dict(method='FULL_PF',job='SAVED_JOB',
            request=self.request,budget=1,approved=True)])
        self.assertEqual((self.output/'ORIGINAL_REQUEST.json').read_bytes(),raw)
        metadata=Caller.read(self.output/'PORTABILITY_INPUTS.json')
        self.assertEqual(metadata['request_canonical_sha256'],Caller.digest(self.request))
        self.assertFalse(result['gateway_cli_start_used'])
        self.assertEqual(result['adoption_count'],1)
        again=self.execute()
        self.assertEqual(again,result);self.assertEqual(self.executed,1)
        self.factory.assert_called_once();self.executor.finish.assert_called_once_with()

    def test_old_unbound_outcome_is_not_read_as_a_new_scientific_execution(self):
        old=self.output/'jobs/FULL_PF/SAVED_JOB/OUTCOME.json';old.parent.mkdir(parents=True)
        old.write_bytes(b'not valid JSON; must never be opened as a result')
        with self.assertRaisesRegex(ValueError,'fresh reevaluation output'):
            self.execute()
        self.factory.assert_not_called();self.assertEqual(self.executed,0)

    def test_unknown_attempt_resume_does_not_reserve_or_execute_again(self):
        self.fail_unknown=True
        with self.assertRaisesRegex(ValueError,'accepted/adopted'):
            self.execute()
        with self.assertRaisesRegex(ValueError,'STOP/UNKNOWN'):
            self.execute()
        self.assertEqual(self.executed,1);self.assertEqual(len(self.store.reservations),1)
        self.factory.assert_called_once()

    def test_expanded_budgets_preparation_hosts_and_source_are_rejected(self):
        variants=[]
        changed=deepcopy(self.round);changed['max_new_q3_per_method']['FULL_PF']=2;variants.append(changed)
        changed=deepcopy(self.round);changed['executor']['distributed']['preparation']['budgets']['signal']=1;variants.append(changed)
        changed=deepcopy(self.round);changed['executor']['distributed']['local_worker']=dict(slots=1);variants.append(changed)
        changed=deepcopy(self.round);changed['executor']['distributed']['pbs']['transport_source']='OTHER_SOURCE';variants.append(changed)
        changed=deepcopy(self.round);changed['executor']['prior_run_dirs']=['/OLD/OUTCOME'];variants.append(changed)
        changed=deepcopy(self.round);changed['price_validation']['compile_approved']=True;variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                Caller.validate(self.service,self.request,self.problem,changed,self.output,self.fixed)
        self.assertEqual(self.executed,0)


if __name__=='__main__':unittest.main()
