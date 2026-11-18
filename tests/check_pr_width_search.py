"""Artificial multi-width ordinary-entry acceptance; no H4 leaves/catalog writes.

The acquisition service and configuration metadata are fixtures.  main(), the
width planner, METHOD_MINIMUM_PRIORITY_V2, shot proposals, Executor's normal
REQUEST/OUTCOME adapter, validation and resume ingestion run unchanged.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import rpe_kappa_schedule as KS
from trottertracks import rpe_method_optimizer as O
from trottertracks import rpe_schedule_search as W
import run_rpe_schedule_search as E


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def configuration(method, delta, kappa='1', ld=1):
    """Artificial metadata only, including the real mapper constants check."""
    h=Fraction(.1);d=KS._frac(delta);k=KS._frac(kappa)
    rows=[dict(m=m,T=O.enc(h*2**m),q_PF=math.ceil(h*2**m/d),structure=dict(signed_weights=[1]))for m in range(13)]
    s=dict(h=O.enc(h),M=12,formula='2nd',finite_K=2,delta_pf=O.enc(d),
           kappa=dict(value=O.enc(k)),split=dict(L_D=ld,lambda_R=O.enc(Fraction(1,10000*ld))))
    body=dict(spec=s,rounds=rows)
    rs=KS.R_vector(KS.constants(body),k)
    for r,R in zip(rows,rs):
        r.update(R_outer=R,physical_candidate_id=hashlib.sha256(
            repr((method,ld,r['m'],r['q_PF'],R)).encode()).hexdigest()[:20])
    cid=hashlib.sha256(json.dumps([method,s,rows],sort_keys=True).encode()).hexdigest()
    return dict(body,configuration_id=cid)


class Reader:
    def read(self, loc):
        return configuration(loc['method'],.1)


class ArtificialCompletion:
    """Fixture substitutes only the unavailable science/price acquisition leaves."""
    def __init__(self, root):
        self.root=Path(root);self.path=self.root/'completion'/'ARTIFICIAL_SERVICE_STATE.json'
        self.state=json.loads(self.path.read_text())if self.path.exists()else dict(records={},stage_calls=0,price_ingestions=0)

    def save(self):write(self.path,self.state)

    def ingest_price_results(self):
        evidence=self.root/'ARTIFICIAL_PRICE_EVIDENCE.json'
        if evidence.exists() and not self.state.get('price_received'):
            self.state.update(price_received=True,price_ingestions=self.state['price_ingestions']+1);self.save()

    def complete(self, req):
        d=req['design'];method=req['method'];delta=KS._frac(d['delta_pf'])
        key=W.width_family_key(method,d['L_D'],d['formula'],d['delta_pf'])+'|k='+d['kappa']
        old=self.state['records'].get(key)
        if old and old['status']=='READY_FOR_SEARCH':return old
        cfg=configuration(method,d['delta_pf'],d['kappa'],d['L_D'])
        # This fixture models the documented Pauli exact-width refusal; it does
        # not turn the rational request into a different binary64 input.
        if method=='PR_PAULI' and delta==Fraction(1,5):
            record=dict(status='HELD_UNSUPPORTED',reason='EXACT_ACTUAL_WIDTH_NOT_BINARY64:m2..m12')
        elif delta!=Fraction(.1) and not self.state.get('price_received'):
            record=dict(status='WAITING_INPUTS',reason='MISSING_ACTUAL_STEP_CALIBRATION',
                        calibration_requests=[dict(method=method,actual_delta_pf=O.enc(delta))])
        else:
            record=dict(status='READY_FOR_SEARCH',configuration_id=cfg['configuration_id'],configuration=cfg,
                        price_scale=2 if delta==Fraction(.1)else 1)
            self.state['stage_calls']+=1
        record.update(key=key,request=req);self.state['records'][key]=record;self.save();return record

    def plan(self, req):
        return dict(status='WAITING_INPUTS',reason='ARTIFICIAL_METADATA_ONLY',readiness=dict(configuration_complete=True),
                    rounds=[dict(m=m,status='NOT_ACQUIRED')for m in range(13)],calibration_requests=[])

    def prepare_family(self, req, allow_science=False):
        self.state['family_preparations']=self.state.get('family_preparations',0)+1;self.save()
        d=req['design']
        if (self.root/'ARTIFICIAL_MISSING_DESCRIPTOR').exists():
            return dict(status='NEEDS_FAMILY_PREPARATION',configuration=None,reason='TARGET_SPLIT_DESCRIPTOR_MISSING',
                        needs=['target_split_descriptor'])
        return dict(status='CONFIGURATION_READY',configuration=configuration(req['method'],d['delta_pf'],d['kappa'],d['L_D']),
                    metadata_only=True)

    def install(self, context, problem, records):
        routes=deepcopy(problem['execution_routes'])
        for r in records:
            if r['status']!='READY_FOR_SEARCH':continue
            req=r['request'];method=req['method'];d=req['design'];cid=r['configuration_id']
            if KS._frac(d['delta_pf'])not in {KS._frac(x)for x in problem['methods'][method]['requested_delta_pf_values']}:continue
            physical=(method,tuple(v['physical_candidate_id']for v in r['configuration']['rounds']))
            same=next((c for c in context['candidates']if c.get('artificial_physical')==physical),None)
            if same is not None:
                r['installed_configuration_id']=same['configuration_id'];continue
            r['installed_configuration_id']=cid
            key=dict(L_D=d['L_D'],formula=d['formula'],delta_over_h=str(KS._frac(d['delta_pf'])/Fraction(.1)),
                     kappa_multiplier=d['kappa'],requested_delta_pf=O.enc(KS._frac(d['delta_pf'])))
            context['candidates'].append(dict(method=method,family='PR',key=key,configuration_id=cid,ready=True,
                prices=[Fraction(r['price_scale'])]*26,caps=[1]*13,name=method+':'+cid,needs=[],labels=[],saved_points=[],artificial_physical=physical,
                price_status='ARTIFICIAL_ONLY',signal_source='ARTIFICIAL_ONLY'))
            context['expected'][cid]=dict(total_configuration_id='ARTIFICIAL_TOTAL_'+cid,binding_id='ARTIFICIAL_BINDING')
            routes[method].append(dict(kind='artificial',configuration_ids=[cid],interpreter=sys.executable,
                view_commit=None,artificial=dict(weights=[1]*13,scale='1/1000',tau='1',
                    force_unknown=(self.root/'ARTIFICIAL_GATE_FAILURE').exists())))
        return routes


class ArtificialDriver(E.WidthDriver):
    def catalog(self):return Reader(),dict(candidates=[])

    def generate_families(self, method, items):
        return [dict(item_id=item['item_id'], **self.generate(method, item['base'], item['changes']))
                for item in items]

    def generate(self, method, base, changes):
        delta=Fraction(.1)*KS._frac(changes['delta_over_h'])
        # The real DF mapper, unlike the Pauli mapper, rejects this grid.
        if method=='PR_DF'and delta==Fraction(.2):
            return dict(status='UNSUPPORTED',reason='Nonintegral PF grid; no implicit delta change')
        return dict(status='GENERATED',configuration=configuration(method,delta))

    def context(self, problem):
        return dict(candidates=[],expected={},tau=Fraction(1),saved=O.SavedBackend([],{}),generate=lambda *args:None)

    def completion_api(self):
        if self.completion is None:self.completion=ArtificialCompletion(self.run)
        return self.completion


class WidthSearchTests(unittest.TestCase):
    def test_exact_input_and_legacy_dispatch(self):
        spec=dict(methods=['PR_PAULI','PR_DF'],search_overrides=dict(requested_widths={'PR_PAULI':[.2,'1/5',.2]}))
        got=W.requested_widths({},spec)['PR_PAULI']
        self.assertEqual([v['exact']for v in got],[str(Fraction(.2)),'1/5'])
        self.assertNotEqual(W.width_family_key('PR_PAULI',1,'2nd',.2),W.width_family_key('PR_PAULI',1,'2nd','1/5'))
        self.assertIsNone(W.requested_widths({},dict(methods=['PR_PAULI'])))
        with self.assertRaises(ValueError):W.requested_widths({},dict(methods=['PR_DF'],search_overrides=spec['search_overrides']))
        # Adding the new declaration must not change the saved round05 file.
        import subprocess
        old=subprocess.check_output(['git','show','d8e806265:configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json'],cwd=ROOT)
        self.assertEqual(old,(ROOT/'configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json').read_bytes())

    def files(self, root, approved=True):
        problem=dict(accuracy=dict(e=1),target=dict(energy_ha=0),rpe=dict(M=12),execution_routes={m:[]for m in ('PR_PAULI','PR_DF')},
            methods={m:dict(L_D_values=[1],generation=dict(formulas=['2nd'],baseline=dict(formula='2nd',delta_over_h='1',kappa_multiplier='1')))
                     for m in ('PR_PAULI','PR_DF')},
            search=dict(rule='METHOD_MINIMUM_PRIORITY_V2',decrement_steps=[1],large_step_headroom='9/10',seed_rules=[],
                        max_steps_per_method=100,max_new_q3_per_method=100,on_missing='STOP'))
        spec=dict(round_id='ARTIFICIAL_MULTI_WIDTH',methods=['PR_PAULI','PR_DF'],science_approved=approved,initial='all',
            width_families=[dict(method=m,L_D=1,configuration=dict(method=m,path='ARTIFICIAL',commit='ARTIFICIAL'))for m in ('PR_PAULI','PR_DF')],
            search_overrides=dict(requested_widths={'PR_PAULI':[.1,.2,'1/5'],'PR_DF':[.1,.05,.2]},
                schedule_wide=dict(max_kappa='1',per_decision=8)))
        write(root/'PROBLEM.json',problem);write(root/'ROUND.json',spec)
        return ['--problem',str(root/'PROBLEM.json'),'--round',str(root/'ROUND.json'),'--run-dir',str(root/'run'),
                '--max-cycles','3','--batch','16']

    def call(self,args):
        with redirect_stdout(io.StringIO()):return E.main(args,driver_factory=ArtificialDriver)

    def test_actual_entry_both_methods_registration_shots_response_resume(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-artificial-')as t:
            root=Path(t);args=self.files(root);first=self.call(args)
            out=json.loads((first.run/'ROUND_RESULT.json').read_text())['methods']
            progress=[json.loads(line)for line in(first.run/'DRIVER_PROGRESS.jsonl').read_text().splitlines()]
            for phase in('SEED','SEED_FAMILY','PLAN','INPUTS','CANDIDATE_INPUTS','V2_BATCH'):
                self.assertTrue(any(v['event']==phase+'_START'for v in progress))
                ends=[v for v in progress if v['event']==phase+'_END']
                self.assertTrue(ends and all(v['wall_seconds']>=0 for v in ends))
            self.assertTrue(all(out[m]['best']for m in ('PR_PAULI','PR_DF')))
            self.assertTrue(any(p['status']=='WAITING_INPUTS'for p in first.state['points']))
            self.assertTrue(any(v['status']=='UNSUPPORTED_MAPPER'and 'Nonintegral'in v['reason']for v in first.state['families'].values()))
            original={m:O.frac(out[m]['best']['cost'])for m in out}
            jobs=list((first.run/'jobs').glob('*/*/OUTCOME.json'));self.assertEqual(len(jobs),2)
            for job in jobs:
                result=json.loads(job.read_text());self.assertTrue(result['artificial']);self.assertEqual(result['status'],O.PASS)
                self.assertEqual(json.loads((job.parent/'PROCESS_RETURN.json').read_text())['returncode'],0)
            write(first.run/'ARTIFICIAL_PRICE_EVIDENCE.json',dict(ARTIFICIAL=True,accepted=True))
            second=self.call(args);out=json.loads((second.run/'ROUND_RESULT.json').read_text())['methods']
            for method in ('PR_PAULI','PR_DF'):
                self.assertLess(O.frac(out[method]['best']['cost']),original[method])
                wanted=Fraction(.2)if method=='PR_PAULI'else Fraction(.05)
                self.assertEqual(O.frac(out[method]['best']['configuration']['requested_delta_pf']),wanted)
            jobs=list((second.run/'jobs').glob('*/*/OUTCOME.json'));self.assertEqual(len(jobs),4)
            counts=json.loads((second.run/'completion'/'ARTIFICIAL_SERVICE_STATE.json').read_text())
            third=self.call(args);counts2=json.loads((third.run/'completion'/'ARTIFICIAL_SERVICE_STATE.json').read_text())
            self.assertEqual(counts2['stage_calls'],counts['stage_calls'])
            self.assertEqual(len(list((third.run/'jobs').glob('*/*/LAUNCH.json'))),4)
            self.assertEqual(counts2['price_ingestions'],1)
            self.assertTrue(all(p['best_N_X']==[1]*13 for p in third.state['points']if p['status']=='EVALUATED'))


    def test_identical_physics_alias_retains_both_declared_widths(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-alias-')as t:
            root=Path(t);args=self.files(root)
            spec=json.loads((root/'ROUND.json').read_text())
            spec['search_overrides']['requested_widths']={'PR_PAULI':[500,600],'PR_DF':[500,600]}
            write(root/'ROUND.json',spec);write(root/'run'/'ARTIFICIAL_PRICE_EVIDENCE.json',dict(ARTIFICIAL=True))
            d=self.call(args)
            self.assertEqual(len(list((d.run/'jobs').glob('*/*/OUTCOME.json'))),2)
            for method in ('PR_PAULI','PR_DF'):
                points=[p for p in d.state['points']if p['method']==method]
                self.assertEqual(len(points),2)
                self.assertTrue(all(p['status']=='EVALUATED'and p.get('best')is not None for p in points))
                self.assertEqual(len({p['configuration_id']for p in points}),2)
                self.assertEqual(len({p['evaluation_configuration_id']for p in points}),1)
            self.call(args)
            self.assertEqual(len(list((d.run/'jobs').glob('*/*/LAUNCH.json'))),2)

    def test_missing_split_bootstrap_uses_its_own_lambda_then_normal_planner(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-bootstrap-')as t:
            root=Path(t);args=self.files(root)
            spec=json.loads((root/'ROUND.json').read_text());problem=json.loads((root/'PROBLEM.json').read_text())
            spec['search_overrides']['requested_widths']={m:[.1]for m in ('PR_PAULI','PR_DF')}
            for setting in problem['methods'].values():setting['L_D_values']=[1,2]
            write(root/'ROUND.json',spec);write(root/'PROBLEM.json',problem)
            d=self.call(args)
            for method in ('PR_PAULI','PR_DF'):
                f=[f for f in d.state['families'].values()if f['method']==method and f['L_D']==2][0]
                self.assertEqual(f['status'],'READY_METADATA')
                self.assertNotIn('reason',f)
                self.assertTrue(f['metadata_kappa_not_a_scientific_candidate'])
                self.assertEqual(KS._frac(f['configuration']['spec']['split']['lambda_R']),Fraction(1,20000))
                p=[p for p in d.state['points']if p['method']==method and p['L_D']==2][0]
                self.assertEqual(p['completion_request']['selection']['rule'],W.RULE)
                self.assertEqual(p['status'],'EVALUATED')
            self.assertEqual(len(list((d.run/'jobs').glob('*/*/OUTCOME.json'))),4)
            self.call(args)
            service=json.loads((d.run/'completion'/'ARTIFICIAL_SERVICE_STATE.json').read_text())
            self.assertEqual(service['family_preparations'],2)
            self.assertEqual(service['stage_calls'],4)

    def test_missing_descriptor_does_not_block_other_split_or_start_science(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-missing-')as t:
            root=Path(t);args=self.files(root)
            spec=json.loads((root/'ROUND.json').read_text());problem=json.loads((root/'PROBLEM.json').read_text())
            spec['search_overrides']['requested_widths']={m:[.1]for m in ('PR_PAULI','PR_DF')}
            for setting in problem['methods'].values():setting['L_D_values']=[1,2]
            write(root/'ROUND.json',spec);write(root/'PROBLEM.json',problem)
            hold=root/'run'/'ARTIFICIAL_MISSING_DESCRIPTOR';hold.parent.mkdir(parents=True);hold.touch()
            d=self.call(args)
            self.assertEqual(len(list((d.run/'jobs').glob('*/*/OUTCOME.json'))),2)
            pending=[f for f in d.state['families'].values()if f['L_D']==2]
            self.assertTrue(all(f['reason']=='TARGET_SPLIT_DESCRIPTOR_MISSING'for f in pending))
            self.assertTrue(all(f['preparation_request']['design']['L_D']==2 for f in pending))
            self.assertEqual(d.state['stop']['reason'],'WAITING_FOR_INPUTS')

    def test_v2_unknown_stop_is_sealed_on_same_run_resume(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-failure-')as t:
            root=Path(t);args=self.files(root)
            marker=root/'run'/'ARTIFICIAL_GATE_FAILURE';marker.parent.mkdir(parents=True);marker.touch()
            d=self.call(args)
            self.assertEqual(d.state['stop']['reason'],'V2_RESULT_REQUIRES_REVIEW')
            jobs=list((d.run/'jobs').glob('*/*/LAUNCH.json'));calls=d.completion_api().state['stage_calls']
            self.assertEqual(len(jobs),2)
            resumed=self.call(args)
            self.assertEqual(len(list((d.run/'jobs').glob('*/*/LAUNCH.json'))),len(jobs))
            self.assertIsNone(resumed.completion)
            self.assertEqual(json.loads((d.run/'completion'/'ARTIFICIAL_SERVICE_STATE.json').read_text())['stage_calls'],calls)

    def test_resume_rejects_changed_scientific_scope_and_accepts_new_price_evidence(self):
        for change in('M','target','loss','formula'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(prefix='pr-width-scope-')as t:
                root=Path(t);args=self.files(root,approved=False);d=self.call(args+['--plan-only'])
                self.assertTrue((d.run/'FIXED_INPUTS.json').exists())
                problem=json.loads((root/'PROBLEM.json').read_text());spec=json.loads((root/'ROUND.json').read_text())
                if change=='M':problem['rpe']['M']=11
                elif change=='target':problem['target']['energy_ha']=1
                elif change=='loss':problem['accuracy']['loss']='ANOTHER_LOSS'
                else:spec['method_overrides']={'PR_PAULI':dict(generation=dict(formulas=['4th(m5_best)']))}
                write(root/'PROBLEM.json',problem);write(root/'ROUND.json',spec)
                with self.assertRaisesRegex(ValueError,'FIXED_INPUTS_CHANGED'):self.call(args+['--plan-only'])
        with tempfile.TemporaryDirectory(prefix='pr-width-price-input-')as t:
            root=Path(t);args=self.files(root,approved=False);d=self.call(args+['--plan-only'])
            spec=json.loads((root/'ROUND.json').read_text());spec['science_approved']=True
            spec['candidate_completion']=dict(price_evidence_results=['ARTIFICIAL_NEW_FIXED_EVIDENCE'])
            write(root/'ROUND.json',spec)
            resumed=self.call(args)
            self.assertTrue((resumed.run/'jobs').exists())

    def test_unsealed_metadata_migration_checks_family_time_grid(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-scope-adopt-')as t:
            root=Path(t);args=self.files(root,approved=False);d=self.call(args+['--plan-only'])
            (d.run/'FIXED_INPUTS.json').unlink()
            self.call(args+['--plan-only'])
            self.assertTrue((d.run/'FIXED_INPUTS.json').exists())
            (d.run/'FIXED_INPUTS.json').unlink()
            state=json.loads((d.run/'KAPPA_STATE.json').read_text())
            next(iter(state['families'].values()))['configuration']['spec']['M']=11
            write(d.run/'KAPPA_STATE.json',state)
            with self.assertRaisesRegex(ValueError,'METADATA_SCOPE_M_CHANGED'):self.call(args+['--plan-only'])

    def test_ready_candidate_rejects_immutable_source_drift_even_with_same_lambda_and_hash(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-source-')as t:
            root=Path(t);args=self.files(root,approved=False);d=self.call(args+['--plan-only'])
            point=next(p for p in d.state['points']if p['method']=='PR_PAULI')
            family=d.state['families'][point['family']]
            baseline=deepcopy(family['configuration'])
            baseline['spec'].update(binding=dict(hamiltonian='H',state='S',snapshot='N',source='FIXED_BINDING_SOURCE'),
                law=dict(model='ideal_finite_RTE_law',source=dict(commit='FIXED_LAW',path='law.py')))
            baseline['spec']['split'].update(lambda_source=dict(commit='FIXED_LAMBDA',path='lambda.json'),
                tail_dictionary=dict(dictionary_sha256='SAME_DICTIONARY_HASH',source=dict(commit='FIXED_DICT',path='dictionary.json')))
            family['configuration']=deepcopy(baseline)
            record=dict(configuration=baseline)
            d.check_completed_configuration(point,record)
            for section,field in(('law','source'),('binding','source'),('tail_dictionary','source'),('split','lambda_source')):
                with self.subTest(section=section):
                    changed=deepcopy(baseline)
                    if section=='tail_dictionary':changed['spec']['split'][section][field]={'commit':'DRIFTED','path':'dictionary.json'}
                    else:changed['spec'][section][field]={'commit':'DRIFTED','path':'source.json'}
                    d.completion_api().state['records']={}
                    d.completion_api().complete=lambda request,changed=changed:dict(status='READY_FOR_SEARCH',key='DRIFT',
                        configuration_id=changed['configuration_id'],configuration=changed)
                    point.update(status='COMPLETION')
                    d.state['cycle']+=1  # Each synthetic drift gets its own bounded preparation wave.
                    d.a.plan_only=False;d.spec['science_approved']=True;d.complete_points()
                    self.assertEqual(point['status'],'HELD')
                    self.assertIn('COMPLETED_CONFIGURATION_MISMATCH:DIFFERENT_',point['reason'])
                    self.assertEqual(KS._frac(changed['spec']['split']['lambda_R']),KS._frac(baseline['spec']['split']['lambda_R']))
                    self.assertEqual(changed['spec']['split']['tail_dictionary']['dictionary_sha256'],'SAME_DICTIONARY_HASH')

    def test_plan_only_never_runs_acquisition_or_executor(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-plan-')as t:
            root=Path(t);args=self.files(root,approved=False);drv=self.call(args+['--plan-only'])
            self.assertFalse((drv.run/'jobs').exists());self.assertFalse((drv.run/'completion').exists())
            self.assertEqual(drv.state['stop']['reason'],'REQUESTED_WIDTH_METADATA_ONLY')
            self.assertTrue(all(p['status']=='WAITING_INPUTS'for p in drv.state['points']))


if __name__=='__main__':unittest.main()
