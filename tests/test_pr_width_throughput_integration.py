"""Ordinary width CLI -> real Completion/registry -> shared artificial Q3 queue.

Scientific providers are fixture leaves; Completion.complete/price refresh,
StepPriceRegistry audit/direct receipt validation, install, V2 and scheduler are
production code. No H4 input or scientific worker is used.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import run_rpe_schedule_search as E
import test_pr_step_price_registry as P
from trottertracks import pr_candidate_completion as C
from trottertracks import rpe_kappa_schedule as KS
from trottertracks import rpe_method_optimizer as O

ORIGINAL_CONFIG=P.config
METHODS=('PR_PAULI','PR_DF')


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(O.jsonable(value),indent=2)+'\n')


def configuration(method,width,kappa='1',ld=None):
    cfg=ORIGINAL_CONFIG(width,method=='PR_PAULI');cfg['spec']['kappa']={'value':P.exact(Fraction(kappa))}
    if ld is not None:assert cfg['spec']['L_D']==ld
    rs=KS.R_vector(KS.constants(cfg),Fraction(kappa))
    for r,R in zip(cfg['rounds'],rs):
        r['R_outer']=R;r['structure'].update(allocation=[R],event_count=r['q_PF']*R)
        name=hashlib.sha256(repr((method,r['m'],r['q_PF'],R,r['actual_delta_pf'])).encode()).hexdigest()
        r['physical_candidate']['R_outer']=R;r['physical_candidate']['id']='ARTIFICIAL:'+name
        r['candidate'].update(R_outer=R,event_count=r['q_PF']*R,id='ARTIFICIAL_RMSE:'+name)
    cfg['id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    return cfg


class World:
    def __init__(self,root):
        self.root=Path(root);self.reader=P.Documents();self.packages={};self.signal_calls=[];self.quote_calls=[]
        self.scientific_price_leaves=[];self.installs=[];self.executors=[];self.publications=[]

    def publish(self,paths,message):
        self.publications.append(message)
        for path in paths:
            p=Path(path)
            for f in ([p]if p.is_file()else p.rglob('*.json')):
                if f.suffix=='.json':self.reader.add(C.rel(f),C.read(f))
        return 'ARTIFICIAL'

    def package(self,cfg,method):
        if cfg['id']in self.packages:return self.packages[cfg['id']]
        with patch.object(P,'config',return_value=deepcopy(cfg)):
            reader,_,_,package=P.pauli_fixture()if method=='PR_PAULI'else P.df_direct_fixture()
        prefix='artificial_evidence/'+cfg['id']+'/'
        def scoped(x):
            if isinstance(x,list):return [scoped(y)for y in x]
            if not isinstance(x,dict):return x
            y={k:scoped(v)for k,v in x.items()}
            if x.get('commit')=='ARTIFICIAL'and x.get('path')and ('ARTIFICIAL',x['path'])in reader.docs:y['path']=prefix+x['path']
            # These are newly namespaced artificial receipts, so recompute only
            # their metadata request/context digests after changing locators.
            if 'request_id'in y and 'physical_context'in y:
                y['context_key']=P.digest(y['physical_context']);copy=dict(y);copy.pop('request_id')
                y['request_id']=P.digest(copy)
            if 'resolved_context_key'in y and 'request'in y:
                y['resolved_context_key']=P.digest(dict(context=y['request']['physical_context'],resolved=y['resolved_context']))
            return y
        for (commit,path),value in reader.docs.items():self.reader.docs[commit,prefix+path]=scoped(value)
        self.packages[cfg['id']]=scoped(package);return self.packages[cfg['id']]

    def quote(self,cfg):
        method='PR_PAULI'if cfg['spec']['split'].get('tail_dictionary')else'PR_DF'
        q=P.missing_quote(cfg,method);q['implementation']=P.implementation(method=='PR_PAULI')
        base=Fraction(cfg['spec']['delta_pf']['numerator'],cfg['spec']['delta_pf']['denominator'])==Fraction(.1)
        for r in q['rows']:
            if base or (r['m']==0 if method=='PR_PAULI'else r['m']>0):
                r.update(unit_price_RZ=P.exact(200000 if base else 7),classification='MODEL_REFERENCE')
        q['complete']=all(r['unit_price_RZ']is not None for r in q['rows'])
        self.quote_calls.append((cfg['id'],q['complete']))
        if not base:self.package(cfg,method)
        return q

    def signal(self,cfg):
        self.signal_calls.append(cfg['id'])
        return dict(packet_source=self.reader.add('signals/'+cfg['id']+'.json',{'ARTIFICIAL':True,'configuration':cfg}),
                    new_signal_calls=0)


WORLD=None


class Completion(C.Completion):
    def __init__(self,directory,settings):
        settings=dict(settings,source='ARTIFICIAL')
        path=WORLD.root/'BASE_REGISTRY.json';put(path,dict(entries=[]));settings['price_registry_file']=str(path)
        super().__init__(directory,settings)

    def service(self,mode,request,d,**kwargs):
        if mode=='df-generate':
            x=request['design'];return dict(status='CONFIGURATION_READY',configuration=configuration('PR_DF',x['delta_pf'],x['kappa'],x['L_D']))
        if mode=='df-price-design':return dict(groups=[dict(status='CALIBRATION_DESIGN_READY')])
        if mode=='intake':
            cfg=WORLD.reader.read(request['configuration']);return dict(binding_id='ARTIFICIAL_BINDING',total_configuration_id='ARTIFICIAL_TOTAL:'+cfg['id'])
        if mode=='quote':
            cfg=WORLD.reader.read(request['configuration'])if 'commit'in request['configuration']else request['configuration']
            return WORLD.quote(cfg)
        raise AssertionError('No unmocked producer service '+mode)

    def pauli_configuration(self,request,d,allow_science=True):
        x=request['design'];return configuration('PR_PAULI',x['delta_pf'],x['kappa'],x['L_D']),{}

    def pauli_signals(self,cfg,prepared,d):return WORLD.signal(cfg)
    def df_signals(self,cfg,*args,**kwargs):return WORLD.signal(cfg)

    def call(self,cmd,d,stage,output=None,science=False):
        if stage!='PAULI_PRICES':raise AssertionError('No scientific child '+stage)
        WORLD.scientific_price_leaves.append(stage)
        return dict(status='WAITING_PRICE_EVIDENCE')

    def df_price(self,*args,**kwargs):
        WORLD.scientific_price_leaves.append('DF_PRICES')
        raise C.Held('PRICE_EVIDENCE_REQUIRED:ARTIFICIAL_EXTERNAL_AUDIT_NOT_YET_PUBLISHED')

    def ingest_price_results(self,packages=None):
        if packages is not None:return super().ingest_price_results(packages)
        # An externally published audit/direct package arrives only after the
        # first READY jobs finish. Actual refresh/install then runs unchanged.
        run=self.directory.parent
        if list((run/'jobs').glob('*/*/OUTCOME.json')):
            return super().ingest_price_results(list(WORLD.packages.values()))
        return []

    def install(self,context,problem,records):
        routes=super().install(context,problem,records)
        for method,rules in routes.items():
            for i,rule in enumerate(rules):
                if rule['kind']in ('pauli_b_common_entry','prdf_d2_common_entry'):
                    # Keep canonical requested/evaluation IDs from actual install;
                    # only the scientific numeric worker is artificial.
                    rules[i]=dict(kind='artificial',configuration_ids=rule['configuration_ids'],interpreter=sys.executable,
                        view_commit=None,artificial=dict(weights=[1]*13,scale='1/1000',tau='1',sleep_seconds=.01))
        WORLD.installs.append(dict(ready=[r['configuration_id']for r in records if r['status']=='READY_FOR_SEARCH'],
            candidates=[c['configuration_id']for c in context['candidates']if c.get('ready')]))
        return routes


class Driver(E.WidthDriver):
    def catalog(self):
        class Reader:
            def read(self,loc):return configuration(loc['method'],.1)
        return Reader(),dict(candidates=[])

    def generate_families(self,method,items):
        result=[]
        for item in items:
            delta=Fraction(.1)*KS._frac(item['changes']['delta_over_h'])
            row={'item_id': item['item_id']}
            if (method=='PR_PAULI'and delta==Fraction(1,5))or(method=='PR_DF'and delta==Fraction(.2)):
                row.update(status='UNSUPPORTED',reason='ARTIFICIAL_MAPPER_EXACT_WIDTH_REFUSAL_NO_ROUNDING')
            else:
                row.update(status='GENERATED',configuration=configuration(method,delta))
            result.append(row)
        return result

    def context(self,problem):
        candidates=[]
        for method,widths in self.widths.items():
            for w in widths:
                candidates.append(dict(method=method,family='PR',configuration_id=None,
                    key=dict(L_D=3 if method=='PR_PAULI'else 8,formula='2nd',delta_pf=P.exact(KS._frac(w['raw']))),
                    caps=[1]*13,ready=False,needs=['PRICE_26_AXIS_MISSING'],prices=None,saved_points=[],labels=[],
                    name='ARTIFICIAL_PLACEHOLDER:'+method+':'+w['exact'],price_status='ARTIFICIAL',signal_source=None))
        return dict(candidates=candidates,expected={},tau=Fraction(1),saved=O.SavedBackend([],{}),generate=lambda *args:None)

    def completion_api(self):
        if self.completion is None:self.completion=Completion(self.run/'completion',self.spec.get('candidate_completion',{}))
        return self.completion


def inputs(root,fast=True):
    problem=dict(accuracy=dict(fixture='ARTIFICIAL'),target=dict(energy_ha=0),rpe=dict(M=12),execution_routes={m:[]for m in METHODS},
        methods={m:dict(L_D_values=[3 if m=='PR_PAULI'else 8],generation=dict(formulas=['2nd'],baseline=dict(formula='2nd',delta_over_h='1',kappa_multiplier='1')))for m in METHODS},
        search=dict(rule='METHOD_MINIMUM_PRIORITY_V2',decrement_steps=[1],large_step_headroom='9/10',seed_rules=[],
                    max_steps_per_method=100,max_new_q3_per_method=100,on_missing='STOP'))
    spec=dict(round_id='ARTIFICIAL_WIDTH_THROUGHPUT',methods=list(METHODS),science_approved=True,initial='all',
        max_new_q3_per_method={m:2 for m in METHODS},candidate_completion=dict(max_attempted_per_method=0),
        executor=dict(scheduler='concurrent'if fast else'legacy',max_concurrent_jobs=2,dp_workers=1,prefix_cache_directory=str(root/'prefix_cache'),prefix_checkpoint_rounds=[5,11]),
        width_families=[dict(method=m,L_D=3 if m=='PR_PAULI'else 8,configuration=dict(method=m,path='ARTIFICIAL',commit='ARTIFICIAL'))for m in METHODS],
        search_overrides=dict(requested_widths={'PR_PAULI':[.1,.2,'1/5'],'PR_DF':[.1,.05,.2]},schedule_wide=dict(max_kappa='1/1000000000',per_decision=8)))
    put(root/'PROBLEM.json',problem);put(root/'ROUND.json',spec)
    return ['--problem',str(root/'PROBLEM.json'),'--round',str(root/'ROUND.json'),'--run-dir',str(root/'run'),'--max-cycles','3','--batch','2']


class WidthThroughput(unittest.TestCase):
    def run_case(self,path,fast=True):
        global WORLD
        world=WORLD=World(path);args=inputs(path,fast)
        actual_factory=E.K.CLI.make_executor
        def factory(*a,**kw):
            executor,optimize=actual_factory(*a,**kw);world.executors.append(executor);return executor,optimize
        with ExitStack()as stack:
            stack.enter_context(patch.object(C,'ROOT',path))
            stack.enter_context(patch.object(C,'value',side_effect=world.reader.read))
            stack.enter_context(patch.object(E,'read_loc',side_effect=world.reader.read))
            stack.enter_context(patch.object(C,'publish',side_effect=world.publish))
            stack.enter_context(patch.object(C,'git',return_value='ARTIFICIAL'))
            stack.enter_context(patch('trottertracks.pr_price_rules.GitReader',return_value=world.reader))
            stack.enter_context(patch.object(E.K.CLI,'make_executor',side_effect=factory))
            with redirect_stdout(io.StringIO()):first=E.main(args,driver_factory=Driver)
            launches={str(p):p.read_bytes()for p in(first.run/'jobs').glob('*/*/LAUNCH.json')}
            signals=len(world.signal_calls);leaves=len(world.scientific_price_leaves)
            with redirect_stdout(io.StringIO()):second=E.main(args,driver_factory=Driver)
        return world,first,second,launches,signals,leaves

    def test_normal_cli_ready_and_audited_late_ready_use_one_queue_and_resume_zero(self):
        with tempfile.TemporaryDirectory(prefix='pr-width-throughput-artificial-')as tmp:
            root=Path(tmp);world,first,again,launches,signals,leaves=self.run_case(root)
            result=json.loads((first.run/'ROUND_RESULT.json').read_text())['methods']
            self.assertTrue(all(result[m]['best']for m in METHODS))
            self.assertEqual(len(launches),4)
            self.assertEqual(launches,{str(p):p.read_bytes()for p in(again.run/'jobs').glob('*/*/LAUNCH.json')})
            self.assertEqual(len(world.signal_calls),signals);self.assertEqual(len(world.scientific_price_leaves),leaves)
            self.assertTrue(all(ex.state.closed for ex in world.executors))
            events=[json.loads(line)for line in(first.run/'PROGRESS.jsonl').read_text().splitlines()]
            enqueued=[r for r in events if r['event']=='ENQUEUE']
            self.assertEqual(len(enqueued),4)
            self.assertEqual(len({r['invocation']for r in enqueued}),1)
            self.assertEqual({m:sum(r['method']==m for r in enqueued)for m in METHODS},{m:2 for m in METHODS})
            invocations=[json.loads(line)for line in(first.run/'INVOCATIONS.jsonl').read_text().splitlines()]
            self.assertTrue(invocations)
            self.assertTrue(all(i['budget']=={m:2 for m in METHODS} and i['launched']=={m:2 for m in METHODS} for i in invocations))
            self.assertEqual({r['method']for r in enqueued},set(METHODS))
            self.assertTrue(all(result[m]['new_calls']<=2 for m in METHODS))
            state=json.loads((first.run/'completion/STATE.json').read_text())
            ready=[r for r in state['records'].values()if r['status']=='READY_FOR_SEARCH']
            self.assertEqual(len(ready),4)
            late=[r for r in ready if r.get('previous_hold')]
            self.assertEqual(len(late),2)
            self.assertTrue(all(r['recovered_by']=='METADATA_REQUERY_NO_SCIENTIFIC_RETRY'for r in late))
            step=json.loads((first.run/'completion/STEP_PRICE_RULES.json').read_text())
            self.assertEqual(len(step['entries']),2)
            self.assertTrue(any(f['status']=='UNSUPPORTED_MAPPER'for f in first.state['families'].values()))
            self.assertNotEqual(KS._frac(.2),KS._frac('1/5'))
            for p in first.state['points']:
                if p['status']=='EVALUATED':
                    self.assertIn('evaluation_configuration_id',p)
                    self.assertEqual(KS._frac(p['requested_delta_pf']),KS._frac(p['completion_request']['design']['delta_pf']))
            output=os.environ.get('PR_WIDTH_QUEUE_EVIDENCE')
            if output:
                target=Path(output);target.mkdir(parents=True,exist_ok=True)
                shutil.copytree(first.run,target/'run',dirs_exist_ok=True)
                put(target/'ARTIFICIAL_DOCUMENTS.json',dict(artificial=True,documents=[dict(commit=c,path=p,value=v)
                    for (c,p),v in sorted(world.reader.docs.items()) if p.startswith(('artificial_evidence/','signals/'))]))
                put(target/'SUMMARY.json',dict(artificial=True,methods=list(METHODS),queue_invocations_with_enqueues=1,
                    launches=len(launches),enqueues=len(enqueued),late_ready=len(late),scoped_entries=len(step['entries']),
                    first_mock_signal_calls=signals,first_mock_price_leaves=leaves,
                    cumulative_budget={m:2 for m in METHODS},launches_by_method={m:2 for m in METHODS},
                    replay_new_signal_calls=len(world.signal_calls)-signals,replay_new_price_leaves=len(world.scientific_price_leaves)-leaves,
                    replay_new_launches=0,actual_H4_signal_compile_Q3=0))


if __name__=='__main__':unittest.main(verbosity=2)
