"""Changed-width completion through the actual Completion and scoped registry.

Only artificial receipt documents and mocked scientific leaves are used. The
metadata registry, partial-step audit intake, pending refresh and installation
are the production implementations.
"""
from copy import deepcopy
from contextlib import nullcontext
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pr_candidate_completion as C,rpe_method_optimizer as O
from trottertracks.pr_step_price_registry import StepPriceRegistry
from test_pr_step_price_registry import (
    Documents, config, df_direct_fixture, df_model, exact, implementation, missing_quote, pauli_fixture,
)


def load_script(name):
    spec=importlib.util.spec_from_file_location('_width_'+name,ROOT/'scripts'/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def enrich(cfg):
    cfg['spec']['kappa']=dict(value=exact(2));return cfg


def test_df_audit_worker_returns_archived_input_outputs_in_owned_leaf(tmp_path):
    runner = tmp_path/'source/scripts/provider.py'
    runner.parent.mkdir(parents=True)
    runner.write_text("""
import json, sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
def main():
    document = Path(sys.argv[sys.argv.index('--input')+1])
    fixed = json.loads(document.read_bytes())
    write(ROOT/fixed['output_directory']/'FROZEN_MODELS.json', {'read_path': fixed['output_directory']})
""")
    doc = tmp_path/'owned/df_prices/INPUTS.json'
    doc.parent.mkdir(parents=True)
    content = dict(output_directory='prior/df_prices', runtime={'provider_root': str(tmp_path)})
    doc.write_text(json.dumps(content))
    subprocess.run([sys.executable, '-B', '-c', C._DF_PRICE_BOOTSTRAP,
                    str(runner), str(tmp_path), 'freeze', '--input', str(doc)], check=True)
    assert C.read(doc) == content
    assert C.read(doc.with_name('FROZEN_MODELS.json'))['read_path'] == content['output_directory']
    assert not (runner.parents[1]/content['output_directory']).exists()


class WidthCompletionTests(unittest.TestCase):
    def setup_completion(self,root,reader):
        path=root/'BASE_REGISTRY.json';C.put(path,dict(entries=[]))
        c=C.Completion(root/'completion',dict(source='ARTIFICIAL',price_registry_file=str(path)))
        def publish(paths,message):
            for path in paths:
                path=Path(path)
                for item in ([path]if path.is_file()else path.rglob('*.json')):
                    if item.suffix=='.json':reader.add(C.rel(item),C.read(item))
            return 'ARTIFICIAL'
        return c,publish

    def test_own_returned_audit_auto_registers_ready_and_resume_launches_zero(self):
        reader,cfg,imp,package=pauli_fixture();enrich(cfg)
        reader.docs['ARTIFICIAL','CONFIG.json']=deepcopy(cfg)
        with tempfile.TemporaryDirectory()as tmp,patch.object(C,'ROOT',Path(tmp)),patch('trottertracks.pr_price_rules.GitReader',return_value=reader):
            c,publisher=self.setup_completion(Path(tmp),reader)
            q=missing_quote(cfg);q['implementation']=imp
            for row in q['rows']:
                if row['m']==0:row.update(unit_price_RZ=exact(3),classification='MODEL_REFERENCE')
            scientific=[]
            def service(mode,request,d,tag=None,**kwargs):
                if mode=='quote':return deepcopy(q)
                if mode=='intake':return dict(binding_id='BIND',total_configuration_id='TOTAL',
                                              configuration=request['configuration'],signal_packet=request['packet'])
                self.fail('Unexpected metadata service '+mode)
            def leaf(cmd,d,stage,output=None,science=False):
                self.assertEqual(stage,'PAULI_PRICES');scientific.append(stage)
                return dict(status='STEP_MODELS_READY',evidence_packages=[package])
            signal=dict(packet_source=reader.add('PACKET.json',dict(status='ARTIFICIAL_COMPLETE')))
            request=dict(method='PR_PAULI',design=dict(L_D=3,formula='2nd',delta_pf=.2,kappa='2',K=2))
            with patch.object(C,'value',side_effect=reader.read),patch.object(C,'publish',side_effect=publisher),patch.object(c,'service',side_effect=service),patch.object(c,'call',side_effect=leaf),patch.object(c,'pauli_configuration',return_value=(cfg,{})),patch.object(c,'pauli_signals',return_value=signal)as signals:
                out=c.complete(request)
                self.assertEqual(out['status'],'READY_FOR_SEARCH',out)
                self.assertEqual(scientific,['PAULI_PRICES']);self.assertEqual(signals.call_count,1)
                quote=reader.read(out['quote'])
                self.assertTrue(quote['complete']);self.assertEqual(quote['rows'][0]['unit_price_RZ'],exact(3))
                self.assertEqual(quote['rows'][2]['model_alternative']['covered_rounds'],list(range(1,13)))
                again=c.complete(request)
                self.assertTrue(again['resumed']);self.assertEqual(scientific,['PAULI_PRICES']);self.assertEqual(signals.call_count,1)
                self.assertEqual(len(C.read(c.step_registry_path)['entries']),1)
                context=dict(candidates=[dict(method='PR_PAULI',family='PR',key=dict(L_D=3,formula='2nd',delta_pf=.2),configuration_id=None,caps={})],expected={})
                problem=dict(methods={'PR_PAULI':dict(requested_delta_pf_values=[exact(.2)])},execution_routes={'PR_PAULI':[]})
                with patch.object(C,'git',return_value='ARTIFICIAL'):
                    routes=c.install(context,problem,[out])
                self.assertTrue(context['candidates'][0]['ready']);self.assertEqual(len(routes['PR_PAULI']),1)

    def test_pending_candidate_rechecks_new_audit_without_signal_or_price_acquisition(self):
        reader,cfg,imp,package=pauli_fixture();enrich(cfg)
        reader.docs['ARTIFICIAL','CONFIG.json']=deepcopy(cfg)
        with tempfile.TemporaryDirectory()as tmp,patch.object(C,'ROOT',Path(tmp)),patch('trottertracks.pr_price_rules.GitReader',return_value=reader):
            c,publisher=self.setup_completion(Path(tmp),reader)
            request=dict(method='PR_PAULI',design=dict(delta_pf=.2));ident=C.key(request)
            rec=dict(key=ident,method='PR_PAULI',request=request,status='WAITING_INPUTS',reason='PRICE_RULE_INCOMPLETE:',configuration=package['configuration_source'],signal={'packet_source':{'commit':'ARTIFICIAL','path':'PACKET'}},intake={'binding_id':'B','total_configuration_id':'T'},registry_at_hold=c.price_registry_digest())
            c.state['records'][ident]=rec;c.save()
            q=missing_quote(cfg);q['implementation']=imp
            with patch.object(C,'value',side_effect=reader.read),patch.object(C,'publish',side_effect=publisher),patch.object(c,'service',return_value=q)as service,patch.object(c,'call',side_effect=AssertionError('No acquisition on pending resume')):
                self.assertEqual(c.complete(request)['status'],'WAITING_INPUTS');self.assertEqual(service.call_count,0)
                c.settings['price_evidence_results']=[package]
                self.assertEqual(c.complete(request)['status'],'WAITING_INPUTS')
                # Only the 0.2 step is now covered: m0 has not silently acquired it.
                self.assertEqual(service.call_count,1)
                q['rows'][0].update(unit_price_RZ=exact(3),classification='MODEL_REFERENCE')
                q['rows'][1].update(unit_price_RZ=exact(3),classification='MODEL_REFERENCE')
                registry=C.read(c.registry);registry['metadata']='new m0 saved price';C.put(c.registry,registry)
                self.assertEqual(c.complete(request)['status'],'READY_FOR_SEARCH')
                self.assertEqual(service.call_count,2)
                self.assertEqual(c.complete(request)['status'],'READY_FOR_SEARCH');self.assertEqual(service.call_count,2)

    def test_df_overlap_and_extrapolation_plan_only_the_missing_rounds(self):
        reader=Documents();cfg=enrich(config(.1,pauli=False));imp=implementation(False)
        reg=StepPriceRegistry(reader)
        df_model(reg,cfg,imp,'R1',varying=False,q_max=64)
        df_model(reg,cfg,imp,'QR',varying=True,q_max=4096)
        cfg['rounds'][-1]['R_outer']=31
        cfg['rounds'][-1]['structure'].update(allocation=[31],event_count=31*4096)
        for key in ('candidate','physical_candidate'):cfg['rounds'][-1][key]['R_outer']=31
        cfg['rounds'][-1]['candidate']['event_count']=31*4096
        cref=reader.add('CONFIG.json',cfg)
        request=dict(method='PR_DF',design=dict(L_D=8,delta_pf=.1))
        ident=C.key(request)
        quote=missing_quote(cfg,'PR_DF');quote['implementation']=imp
        with tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)), \
             patch('trottertracks.pr_price_rules.GitReader',return_value=reader):
            c,_=self.setup_completion(Path(tmp),reader)
            C.put(c.step_registry_path,reg.state)
            old_reason='PRICE_EVIDENCE_REQUIRED:'+C.PRICE_SOURCE_CONFLICT
            old_digest=C.key(dict(legacy=C.read(c.registry),steps=reg.state))
            record=dict(key=ident,method='PR_DF',request=request,status='WAITING_INPUTS',reason=old_reason,
                        configuration=cref,signal={'packet_source':'FIXED'},intake={'binding_id':'B'},
                        registry_at_hold=old_digest)
            c.state['records'][ident]=record;c.save()

            runner=load_script('run_pr_df_candidate_prices.py')
            planner=load_script('pr_candidate_services.py')
            def service(mode,payload,*_args,**_kwargs):
                if mode=='df-generate':return dict(status='CONFIGURATION_READY',configuration=cfg)
                if mode=='df-plan':return dict(rounds=[],reuse_rounds=list(range(13)))
                if mode=='quote':return deepcopy(quote)
                if mode=='df-price-design':return planner.df_price_disposition(cfg,payload['price_quote'],runner)
                self.fail('Unexpected metadata service '+mode)

            with patch.object(C,'value',side_effect=reader.read),patch.object(c,'service',side_effect=service) as run, \
                 patch.object(c,'call',side_effect=AssertionError('No scientific acquisition')), \
                 patch.object(c,'df_price',side_effect=AssertionError('No duplicate calibration')):
                plan=c.plan(request)
                requests=plan['calibration_requests']
                self.assertEqual([g['covered_rounds'] for g in requests],[[m] for m in [*range(7),12]])
                self.assertTrue(all(g['direct_resolution'] and g['design']['holdout_round'] is None
                                    for g in requests))
                self.assertIn(C.PRICE_EXTRAPOLATION_HOLD,plan['reason'])
                self.assertEqual({a['m'] for a in plan['extrapolated_price_axes']},{12})
                self.assertEqual({a['m'] for a in plan['conflicting_price_axes']},set(range(7)))
                self.assertEqual(run.call_count,4)

    def test_df_price_not_evaluated_keeps_signal_and_resumes_after_saved_price_arrives(self):
        reader=Documents();cfg=enrich(config(.1,pauli=False))
        request=dict(method='PR_DF',design=dict(L_D=8,delta_pf=.1,kappa='2'))
        hold=missing_quote(cfg,'PR_DF')
        hold['rows'][2]['needs']=dict(reasons=[C.PRICE_SOURCE_CONFLICT],entry_ids=['a','b'],
                                     source_kind='DIRECT_RECEIPTS')
        ready=deepcopy(hold);ready['complete']=True
        for row in ready['rows']:
            row['unit_price_RZ']=exact(2)

        with tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)):
            c,publisher=self.setup_completion(Path(tmp),reader)
            signal=dict(packet_source=reader.add('PACKET.json',dict(status='ARTIFICIAL_COMPLETE')))
            modes=[]
            def service(mode,*args,**kwargs):
                modes.append(mode)
                if mode=='df-generate':return dict(status='CONFIGURATION_READY',configuration=cfg)
                if mode=='df-plan':return dict(rounds=[dict(decision='REQUEST',m=4)])
                if mode=='intake':return dict(binding_id='BIND',total_configuration_id='TOTAL')
                if mode=='quote':return deepcopy(ready)
                self.fail('Unexpected stage: '+mode)

            with patch.object(c,'service',side_effect=service), patch.object(C,'value',side_effect=reader.read), \
                 patch.object(C,'publish',side_effect=publisher), patch.object(c,'initial_quote',return_value=hold), \
                 patch.object(c,'price',return_value=dict(hold,price_decisions=[
                     dict(m=1,axis='X',action='NO_SAFE_PRICE_ACTION',new_calculation=False)])) as price, \
                 patch.object(c,'df_signals',return_value=signal) as signals:
                held=c.complete(request)
                self.assertEqual(held['status'],'PRICE_NOT_EVALUATED')
                self.assertEqual(held['signal'],signal)
                self.assertEqual(c.price_deferred_keys(),{held['key']})
                self.assertEqual(modes,['df-generate','df-plan','intake'])
                self.assertTrue(c.complete(request)['resumed'])
                C.put(c.registry,dict(entries=[],revision='fresh saved price evidence'))
                self.assertFalse(c.price_deferred_keys())
                accepted=c.complete(request)
                self.assertEqual(accepted['status'],'READY_FOR_SEARCH',accepted)
                self.assertEqual(accepted['configuration_id'],cfg['id'])
                self.assertEqual(accepted['directory'],held['directory'])
                self.assertEqual(modes,['df-generate','df-plan','intake','quote'])
                self.assertEqual(signals.call_args.kwargs['plan']['rounds'],[dict(decision='REQUEST',m=4)])
                signals.assert_called_once()
                price.assert_called_once()

    def test_df_original_owner_resumes_and_shared_request_adopts_once(self):
        reader,cfg,imp,package=df_direct_fixture()
        quote=missing_quote(cfg,'PR_DF');quote['implementation']=imp
        for row in quote['rows']:
            if row['m']:
                row.update(unit_price_RZ=exact(10),source={'kind':'fixed'})
            else:
                row['needs']=dict(reasons=[C.PRICE_EXTRAPOLATION_HOLD])
        group=dict(group_id='shared', covered_rounds=[0], direct_resolution=True,
                   axis_keys=[dict(m=0,axis=a) for a in ('X','Y')],required_saved_companions=[],
                   status='CALIBRATION_DESIGN_READY')
        with (tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)),
              patch.object(C,'value',side_effect=reader.read),
              patch('trottertracks.pr_price_rules.GitReader',return_value=reader)):
            c,_=self.setup_completion(Path(tmp),reader)
            cref=package['configuration_source']
            a,b=(c.directory/'candidates'/name for name in ('owner','alias'))
            with (patch.object(c,'df_price_plan',return_value=dict(groups=[group])),
                  patch.object(c,'df_price_group',side_effect=[C.PreparationWait('owned'),package]) as worker):
                with self.assertRaises(C.PreparationWait):c.df_price(cref,quote,a,c.registry)
                with self.assertRaises(C.PreparationWait):c.df_price(cref,quote,b,c.registry)
                self.assertEqual(worker.call_count,1)
                c.df_price(cref,quote,a,c.registry)
                self.assertEqual(worker.call_args_list[0].args[2],worker.call_args_list[1].args[2])
                c.df_price(cref,quote,b,c.registry)
                self.assertEqual(worker.call_count,2)
                self.assertEqual(len(C.read(c.step_registry_path)['entries']),1)

    def test_df_prior_owner_finishes_before_audit_and_is_never_recompiled(self):
        reader,cfg,imp,package=df_direct_fixture()
        quote=missing_quote(cfg,'PR_DF')
        quote['implementation']=imp
        group=dict(group_id='prior', covered_rounds=[0], direct_resolution=True,
                   axis_keys=[dict(m=0,axis=a) for a in ('X','Y')],required_saved_companions=[],
                   status='CALIBRATION_DESIGN_READY')
        with (tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)),
              patch.object(C,'value',side_effect=reader.read),
              patch('trottertracks.pr_price_rules.GitReader',return_value=reader)):
            root=Path(tmp)
            c,_=self.setup_completion(root,reader)
            previous=root/'prior-run/completion'
            owner=previous/'candidates/cfg/df_price_groups/prior'
            leaf=owner/'evidence-original'
            base=leaf/'df_prices'
            result_path=base/'calibration/acquisition/RESULT.json'
            C.put(previous/'df-price-owners/prior.json',dict(directory=str(owner)))
            C.put(owner/'EXECUTION_LEAF.json',dict(name=leaf.name))
            cref=package['configuration_source']
            current=c.directory/'candidates/current'
            with (patch.object(c,'df_price_owner_roots',return_value=[previous]),
                  patch.object(c,'df_price_plan',return_value=dict(groups=[group])),
                  patch.object(C,'git',return_value='ARTIFICIAL'),
                  patch.object(c,'df_price_group',return_value=package) as acquire):
                self.assertEqual(c.df_price(cref,quote,current,c.registry),[])
                acquire.assert_not_called()
                [first,_]=C.read(current/'DF_PRICE_DECISIONS.json')
                self.assertFalse(first['new_calculation'])
                self.assertEqual(first['action'],'RECONCILE_ORIGINAL_ATTEMPT')
                decision=dict(price_actions=[first])
                before=c.df_price_evidence_snapshot(decision)
                original_candidate=dict(id='fixed',q=1,R_outer=1)
                inputs=dict(source='fixed',output_directory=C.rel(base),cells=[dict(role='CALIBRATION',m=0,indices=[0],
                    cohort='original-cohort',request=dict(physical_context=dict(physical_candidate=original_candidate)))])
                witness={axis:dict(status='REUSE',index=0,axis=axis,
                    path_receipt=dict(commit='WORKING',path='path.json'),
                    receipt=dict(commit='WORKING',path=axis+'.json')) for axis in ('X','Y')}
                result=dict(status='ACQUIRED',source='fixed',inputs_commit='ARTIFICIAL',
                    rows=[dict(m=0,index=0,status='VALID',axes=witness)])
                reader.add(C.rel(base/'INPUTS.json'),inputs)
                reader.add(C.rel(result_path),result)
                C.put(base/'INPUTS.json',inputs)
                C.put(result_path,result)
                C.put(leaf/'DF_CALIBRATION.RETURN.json',dict(returncode=0,signal=None,source='fixed'))
                self.assertNotEqual(c.df_price_evidence_snapshot(decision),before)
                self.assertEqual(c.df_price(cref,quote,current,c.registry),[package])
                recovered=acquire.call_args.kwargs['recovered_axes']['rows']
                self.assertEqual(len(recovered),2)
                for row in recovered:
                    [alternative]=row['direct_alternatives']
                    self.assertEqual(alternative['cohort'],'original-cohort')
                    self.assertEqual(alternative['path_witnesses'][0]['receipt']['commit'],'ARTIFICIAL')
                self.assertTrue(all(a['action']=='AUDIT_ORIGINAL_COMPILE' and not a['new_calculation']
                                    for a in C.read(current/'DF_PRICE_DECISIONS.json')))
                c.df_price(cref,quote,current,c.registry)
                acquire.assert_called_once()

    def test_df_recovered_wrapper_axes_run_audit_without_compile_reservation(self):
        reader,_,imp,package=df_direct_fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)):
            c,publisher=self.setup_completion(Path(tmp),reader)
            C.put(c.registry,dict(entries=[dict(name='DF_LD7',implementation=imp)]))
            root=c.directory/'recovered'
            group=dict(covered_rounds=[0],direct_resolution=True,required_saved_companions=[])
            evidence=dict(schema='PR_DF_PARTIAL_AXIS_QUOTES_V1',rows=[])
            stages=[]
            def call(cmd,d,stage,*args,science=False):
                stages.append((stage,science))
                if stage=='DF_PRICE_PREPARE':
                    self.assertIn('--saved-axis-quotes',cmd)
                    C.put(d/'df_prices/INPUTS.json',dict(cells=[dict(m=0,role='CALIBRATION',
                        axis_evidence={'0':{axis:dict(status='REUSE') for axis in ('X','Y')}})],
                        model_design={'holdout_round':None}))
            with (patch.object(c,'call',side_effect=call),patch.object(C,'publish',side_effect=publisher),
                  patch.object(c,'df_saved_axis_objects',return_value={})):
                got=c.df_price_group(package['configuration_source'],dict(rows=[]),root,c.registry,group,
                                     recovered_axes=evidence)
            self.assertEqual(got['calibration_compile_axes'],[])
            self.assertEqual(stages,[('DF_PRICE_PREPARE',False),('DF_CALIBRATION',False),('DF_MODEL_FREEZE',True)])

    def test_df_different_price_group_can_recover_original_wrappers(self):
        reader,cfg,imp,package=df_direct_fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)):
            c,publisher=self.setup_completion(Path(tmp),reader)
            C.put(c.registry,dict(entries=[dict(name='DF_LD7',implementation=imp)]))
            prior=c.directory/'candidates/prior/df_price_groups/other/evidence-old'
            cell=dict(m=0,role='CALIBRATION',request=dict(axes=['X','Y'],physical_context={
                'physical_candidate':cfg['rounds'][0]['physical_candidate']}))
            C.put(prior/'df_prices/INPUTS.json',dict(cells=[cell]))
            marker=prior/'DF_CALIBRATION.DISTRIBUTED.json'
            C.put(marker,dict(job_id='prep-original'))
            current=c.directory/'candidates/current/df_price_groups/new'
            group=dict(covered_rounds=[0],direct_resolution=True,required_saved_companions=[])
            quote=dict(rows=[])
            stages=[]
            def call(cmd,d,stage,*args,science=False):
                stages.append((stage,science))
                if stage=='DF_PRICE_PREPARE':
                    status='REUSE' if '--saved-axis-quotes' in cmd else 'MISSING'
                    C.put(d/'df_prices/INPUTS.json',dict(cells=[dict(cell,axis_evidence={
                        '0':{axis:dict(status=status) for axis in ('X','Y')}})],
                        model_design={'holdout_round':None}))
            evidence=dict(schema='PR_DF_PARTIAL_AXIS_QUOTES_V1',rows=[])
            with (patch.object(c,'call',side_effect=call),patch.object(C,'publish',side_effect=publisher),
                  patch.object(c,'df_original_axis_quotes',side_effect=[None,evidence]) as recover,
                  patch.object(c,'df_saved_axis_objects',return_value={})):
                with self.assertRaises(C.DFOriginalPriceAttempt) as held:
                    c.df_price_group(package['configuration_source'],quote,current,c.registry,group)
                self.assertEqual(held.exception.marker,marker)
                self.assertFalse((current/'EXECUTION_LEAF.json').exists())
                record=dict(price_actions=[dict(original_marker=str(marker))])
                before=c.df_price_evidence_snapshot(record)
                C.put(prior/'DF_CALIBRATION.RETURN.json',dict(returncode=0))
                self.assertNotEqual(c.df_price_evidence_snapshot(record),before)
                result=c.df_price_group(package['configuration_source'],quote,current,c.registry,group)
            self.assertEqual([entry.args for entry in recover.call_args_list],[(prior,),(prior,)])
            self.assertEqual(result['calibration_compile_axes'],[])
            self.assertIn(('DF_MODEL_FREEZE',True),stages)
            self.assertNotIn(('DF_CALIBRATION',True),stages)

    def test_df_normal_resume_acquires_exact_price_and_keeps_original_signal(self):
        reader,cfg,imp,package=df_direct_fixture()
        enrich(cfg)
        reader.docs['ARTIFICIAL','CONFIG.json']=deepcopy(cfg)
        quote=missing_quote(cfg,'PR_DF');quote['implementation']=imp
        for row in quote['rows']:
            if row['m']:
                row.update(unit_price_RZ=exact(10),source={'commit':'ARTIFICIAL','path':'fixed'})
            else:
                row['needs']=dict(reasons=[C.PRICE_SOURCE_CONFLICT],entry_ids=['old_model','other_model'])
        planner=load_script('pr_candidate_services.py')
        runner=load_script('run_pr_df_candidate_prices.py')
        def service(mode,payload,*args,**kwargs):
            if mode=='quote':return deepcopy(quote)
            if mode=='df-price-design':return planner.df_price_disposition(cfg,payload['price_quote'],runner)
            self.fail('Original signal and configuration should have been reused: '+mode)
        with (tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)),
              patch.object(C,'value',side_effect=reader.read),
              patch('trottertracks.pr_price_rules.GitReader',return_value=reader)):
            c,publisher=self.setup_completion(Path(tmp),reader)
            request=dict(method='PR_DF',design=dict(L_D=3,kappa='2',delta_pf=.2))
            ident=C.key(request);cref=package['configuration_source']
            signal=dict(packet_source={'commit':'ARTIFICIAL','path':'ORIGINAL_SIGNAL'})
            c.state['records'][ident]=dict(status='WAITING_INPUTS',key=ident,method='PR_DF',selection=None,
                request=request,directory=C.rel(c.directory/'candidates'/ident),configuration=cref,
                signal=signal,intake={'binding_id':'ORIGINAL_INTAKE'},registry_at_hold='previous-policy',
                reason='PRICE_EVIDENCE_REQUIRED:'+C.PRICE_SOURCE_CONFLICT)
            with (patch.object(c,'service',side_effect=service), patch.object(C,'publish',side_effect=publisher),
                  patch.object(c,'df_price_group',return_value=package) as acquire,
                  patch.object(c,'df_signals',side_effect=AssertionError('No signal restart'))):
                record=c.complete(request)
                self.assertEqual(record['status'],'READY_FOR_SEARCH',record.get('reason'))
                self.assertEqual(record['signal'],signal)
                self.assertEqual(acquire.call_args.args[4]['covered_rounds'],[0])
                self.assertEqual(record['prices'][:2],[exact(31),exact(32)])
                self.assertTrue(c.complete(request)['resumed'])
                self.assertEqual(acquire.call_count,1)

    def test_df_direct_leaf_protects_already_priced_companion(self):
        reader,cfg,imp,package=df_direct_fixture()
        quote=missing_quote(cfg,'PR_DF')
        quote['rows'][0].update(unit_price_RZ=exact(31),source={'kind':'fixed'})
        cell=dict(m=0,axis_evidence={'0':{'X':dict(status='MISSING'),'Y':dict(status='MISSING')}})
        group=dict(covered_rounds=[0],direct_resolution=True,required_saved_companions=['X'])
        with tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)):
            c,_=self.setup_completion(Path(tmp),reader)
            def call(cmd, directory, stage, *args, **kwargs):
                self.assertEqual(stage,'DF_PRICE_PREPARE')
                C.put(directory/'df_prices/INPUTS.json',dict(cells=[cell],model_design={'holdout_round':None}))
            with patch.object(c,'call',side_effect=call) as execute:
                with self.assertRaisesRegex(C.Held,'DIRECT_PAIR_REQUIRES_SAVED_COMPANION'):
                    c.df_price_group(package['configuration_source'],quote,c.directory/'direct',c.registry,group)
                self.assertEqual(execute.call_count,1)
                self.assertFalse((c.directory/'direct/EXECUTION_LEAF.json').exists())
            # An admissible companion makes only the missing axis runnable.
            cell['axis_evidence']['0']['X']=dict(status='REUSE',C=31)
            c.stage_executor=SimpleNamespace(handles=lambda stage:stage=='DF_CALIBRATION')
            launched=[]
            def worker(cmd, directory, stage, *args, **kwargs):
                if stage=='DF_PRICE_PREPARE':
                    return call(cmd,directory,stage)
                if stage=='DF_CALIBRATION':
                    self.assertTrue(kwargs['science'])
                    fixed=C.read(directory/'df_prices/INPUTS.json')['cells'][0]['axis_evidence']['0']
                    launched.extend(axis for axis,value in fixed.items() if value['status']=='MISSING')
                    raise C.PreparationWait('fixture leaves original owner outstanding')
                self.fail('Unexpected scientific stage '+stage)
            with (patch.object(c,'call',side_effect=worker), patch.object(C,'publish',return_value='ARTIFICIAL')):
                with self.assertRaises(C.PreparationWait):
                    c.df_price_group(package['configuration_source'],quote,c.directory/'direct',c.registry,group)
            self.assertEqual(launched,['Y'])
            self.assertTrue((c.directory/'direct/EXECUTION_LEAF.json').is_file())

    def test_pauli_waits_on_confirmed_price_hold_and_resumes_from_its_configuration(self):
        reader,cfg,_,_=pauli_fixture();enrich(cfg)
        request=dict(method='PR_PAULI',design=dict(L_D=3,formula='2nd',delta_pf=.2,kappa='2',K=2))
        hold=missing_quote(cfg)
        hold['rows'][0]['needs']=dict(reasons=[C.PRICE_SOURCE_CONFLICT],entry_ids=['a','b'])
        ready=deepcopy(hold);ready['complete']=True
        for row in ready['rows']:row['unit_price_RZ']=exact(2)
        with tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)):
            c,publisher=self.setup_completion(Path(tmp),reader)
            signal=dict(packet_source=reader.add('PACKET.json',dict(status='ARTIFICIAL_COMPLETE')))
            with patch.object(C,'value',side_effect=reader.read), patch.object(C,'publish',side_effect=publisher), \
                 patch.object(c,'pauli_configuration',return_value=(cfg,{})) as configuration, \
                 patch.object(c,'initial_quote',side_effect=[hold,ready]), patch.object(c,'price',return_value=ready) as price, \
                 patch.object(c,'pauli_signals',return_value=signal) as signals, \
                 patch.object(c,'service',return_value=dict(binding_id='BIND',total_configuration_id='TOTAL')) as service:
                held=c.complete(request)
                self.assertEqual(held['status'],'WAITING_INPUTS')
                self.assertEqual(c.price_deferred_keys(),{held['key']})
                self.assertTrue(c.complete(request)['resumed'])
                signals.assert_not_called();price.assert_not_called();service.assert_not_called()
                C.put(c.registry,dict(entries=[],revision='new admissible price'))
                accepted=c.complete(request)
                self.assertEqual(accepted['status'],'READY_FOR_SEARCH',accepted)
                self.assertEqual(configuration.call_args.args[0]['configuration'],held['configuration'])
                signals.assert_called_once();price.assert_called_once()

    def test_distributed_pauli_keeps_independent_signal_when_price_conflicts(self):
        reader,cfg,_,_=pauli_fixture();enrich(cfg)
        request=dict(method='PR_PAULI',design=dict(L_D=3,formula='2nd',delta_pf=.2,kappa='2',K=2))
        hold=missing_quote(cfg)
        hold['rows'][0]['needs']=dict(reasons=[C.PRICE_SOURCE_CONFLICT],entry_ids=['a','b'])
        ready=deepcopy(hold);ready['complete']=True
        for row in ready['rows']:
            row['unit_price_RZ']=exact(2)
        with tempfile.TemporaryDirectory() as tmp, patch.object(C,'ROOT',Path(tmp)):
            c,publisher=self.setup_completion(Path(tmp),reader)
            signal=dict(packet_source=reader.add('PACKET.json',dict(status='ARTIFICIAL_COMPLETE')))
            def service(mode,*args,**kwargs):
                if mode=='intake':
                    return dict(binding_id='BIND',total_configuration_id='TOTAL')
                self.assertEqual(mode,'quote')
                return deepcopy(ready)
            with (patch.object(C,'value',side_effect=reader.read),patch.object(C,'publish',side_effect=publisher),
                  patch.object(c,'pauli_configuration',return_value=(cfg,{})),patch.object(c,'quote_with_steps',side_effect=lambda _,q:q),
                  patch.object(c,'initial_quote',return_value=hold),patch.object(c,'price',return_value=hold),
                  patch.object(c,'pauli_signals',return_value=signal) as signals,patch.object(c,'service',side_effect=service)):
                old=c.complete(request)
                self.assertTrue(old['signal_deferred_for_price'])
                signals.assert_not_called()
                c.stage_executor=object()
                self.assertNotIn(old['key'],c.price_deferred_keys())
                held=c.complete(request)
                self.assertEqual(held['status'],'PRICE_NOT_EVALUATED')
                self.assertEqual(held['key'],old['key'])
                self.assertEqual(held['signal'],signal)
                self.assertEqual(held['price_actions'][0]['recheck_on'],'PRICE_REGISTRY_CHANGED')
                self.assertTrue(c.complete(request)['resumed'])
                C.put(c.registry,dict(entries=[],revision='new admissible price'))
                accepted=c.complete(request)
                self.assertEqual(accepted['status'],'READY_FOR_SEARCH')
                self.assertEqual(accepted['signal'],signal)
                signals.assert_called_once()

    def test_df_direct_group_does_not_expand_to_other_rounds(self):
        reader,cfg,imp,package=df_direct_fixture();enrich(cfg)
        reader.docs['ARTIFICIAL','CONFIG.json']=deepcopy(cfg)
        with tempfile.TemporaryDirectory()as tmp,patch('trottertracks.pr_price_rules.GitReader',return_value=reader):
            c,_=self.setup_completion(Path(tmp),reader);c.ingest_price_results([package])
            q=missing_quote(cfg,'PR_DF');q['implementation']=imp
            result=c.quote_with_steps(cfg,q)
            self.assertFalse(result['complete']);self.assertIsNotNone(result['rows'][0]['unit_price_RZ'])
            self.assertTrue(all(r['unit_price_RZ']is None for r in result['rows'][2:]))
            modified=deepcopy(cfg);modified['spec']['profile']['compiler']='OTHER'
            self.assertFalse(any(r['unit_price_RZ']is not None for r in c.quote_with_steps(modified,dict(q,configuration_id=modified['id']))['rows']))

    def test_failed_science_never_reopens_when_price_evidence_changes(self):
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion);c.directory=Path(tmp);c.registry=Path(tmp)/'r.json';C.put(c.registry,dict(entries=[]))
            rec=dict(status='HELD_WITH_REASON',reason='STAGE_FAILED:MEAN',scientific_failure=True,configuration={'commit':'A','path':'B'},signal={'x':1},intake={'x':1})
            with patch.object(c,'service',side_effect=AssertionError('STOP is terminal')):
                self.assertIsNone(c.refresh_price_hold(rec))

    def test_physical_alias_contract_keeps_dict_state_law_target_and_ignores_declaration_only(self):
        cfg=enrich(config());alias=deepcopy(cfg);alias['id']='ALIAS';alias['spec'].update(delta_pf=exact(.3),kappa=dict(value=exact(3)))
        self.assertEqual(C.physical_evaluation_contract(cfg),C.physical_evaluation_contract(alias))
        for mutate in (lambda c:c['spec']['split']['tail_dictionary'].update(dictionary_sha256='OTHER'),
                       lambda c:c['spec']['binding'].update(state='OTHER'),
                       lambda c:c['spec']['law']['source'].update(commit='OTHER'),
                       lambda c:c['spec'].update(target={'energy':'OTHER'})):
            changed=deepcopy(alias);mutate(changed)
            self.assertNotEqual(C.physical_evaluation_contract(cfg),C.physical_evaluation_contract(changed))

    def test_early_ready_alias_skips_all_scientific_leaves(self):
        reader,cfg,imp,package=pauli_fixture();enrich(cfg)
        old=deepcopy(cfg);old['id']='OLD_PHYSICAL';oldref=reader.add('OLD.json',old)
        cfg['spec']['delta_pf']=exact(.3)
        request=dict(method='PR_PAULI',design=dict(L_D=3,formula='2nd',delta_pf=.3,kappa='2',K=2))
        with tempfile.TemporaryDirectory()as tmp,patch.object(C,'ROOT',Path(tmp)):
            c,publisher=self.setup_completion(Path(tmp),reader)
            c.state['records']['old']=dict(status='READY_FOR_SEARCH',method='PR_PAULI',configuration=oldref,
                signal={'packet_source':{'commit':'ARTIFICIAL','path':'P'}},intake={'binding_id':'B','total_configuration_id':'T'},
                quote={'commit':'ARTIFICIAL','path':'Q'},prices=[exact(3)]*26,schedule=[],request={})
            with patch.object(C,'value',side_effect=reader.read),patch.object(C,'publish',side_effect=publisher),patch.object(c,'pauli_configuration',return_value=(cfg,{})),patch.object(c,'pauli_signals',side_effect=AssertionError('No alias signal')),patch.object(c,'price',side_effect=AssertionError('No alias price')):
                out=c.complete(request)
            self.assertEqual(out['status'],'READY_FOR_SEARCH',out)
            self.assertEqual(out['reused_physical_configuration'],oldref);self.assertEqual(out['new_scientific_calls'],0)

    def test_install_refuses_underbound_legacy_alias_and_honors_declared_widths(self):
        reader,cfg,imp,package=pauli_fixture();enrich(cfg)
        other=deepcopy(cfg);other['id']='OTHER_DICT'
        other['spec']['split']['tail_dictionary']['dictionary_sha256']='DIFFERENT'
        oldref=reader.add('OLD.json',cfg);newref=reader.add('NEW.json',other)
        qref=reader.add('Q.json',dict(complete=True,rows=[]))
        candidate=dict(method='PR_PAULI',family='PR',configuration_id=cfg['id'],generated_configuration=cfg,
            configuration_source=oldref,ready=True,caps={},schedule_keys=O.schedule_keys(cfg),key=dict(L_D=3))
        self.assertEqual(O.schedule_keys(cfg),O.schedule_keys(other))  # intentionally exercise the legacy collision
        packet={'commit':'ARTIFICIAL','path':'PACKET'}
        rec=dict(method='PR_PAULI',status='READY_FOR_SEARCH',configuration=newref,
            signal={'packet_source':packet},
            intake={'binding_id':'B','total_configuration_id':'T','configuration':newref,'signal_packet':packet},
            prices=[exact(3)]*26,quote=qref)
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion);c.directory=Path(tmp);c.state={'records':{}};c.state_path=Path(tmp)/'STATE.json';c.lock=threading.RLock()
            ctx=dict(candidates=[candidate],expected={});problem=dict(methods={'PR_PAULI':dict(requested_delta_pf_values=[exact(.2)])},execution_routes={'PR_PAULI':[]})
            with patch.object(C,'value',side_effect=reader.read),patch.object(C,'git',return_value='ARTIFICIAL'):
                c.install(ctx,problem,[rec])
                self.assertEqual(len(ctx['candidates']),2);self.assertEqual(rec['installed_configuration_id'],other['id'])
                self.assertFalse(candidate.get('physical_aliases'))
                blocked=deepcopy(rec);blocked['configuration']=reader.add('THIRD.json',dict(other,id='THIRD'))
                problem['methods']['PR_PAULI']['requested_delta_pf_values']=[exact(.1)]
                c.install(ctx,problem,[blocked]);self.assertEqual(len(ctx['candidates']),2)

    def test_scoped_audit_metadata_validates_only_covered_actual_step(self):
        module=load_script('pauli_split_cal_audit_request.py')
        reader,cfg,imp,package=pauli_fixture();binding=reader.read(package['binding_source'])
        binding['step_seeds']=[[7],[8]]
        request=binding['request'];summaries=[];receipts={}
        for i in range(2):
            prints={}
            for axis,native in(('X','cosine'),('Y','sine')):
                fingerprint=axis+str(i);prints[native]=dict(fingerprint=fingerprint,cache_reused=False)
                receipts[axis+str(i)]=dict(axis=axis,index=i,request=request,cohort=binding['cohort'],seeds=binding['step_seeds'][i],
                    record=dict(fingerprint=fingerprint,metrics={k:1 for k in module.A.METRICS}),artifacts_private=[],
                    native_cache=dict(path='/tmp/ARTIFICIAL_NO_QPY_'+fingerprint+'.json',sha256='TOY'))
            summaries.append(dict(index=i,result=dict(fingerprints=prints)))
        result=dict(status='FINITE_LD3_CAL_PATHS_COMPLETE',summaries=summaries)
        spec=dict(out='TOY',prefix=3,q=8,R=1,paths=2,cell='Q8R1',config_id=cfg['id'],config={'commit':'ARTIFICIAL','path':'CONFIG.json'},covered_rounds=list(range(1,13)),calibration_domain={'actual_step':exact(.2)})
        def raw(commit,path):
            if path.endswith('COMPILE_RECEIPTS.json'):return receipts
            if path.endswith('RESULT.json'):return result
            if path.endswith('BINDING.json'):return binding
            if path=='CONFIG.json':return cfg
            self.fail(path)
        with tempfile.TemporaryDirectory()as tmp,patch.object(module,'ROOT',Path(tmp)),patch.object(module.A,'git_json',side_effect=raw):
            module.build(spec,'ARTIFICIAL')
            data=json.loads((Path(tmp)/'TOY/INPUTS_AUDIT.json').read_text())
            self.assertEqual(sorted(map(int,data['round_T'])),list(range(1,13)))
            spec['covered_rounds']=list(range(13))
            with self.assertRaisesRegex(module.A.AuditStop,'ACTUAL_STEP_COVERAGE'):module.build(spec,'ARTIFICIAL')

    def test_multistep_production_wrapper_runs_every_selected_group_and_preserves_packages(self):
        module=load_script('run_pr_pauli_candidate_prices.py');calls=[]
        groups=[dict(request_id='CELL:early',covered_rounds=[0]),dict(request_id='CELL:later',covered_rounds=list(range(1,13)))]
        def leaf(request,execute=False,push=True):
            calls.append((request.get('group_request_id'),execute))
            if not execute:return dict(status='CALIBRATION_PLANNED_NOT_EXECUTED',calibration_requests=groups,needs=[])
            return dict(status='SCOPED_STEP_MODEL_READY',evidence_packages=[{'group':request['group_request_id']}])
        request=dict(scoped_registration=True,configuration={'id':'TOY'},directory='tmp/TOY',private_root='/tmp/TOY')
        with patch.object(module,'_run_one',side_effect=leaf):result=module.run(request,execute=True,push=False)
        self.assertEqual(result['status'],'STEP_MODELS_READY');self.assertEqual(len(result['evidence_packages']),2)
        self.assertEqual(calls,[(None,False),('CELL:early',True),('CELL:later',True)])

    def test_scoped_price_real_coordinator_chain_publishes_each_step_then_resumes_zero(self):
        module=load_script('run_pr_pauli_candidate_prices.py')
        with tempfile.TemporaryDirectory()as tmp:
            workspace=Path(tmp)/'owned';workspace.mkdir()
            for name in module.A_FILES+module.AUDIT_FILES:
                target=workspace/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'FIXED')
            cfg=dict(id='ARTIFICIAL_SCOPED',spec={'L_D':3})
            request=dict(workspace=str(workspace),configuration=cfg,prepared_item={'configuration':cfg},
                configuration_source={'commit':'INPUTS','path':'CONFIG.json'},directory='results/selected',
                price_quote=dict(configuration_id=cfg['id'],policy='MODEL',complete=False,rows=[]),
                registry_file=str(Path(tmp)/'registry.json'),private_root=str(Path(tmp)/'private'),
                scoped_registration=True,execution_source='NEW_METADATA_SOURCE')
            groups=[dict(request_id='CELL:'+str(i),covered_rounds=rounds,driver_request={'driver_spec':{}})
                    for i,rounds in enumerate(([0],list(range(1,13))))]
            calls=[];publications=[]
            def commit(w,paths,message):
                publications.append(message);return 'COMMIT'+str(len(publications))
            def command(w,cmd,log,marker=None):
                calls.append(cmd);Path(log).write_text('ARTIFICIAL_NO_WORKER')
                if 'scripts/prepare_pauli_calibration.py'in cmd:
                    module.OPS.write(cmd[cmd.index('--output')+1],dict(status='CALIBRATION_GROUPS_READY_NOT_EXECUTED',
                        calibration_requests=deepcopy(groups),needs=[],implementation={'fixture':'GROUP_IMPLEMENTATION'}))
                elif '-c'in cmd:
                    group=module.OPS.read(cmd[-4]);ops=module.OPS.read(cmd[-3])
                    spec=dict(prefix=3,base=ops['base'],private=ops['private'],cell=dict(q=8,R=1,T=.8 if group['covered_rounds']==[0]else 1.6),
                        covered_rounds=group['covered_rounds'],calibration_domain=dict(actual_step=exact(.1 if group['covered_rounds']==[0]else .2)))
                    module.OPS.write(cmd[-1],dict(driver_spec=spec))
                else:
                    spec=module.OPS.read(cmd[cmd.index('--request')+1])['driver_spec'];base=workspace/spec['base']
                    if '--freeze'in cmd:module.OPS.write(base/'BINDING.json',{'fixture':'BINDING'})
                    elif '--execute'in cmd:
                        module.OPS.write(base/'run01/RESULT.json',dict(status='FINITE_LD3_CAL_PATHS_COMPLETE',compiles=4,summaries=[{},{}],
                            source={'commit':'FIXED_WORKER'},binding={'commit':cmd[cmd.index('--execute')+1]}))
                    elif '--analyze'in cmd:
                        module.OPS.write(base/'PRICE_26AXIS.json',dict(status='PRICE_RULE_SUPPORTED_MODEL_REFERENCE'))
                        module.OPS.write(base/'audit_run01/AUDIT_RESULT.json',dict(rule_supported=True))
                return 0
            with patch.object(module.OPS,'owned_checkout_lock',lambda w:nullcontext()),patch.object(module.subprocess,'check_output',return_value=b'FIXED'),patch.object(module.OPS,'git',return_value='OWNED'),patch.object(module.OPS,'commit',side_effect=commit),patch.object(module.OPS,'publish',return_value='OWNED'),patch.object(module,'command',side_effect=command):
                first=module.run(request,execute=True,push=False);count=len(calls)
                again=module.run(request,execute=True,push=False)
            self.assertEqual(first['status'],'STEP_MODELS_READY',first);self.assertEqual(again['status'],'STEP_MODELS_READY')
            self.assertEqual(len(calls),count);self.assertEqual(len(first['evidence_packages']),2)
            self.assertEqual([g['preparation_group']['covered_rounds']for g in first['evidence_packages']],[[0],list(range(1,13))])
            self.assertTrue(all(g['raw_source']['commit']!=g['price_source']['commit']for g in first['evidence_packages']))
            self.assertEqual(sum('--execute'in cmd for cmd in calls),2)
            self.assertTrue(all(g['legacy_registry_modified']is False for g in first['group_results']))
            # A terminal scientific failure is not reopened by calling the
            # coordinator again, even before any sampling marker was written.
            failed_path=workspace/'results/selected/step_groups/0/PRICE_STATE.json'
            failed=module.OPS.read(failed_path);failed['status']='HELD_CALIBRATION_METADATA'
            module.OPS.write(failed_path,failed)
            with patch.object(module.OPS,'owned_checkout_lock',lambda w:nullcontext()),patch.object(module.subprocess,'check_output',return_value=b'FIXED'),patch.object(module,'command',side_effect=AssertionError('No scientific retry')):
                held=module.run(request,execute=True,push=False)
            self.assertEqual(held['status'],'HELD_CALIBRATION_METADATA')

    def test_df_family_metadata_missing_descriptor_does_not_start_science(self):
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion);c.directory=Path(tmp);c.state={'records':{}};c.settings={}
            request=dict(method='PR_DF',design=dict(L_D=8,formula='2nd',delta_pf=.05,kappa='2',K=2))
            with patch.object(c,'service',return_value=dict(status='NEEDS_DF_TAIL_DESCRIPTOR',missing=['descriptor']))as service,patch.object(c,'call',side_effect=AssertionError('No scientific descriptor')):
                out=c.prepare_family(request)
            self.assertEqual(out['status'],'NEEDS_FAMILY_PREPARATION');self.assertEqual(service.call_count,1)
            self.assertEqual(out['new_signal_calls'],0);self.assertEqual(out['new_price_calls'],0)


if __name__=='__main__':unittest.main(verbosity=2)


def test_exploratory_price_conflict_does_not_fan_out_into_direct_compiles():
    import pr_candidate_services as services
    from types import SimpleNamespace

    known = dict(m=0, axis='X', physical_id='axis-x', unit_price_RZ=10)
    missing = dict(m=0, axis='Y', physical_id='axis-y', unit_price_RZ=None,
                   needs={'entry_ids': ['competing-model']})
    plan = dict(groups=[dict(status='CALIBRATION_DESIGN_READY', axis_keys=[missing])],
                round_actions=[dict(m=0, axis='Y', action='CALIBRATE')])
    runner = SimpleNamespace(design_groups=lambda config, quote: plan)
    outcome = services.df_price_disposition({}, dict(rows=[known, missing]), runner, exploratory=True)
    assert outcome['groups'] == []
    assert outcome['round_actions'][0]['reasons'] == ['EXPLORATORY_PRICE_NOT_IDENTIFIED']
