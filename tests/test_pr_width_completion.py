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
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pr_candidate_completion as C,rpe_method_optimizer as O
from trottertracks.pr_step_price_registry import StepPriceRegistry
from test_pr_step_price_registry import pauli_fixture,df_direct_fixture,missing_quote,config,exact


def load_script(name):
    spec=importlib.util.spec_from_file_location('_width_'+name,ROOT/'scripts'/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def enrich(cfg):
    cfg['spec']['kappa']=dict(value=exact(2));return cfg


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
                if mode=='intake':return dict(binding_id='BIND',total_configuration_id='TOTAL')
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
        rec=dict(method='PR_PAULI',status='READY_FOR_SEARCH',configuration=newref,
            signal={'packet_source':{'commit':'ARTIFICIAL','path':'PACKET'}},intake={'binding_id':'B','total_configuration_id':'T'},prices=[exact(3)]*26,quote=qref)
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
