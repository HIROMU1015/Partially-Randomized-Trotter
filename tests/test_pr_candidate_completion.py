"""Common B2 callback tests with artificial acquisition; no scientific providers."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import pr_candidate_completion as C,rpe_method_optimizer as O,rpe_priority_search as P


class CompletionTests(unittest.TestCase):
    def test_restored_intake_uses_checked_locators_and_preserves_typed_inputs(self):
        config = dict(commit='prepared-config', path='config.json')
        packet = dict(commit='prepared-packet', path='packet.json')
        checked_config = dict(config, commit='intake-config')
        checked_packet = dict(packet, commit='intake-packet')
        record = dict(configuration=config, signal=dict(packet_source=packet),
                      intake=dict(configuration=checked_config, signal_packet=checked_packet))
        objects = {'prepared-config': {'x': 1}, 'prepared-packet': {'signal': [1]},
                   'intake-config': {'x': 1}, 'intake-packet': {'signal': [1]}}
        with patch.object(C, 'value', side_effect=lambda ref: objects[ref['commit']]):
            self.assertEqual(C.intake_locators(record), (checked_config, checked_packet))
            objects['intake-packet'] = {'signal': [True]}
            with self.assertRaisesRegex(ValueError, 'canonical intake receipt'):
                C.intake_locators(record)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'DF price worker uses Linux fork')
    def test_df_price_bootstrap_keeps_the_worker_importable_for_fork_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner=Path(tmp)/'scripts'/'price.py'
            runner.parent.mkdir()
            runner.write_text("""from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def fixture():
    return 23

def main():
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('fork')) as pool:
        print(pool.submit(fixture).result(timeout=5))
    return 0
""")
            output=subprocess.check_output([sys.executable,'-B','-c',C._DF_PRICE_BOOTSTRAP,str(runner),tmp],
                text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')))
            self.assertEqual(output.strip(),'23')

    def test_saved_ready_resume_does_not_call_acquisition(self):
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion);c.directory=Path(tmp);c.state={'records':{}};c.lock=__import__('threading').RLock();c.state_path=Path(tmp)/'STATE.json'
            r=dict(method='PR_DF',configuration={'fixture':'ARTIFICIAL'})
            c.state['records'][C.key(r)]=dict(status='READY_FOR_SEARCH',configuration_id='ARTIFICIAL')
            with patch.object(c,'pauli_configuration',side_effect=AssertionError('no acquisition')):
                out=c.complete(r)
            self.assertTrue(out['resumed']);self.assertEqual(out['configuration_id'],'ARTIFICIAL')

    def test_prior_failure_lookup_survives_concurrent_candidate_registration(self):
        request = {'method': 'PR_DF', 'configuration': 'selected'}
        config = {'id': 'selected', 'spec': {}, 'rounds': []}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = root / 'fixture_registry.json'
            C.put(registry, {'entries': []})
            with patch.object(C, 'ROOT', root):
                completion = C.Completion(root / 'run', {'source': 'fixture', 'price_registry_file': registry})
                records = completion.state['records']
                records.update({
                    name: dict(key=name, method='PR_DF', status='HELD_TECHNICAL_FAILURE',
                               scientific_failure=True, configuration={'path': name})
                    for name in ('unrelated', 'physical_twin')
                })

                def read_previous(locator):
                    if locator['path'] == 'unrelated':
                        def register():
                            with completion.lock:
                                records['late'] = {'status': 'SELECTED'}
                        new_candidate = threading.Thread(target=register)
                        new_candidate.start()
                        new_candidate.join(timeout=2)
                        self.assertFalse(new_candidate.is_alive())
                        return dict(config, spec={'different': True})
                    return config

                with patch.object(completion, 'saved_df_descriptor', return_value=None), \
                     patch.object(completion, 'service', return_value=dict(status='CONFIGURATION_READY', configuration=config)), \
                     patch.object(completion, 'initial_quote', side_effect=AssertionError('prior failure must hold')), \
                     patch.object(C, 'publish', return_value='fixed'), patch.object(C, 'value', side_effect=read_previous):
                    result = completion.complete(request)
                self.assertEqual(result['reason'], 'PRIOR_PHYSICAL_PREPARATION_FAILED_NO_AUTOMATIC_RETRY:physical_twin')
                self.assertTrue(result['scientific_failure'])
                self.assertIn('late', records)

    def test_published_v2_missing_order_not_reimplemented(self):
        candidates=[dict(name='PR_DF:'+str(n),ready=False,configuration_id=None,key=dict(L_D=n),needs=['SIGNAL'])for n in (9,6,8,5)]
        ready=dict(key=dict(L_D=7));states={'saved':type('State',(),{'c':ready,'best':(Fraction(10),None,None)})()}
        rows=P.new_ld_requests(candidates,states,Fraction(10),{}, {})
        self.assertEqual([r['configuration']['L_D']for r in rows],[6,8,5,9])
        self.assertTrue(all(r['kind']=='NEW_L_D'for r in rows))

    def test_absolute_kappa_is_translated_against_actual_base_without_fixing_width_formula(self):
        request=dict(method='PR_PAULI',design=dict(L_D=4,formula='4th(m5_best)',delta_over_h='2',kappa='2',K=2))
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion);c.settings={}
            base=dict(spec=dict(kappa=dict(value=dict(numerator=7,denominator=1))))
            result=dict(spec=dict(kappa=dict(value=dict(numerator=2,denominator=1))))
            def call(cmd,d,tag,*args,**kwargs):return dict(items=[dict(status='GENERATED',configuration=base if tag=='A_BASE'else result)])
            with patch.object(c,'call',side_effect=call):config,_=c.pauli_configuration(request,Path(tmp))
            sent=json.loads((Path(tmp)/'A_DESIGN.REQUEST.json').read_text())['items'][0]['changes']
            self.assertEqual(sent['kappa_multiplier'],'2/7')
            self.assertEqual(sent['formula'],'4th(m5_best)');self.assertEqual(sent['delta_over_h'],'2')
            self.assertEqual(config,result)

    def test_pauli_signal_evidence_change_preserves_old_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = Path('attempt')
            previous = root/relative
            previous.mkdir()
            config = {'id': 'fixed'}
            C.put(previous/'CONFIGURATION.json', config)
            C.put(previous/'SAVED_PACKETS.json', ['old'])
            inputs = C.Completion.pauli_signal_inputs
            refreshed, packets = inputs(root, relative, config, None, ['new'])
            self.assertEqual(refreshed.parent, relative)
            self.assertEqual(packets, ['new'])
            self.assertEqual(C.read(previous/'SAVED_PACKETS.json'), ['old'])
            (previous/'SIGNALS_STARTED.json').touch()
            with self.assertRaisesRegex(C.Held, 'NO_AUTOMATIC_RETRY'):
                inputs(root, relative, config, None, ['new'])
            C.put(previous/'SIGNAL_STATE.json', {'status': 'SIGNALS_READY'})
            self.assertEqual(inputs(root, relative, config, None, ['new']), (relative, ['old']))

    def test_pauli_price_uses_frozen_preparation_only_for_unchanged_calibration_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'PREPARATION.json'
            saved = dict(configuration={'id': 'fixed'}, schedule=[1], signal_plan={'REQUEST': 13},
                         preparation_status='GENERATED_CONFIGURATION')
            C.put(path, saved)
            current = dict(saved, signal_plan={'REUSE': 12},
                           preparation_status='REUSED_LOCAL_METADATA_CONFIGURATION')
            self.assertEqual(C.Completion.pauli_price_preparation(path, current), saved)
            changed = dict(current, schedule=[2])
            self.assertEqual(C.Completion.pauli_price_preparation(path, changed), changed)

    def test_pauli_missing_saved_input_is_held_before_science(self):
        request = dict(design=dict(L_D=4))
        paths = {'/fixed/receipt.json': '/local/receipt.json'}
        with tempfile.TemporaryDirectory() as tmp:
            completion = C.Completion.__new__(C.Completion)
            completion.settings = {'pauli_saved_inputs': paths}
            detail = dict(status='NEEDS_INPUT_SOURCE', missing_inputs=['receipt'], items=[])
            with patch.object(completion, 'call', return_value=detail) as call:
                with self.assertRaisesRegex(C.Held, 'A_PREPARATION:NEEDS_INPUT_SOURCE.*receipt'):
                    completion.pauli_configuration(request, Path(tmp))
            command = call.call_args.args[0]
            self.assertEqual(Path(command[2]).name, 'rpe_pauli_saved_inputs.py')
            self.assertEqual(json.loads(command[3]), paths)
            self.assertFalse(call.call_args.kwargs['science'])

    def test_legacy_pauli_missing_input_is_rechecked_only_without_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'receipt.json'
            source.touch()
            c = C.Completion.__new__(C.Completion)
            c.settings = {'pauli_saved_inputs': {'/old/receipt.json': str(source)}}
            record = dict(method='PR_PAULI', reason='IndexError: list index out of range')
            C.put(root/'A_BASE.json', dict(status='NEEDS_INPUT_SOURCE', items=[],
                                         signal_price_Q3_evaluation_performed=False))
            C.put(root/'A_BASE.STARTED.json', dict(science=False, source='fixed'))
            C.put(root/'A_BASE.RETURN.json', dict(returncode=0, signal=None, source='fixed'))
            source.unlink()
            self.assertFalse(c.recoverable_pauli_input_hold(record, root))
            # The mapped input is external to the attempt evidence directory.
            attempt = root/'attempt'
            attempt.mkdir()
            for path in root.glob('A_BASE.*'):
                path.rename(attempt/path.name)
            source.touch()
            self.assertTrue(c.recoverable_pauli_input_hold(record, attempt))
            (attempt/'PAULI_SIGNALS.STARTED.json').touch()
            self.assertFalse(c.recoverable_pauli_input_hold(record, attempt))

    def test_registered_no_drop_selected_for_ld7_ld8_ld9_without_science(self):
        for prefix in (7,8,9):
            with self.subTest(prefix=prefix),tempfile.TemporaryDirectory()as tmp:
                c=C.Completion.__new__(C.Completion);c.settings={};calls=[]
                base=dict(spec=dict(L_D=prefix,kappa=dict(value=dict(numerator=7,denominator=1))))
                def call(cmd,d,tag,*args,**kwargs):
                    calls.append((cmd,kwargs))
                    self.assertIn('--p0-builder',cmd);self.assertEqual(cmd[cmd.index('--p0-builder')+1],'no-drop')
                    self.assertNotIn('--prepare-missing',cmd);self.assertFalse(kwargs.get('science'))
                    return dict(items=[dict(status='GENERATED',preparation_status='REUSED_CONFIGURATION',
                        dictionary_builder_id='P0_NO_INTERMEDIATE_DROP_V1',configuration=base)])
                with patch.object(c,'call',side_effect=call):
                    _,item=c.pauli_configuration(dict(design=dict(L_D=prefix,formula='4th(m5_best)',delta_over_h='2',kappa='3',K=2)),Path(tmp))
                sent=json.loads((Path(tmp)/'A_DESIGN.REQUEST.json').read_text())['items'][0]['changes']
                self.assertEqual(sent,dict(formula='4th(m5_best)',K=2,kappa_multiplier='3/7',delta_over_h='2'))
                self.assertEqual(item['dictionary_builder_id'],'P0_NO_INTERMEDIATE_DROP_V1')
                self.assertEqual(len(calls),2)

    def test_descriptor_envelope_is_passed_to_all_signal_stages(self):
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion);c.settings={};c.state={'records':{}}
            config=dict(id='fixture',spec=dict(L_D=4,binding={'H':'fixed'}))
            prepared=dict(descriptor=dict(status='READY',descriptor={'tail':'own'},source={'commit':'fixed','path':'DESCRIPTOR.json'}))
            def call(cmd,d,stage,output,**kwargs):
                req=json.loads((d/'PAULI_SIGNAL_REQUEST.json').read_text())
                self.assertEqual(req['tail_descriptor'],dict(prefix=4,binding={'H':'fixed'},descriptor={'tail':'own'},source={'commit':'fixed','path':'DESCRIPTOR.json'}))
                self.assertTrue(req['directory'].endswith('/descriptor_envelope_v2'))
                return dict(status='SIGNALS_READY')
            with patch.object(c,'call',side_effect=call):c.pauli_signals(config,prepared,Path(tmp))

    def test_df_descriptor_is_shared_across_kappa_requests(self):
        c=C.Completion.__new__(C.Completion)
        loc=dict(commit='fixed',path='DESCRIPTOR.json')
        c.state={'records':{'saved':dict(method='PR_DF',design={'L_D':6},request={'tail_descriptor':loc})}}
        self.assertEqual(c.saved_df_descriptor(dict(design={'L_D':6,'kappa':'3'})),loc)
        self.assertIsNone(c.saved_df_descriptor(dict(design={'L_D':5,'kappa':'3'})))
        canonical=dict(spec=dict(split=dict(lambda_source=loc)))
        self.assertEqual(c.saved_df_descriptor(dict(configuration=canonical)),loc)
        c.settings={'saved_df_descriptors':{'8':loc}}
        self.assertEqual(c.saved_df_descriptor(dict(design={'L_D':8,'kappa':'2'})),loc)
        self.assertIsNone(c.saved_df_descriptor(dict(design={'L_D':7,'kappa':'2'})))

    def test_scoped_price_conflict_delegates_to_planner_and_complete_price_skips_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            c=C.Completion.__new__(C.Completion)
            c.directory=path
            c.registry=path/'PRICE_RULES.json'
            C.put(c.registry, {'entries': []})
            need=dict(m=3,axis='X',unit_price_RZ=None,needs=dict(reasons=[C.PRICE_SOURCE_CONFLICT],
                                                               entry_ids=['scoped1','scoped2']))
            quote=dict(complete=False,rows=[need])
            def plans(*args):
                raise C.Held('PRICE_EVIDENCE_REQUIRED: scoped conflict')
            with patch.object(c,'path',return_value='FIXTURE'),patch.object(c,'quote_with_steps',side_effect=lambda cfg,q:q), \
                 patch.object(c,'df_price',side_effect=plans) as calibrate,patch.object(c,'ingest_price_results'):
                with self.assertRaisesRegex(C.Held, 'scoped conflict'):
                    c._price({}, {}, {}, {'method':'PR_DF'},path,quote)
                calibrate.assert_called_once()
                complete=dict(complete=True,rows=[dict(need,unit_price_RZ=5)])
                self.assertEqual(c._price({}, {}, {}, {'method':'PR_DF'},path,complete),complete)
                calibrate.assert_called_once()

    def test_ready_unpriced_df_retains_q3_and_reuses_price_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'candidate'
            folder.mkdir()
            c = C.Completion.__new__(C.Completion)
            c.directory = root
            c.settings = {'exploratory_cost': True}
            old = dict(request={}, method='PR_DF', status='READY_FOR_SEARCH', prices=None,
                       configuration_id='candidate', configuration={'fixed': True},
                       directory='candidate', signal={'saved': True}, intake={'binding_id': 'saved'})
            c.state = {'records': {'candidate': old}}
            from types import SimpleNamespace
            store = SimpleNamespace(row=lambda job: {'state': 'QUEUED'})
            c.stage_executor = SimpleNamespace(executor=SimpleNamespace(store=store))
            with patch.object(C,'ROOT',root), patch.object(C,'value',return_value={}), \
                 patch.object(c,'price_registry_digest',return_value='prices'), \
                 patch.object(c,'df_price_evidence_snapshot',return_value='evidence'), \
                 patch.object(c,'initial_quote',return_value={'complete': False}), \
                 patch.object(c,'df_price',side_effect=C.PreparationWait('waiting',job_id='original')) as price, \
                 patch.object(c,'save'):
                for _ in range(3):
                    c.prepare_shortlist_prices({'candidate'}, stop_requested=lambda: False)
            self.assertEqual(price.call_count, 1)
            self.assertEqual(old['shortlist_price_attempt']['job_id'], 'original')
            self.assertEqual(old['signal'], {'saved': True})
            self.assertEqual(old['status'], 'READY_FOR_SEARCH')

    def test_new_registry_recovers_price_hold_without_scientific_retry(self):
        with tempfile.TemporaryDirectory()as tmp:
            c=C.Completion.__new__(C.Completion);c.directory=Path(tmp);c.registry=Path(tmp)/'registry.json'
            C.put(c.registry,{'entries':['new published evidence']})
            r=dict(key='selected',status='HELD_WITH_REASON',reason='price missing',configuration={'commit':'cfg','path':'cfg'},signal={'packet':'fixed'},intake={'binding':'fixed'},registry_at_hold='old')
            cfg=dict(spec=dict(L_D=6,formula='2nd',kappa={'value':'2'},delta_pf='0.1'),rounds=[])
            q=dict(complete=True,rows=[{'unit_price_RZ':{'numerator':7,'denominator':1}}])
            with patch.object(C,'ROOT',Path(tmp)),patch.object(c,'service',return_value=q)as service,patch.object(c,'save'),patch.object(C,'publish',return_value='published'),patch.object(C,'value',return_value=cfg):
                out=c.refresh_price_hold(r)
            self.assertEqual(out['status'],'READY_FOR_SEARCH')
            self.assertEqual(out['previous_hold']['reason'],'price missing')
            self.assertEqual(service.call_args.args[0],'quote')
            self.assertEqual(out['recovered_by'],'METADATA_REQUERY_NO_SCIENTIFIC_RETRY')

    def test_completion_does_not_treat_pf_as_a_df_scientific_request(self):
        c=C.Completion.__new__(C.Completion)
        with self.assertRaisesRegex(ValueError,'PR_CANDIDATE_COMPLETION_ONLY'):
            c.complete(dict(method='FULL_PF'))

    def test_caller_published_registry_is_used_without_manual_registration(self):
        with tempfile.TemporaryDirectory()as tmp:
            registry=dict(entries=[dict(name='B2_PUBLISHED_NEW_STEP',model_source={'commit':'fixed','path':'model'})])
            with patch.object(C,'value',return_value=registry)as reader:
                c=C.Completion(Path(tmp),dict(source='fixed-code',price_registry_source={'commit':'b2','path':'registry'}))
            self.assertEqual(C.read(c.registry),registry)
            self.assertEqual(reader.call_args.args[0],{'commit':'b2','path':'registry'})

    def test_normal_cli_contains_callback_and_df_route_is_generic(self):
        spec=importlib.util.spec_from_file_location('_completion_cli',ROOT/'scripts/run_rpe_method_optimizer.py');cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        import inspect
        self.assertIn('prepare_for_search',inspect.getsource(cli.round_mode))
        spec=importlib.util.spec_from_file_location('_completion_q3',ROOT/'scripts/run_rpe_optimizer_q3_job.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
        inputs=dict(cases=[dict(evaluation_target={'fixture':1})],statistical_contract={},options={},adopted_evaluation={},acceleration={})
        r=dict(request_id='fixture',configuration_id='cfg',N_X=[1]*13,N_Y=[2]*13,route=dict(kind='prdf_d2_common_entry',contract_template={'fixture':1},configurations={'cfg':dict(configuration={'fixture':1},signal_packet={'fixture':2},binding_id='b',total_configuration_id='t')}))
        with patch.object(q,'git_json',return_value=inputs):case,actual=q.build(ROOT,r)
        self.assertEqual(case['signal_packet_kind'],'A_BUNDLE_REQUIRES_EXISTING_K_ACCEPT_SAVED')
        self.assertEqual(case['original_configuration_id'],'cfg');self.assertEqual(case['allocation']['shots'][-1]['N_Y'],2)


class Continuation(unittest.TestCase):
    def test_next_selected_candidate_uses_same_executor_and_total_budget(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory()as tmp:
            completion=SimpleNamespace(directory=Path(tmp),state={'records':{'old':{'status':'READY_FOR_SEARCH','configuration_id':'a'}}})
            ex=SimpleNamespace(routes={},expected={},launched=lambda m:1)
            spec=dict(methods=['PR_DF'],max_new_q3_per_method={'PR_DF':2})
            result={'PR_DF':dict(stop_reason='INPUTS',new_calls=1,input_requests=[{'kind':'NEW_L_D'}])}
            context={'candidates':[],'expected':{}}
            def prepare(*args,**kwargs):
                self.assertIs(kwargs['completion'],completion)
                self.assertEqual(kwargs['selection_results'],result)
                completion.state['records']['next']=dict(status='READY_FOR_SEARCH',configuration_id='b')
                return context,{'PR_DF':['original_route']},completion
            final={'PR_DF':dict(stop_reason='BUDGET',new_calls=1,input_requests=[])}
            with patch.object(C,'prepare_for_search',side_effect=prepare),patch.object(O,'apply_published'),patch.object(O,'optimize',return_value=final)as opt:
                out=C.continue_selected_preparation(ROOT,{},context,spec,tmp,completion,ex,result)
            self.assertIs(out,final);self.assertIs(opt.call_args.kwargs['executor'],ex)
            self.assertEqual(opt.call_args.kwargs['max_new'],spec['max_new_q3_per_method'])
            self.assertEqual(ex.routes,{'PR_DF':['original_route']})

if __name__=='__main__':unittest.main(verbosity=2)
