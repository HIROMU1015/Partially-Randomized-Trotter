"""Relocated PR metadata and descriptor source routing; synthetic inputs only."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_candidate_completion as C

spec=importlib.util.spec_from_file_location('descriptor_path_fixture',ROOT/'scripts/derive_pr_df_candidate_descriptor.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
spec=importlib.util.spec_from_file_location('metadata_path_fixture',ROOT/'scripts/pr_candidate_services.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)


class ControllerPaths(unittest.TestCase):
    def test_metadata_after_completed_descriptor_can_resume_without_rederivation(self):
        from trottertracks.pr_df_descriptor_extension import BINDING_KEYS, INDEX, DERIVATION
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'old';p=parent/'completion/candidates/fixed';p.mkdir(parents=True)
            record=dict(method='PR_DF',status='HELD_WITH_REASON',reason='STAGE_FAILED:df-plan:None:None:'+str(p/'df-plan.log'))
            for stage in ('df-generate','INITIAL_QUOTE','DF_GENERATE_WITH_DESCRIPTOR','df-plan'):
                C.put(p/(stage+'.STARTED.json'),dict(science=False))
                C.put(p/(stage+'.RETURN.json'),dict(returncode=1 if stage=='df-plan'else 0,signal=None))
            binding={k:'ARTIFICIAL'for k in BINDING_KEYS}
            source=dict(commit='SOURCE',caller='descriptor.py');configuration=dict(commit='CONFIG',path='config.json')
            C.put(p/'CONFIGURATION.json',dict(spec=dict(L_D=6,binding=binding)))
            C.put(p/'DF_TAIL_DESCRIPTOR.STARTED.json',dict(source=source,configuration=configuration,prefix=6))
            C.put(p/'DF_TAIL_DESCRIPTOR.json',dict(schema='PR_DF_TAIL_DESCRIPTOR_V1',prefix=6,
                descriptor=dict(lambda_R=.25,scalar=.5),binding=binding,execution_source=source,
                configuration_source=configuration,saved_index=INDEX,derivation_source=DERIVATION,input_source={'fixed':True}))
            C.put(p/'DF_DESCRIPTOR.RETURN.json',dict(returncode=0,signal=None,automatic_retry=False,source='SOURCE',job_id='fixed'))
            (p/'df-plan.log').write_text("validated_packet_index\nKeyError: 'means'\n")
            self.assertTrue(C.Completion.metadata_path_failure(record,p))
            C.put(parent/'STOP_REQUESTED.json',dict(reason='USER_REQUESTED_STOP'))
            C.put(parent/'LAUNCH.json',dict(invocation_id='same'))
            C.put(parent/'EXIT.json',dict(invocation_id='same',returncode=0))
            C.put(parent/'distributed/relay/STATUS.json',dict(stopped=True))
            self.assertTrue(C.Completion.metadata_path_failure(record,root/'new/completion/candidates/fixed'))
            C.put(p/'DF_TAIL_DESCRIPTOR.STOP.json',dict(reason='FAILED'))
            self.assertFalse(C.Completion.metadata_path_failure(record,p))
            self.assertFalse(C.Completion.metadata_path_failure(record,root/'new/completion/candidates/fixed'))

    def test_missing_committed_metadata_is_repaired_only_with_identical_recovered_twin(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'candidates/current';p.mkdir(parents=True)
            config=dict(id='generated',spec=dict(L_D=9,formula='2nd',kappa=dict(value='1'),delta_pf='1/10'),rounds=[])
            request=dict(method='PR_DF',design=dict(L_D=9,formula='2nd',kappa='1',delta_pf='1/10'))
            locator='a'*40+':search-evidence/old/completion/candidates/fixed/CONFIGURATION.json'
            record=dict(method='PR_DF',status='HELD_TECHNICAL_FAILURE',request=request,
                reason="CalledProcessError: Command '"+str(['git','show',locator])+"' returned non-zero exit status 128.")
            C.put(p/'df-generate.STARTED.json',dict(science=False))
            C.put(p/'df-generate.RETURN.json',dict(returncode=0,signal=None))
            C.put(p/'df-generate.json',dict(status='CONFIGURATION_READY',configuration=config))
            C.put(p/'CONFIGURATION.json',config);C.put(p/'df-generate.INPUT.json',request)
            (p/'df-generate.log').write_text('metadata completed')
            C.put(p.parent.parent/'STATE.json',dict(records=dict(previous=dict(method='PR_DF',
                configuration=dict(commit='a'*40,path=locator.split(':',1)[1]),configuration_id='generated',request=request))))
            with patch.object(C,'git',return_value=json.dumps(config)):
                self.assertTrue(C.Completion.metadata_path_failure(record,p))
                other=dict(config,spec=dict(config['spec'],L_D=8))
                with patch.object(C,'git',return_value=json.dumps(other)):
                    self.assertFalse(C.Completion.metadata_path_failure(record,p))
                C.put(p/'DF_SIGNALS.STARTED.json',dict(science=True))
                self.assertFalse(C.Completion.metadata_path_failure(record,p))

    def test_service_carries_saved_path_map_without_changing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            c=C.Completion.__new__(C.Completion)
            c.directory=Path(directory);c.source='OLD'
            c.settings=dict(pauli_saved_inputs={'/old/context':'/saved/context'},
                            preparation_path_map={'/old/snapshot':'/saved/snapshot'},
                            df_descriptor_source='NEW')
            c.call=Mock(return_value={})
            request={'configuration':'fixed'}
            c.service('df-plan',request,Path(directory))
            carried=json.loads((Path(directory)/'df-plan.INPUT.json').read_text())
            self.assertEqual(carried['preparation_path_map'],{'/old/context':'/saved/context','/old/snapshot':'/saved/snapshot'})
            self.assertEqual(request,{'configuration':'fixed'})
            self.assertEqual(c.stage_source('DF_DESCRIPTOR'),'NEW')
            self.assertEqual(c.stage_source('DF_SIGNALS'),'OLD')

    def test_metadata_entry_sets_map_before_df_provider(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{},clear=False):
            p=Path(directory);inp=p/'input.json';out=p/'output.json'
            inp.write_text(json.dumps(dict(configuration={'fixed':True},preparation_path_map={'/old/context':'/saved/context'})))
            def entry(*args):
                self.assertEqual(json.loads(os.environ['RPE_PREPARATION_PATH_MAP'])['/old/context'],'/saved/context')
                return SimpleNamespace(plan=lambda cfg:dict(status='ARTIFICIAL_METADATA'))
            with patch.object(sys,'argv',['service','df-plan','--input',str(inp),'--output',str(out)]),patch.object(S,'df_entry',side_effect=entry):
                S.main()
            self.assertEqual(json.loads(out.read_text())['status'],'ARTIFICIAL_METADATA')

    def test_descriptor_uses_same_frozen_loader_with_relocated_hashed_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{},clear=False):
            p=Path(directory);context=p/'context.json';binding=dict(system='ARTIFICIAL',rank=10)
            context.write_text(json.dumps(dict(binding=binding)))
            old_index=dict(T20_reference_files=dict(H4=dict(path='/old/context.json',sha256=hashlib.sha256(context.read_bytes()).hexdigest())),
                saved_inputs=dict(H4=dict(inputs=dict(snapshot=dict(path='/old/snapshot.npz')))))
            namespace=dict(p=SimpleNamespace(same=lambda a,b:a==b))
            exec("def load_context(root,binding):\n    return p.SavedArtifacts(root).read(None),binding\n",namespace)
            completion=SimpleNamespace(load_context=namespace['load_context'])
            os.environ['RPE_PREPARATION_PATH_MAP']=json.dumps({'/old':str(p)})
            with patch.object(D,'read',return_value=deepcopy(old_index)):
                loader,original,index=D.saved_context(completion,dict(spec=dict(binding=binding)))
            loaded,_=loader(ROOT,binding)
            self.assertEqual(loaded['saved_inputs']['H4']['inputs']['snapshot']['path'],str(p/'snapshot.npz'))
            self.assertEqual(index,old_index)  # original input provenance retained
            self.assertEqual(original['binding'],binding)

    def test_metadata_repair_rejects_any_scientific_or_ambiguous_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);record=dict(method='PR_DF',status='HELD_WITH_REASON',reason='STAGE_FAILED:df-plan:None')
            for stage in ['df-generate','INITIAL_QUOTE','df-plan']:
                C.put(p/(stage+'.STARTED.json'),dict(science=False))
                C.put(p/(stage+'.RETURN.json'),dict(returncode=1 if stage=='df-plan'else 0,signal=None))
            (p/'df-plan.log').write_text("FileNotFoundError: [Errno 2] No such file or directory: '/old/context'\n")
            self.assertTrue(C.Completion.metadata_path_failure(record,p))
            (p/'df-plan.log').write_text("validated_packet_index\nKeyError: 'means'\n")
            self.assertTrue(C.Completion.metadata_path_failure(record,p))
            (p/'df-plan.log').write_text("other_function\nKeyError: 'means'\n")
            self.assertFalse(C.Completion.metadata_path_failure(record,p))
            (p/'df-plan.log').write_text("validated_packet_index\nKeyError: 'means'\n")
            C.put(p/'DF_SIGNALS.STARTED.json',dict(science=True))
            self.assertFalse(C.Completion.metadata_path_failure(record,p))
            (p/'DF_SIGNALS.STARTED.json').unlink()
            (p/'df-plan.RETURN.json').unlink()
            self.assertFalse(C.Completion.metadata_path_failure(record,p))

    def test_explicit_metadata_repair_preserves_old_failure_and_scientific_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);old=p/'old';identifier='metadata';candidate=old/'candidates'/identifier
            candidate.mkdir(parents=True)
            record=dict(method='PR_DF',status='HELD_WITH_REASON',reason='STAGE_FAILED:df-plan:None',scientific_failure=True)
            for stage in ['df-generate','INITIAL_QUOTE','df-plan']:
                C.put(candidate/(stage+'.STARTED.json'),dict(science=False))
                C.put(candidate/(stage+'.RETURN.json'),dict(returncode=1 if stage=='df-plan'else 0,signal=None))
            (candidate/'df-plan.log').write_text("FileNotFoundError: [Errno 2] No such file or directory: '/old/context'\n")
            scientific=dict(method='PR_DF',status='HELD_WITH_REASON',reason='NUMERICAL_GATE',scientific_failure=True)
            prior=dict(records={identifier:record,'scientific':scientific})
            state=old/'STATE.json';C.put(state,prior);original=state.read_bytes()
            registry=p/'registry.json';C.put(registry,dict(entries=[]))
            settings=dict(source='NEW',price_registry_file=str(registry),protected_stops_from=[dict(commit='OLD',path='state.json')],
                metadata_repair_from=[dict(state_file=str(state),sha256=hashlib.sha256(original).hexdigest(),record_keys=[identifier])])
            with patch.object(C,'value',return_value=prior):c=C.Completion(p/'new',settings)
            self.assertNotIn(identifier,c.state['records'])
            self.assertEqual(c.state['metadata_repairs'][identifier]['original_record'],record)
            self.assertEqual(c.state['records']['scientific']['reason'],'NUMERICAL_GATE')
            self.assertEqual(state.read_bytes(),original)


if __name__=='__main__':unittest.main()
