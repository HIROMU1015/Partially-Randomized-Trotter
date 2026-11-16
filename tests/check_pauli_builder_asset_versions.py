"""Builder-aware preparation/cache tests with artificial JSON sources only.

No real operator, factor, mean, price or circuit is evaluated. Existing saved
candidate tests cover B's mapper; these fixtures isolate version selection.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pauli_candidate_preparation as A


class ArtificialSources:
    def __init__(self,folder):
        self.folder=Path(folder);self.binding=dict(system='ARTIFICIAL',hamiltonian='TOY_H',state='TOY_STATE',snapshot='TOY_INPUT')
        self.data={};self.registry=self.folder/'REGISTRY.json'
        context=self.folder/'CONTEXT.json';context.write_text(json.dumps(dict(prefixes={})))
        receipt=self.folder/'RECEIPT.json';receipt.write_text(json.dumps(dict(rank=10,
            arrays=dict(lambdas=dict(shape=[10]),g_matrices=dict(shape=[10,2,2])))))
        self.meta=dict(binding=self.binding,context_source=dict(path=str(context),sha256='ARTIFICIAL'),
            inputs={'H4_tol_receipt.json':dict(path=str(receipt))})
        self.registry.write_text(json.dumps(dict(entries=[])))

    def read(self,root,ref):
        key=(ref['commit'],ref['path'])
        if key in self.data:value=deepcopy(self.data[key])
        elif ref['commit']==A.VARIANTS:
            value=dict(id='ARTIFICIAL_VARIANT',spec=dict(binding=self.binding,L_D=3,split={}))
        else:raise FileNotFoundError('Missing artificial source '+str(key))
        for part in ref.get('selector',[]):value=value[part]
        return value

    def product(self,builder,*,prefix=7,suffix=''):
        tag=('OLD' if builder==A.LEGACY_BUILDER else 'NEW')+suffix
        payload=dict(prefix=prefix,binding=self.binding,lambda_P=.25,pauli_identity=.125,saved_DF_scalar=.125,
            mapping='ARTIFICIAL_ONLY',terms=[dict(label_qiskit='Z',coefficient=.25)],
            collector_metadata=dict(coefficient_cutoff=0.,fixture_only=True))
        if builder==A.NO_DROP_BUILDER:
            payload['builder']=dict(id=builder,source=dict(commit='ARTIFICIAL_BUILDER'+suffix,path='builder.py'))
        source=dict(commit='ARTIFICIAL',path=tag+'_DICTIONARY.json')
        result_source=dict(commit='ARTIFICIAL',path=tag+'_P0.json')
        descriptor_source=dict(commit='ARTIFICIAL',path=tag+'_DESCRIPTOR.json')
        digest=A.E.L.canonical_sha(payload);metrics=dict(ARTIFICIAL_tail_residual=0.)
        row=dict(prefix=prefix,status='P0_OPERATOR_DICTIONARY_GATES_PASS_NOT_P1_OR_RPE_QUALIFIED',
            dictionary_sha256=digest,identity_metrics=metrics,saved_DF_descriptor=dict(scalar=.125,lambda_R=.5))
        result=dict(status='P0_LD%d_DICTIONARY_COMPLETE_NOT_P1_OR_RPE_QUALIFIED'%prefix,binding=self.binding,
            prefixes=[row],checks=[dict(name='ARTIFICIAL_ONLY',value=0.,threshold=1e-9,PASS=True)])
        if builder==A.NO_DROP_BUILDER:
            result['builder']=deepcopy(payload['builder']);row['builder']=deepcopy(payload['builder'])
        self.data[source['commit'],source['path']]=deepcopy(payload)
        self.data[result_source['commit'],result_source['path']]=result
        self.data[descriptor_source['commit'],descriptor_source['path']]=dict(scalar=.125,lambda_R=.5)
        return dict(status='SPLIT_READY',item=dict(payload=payload,dictionary_sha256=digest,source=source,
            P0_result_source=result_source,P0_metrics=metrics),descriptor=dict(status='ARTIFICIAL_SAVED',
                descriptor=dict(scalar=.125,lambda_R=.5),source=descriptor_source))

    def guards(self):
        stack=ExitStack()
        for obj,name in ((np,'load'),(A.E.p0,'load_saved'),(A.E.provider,'evaluate')):
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('No scientific leaf')))
        stack.enter_context(patch.object(A.E.p0,'metadata',return_value=deepcopy(self.meta)))
        stack.enter_context(patch.object(A.E,'SAVED_PACKETS',()))
        stack.enter_context(patch.object(A.E,'read_ref',side_effect=self.read))
        stack.enter_context(patch.object(A.E.L,'read',return_value=dict(entries=[])))
        stack.enter_context(patch.object(A,'DICTIONARY_REGISTRY',str(self.registry)))
        stack.enter_context(patch.object(A,'generator',return_value=(SimpleNamespace(validate=lambda c:None),None)))
        stack.enter_context(patch.object(A.Assets,'baseline_from_item',side_effect=lambda p,item:dict(_item=item)))
        stack.enter_context(patch.object(A.E,'dictionary_item',side_effect=lambda root,c:deepcopy(c['_item'])))
        return stack


class BuilderVersions(unittest.TestCase):
    def test_old_cache_and_new_cache_are_both_retained_and_fresh_entry_selects_correct_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=ArtificialSources(tmp);old=f.product(A.LEGACY_BUILDER);new=f.product(A.NO_DROP_BUILDER)
            cache=Path(tmp)/'CACHE.json'
            cache.write_text(json.dumps(dict(binding=f.binding,splits={'7':old},configurations=[])))
            with f.guards():
                assets=A.Assets(ROOT,cache=cache,builder_id=A.NO_DROP_BUILDER)
                self.assertIsNone(assets.dictionary(7))
                self.assertEqual(assets.descriptor(7)['descriptor'],old['descriptor']['descriptor'])
                assets.save_split(7,new)
                saved=json.loads(cache.read_text())
                self.assertEqual(saved['splits']['7'],old)
                self.assertEqual(saved['split_versions']['7'][A.NO_DROP_BUILDER],new)
                auto=A.Assets(ROOT,cache=cache)
                explicit_old=A.Assets(ROOT,cache=cache,builder_id=A.LEGACY_BUILDER)
                explicit_new=A.Assets(ROOT,cache=cache,builder_id=A.NO_DROP_BUILDER)
                self.assertEqual(auto.dictionary(7),new['item'])
                self.assertEqual(explicit_old.dictionary(7),old['item'])
                self.assertEqual(explicit_new.dictionary(7),new['item'])

    def test_mistagged_changed_or_failed_cached_new_dictionary_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=ArtificialSources(tmp);new=f.product(A.NO_DROP_BUILDER);cache=Path(tmp)/'CACHE.json'
            def save(product,key=A.NO_DROP_BUILDER):
                cache.write_text(json.dumps(dict(binding=f.binding,splits={},
                    split_versions={'7':{key:product}},configurations=[])))
            with f.guards():
                save(new,A.LEGACY_BUILDER)
                with self.assertRaisesRegex(ValueError,'TAG_DIFFERS'):A.Assets(ROOT,cache=cache)
                changed=deepcopy(new);changed['item']['payload']['terms'][0]['coefficient']=.5;save(changed)
                with self.assertRaisesRegex(ValueError,'SOURCE_OR_GATES'):A.Assets(ROOT,cache=cache)
                save(new);loc=new['item']['P0_result_source'];f.data[loc['commit'],loc['path']]['checks'][0]['PASS']=False
                with self.assertRaisesRegex(ValueError,'SOURCE_OR_GATES'):A.Assets(ROOT,cache=cache)

    def test_registry_uses_fixed_locators_and_preparation_does_not_need_cache_to_find_new_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=ArtificialSources(tmp);new=f.product(A.NO_DROP_BUILDER)
            entry=dict(prefix=7,builder_id=A.NO_DROP_BUILDER,dictionary_source=new['item']['source'],
                p0_result_source=new['item']['P0_result_source'],descriptor_source=new['descriptor']['source'])
            f.registry.write_text(json.dumps(dict(entries=[entry])))
            with f.guards():
                first=A.Assets(ROOT);second=A.Assets(ROOT)
                self.assertEqual(first.dictionary(7),new['item'])
                self.assertEqual(second.dictionary(7),new['item'])
                self.assertEqual(first.descriptor(7)['descriptor'],new['descriptor']['descriptor'])
                self.assertIsNone(A.Assets(ROOT,builder_id=A.LEGACY_BUILDER).dictionary(7))

    def test_fixed_builder_source_mismatch_is_not_silently_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=ArtificialSources(tmp);new=f.product(A.NO_DROP_BUILDER);other=f.product(A.NO_DROP_BUILDER,suffix='_OTHER_SOURCE')
            with f.guards():
                a=A.Assets(ROOT);a.save_split(7,new)
                with self.assertRaisesRegex(ValueError,'SELECT_FIXED_SOURCE'):a.save_split(7,other)
                explicit_old=A.Assets(ROOT,builder_id=A.LEGACY_BUILDER)
                with self.assertRaisesRegex(ValueError,'DIFFERS_FROM_REQUEST'):explicit_old.save_split(7,new)

    def test_cached_new_builder_cannot_override_a_different_published_new_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=ArtificialSources(tmp);published=f.product(A.NO_DROP_BUILDER)
            cached=f.product(A.NO_DROP_BUILDER,suffix='_OTHER_SOURCE');cache=Path(tmp)/'CACHE.json'
            original=dict(binding=f.binding,splits={},
                split_versions={'7':{A.NO_DROP_BUILDER:cached}},configurations=[])
            cache.write_text(json.dumps(original))
            f.registry.write_text(json.dumps(dict(entries=[dict(prefix=7,builder_id=A.NO_DROP_BUILDER,
                dictionary_source=published['item']['source'],
                p0_result_source=published['item']['P0_result_source'],
                descriptor_source=published['descriptor']['source'])])))
            with f.guards(),self.assertRaisesRegex(ValueError,'SELECT_FIXED_SOURCE'):
                A.Assets(ROOT,cache=cache)
            self.assertEqual(json.loads(cache.read_text()),original)

    def test_registry_configuration_must_bind_the_selected_new_dictionary(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=ArtificialSources(tmp);new=f.product(A.NO_DROP_BUILDER)
            loc=dict(commit='ARTIFICIAL',path='CONFIGURATION.json')
            entry=dict(prefix=7,builder_id=A.NO_DROP_BUILDER,dictionary_source=new['item']['source'],
                p0_result_source=new['item']['P0_result_source'],descriptor_source=new['descriptor']['source'],configuration_source=loc)
            f.registry.write_text(json.dumps(dict(entries=[entry])))
            config=dict(id='ARTIFICIAL_NEW',spec=dict(binding=f.binding,L_D=7,
                split=dict(tail_dictionary=A.E.L.dictionary_identity(new['item']))))
            with f.guards(),patch.object(A.E,'load_configuration',return_value=(config,loc)):
                a=A.Assets(ROOT);self.assertIn((config,loc),a.configurations)
                config['spec']['split']['tail_dictionary']['dictionary_sha256']='OLD_DICTIONARY'
                with self.assertRaisesRegex(ValueError,'REGISTERED_CONFIGURATION_DICTIONARY'):A.Assets(ROOT)

    def test_saved_input_descriptor_can_be_reused_despite_old_dictionary_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=ArtificialSources(tmp)
            loc=('9fa9525b3','docs/research/b2_pauli_split_p0_20261116/LD8/FIXED_INPUTS.json')
            f.data[loc]=dict(binding=f.binding,saved_tail_descriptors={'8':dict(scalar=.125,lambda_R=.5)})
            with f.guards():
                assets=A.Assets(ROOT);descriptor=assets.descriptor(8)
                self.assertEqual(descriptor['status'],'SAVED_P0_INPUT_DESCRIPTOR')
                self.assertEqual(descriptor['source']['selector'],['saved_tail_descriptors','8'])
                self.assertIsNone(assets.dictionary(8))
                self.assertEqual(descriptor['descriptor'],dict(scalar=.125,lambda_R=.5))


if __name__=='__main__':unittest.main(verbosity=2)
