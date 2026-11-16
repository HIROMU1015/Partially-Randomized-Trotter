"""Artificial supplied-descriptor source/contract adapter tests; no derivation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_df_descriptor_extension',ROOT/'src/trottertracks/pr_df_descriptor_extension.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)


def fixture():
    loc=dict(commit='ARTIFICIAL_SOURCE',path='descriptor.json')
    binding=dict(system='H4',basis='ARTIFICIAL',geometry='ARTIFICIAL',rank=10,rank_policy='ARTIFICIAL',hamiltonian='H',state='STATE',snapshot='SNAPSHOT')
    cfg=dict(spec=dict(L_D=6,binding=binding,split=dict(lambda_R={'numerator':1,'denominator':4},lambda_source=loc)))
    doc=dict(schema='PR_DF_TAIL_DESCRIPTOR_V1',prefix=6,binding=binding,descriptor=dict(lambda_R=.25,scalar=.125),derivation_source=E.DERIVATION,saved_index=E.INDEX,execution_source={'fixture':'DERIVE'},input_source={'fixture':'INPUT'})
    return loc,cfg,doc


class DescriptorTests(unittest.TestCase):
    def test_binding_source_prefix_checks(self):
        loc,cfg,doc=fixture();self.assertEqual(E.validate(doc,loc,cfg),doc['descriptor'])
        for field in ('prefix','binding','derivation_source'):
            bad=deepcopy(doc)
            if field=='prefix':bad[field]=8
            elif field=='binding':bad[field]['state']='OTHER'
            else:bad[field]={}
            with self.assertRaises(ValueError):E.validate(bad,loc,cfg)

    def test_entry_replaces_only_verified_descriptor_index_assumption(self):
        loc,cfg,doc=fixture()
        class Base:
            def __init__(self,root):self.info={'prefixes':{'7':{'lambda_R':.1,'scalar':.01}},'descriptor_sha256':'ORIGINAL'}
            def _support(self,configuration):return ['TAIL_LAMBDA_OR_SOURCE_NOT_SUPPORTED']+configuration.get('other_failures',[])
        with patch.object(E,'read',return_value=doc):entry=E.entry(ROOT,loc,Base)
        self.assertEqual(entry._support(cfg),[])
        bad=deepcopy(cfg);bad['other_failures']=['FORMULA_NOT_SUPPORTED'];self.assertEqual(entry._support(bad),['FORMULA_NOT_SUPPORTED'])
        bad=deepcopy(cfg);bad['spec']['split']['lambda_source']={'fixture':'FOREIGN'}
        self.assertIn('SUPPLIED_DF_DESCRIPTOR_LAMBDA_OR_SOURCE_MISMATCH',entry._support(bad))
        self.assertEqual(entry.info['descriptor_sha256'],'ORIGINAL')

    def test_context_passes_explicit_descriptor_to_unchanged_factory(self):
        loc,cfg,doc=fixture();calls=[]
        with tempfile.TemporaryDirectory()as d:
            saved=Path(d)/'context.json';saved.write_text(json.dumps({'binding':cfg['spec']['binding']}));raw=saved.read_bytes()
            index={'T20_reference_files':{'H4':{'path':str(saved),'sha256':hashlib.sha256(raw).hexdigest()}}}
            loaded=SimpleNamespace(matrices=['ARTIFICIAL_MATRICES'],state=['ARTIFICIAL_STATE'],ham=SimpleNamespace(constant=2.))
            completion=SimpleNamespace(load_context=lambda root,b:(loaded,{'fixture':'STATE_METADATA'}))
            def make(*args,**kwargs):calls.append((args,kwargs));return SimpleNamespace()
            provider=SimpleNamespace(__package__='_artificial_provider',make_context=make)
            with patch.object(E,'read',side_effect=[doc,index]),patch.dict(sys.modules,{'_artificial_provider.h4_pr_rmse_completion':completion}):
                ctx=E.context(ROOT,cfg,loc,provider)
            self.assertEqual(calls[0][1]['binding'],cfg['spec']['binding']);self.assertEqual(calls[0][1]['tails']if'tails'in calls[0][1]else calls[0][0][3],{6:dict(doc['descriptor'],source=loc)})
            self.assertEqual(ctx.state_metadata,{'fixture':'STATE_METADATA'})


if __name__=='__main__':unittest.main(verbosity=2)
