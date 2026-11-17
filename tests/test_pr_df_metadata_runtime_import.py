"""The normal controller may load the fixed DF metadata after its own package.

Synthetic legacy modules only: no physics, sampling, compiler or worker body.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import trottertracks


class RuntimeImport(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.view=Path(self.temp.name);(self.view/'scripts').mkdir()
        (self.view/'src/trottertracks').mkdir(parents=True)
        self.files={
            'scripts/run_pr_df_ld_split.py':b'from trottertracks import p1_efficient_zz as P\nmarker=P.marker\n',
            'src/trottertracks/p1_efficient_zz.py':b'marker="synthetic fixed metadata helper"\n',
            'src/trottertracks/efficient_controlled_zz.py':b'marker="synthetic lowering"\n'}
        for name,raw in self.files.items():(self.view/name).write_bytes(raw)
        spec=importlib.util.spec_from_file_location('_metadata_df_prices',ROOT/'scripts/run_pr_df_candidate_prices.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.module.FIXED_VIEW=self.view
        self.original_path=list(trottertracks.__path__);self.original_sys_path=list(sys.path)
        self.old_modules={};self.old_attributes={}
        for name in ('p1_efficient_zz','efficient_controlled_zz'):
            self.old_modules[name]=sys.modules.pop('trottertracks.'+name,None)
            self.old_attributes[name]=getattr(trottertracks,name,None)
            if hasattr(trottertracks,name):delattr(trottertracks,name)
        self.addCleanup(self.restore)

    def restore(self):
        trottertracks.__path__[:]=self.original_path;sys.path[:]=self.original_sys_path
        for name,old in self.old_modules.items():
            sys.modules.pop('trottertracks.'+name,None)
            if old is not None:sys.modules['trottertracks.'+name]=old
            if hasattr(trottertracks,name):delattr(trottertracks,name)
            if self.old_attributes[name] is not None:setattr(trottertracks,name,self.old_attributes[name])

    def run_import(self):
        with patch.object(self.module.subprocess,'check_output',return_value=self.module.VIEW_COMMIT+'\n'),\
             patch.object(self.module,'blob',side_effect=lambda commit,name:self.files[name]):
            return self.module.runtime()

    def test_already_loaded_controller_package_gets_only_temporary_legacy_fallback(self):
        result=self.run_import()
        self.assertEqual(result.marker,'synthetic fixed metadata helper')
        self.assertEqual(result.ROOT,ROOT)
        self.assertEqual(list(trottertracks.__path__),self.original_path)
        self.assertEqual(sys.path,self.original_sys_path)
        self.assertIs(self.module.runtime(),result)

    def test_conflicting_loaded_helper_is_rejected_and_paths_restored(self):
        wrong=self.view/'wrong.py';wrong.write_text('marker="other SOURCE"\n')
        module=types.ModuleType('trottertracks.p1_efficient_zz');module.__file__=str(wrong)
        sys.modules[module.__name__]=module
        with self.assertRaisesRegex(ValueError,'Already loaded DF helper SOURCE differs'):
            self.run_import()
        self.assertEqual(list(trottertracks.__path__),self.original_path)
        self.assertEqual(sys.path,self.original_sys_path)


class SavedScalarMetadata(unittest.TestCase):
    def test_old_configuration_uses_original_descriptor_without_relabeling(self):
        spec=importlib.util.spec_from_file_location('_scalar_df_prices',ROOT/'scripts/run_pr_df_candidate_prices.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory()as temp:
            path=Path(temp)/'saved.json'
            path.write_text(json.dumps(dict(prefixes={'2':dict(lambda_R=2.5,scalar=.125)})))
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            source=dict(index=module.INDEX,context_sha256=sha,selector=['prefixes','2'])
            config=dict(spec=dict(L_D=2,split=dict(lambda_R=2.5,lambda_source=source),binding=dict(hamiltonian='synthetic H')))
            original=deepcopy(config)
            def read(locator):
                if locator==module.INDEX:return dict(T20_reference_files=dict(H4=dict(path=str(path),sha256=sha)))
                if locator==module.SCALAR_CONSTANT:return dict(value=1.,source=dict(hamiltonian_hash='synthetic H'))
                raise AssertionError('Unexpected metadata SOURCE')
            with patch.object(module,'read',side_effect=read),\
                 patch.object(module,'runtime',side_effect=AssertionError('scalar metadata must not prepare a native context')):
                resolved=module.scalar_metadata(config)
                self.assertEqual(resolved['scalar'],1.125)
                self.assertEqual(resolved['tail_source'],source)
                self.assertEqual(resolved['record']['application'],'ONCE_AT_TOTAL_T')
                self.assertEqual(config,original)
                bad=deepcopy(config);bad['spec']['split']['lambda_R']=3.
                with self.assertRaisesRegex(ValueError,'Candidate tail binding differs'):module.scalar_metadata(bad)


if __name__=='__main__':unittest.main()
