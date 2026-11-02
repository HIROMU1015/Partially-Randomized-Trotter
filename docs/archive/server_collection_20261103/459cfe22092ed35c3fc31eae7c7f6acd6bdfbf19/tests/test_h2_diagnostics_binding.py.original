"""Artificial metadata/bytes only. No scientific imports, NPZ decode or kernels."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h2_diagnostics_binding as b
import h2_common_production_contract as c


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h2-binding-synthetic-')
        self.root=Path(self.tmp.name)

    def tearDown(self):self.tmp.cleanup()

    def make_runtime(self):
        rows=[];packages={}
        for name,version in [('numpy','1.26.4'),('qiskit','1.3.0')]:
            p=self.root/(name+'_METADATA');p.write_text('Name: '+name+'\nVersion: '+version+'\n')
            rows.append(dict(path=str(p),sha256=b.sha(p),role='runtime:synthetic'))
            packages[name]=dict(path=str(p),sha256=b.sha(p))
        manifest=self.root/'manifest.json'
        b.write(manifest,dict(files=rows,scientific_imports_observed=False,
            unresolved_elf_dependencies=[],package_metadata=packages))
        cache=str(self.root/'ABSENT_CACHE')
        runtime=dict(closure_manifest=str(manifest),closure_mode='STATIC_CLOSED_IMPORT_SET_V1',
            inventory_sha256=b.digest(rows),python_target=str(Path(sys.executable).resolve()),
            cache_prefix=cache,loaded_closure_complete=False)
        cfg=dict(runtime=runtime,files=rows+[dict(path=str(manifest),sha256=b.sha(manifest),role='manifest')])
        return cfg

    def verify(self,cfg):
        with patch.object(b.sys,'flags',SimpleNamespace(isolated=1,no_site=1)),\
             patch.object(b.sys,'dont_write_bytecode',True),\
             patch.object(b.sys,'_xoptions',{'pycache_prefix':cfg['runtime']['cache_prefix']}),\
             patch.dict(os.environ,{'LD_LIBRARY_PATH':'','LD_PRELOAD':''}),\
             patch.object(b,'verify_mapped_runtime'):
            b.verify_runtime_binding(cfg)

    def test_static_seal_does_not_claim_observed_science_imports(self):
        cfg=self.make_runtime();self.verify(cfg)
        cfg['runtime']['loaded_closure_complete']=True
        with self.assertRaisesRegex(ValueError,'NO_FABRICATED'):self.verify(cfg)

    def test_runtime_inventory_mismatch_rejected(self):
        cfg=self.make_runtime();cfg['runtime']['inventory_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'CLOSURE_DIGEST'):self.verify(cfg)

    def test_missing_runtime_file_not_zero_filled(self):
        cfg=self.make_runtime();cfg['files'].pop(0)
        with self.assertRaisesRegex(ValueError,'INCOMPLETE_RUNTIME'):self.verify(cfg)

    def test_unresolved_native_library_rejected(self):
        cfg=self.make_runtime();p=Path(cfg['runtime']['closure_manifest']);r=b.read(p)
        r['unresolved_elf_dependencies']=[{'needed':'synthetic_missing.so'}];b.write(p,r)
        with self.assertRaisesRegex(ValueError,'UNRESOLVED_ELF'):self.verify(cfg)

    def test_package_version_mismatch_rejected(self):
        cfg=self.make_runtime();Path(cfg['files'][0]['path']).write_text('Version: 2.0\n')
        with self.assertRaisesRegex(ValueError,'PACKAGE_METADATA'):self.verify(cfg)

    def test_preexisting_bytecode_cache_rejected(self):
        cfg=self.make_runtime();Path(cfg['runtime']['cache_prefix']).mkdir()
        with self.assertRaisesRegex(ValueError,'BYTECODE_CACHE'):self.verify(cfg)

    def test_nonisolated_flags_rejected(self):
        cfg=self.make_runtime()
        with self.assertRaisesRegex(ValueError,'ISOLATED_RUNTIME'):b.verify_runtime_binding(cfg)

    def test_dynamic_loader_override_rejected(self):
        cfg=self.make_runtime()
        with patch.object(b.sys,'flags',SimpleNamespace(isolated=1,no_site=1)),\
             patch.object(b.sys,'dont_write_bytecode',True),\
             patch.object(b.sys,'_xoptions',{'pycache_prefix':cfg['runtime']['cache_prefix']}),\
             patch.dict(os.environ,{'LD_PRELOAD':'/synthetic/unsealed.so'}):
            with self.assertRaisesRegex(ValueError,'LOADER_ENVIRONMENT'):b.verify_runtime_binding(cfg)

    def test_executable_mapping_requires_actual_sealed_bytes(self):
        p=self.root/'synthetic_native.so';p.write_bytes(b'ARTIFICIAL')
        cfg={'files':[{'path':str(p),'sha256':b.sha(p)}]}
        maps='1000-2000 r-xp 00000000 00:00 1 '+str(p)+'\n'
        self.assertEqual(b.verify_mapped_runtime(cfg,maps),[str(p)])
        p.write_bytes(b'TAMPERED')
        with self.assertRaisesRegex(ValueError,'UNSEALED_EXECUTABLE'):b.verify_mapped_runtime(cfg,maps)

    def test_unknown_or_deleted_executable_mapping_rejected(self):
        for path in ['/unsealed/synthetic.so','/unsealed/synthetic.so (deleted)']:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    b.verify_mapped_runtime({'files':[]},'1000-2000 r-xp 0 00:00 1 '+path)

    def bundle(self):
        run=self.root/'synthetic_run';(run/'results').mkdir(parents=True)
        expected=dict(source_commit='a'*40,binding={'state':'b'*64},physical={'synthetic':True},
            native_policy='synthetic:native',allowance_policy='synthetic:allowance',
            reference_policy='synthetic:reference',runtime={'synthetic':True},budget={'path_signal':8},
            candidates=[{'id':'SYNTHETIC_A'},{'id':'SYNTHETIC_B'}])
        marker=run/'CONSUMED.json';b.write(marker,{'state':'CONSUMED_NO_RETRY_NO_REFUND',
            'grant':{'contract_sha256':b.digest(expected),'budget':expected['budget'],'run_root':str(run)}})
        (run/'ledger').mkdir()
        b.write(run/'ledger/000001.json',dict(kind='path_signal',amount=8,nonrefundable=True,
            contract_sha256=b.digest(expected)))
        for prefix in (1,3):b.write(run/'results'/('preparation_'+str(prefix)+'.json'),{'synthetic':True})
        for row in expected['candidates']:
            cid=row['id'];v=run/'results'/(cid+'_vectors.npz');v.write_bytes(b'ARTIFICIAL_NOT_A_REAL_NPZ')
            b.write(run/'results'/(cid+'_diagnostics.json'),dict(candidate=cid,native_policy=expected['native_policy'],
                strict_diagnostic_limit=1e-9,phase_alignment=False,postcompile_correction=False,scalar_applications=1,
                vector_archive=dict(path=v.name,sha256=b.sha(v))))
        result_files=[dict(path=str(p.relative_to(run)),sha256=b.sha(p)) for p in sorted((run/'results').iterdir())]
        completion=dict(status='CONTRACT_COMPLETE_NOT_SCIENTIFIC_GO',result={'status':'DIAGNOSTICS_ONLY'},
            used=expected['budget'],marker_sha256=b.sha(marker),contract_sha256=b.digest(expected),
            result_files=result_files,**{k:expected[k] for k in ('source_commit','binding','physical','native_policy','allowance_policy','reference_policy','runtime')})
        b.write(run/'completion.json',completion);return run,expected

    def refresh(self,run):
        p=run/'completion.json';r=b.read(p)
        r['result_files']=[dict(path=str(q.relative_to(run)),sha256=b.sha(q)) for q in sorted((run/'results').iterdir())]
        b.write(p,r)

    def accept(self,run,cfg):
        return b.accepted_diagnostics_receipt(run,cfg,completion_sha256=b.sha(run/'completion.json'))

    def test_accepted_bundle_is_metadata_only_and_grant_not_reusable(self):
        run,cfg=self.bundle();r=self.accept(run,cfg)
        self.assertFalse(r['prior_grant_reusable']);self.assertFalse(r['science_GO'])
        self.assertFalse(r['numerical_values_evaluated'])

    def test_partial_diagnostics_cannot_be_accepted(self):
        run,cfg=self.bundle();b.write(run/'STOP.json',{'partial':True})
        with self.assertRaisesRegex(ValueError,'PARTIAL'):self.accept(run,cfg)

    def test_missing_vectors_rejected(self):
        run,cfg=self.bundle();(run/'results/SYNTHETIC_A_vectors.npz').unlink();self.refresh(run)
        with self.assertRaisesRegex(ValueError,'COMPLETE_DIAGNOSTIC_BUNDLE'):self.accept(run,cfg)

    def test_result_byte_mutation_rejected(self):
        run,cfg=self.bundle();(run/'results/SYNTHETIC_A_vectors.npz').write_bytes(b'TAMPERED')
        with self.assertRaisesRegex(ValueError,'RESULT_HASH'):self.accept(run,cfg)

    def test_source_runtime_reference_identity_rejected(self):
        for key,value in [('source_commit','c'*40),('runtime',{'different':True}),('reference_policy','DIFFERENT')]:
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory(prefix='h2-binding-case-') as tmp:
                    old=self.root;self.root=Path(tmp)
                    try:
                        run,cfg=self.bundle();cfg[key]=value
                        with self.assertRaises(ValueError):self.accept(run,cfg)
                    finally:self.root=old

    def test_consumed_marker_change_rejected(self):
        run,cfg=self.bundle();(run/'CONSUMED.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'CONSUMED_MARKER'):self.accept(run,cfg)

    def test_different_or_boolean_budget_rejected(self):
        run,cfg=self.bundle();p=run/'completion.json';r=b.read(p);r['used']={'path_signal':True};b.write(p,r)
        with self.assertRaisesRegex(ValueError,'BUDGET'):self.accept(run,cfg)

    def test_relaxed_tolerance_phase_fit_or_scalar_duplication_rejected(self):
        run,cfg=self.bundle();p=run/'results/SYNTHETIC_A_diagnostics.json';original=b.read(p)
        for field,value in [('strict_diagnostic_limit',1e-7),('phase_alignment',True),('postcompile_correction',True),('scalar_applications',2)]:
            r=copy.deepcopy(original);r[field]=value;b.write(p,r);self.refresh(run)
            with self.assertRaises(ValueError):self.accept(run,cfg)

    def test_completion_requires_external_byte_seal(self):
        run,cfg=self.bundle();value=b.sha(run/'completion.json')
        with (run/'completion.json').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'EXTERNALLY_SEALED'):
            b.accepted_diagnostics_receipt(run,cfg,completion_sha256=value)

    def test_original_diagnostic_ledger_cannot_be_missing_or_underbudget(self):
        run,cfg=self.bundle();p=run/'ledger/000001.json';row=b.read(p)
        row['amount']=7;b.write(p,row)
        with self.assertRaisesRegex(ValueError,'LEDGER_BUDGET'):self.accept(run,cfg)
        p.unlink()
        with self.assertRaisesRegex(ValueError,'LEDGER_REQUIRED'):self.accept(run,cfg)

    def test_diagnostics_counts_unchanged(self):
        expected=dict.fromkeys(c.KINDS,0);expected.update(bootstrap=1,setup=1,prepare=2,diagnose=4,path_signal=8,reference_rebind=4)
        self.assertEqual(c.counts('DIAGNOSTICS'),expected)

    def test_no_scientific_modules_imported(self):
        self.assertFalse(any(x.split('.')[0] in {'numpy','scipy','qiskit','trotterlib','trottertracks'} for x in sys.modules))

    def test_static_ast_import_discovery_does_not_execute(self):
        p=self.root/'synthetic.py';p.write_text('import imaginary_package\nraise RuntimeError("NEVER_EXECUTE")\n')
        self.assertEqual(b.imports([p]),{'imaginary_package'})

    def test_launcher_rejects_execute_before_contract_read(self):
        r=subprocess.run([sys.executable,'-I','-S','-B',str(ROOT/'scripts/launch_h2_diagnostics_binding.py'),
            '--contract','/absent','--execute'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('APPROVAL_AND_NEW_GRANT_REQUIRED',r.stderr)

    def test_new_serialization_is_only_in_future_science_join(self):
        tree=ast.parse((ROOT/'scripts/h2_common_production_diagnostics.py').read_text())
        join=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='join')
        text=ast.unparse(join)
        for token in ['np.savez','strict_diagnostic_limit=1e-09','scalar_applications=1']:
            self.assertIn(token,text)

    def test_fixed_git_batch_preserves_bytes_and_rejects_missing(self):
        path='scripts/run_h2_common_production.py';rev='59e5ddddb9306e9c076cb3d3e34e15264cc3192d'
        rows=b.git_batch(rev,[path]);self.assertEqual(rows[0][1],b.git('show',rev+':'+path))
        with self.assertRaises(ValueError):b.git_batch(rev,['missing_synthetic_file'])


if __name__=='__main__':unittest.main(verbosity=2)
