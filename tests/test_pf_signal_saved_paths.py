"""Attempt-local PF saved input relocation, with numerical leaves intercepted."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import trottertracks
from trottertracks import baseline_signal_numerics as N
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from trottertracks.rpe_preparation_bridge import DistributedPreparationBridge
from trottertracks import rpe_preparation_worker as W


def wrapper():
    spec = importlib.util.spec_from_file_location('_pf_saved_path_wrapper', ROOT/'scripts/run_pf_preparation_stage.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PFSignalSavedPaths(unittest.TestCase):
    def test_explicit_blas_profile_reaches_probe_before_import_and_preserves_legacy(self):
        with patch.dict(os.environ, {'OPENBLAS_NUM_THREADS':'99'}, clear=False):
            legacy=W.preparation_environment({})
            self.assertEqual(legacy['OPENBLAS_NUM_THREADS'],'99')
            env=W.preparation_environment({'openblas_coretype':'HASWELL'})
            self.assertEqual(env['OPENBLAS_CORETYPE'],'HASWELL')
            self.assertTrue(all(env[key]=='1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')))
            self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'],'99')
        result=SimpleNamespace(returncode=0,stdout=json.dumps(dict(executable=sys.executable,
            version_info=list(sys.version_info),packages={})))
        with patch.object(W.subprocess,'run',return_value=result) as process:
            W._runtime(dict(interpreter=sys.executable,openblas_coretype='HASWELL'),{})
        self.assertEqual(process.call_args.kwargs['env']['OPENBLAS_CORETYPE'],'HASWELL')
        self.assertEqual(process.call_args.kwargs['env']['OPENBLAS_NUM_THREADS'],'1')
        with self.assertRaisesRegex(ValueError,'Unsupported fixed'):
            W.preparation_environment({'openblas_coretype':'UNSUPPORTED'})

    def test_frozen_loader_relocates_missing_absolute_inputs_and_retains_hash_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            old = base/'missing-original'
            saved = base/'worker-inputs'; saved.mkdir()
            receipt = dict(snapshot_path=str(old/'snapshot.npz'))
            binding = dict(system='ARTIFICIAL_H4', hamiltonian='H', state='S', snapshot='X', rank=10)
            contents = {'receipt.json': json.dumps(receipt).encode(),
                        'snapshot.npz': b'ARTIFICIAL_NO_ARRAYS',
                        'context.json': json.dumps(dict(binding=binding, prefixes={})).encode()}
            entries = {}
            for name, raw in contents.items():
                (saved/name).write_bytes(raw)
                entries[name] = dict(path=str(old/name), sha256=hashlib.sha256(raw).hexdigest())
            index = dict(saved_inputs={'H4': {'inputs': {
                'H4_tol_snapshot.npz': entries['snapshot.npz'],
                'H4_tol_receipt.json': entries['receipt.json']}}},
                T20_reference_files={'H4': entries['context.json']})
            preparation = ModuleType('trottertracks.pf_evidence_preparation')
            preparation.N = SimpleNamespace(load_shared_h4=N.load_shared_h4)
            pipeline = ModuleType('trottertracks.pf_pr_rpe_pipeline')
            pipeline.SavedArtifacts = lambda root: SimpleNamespace(read=lambda locator: index)
            pipeline.same = lambda a,b: a == b
            science = ModuleType('trottertracks.r_price_phase1_science')
            observed = []

            def functions(root, source, path, names, scope):
                self.assertEqual(names, ('array_record', 'load'))
                def load(request):
                    mapping = request['mapping']
                    record = scope['c'].read(mapping['tol_receipt'])
                    self.assertEqual(scope['c'].sha(record['snapshot_path']), entries['snapshot.npz']['sha256'])
                    observed.append(mapping)
                    return SimpleNamespace(n_blocks=10,n_qubits=8), None, None, [0]*36, {}
                scope['load'] = load
            science.functions = functions
            modules = {'trottertracks.pf_evidence_preparation': preparation,
                       'trottertracks.pf_pr_rpe_pipeline': pipeline,
                       'trottertracks.r_price_phase1_science': science}
            w = wrapper()
            with patch.dict(sys.modules, modules), \
                 patch.object(trottertracks,'pf_evidence_preparation',preparation,create=True), \
                 patch.object(trottertracks,'pf_pr_rpe_pipeline',pipeline,create=True), \
                 patch.object(N,'kernels',return_value=SimpleNamespace(make_context=lambda *a:'ARTIFICIAL_CONTEXT')):
                with self.assertRaises(FileNotFoundError):
                    preparation.N.load_shared_h4(base,binding)
                with patch.dict(os.environ, {'RPE_PREPARATION_PATH_MAP': json.dumps({str(old):str(saved)})}):
                    relocated = w.bind_saved_signal_paths()
                    self.assertIs(relocated.__code__, N.load_shared_h4.__code__)
                    context, actual_binding, metadata = relocated(base,binding)
                    self.assertEqual(context,'ARTIFICIAL_CONTEXT')
                    self.assertEqual(actual_binding,binding)
                    self.assertEqual(metadata['source']['inputs'],index['saved_inputs']['H4']['inputs'])
                    (saved/'receipt.json').write_bytes(b'changed')
                    with self.assertRaisesRegex(ValueError,'SOURCE bytes differ'):
                        relocated(base,binding)
            self.assertEqual(observed[0]['tol_snapshot'],str(old/'snapshot.npz'))
            self.assertFalse(old.exists())

    def test_bridge_captures_snapshot_receipt_context_with_fixed_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);saved=base/'saved';saved.mkdir()
            entries={}
            for name in ('receipt.json','snapshot.npz','context.json'):
                raw=('ARTIFICIAL-'+name).encode();(saved/name).write_bytes(raw)
                entries[name]=dict(path='/absent/shared/H4/'+name,sha256=hashlib.sha256(raw).hexdigest())
            index=dict(T20_reference_files={'H4':entries['context.json']},
                       saved_inputs={'H4':{'inputs':{k:v for k,v in entries.items() if k!='context.json'}}})
            bridge=DistributedPreparationBridge(SimpleNamespace(run_dir=base/'run',artifact_cache=ArtifactCache(base/'cache')),base)
            inputs,bindings,hashes={},{},[]
            with patch('trottertracks.rpe_preparation_bridge.subprocess.check_output',return_value=json.dumps(index).encode()):
                bridge._saved_h4_snapshot(base,saved,inputs,bindings,hashes)
                self.assertEqual(len(inputs),3)
                self.assertEqual(len(hashes),3)
                self.assertEqual(bindings['/absent/shared/H4/receipt.json'],'@WORK@/private/df-snapshot/receipt.json')
                self.assertEqual(bridge.executor.artifact_cache.path(inputs['private/df-snapshot/snapshot.npz']).read_bytes(),
                                 (saved/'snapshot.npz').read_bytes())
                (saved/'snapshot.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'snapshot digest differs'):
                    bridge._saved_h4_snapshot(base,saved,{},{},[])

    def test_only_signal_installs_the_path_adapter_before_fixed_provider(self):
        w=wrapper();w.ROOT=Path('/synthetic/provider');w.publish=lambda *a:'ARTIFICIAL_PUBLICATION'
        order=[]
        provider=SimpleNamespace(signals=lambda *a:order.append('signal'))
        with patch.object(w,'bind_saved_signal_paths',side_effect=lambda:order.append('paths')):
            result=w.run(provider,dict(value={},relative_output='owned',phase='signal',source={'commit':'FIXED'}))
        self.assertEqual(order,['paths','signal'])
        self.assertEqual(result['status'],'SIGNALS_READY')


if __name__=='__main__':
    unittest.main()
