"""Synthetic storage fault injection and authorization; no molecular inputs."""
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_pr_pilot_storage_runtime as s
import h46_pr_pilot_contract as c

class RuntimeStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,stage='H4_DESIGN_PILOT',
          run_root=str(self.root),output=str(self.root/'output'),marker=str(self.root/'CONSUMED.json'),
          grant={'synthetic':True},tasks=[{'id':'TOY'}],storage=s.storage_policy())
        self.q=self.root/'output/TOY/quarantine_cosine';self.q.mkdir(parents=True)
        s.save(self.cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=self.cfg['grant']))
        self.path=self.q/'wrapper.qpy';self.payload=bytes(range(256))*100
    def tearDown(self):self.tmp.cleanup()
    def write(self,**kw):return s.write_binary(self.cfg,self.path,lambda f:f.write(self.payload),**kw)
    def fault(self,at):
        def f(stage):
            if stage==at:raise RuntimeError('INJECTED_'+stage)
        return f
    def test_raw_and_archive_hashes_distinct(self):
        r=self.write();self.assertFalse(self.path.exists());self.assertNotEqual(r['sha256'],r['archive_sha256'])
        m,ap=s.manifest_for(self.cfg,self.path);self.assertTrue(m['byte_storage_PASS']);self.assertIsNone(m['native_scientific_PASS'])
        self.assertEqual(s.codec.verify_archive(ap,m['codec_receipt'])['sha256'],r['sha256'])
    def test_restore_to_new_area_exact_bytes(self):
        self.write();dest=self.root/'restoration/toy.qpy'
        with s.resolve(self.cfg,self.path,dest) as p:self.assertEqual(p.read_bytes(),self.payload)
        self.assertFalse(dest.exists());self.assertTrue(self.path.with_name('wrapper.qpy.zlib').exists())
    def test_restore_to_original_or_outside_rejected(self):
        self.write()
        for dest in (self.path,self.root.parent/'outside.qpy'):
            with self.assertRaises(ValueError):
                with s.resolve(self.cfg,self.path,dest):pass
    def test_preexisting_raw_never_overwritten(self):
        self.path.write_bytes(b'OLD_RAW')
        with self.assertRaises(ValueError):self.write()
        self.assertEqual(self.path.read_bytes(),b'OLD_RAW')
    def test_outside_raw_never_removed(self):
        p=self.root/'old.qpy';p.write_bytes(b'KEEP')
        with self.assertRaises(ValueError):s.release_new_raw(self.cfg,p)
        self.assertEqual(p.read_bytes(),b'KEEP')
    def test_foreign_task_rejected(self):
        cfg=dict(self.cfg,tasks=[{'id':'FOREIGN'}])
        with self.assertRaises(ValueError):s.write_binary(cfg,self.path,lambda f:None)
    def test_symlink_rejected(self):
        other=self.root/'old.qpy';other.write_bytes(b'KEEP');self.path.symlink_to(other)
        with self.assertRaises(ValueError):self.write()
        self.assertEqual(other.read_bytes(),b'KEEP')
    def test_unconsumed_grant_rejected(self):
        Path(self.cfg['marker']).unlink()
        with self.assertRaises(FileNotFoundError):self.write()
    def test_other_grant_rejected(self):
        cfg=dict(self.cfg,grant={'foreign':True})
        with self.assertRaisesRegex(ValueError,'CONSUMED'):s.write_binary(cfg,self.path,lambda f:None)
    def test_storage_policy_change_rejected(self):
        cfg=copy.deepcopy(self.cfg);cfg['storage']['level']=6
        with self.assertRaises(ValueError):s.write_binary(cfg,self.path,lambda f:None)
    def test_writer_crash_keeps_partial(self):
        def writer(f):f.write(b'PARTIAL');raise RuntimeError('WRITER_CRASH')
        with self.assertRaises(RuntimeError):s.write_binary(self.cfg,self.path,writer)
        self.assertEqual(self.path.with_name(self.path.name+'.raw.partial').read_bytes(),b'PARTIAL')
        self.assertFalse(self.path.with_name(self.path.name+'.archive.json').exists())
    def test_raw_durable_crash_keeps_raw_no_resume(self):
        with self.assertRaises(RuntimeError):self.write(fault=self.fault('raw_durable'))
        self.assertEqual(self.path.read_bytes(),self.payload)
        with self.assertRaises(ValueError):self.write()
    def test_archive_before_manifest_crash_keeps_raw(self):
        with self.assertRaises(RuntimeError):self.write(fault=self.fault('archive_durable'))
        self.assertTrue(self.path.exists());self.assertTrue(self.path.with_name(self.path.name+'.zlib').exists())
        with self.assertRaises(FileNotFoundError):s.release_new_raw(self.cfg,self.path)
    def test_manifest_crash_recovers_byte_storage_only(self):
        with self.assertRaises(RuntimeError):self.write(fault=self.fault('manifest_durable'))
        self.assertTrue(self.path.exists())
        m=s.release_new_raw(self.cfg,self.path);self.assertFalse(self.path.exists());self.assertIsNone(m['native_scientific_PASS'])
        self.assertEqual(m,s.release_new_raw(self.cfg,self.path))
    def test_before_release_crash_keeps_raw(self):
        with self.assertRaises(RuntimeError):self.write(fault=self.fault('before_raw_release'))
        self.assertEqual(self.path.read_bytes(),self.payload)
    def test_corruption_prevents_release(self):
        with self.assertRaises(RuntimeError):self.write(fault=self.fault('manifest_durable'))
        ap=self.path.with_name(self.path.name+'.zlib');ap.write_bytes(ap.read_bytes()[:-1])
        with self.assertRaises(ValueError):s.release_new_raw(self.cfg,self.path)
        self.assertTrue(self.path.exists())
    def test_changed_raw_prevents_release(self):
        with self.assertRaises(RuntimeError):self.write(fault=self.fault('manifest_durable'))
        self.path.write_bytes(b'CHANGED')
        with self.assertRaises(ValueError):s.release_new_raw(self.cfg,self.path)
        self.assertTrue(self.path.exists())
    def test_manifest_identity_tamper_rejected(self):
        self.write();mp=self.path.with_name(self.path.name+'.archive.json');m=s.read(mp);m['SOURCE']='c'*40;mp.write_text(json.dumps(m))
        with self.assertRaises(ValueError):s.manifest_for(self.cfg,self.path)
    def test_disk_failure_preserves_new_raw(self):
        with patch.object(s.codec.shutil,'disk_usage',return_value=type('Disk',(),{'free':0})()):
            with self.assertRaisesRegex(ValueError,'SCRATCH_HEADROOM'):self.write()
        self.assertEqual(self.path.read_bytes(),self.payload)
    def test_durable_save_refuses_replacement(self):
        p=self.root/'receipt.json';s.save(p,{'a':1})
        with self.assertRaises(FileExistsError):s.save(p,{'a':2})
        self.assertEqual(s.read(p),{'a':1})
    def test_incomplete_pair_cannot_pass_storage_audit(self):
        self.write()
        with self.assertRaises(FileNotFoundError):s.audit_task(self.cfg,{'id':'TOY'})
    def test_allocation_uses_resources_and_saved_demand(self):
        a=s.allocation(64,500,{'H4':396,'H6':396},{'H4':4,'H6':16},{'H4':1,'H6':5})
        self.assertLessEqual(sum(a.values()),64);self.assertLessEqual(a['H4']*4+a['H6']*16,500)
        self.assertGreater(a['H6'],a['H4']);self.assertGreater(a['H6'],4)
    def test_no_joint_admission_without_ram(self):
        with self.assertRaises(ValueError):s.allocation(64,10,{'H4':396,'H6':396},{'H4':4,'H6':16},{'H4':1,'H6':5})
    def test_draft_rejected_before_storage_or_inputs(self):
        with patch.object(c,'verify',side_effect=AssertionError('NO_INPUT_READ')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_PILOT'):c.authorize({'authorized':False,'grant':{'valid':False}})
    def test_unadopted_sample_count_rejected(self):
        dec={k:True for k in ('pilot_design_adopted','nominal_variance_precision_limit_accepted',
          'rare_event_limit_accepted','resources_policy_accepted','unit_cost_only_accepted')}
        dec.update(formal_shot_policy_adopted=False,AD03='DEFERRED',adaptive_top_up=False,pooling=False,
          n_P_by_candidate={r['id']:9 for r in c.design.groups() if r['system']=='H4'})
        with self.assertRaisesRegex(ValueError,'ADOPTED_33'):
            c.authorize(dict(authorized=True,grant={'valid':True},stage='H4_DESIGN_PILOT',
              cohort='DESIGN_PILOT',decisions=dec,resources={'proxy_policy_adopted':True}))

    def test_common_integrity_failure_does_not_confuse_local_numeric_or_PSI(self):
        from launch_h46_pr_pilot_acquisition import common_integrity_failure
        for text in ('NUMERIC_DISAGREEMENT:compiled_vector_cosine','MEMORY_PSI 1.0','OWN_COST_STAGE_STOP'):
            self.assertFalse(common_integrity_failure(text))
        for text in ('SEALED_BYTES:input','STORED_BYTES_OR_SHA_MISMATCH','COMMITTED_PILOT_SOURCE:file'):
            self.assertTrue(common_integrity_failure(text))

@unittest.skipUnless(os.environ.get('M5_STORAGE_QPY_TEST'),'fixed Qiskit runtime required')
class SyntheticFormatsTests(RuntimeStorageTests):
    # Inherits fault tests in the actual fixed scientific interpreter too.
    def test_QPY_NPZ_original_bytes_exact(self):
        import numpy as np
        from qiskit import QuantumCircuit,qpy
        qc=QuantumCircuit(2);qc.h(0);qc.cx(0,1)
        for name,writer in [('wrapper.qpy',lambda f:qpy.dump(qc,f)),
          ('vectors.npz',lambda f:np.savez(f,low=np.arange(4),before=np.arange(4),after=np.arange(4)))]:
            p=self.q/name;buf=io.BytesIO();writer(buf);expected=buf.getvalue()
            s.write_binary(self.cfg,p,writer)
            with s.resolve(self.cfg,p,self.root/'restoration'/name) as restored:
                self.assertEqual(restored.read_bytes(),expected)

class CompletionProjectionTests(unittest.TestCase):
    def test_missing_results_remain_unknown_and_private_paths_not_exported(self):
        from finalize_h46_pr_pilot_acquisition import project
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(SOURCE='a'*40,stage='H4_DESIGN_PILOT',plan_sha256='b'*64,
              run_root=tmp,tasks=[{'id':'TOY'}],call_scope={'compile_axes':2},
              supervisor={'exit':str(Path(tmp)/'exit_PRIVATE.json')})
            r=project(cfg)
            self.assertEqual(r['status'],'STOP_END_AUDIT_MISSING')
            self.assertIsNone(r['compressed_byte_total']);self.assertIsNone(r['worker_peak_RSS_KiB'])
            self.assertFalse(r['native_scientific_full_stage_PASS']);self.assertFalse(r['statistics_executed'])
            self.assertIsNone(r['n_E']);self.assertNotIn(tmp,json.dumps(r))

if __name__=='__main__':unittest.main()
