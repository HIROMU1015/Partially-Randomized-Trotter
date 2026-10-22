"""Original scalar NPY regression and before-grant rejection, synthetic only."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import pf_common_run_contract as contract
from pf_common_run_native import read_snapshot_arrays
from pf_common_snapshot_receipt import check_original_snapshot


class SnapshotReceipt(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np=np
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.arrays=dict(constant=np.array(.17),metadata_json=np.array('{"kind":"SYNTHETIC"}'),
                         sector_state_vector=np.array([.6+.2j,.1-.3j]))
        self.path=self.root/'snapshot.npz'
        np.savez(self.path,**self.arrays)
        self.receipt=dict(sha256=contract.sha(self.path),bytes=self.path.stat().st_size,
                         arrays={k:dict(shape=list(v.shape),dtype=v.dtype.str,
                         sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())
                         for k,v in self.arrays.items()})

    def check(self,receipt=None):
        return check_original_snapshot(self.path,receipt or self.receipt,
                                       expected_sha256=self.receipt['sha256'])

    def test_original_numeric_unicode_scalars_and_complex_vector(self):
        result=self.check()
        self.assertEqual(result['array_count'],3)
        self.assertEqual(result['arrays']['constant']['shape'],[])
        self.assertEqual(result['arrays']['metadata_json']['shape'],[])
        self.assertEqual(result['scientific_actions'],0)
        self.assertEqual(result['new_grants'],0)

    def test_real_loader_array_path_preserves_original_shapes_and_bytes(self):
        result=read_snapshot_arrays(self.path,self.receipt,system='H6')
        for name,value in self.arrays.items():
            self.assertEqual(result[name].shape,value.shape)
            self.assertEqual(result[name].dtype,value.dtype)
            self.assertTrue(self.np.array_equal(result[name],value))

    def test_migrated_H2_H4_scalar_receipt_convention_is_preserved(self):
        legacy=copy.deepcopy(self.receipt)
        for name,value in self.arrays.items():
            legacy['arrays'][name]['shape']=list(self.np.ascontiguousarray(value).shape)
        for system in ('H2','H4'):
            result=read_snapshot_arrays(self.path,legacy,system=system)
            self.assertEqual(result['constant'].shape,())
        with self.assertRaisesRegex(ValueError,'EXACT_SCIENTIFIC_ARRAY:constant'):
            read_snapshot_arrays(self.path,legacy,system='H6')

    def test_wrong_shape_dtype_or_payload_hash_rejected_in_both_paths(self):
        for name,key,value in (('constant','shape',[1]),('constant','dtype','<i8'),
                               ('metadata_json','sha256','0'*64)):
            receipt=copy.deepcopy(self.receipt);receipt['arrays'][name][key]=value
            with self.subTest(name=name,key=key):
                with self.assertRaisesRegex(ValueError,'ARRAY_RECEIPT:'+name):self.check(receipt)
                with self.assertRaisesRegex(ValueError,'EXACT_SCIENTIFIC_ARRAY:'+name):
                    read_snapshot_arrays(self.path,receipt,system='H6')

    def test_archive_SHA_and_binding_SHA_are_both_checked(self):
        receipt=copy.deepcopy(self.receipt);receipt['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'ARCHIVE_SHA256'):self.check(receipt)
        with self.assertRaisesRegex(ValueError,'BINDING_SHA256'):
            check_original_snapshot(self.path,self.receipt,expected_sha256='0'*64)

    def test_missing_array_rejected_in_both_paths(self):
        receipt=copy.deepcopy(self.receipt);del receipt['arrays']['metadata_json']
        with self.assertRaisesRegex(ValueError,'ARRAY_NAMES'):self.check(receipt)
        with self.assertRaisesRegex(ValueError,'ARRAY_NAMES'):
            read_snapshot_arrays(self.path,receipt,system='H6')

    def test_duplicate_zip_member_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(self.path,'a') as archive:
                archive.writestr('constant.npy',archive.read('constant.npy'))
        receipt=copy.deepcopy(self.receipt)
        receipt.update(sha256=contract.sha(self.path),bytes=self.path.stat().st_size)
        with self.assertRaisesRegex(ValueError,'DUPLICATE_MEMBER'):
            check_original_snapshot(self.path,receipt,expected_sha256=receipt['sha256'])

    def test_fortran_order_is_not_silently_reinterpreted(self):
        value=self.np.asfortranarray(self.np.array([[.1,.2],[.3,.4]]))
        self.np.savez(self.path,constant=value)
        receipt=dict(sha256=contract.sha(self.path),bytes=self.path.stat().st_size,
            arrays=dict(constant=dict(shape=[2,2],dtype='<f8',sha256='0'*64)))
        with self.assertRaisesRegex(ValueError,'ORIGINAL_C_ORDER'):
            check_original_snapshot(self.path,receipt)

    def test_preflight_does_not_import_numpy(self):
        receipt_path=self.root/'receipt.json';receipt_path.write_text(json.dumps(self.receipt))
        code=("import sys;sys.path.insert(0,sys.argv[1]);"
              "from pf_common_snapshot_receipt import check_original_snapshot;"
              "import json;from pathlib import Path;"
              "check_original_snapshot(sys.argv[2],json.loads(Path(sys.argv[3]).read_text()));"
              "assert 'numpy' not in sys.modules;print('STDLIB_ONLY_PASS')")
        result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'scripts'),str(self.path),str(receipt_path)],
                              check=True,capture_output=True,text=True)
        self.assertEqual(result.stdout.strip(),'STDLIB_ONLY_PASS')

    def plan(self,receipt):
        private=self.root/'pack';private.mkdir()
        receipt_path=self.root/'receipt.json';receipt_path.write_text(json.dumps(receipt))
        plan=dict(SOURCE='SYNTHETIC',system='H6',private_root=str(private),retry=False,resume=False,
                  closure=[],input_paths=dict(snapshot=str(self.path),receipt=str(receipt_path)),
                  binding=dict(snapshot=self.receipt['sha256']),
                  approval=dict(execution_authorized=True,no_intermediate_reapproval=True))
        contract.save(private/'plan_APPROVED.json',plan)
        contract.save(private/'approval_receipt.json',dict(SOURCE='SYNTHETIC',execution_authorized=True,
                      plan_sha256=contract.sha(private/'plan_APPROVED.json')))
        return private

    def test_shape_error_rejected_before_stage_grant_marker_and_ledger(self):
        receipt=copy.deepcopy(self.receipt);receipt['arrays']['constant']['shape']=[1]
        private=self.plan(receipt)
        with self.assertRaisesRegex(ValueError,'ARRAY_RECEIPT:constant'):
            contract.Stage(private,'METADATA',dict(snapshot_restore=1))
        self.assertFalse((private/'METADATA').exists())

    def test_valid_snapshot_allows_synthetic_stage_with_zero_consumption(self):
        private=self.plan(self.receipt)
        stage=contract.Stage(private,'METADATA',dict(snapshot_restore=1))
        self.assertTrue((stage.root/'grant.json').exists())
        self.assertEqual(stage.total(),dict(snapshot_restore=0))
        self.assertEqual(list((stage.root/'ledger').iterdir()),[])


if __name__=='__main__':unittest.main()
