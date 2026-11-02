"""Temporary-repository B authority and no-retry checks; artificial data only."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


c=module('b_authority_fixture',ROOT/'src/trotterlib/h4_precision_control.py')
fixtures=module('bc_math_fixtures',Path(__file__).with_name('test_h4_precision_bc.py'))


class AuthorityChecks(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='h4-b-artificial-',dir='/tmp');self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        subprocess.run(['git','init','-q',str(self.root)],check=True)
        (self.root/'code.py').write_text('# ARTIFICIAL authority fixture only\n')
        (self.root/'saved.json').write_text('{"scope":"ARTIFICIAL_ONLY"}\n')
        subprocess.run(['git','-C',str(self.root),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.root),'-c','user.name=Artificial Test','-c','user.email=test@example.invalid','commit','-qm','artificial authority'],check=True)
        self.pack=self.root/'b/B_run01'
        self.auth=self.root/'authorization.txt';self.auth.write_bytes('段階Bの固定40軸\r\n段階C 実H4は未承認\r\n'.encode())
        self.inputs=dict(source_commit='ARTIFICIAL',scientific_SOURCE='ARTIFICIAL',
            files={'saved.json':c.sha(self.root/'saved.json')},rows=fixtures.fixture_rows(),rho=.01)
        patches=[patch.object(c,'ROOT',self.root),patch.object(c,'BASE_PATH','b'),patch.object(c,'CODE_FILES',('code.py',)),
                 patch.object(c,'collect_inputs',side_effect=lambda:deepcopy(self.inputs))]
        for p in patches:p.start();self.addCleanup(p.stop)

    def prepare(self):return c.prepare(self.pack,self.auth)

    def test_prepare_freezes_contract_source_input_budget_and_authorization(self):
        grant=self.prepare();self.assertTrue(grant['B_valid']);self.assertFalse(grant['C_valid'])
        contract=c.read(self.pack/'contract.json')
        self.assertEqual(contract['budget']['axis_diagnostics'],40)
        self.assertEqual(contract['budget']['signal_calculation'],0)
        self.assertEqual((self.pack/'authorization.txt').read_bytes(),self.auth.read_bytes())
        self.assertFalse((self.pack/'CONSUMED.json').exists())

    def test_rename_namespace_and_prepare_twice_refused(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError,'ALREADY_RESERVED'):self.prepare()
        moved=self.root/'b/B_other';self.pack.rename(moved)
        with self.assertRaisesRegex(ValueError,'NAMESPACE'):c.check_authority(moved)

    def test_modified_source_or_input_refused_before_consumption(self):
        self.prepare();(self.root/'code.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError,'SOURCE_FILE_CHANGED'):c.run(self.pack)
        self.assertFalse((self.pack/'CONSUMED.json').exists())
        (self.root/'code.py').write_text('# ARTIFICIAL authority fixture only\n')
        (self.root/'saved.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'OLD_INPUT_CHANGED'):c.run(self.pack)

    def test_modified_scalar_pack_or_scope_refused(self):
        self.prepare();data=c.read(self.pack/'scalar_inputs.json');data['rows'][0]['saved_u']=.3
        (self.pack/'scalar_inputs.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'PACK_BYTES_CHANGED'):c.run(self.pack)
        self.assertFalse((self.pack/'CONSUMED.json').exists())

    def test_C_scope_never_accepted_by_B_runner(self):
        self.prepare();grant=c.read(self.pack/'grant.json');grant['C_valid']=True
        (self.pack/'grant.json').write_text(json.dumps(grant))
        with self.assertRaisesRegex(ValueError,'B_ONLY_SCOPE'):c.run(self.pack)

    def test_artificial_e2e_and_consumed_grant_no_retry(self):
        self.prepare();result=c.run(self.pack)
        self.assertEqual(result['axes'],40)
        self.assertEqual(len(list((self.pack/'axes').glob('*.json'))),40)
        # Explicit artificial receipt used only for testing saved verifier.
        c.save(self.pack/'native_process.json',dict(exit_code=0,scope='ARTIFICIAL_TEST_RECEIPT_ONLY'))
        self.assertEqual(c.verify(self.pack)['axes'],40)
        with self.assertRaisesRegex(ValueError,'ALREADY_CONSUMED'):c.run(self.pack)

    def test_failure_preserves_partial_rows_and_never_retries(self):
        self.prepare();real=c.budget.diagnose_axis;calls=[]
        def fail_second(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:raise ValueError('ARTIFICIAL_INJECTED_FAILURE')
            return real(*args,**kwargs)
        with patch.object(c.budget,'diagnose_axis',side_effect=fail_second):
            with self.assertRaisesRegex(ValueError,'ARTIFICIAL_INJECTED_FAILURE'):c.run(self.pack)
        stop=c.read(self.pack/'STOP.json')
        self.assertEqual((stop['completed_axes'],stop['axis_diagnostics_started']),(1,2))
        self.assertTrue((self.pack/'axes/00.json').exists())
        with self.assertRaisesRegex(ValueError,'ALREADY_CONSUMED'):c.run(self.pack)

    def test_native_failure_is_recorded_without_second_dispatch(self):
        self.prepare()
        checked=c.check_authority(self.pack)
        with patch.object(c,'check_authority',return_value=checked), \
             patch.object(c.subprocess,'run',return_value=subprocess.CompletedProcess([],7)) as dispatch:
            with self.assertRaisesRegex(ValueError,'NATIVE_FAILURE'):c.launch(self.pack)
        self.assertEqual(dispatch.call_count,1)
        self.assertEqual(c.read(self.pack/'native_process.json')['exit_code'],7)
        self.assertFalse(c.read(self.pack/'native_process.json')['original_native_completion_verified'])

    def test_runtime_binding_and_output_tampering_refused(self):
        self.prepare()
        with patch.object(c,'runtime',return_value={'changed':True}):
            with self.assertRaisesRegex(ValueError,'RUNTIME_CHANGED'):c.run(self.pack)
        c.run(self.pack);c.save(self.pack/'native_process.json',dict(exit_code=0,scope='ARTIFICIAL_ONLY'))
        (self.pack/'diagnostics.csv').write_text('altered')
        with self.assertRaisesRegex(ValueError,'OUTPUT_CHANGED'):c.verify(self.pack)


class OriginalInputBinding(unittest.TestCase):
    def test_committed_inventory_must_match_original_axis_and_signed_bias(self):
        with tempfile.TemporaryDirectory(prefix='h4-b-input-fixture-',dir='/tmp') as tmp:
            root=Path(tmp);rows=fixtures.fixture_rows();inventory=dict(rows=[],cases={})
            for case in c.budget.CASES:
                folder=root/'artifacts/non_df_h4_pilot/2026-10-23/run01/run'/case
                folder.mkdir(parents=True);precision=[]
                normalization=dict(B=1.,log_B=0.,normalization_certificate_verified=False)
                biases=dict(real=1e-5,imag=-1e-5)
                inventory['cases'][case]=dict(signed_bias=biases,normalization=normalization)
                (folder/'result.json').write_text(json.dumps(dict(signal=dict(signed_bias=biases),normalization=normalization)))
                for epsilon in c.budget.EPSILONS:
                    selected=[x for x in rows if x['case']==case and x['epsilon_sig']==epsilon]
                    axes={x['axis']:dict(epsilon=x['epsilon_axis'],bias=x['absolute_saved_bias'],allowance=x['saved_u'],
                        headroom=x['saved_headroom'],status='UNDETERMINED',shots=None) for x in selected}
                    precision.append(dict(epsilon_sig=epsilon,shots=dict(N_total=None,axes=axes),total_RZ=None))
                    inventory['rows'].append(dict(case=case,epsilon_sig=epsilon,axes={a:dict(saved=v) for a,v in axes.items()},stored_N_total=None,stored_G=None))
                (folder/'precision_rows.json').write_text(json.dumps(precision))
            inv=root/c.INPUT_PATHS[0];inv.parent.mkdir(parents=True);inv.write_text(json.dumps(inventory))
            (root/c.INPUT_PATHS[1]).write_text(json.dumps(dict(numerical=dict(maximum_u_over_headroom=.01))))
            subprocess.run(['git','init','-q',tmp],check=True)
            subprocess.run(['git','-C',tmp,'add','.'],check=True)
            subprocess.run(['git','-C',tmp,'-c','user.name=Artificial','-c','user.email=test@example.invalid','commit','-qm','synthetic inputs'],check=True)
            commit=subprocess.check_output(['git','-C',tmp,'rev-parse','HEAD'],text=True).strip()
            with patch.object(c,'ROOT',root),patch.object(c,'BASE_REF',commit):
                self.assertEqual(len(c.collect_inputs()['rows']),40)
            inventory['cases']['pauli_jw_PF']['signed_bias']['real']=.3
            inv.write_text(json.dumps(inventory))
            subprocess.run(['git','-C',tmp,'add','.'],check=True)
            subprocess.run(['git','-C',tmp,'-c','user.name=Artificial','-c','user.email=test@example.invalid','commit','-qm','synthetic mismatch'],check=True)
            commit=subprocess.check_output(['git','-C',tmp,'rev-parse','HEAD'],text=True).strip()
            with patch.object(c,'ROOT',root),patch.object(c,'BASE_REF',commit):
                with self.assertRaisesRegex(ValueError,'SIGNED_BIAS_NOT_ORIGINAL'):c.collect_inputs()


if __name__=='__main__':unittest.main(verbosity=2)
