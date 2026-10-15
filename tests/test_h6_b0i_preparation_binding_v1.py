"""Synthetic preparation identity checks plus frozen-metadata regressions."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('identity_artificial_fixture',ROOT/'tests/h6_pilot_fixtures_v1.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
r=f.r;b=r.adapter_module.b;identity=r.adapter_module.identity_binding


class PreparationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'],prefix='h6-identity-mock-')
        self.base=Path(self.tmp.name);self.root=self.base/'source';self.output=self.base/'output'
        self.c,_=f.setup(self.root,self.output);self.prepared={};rows=[]
        for prefix,q in r.PILOT:
            old=f.f.h6_prepared(prefix);current=deepcopy(old);current.preparation_hash=f'{prefix:064x}'
            self.prepared[prefix]=current;binding=b.binding_for(SimpleNamespace(contract=self.c),prefix,q)
            legacy=b.blueprint(old,binding,prefix=prefix,q=q);actual=b.blueprint(current,binding,prefix=prefix,q=q)
            self.c['binding']['b0_fingerprints'][f'{prefix}/{q}']=r.g.d0.digest(actual)
            rows.append(dict(prefix=prefix,q=q,legacy_blueprint=legacy,legacy_digest=r.g.d0.digest(legacy),
                current_blueprint=actual,current_digest=r.g.d0.digest(actual)))
        self.c['native_provider']=dict(SOURCE='a'*40,source_hashes={'artificial.py':'b'*64})
        self.value=dict(schema='h6_current_profile_preparation_identity_v1',
            profile_digest=r.g.d0.digest(self.c['profile']),input_receipts_digest=r.g.d0.digest(self.c['input_receipts']),
            provider_SOURCE=self.c['native_provider']['SOURCE'],provider_closure_digest=r.g.d0.digest(self.c['native_provider']['source_hashes']),
            input_array_generation=False,native_build_compile_performed=False,cases=rows)
        self.c['preparation_identity']=dict(policy=identity.POLICY,reference=dict(path='identity.json'))
        self.write_reference();self.refresh()
    def tearDown(self):self.tmp.cleanup()
    def write_reference(self):
        raw=(json.dumps(self.value,sort_keys=True,allow_nan=False)+'\n').encode();(self.root/'identity.json').write_bytes(raw)
        sha=r.g.sha(raw);self.c['source_hashes']['identity.json']=sha
        self.c['preparation_identity']['reference'].update(bytes=len(raw),sha256=sha)
    def refresh(self):
        self.auth=f.f.authorization(self.c,self.output)
        self.grant=r.g.authorization_gate(self.c,self.auth,contract_sha=r.g.d0.digest(self.c),
            authorization_sha=r.g.d0.digest(self.auth),output=self.output,scope=r.SCOPE,synthetic=True)
    def check(self,prefix=5,q=1,prepared=None):
        return identity.checked_prepared(self.grant,prepared or self.prepared[prefix],prefix=prefix,q=q,bridge=b)
    def test_fixed_current_identity_passes_and_keeps_legacy_difference_explicit(self):
        for prefix,q in r.PILOT:
            prepared,audit=self.check(prefix,q)
            self.assertIs(prepared,self.prepared[prefix]);self.assertEqual(audit['field_differences'],{})
            self.assertEqual(set(audit['legacy_field_differences']),{'preparation_hash'})
            self.assertFalse(audit['legacy_bit_identity_claimed'])
    def test_unexpected_current_digest_has_actual_expected_field_diagnostic(self):
        changed=deepcopy(self.prepared[5]);changed.preparation_hash='f'*64
        with self.assertRaises(identity.PreparationBindingError) as caught:self.check(prepared=changed)
        d=caught.exception.preparation_diagnostic
        self.assertNotEqual(d['actual_digest'],d['current_expected_digest'])
        self.assertEqual(d['field_differences']['preparation_hash']['actual'],'f'*64)
    def test_original_bridge_accepts_pinned_current_blueprint_without_changing_math(self):
        fragments=deepcopy(self.c['fixture_fragments'])
        for fragment in fragments:
            fragment['G']=tuple(tuple(complex(z['real'],z['imag']) for z in row) for row in fragment['G'])
        request=b.prepare(self.grant,prefix=5,q=1,binding=b.binding_for(self.grant,5,1),
            load_prepared=lambda _:self.check()[0],load_fragments=lambda _:fragments,
            saved_record=deepcopy(self.c['fixture_record']),scalar_input_error_upper=0.,rotation_roundoff_upper=1e-12)
        self.assertEqual(request['blueprint'],self.value['cases'][0]['current_blueprint'])
        self.assertFalse(request['signal']['allowance']['certified'])
    def test_old_digest_in_current_contract_is_rejected(self):
        self.c['binding']['b0_fingerprints']['5/1']=self.value['cases'][0]['legacy_digest'];self.refresh()
        with self.assertRaisesRegex(r.g.Stop,'IDENTITY_OR_SCIENTIFIC'):self.check()
    def test_evidence_bytes_or_closure_tamper_rejected(self):
        (self.root/'identity.json').write_text('{}')
        with self.assertRaisesRegex(r.g.Stop,'BYTES_SHA'):self.check()
        self.write_reference();self.c['source_hashes'].pop('identity.json');self.refresh()
        with self.assertRaisesRegex(r.g.Stop,'BYTES_SHA'):self.check()
    def test_profile_input_provider_changes_cannot_reuse_preparation_receipt(self):
        for field in ('profile','input_receipts','provider_SOURCE','provider_closure'):
            with self.subTest(field=field):
                saved=deepcopy(self.c)
                if field=='profile':self.c['profile']['python']='CHANGED'
                elif field=='input_receipts':self.c['input_receipts']=[dict(logical_name='NEW')]
                elif field=='provider_SOURCE':self.c['native_provider']['SOURCE']='c'*40
                else:self.c['native_provider']['source_hashes']['artificial.py']='c'*64
                self.refresh()
                with self.assertRaisesRegex(r.g.Stop,'PROFILE_INPUT_PROVIDER'):self.check()
                self.c=saved;self.refresh()
    def test_changed_scientific_blueprint_field_not_whitelisted(self):
        self.value['cases'][0]['current_blueprint']['constant']+=1.;self.write_reference();self.refresh()
        with self.assertRaisesRegex(r.g.Stop,'IDENTITY_OR_SCIENTIFIC'):self.check()
    def test_registry_extension_and_path_escape_refused(self):
        self.value['cases'].append(deepcopy(self.value['cases'][0]));self.write_reference();self.refresh()
        with self.assertRaisesRegex(r.g.Stop,'REFERENCE_CASES'):self.check()
        self.c['preparation_identity']['reference']['path']='../escape.json';self.refresh()
        with self.assertRaisesRegex(r.g.Stop,'REFERENCE_PATH'):self.check()
    def execute_failure(self,fail_diagnostic_write=False):
        audit=dict(prefix=5,field_differences={'preparation_hash':dict(expected='fixed',actual='changed')})
        class Fail(f.FakeAdapter):
            def prepare(obj,prefix,q):raise identity.PreparationBindingError('CURRENT_PROFILE_B0_BLUEPRINT_MISMATCH',audit)
        original=r.g.Artifacts.write
        def write(obj,name,value,*args,**kwargs):
            if fail_diagnostic_write and name=='preparation_mismatch_v1.json':raise OSError('DIAGNOSTIC_WRITE_FAILED')
            return original(obj,name,value,*args,**kwargs)
        with patch.object(r.g.Artifacts,'write',write):
            return r.run(self.c,self.auth,contract_sha=r.g.d0.digest(self.c),authorization_sha=r.g.d0.digest(self.auth),
                output=self.output,root=self.root,mapping_loader=lambda:{},synthetic=True,
                monitor_factory=f.f.FakeMonitor,factory=Fail)
    def test_failure_persists_field_differences_before_terminal_and_compile_zero(self):
        self.assertEqual(self.execute_failure(),2)
        audit=json.loads((self.output/'preparation_mismatch_v1.json').read_text());self.assertIn('preparation_hash',audit['field_differences'])
        t=json.loads((self.output/'terminal_v1.json').read_text());self.assertIn('CURRENT_PROFILE_B0',t['first_stop']['reason'])
        self.assertEqual(t['calls_reserved_before_operations']['compile'],0)
    def test_diagnostic_io_error_never_overwrites_first_stop(self):
        self.assertEqual(self.execute_failure(True),2)
        t=json.loads((self.output/'terminal_v1.json').read_text());self.assertIn('CURRENT_PROFILE_B0',t['first_stop']['reason'])
        self.assertIn('DIAGNOSTIC_WRITE_FAILED',t['preparation_diagnostic_write_error'])
    def test_unauthorized_run_never_checks_preparation_or_reaches_loader(self):
        factory=Mock();loader=Mock()
        with patch.object(identity,'validate_evidence',side_effect=AssertionError('SHOULD_NOT_RUN')):
            with self.assertRaises(r.g.Stop):
                r.run(self.c,{},contract_sha=r.g.d0.digest(self.c),authorization_sha='0'*64,
                    output=self.output,root=self.root,mapping_loader=loader,synthetic=True,factory=factory)
        factory.assert_not_called();loader.assert_not_called()
    def test_frozen_real_metadata_all_three_are_current_profile_and_legacy_distinct(self):
        path=ROOT/'artifacts/resource_applicability/track_a_h6_b0i_blueprint_fix_v1/2026-10-15/current_profile_preparation_identity_v1.json'
        value=json.loads(path.read_text());self.assertEqual(len(value['cases']),3)
        for row in value['cases']:
            self.assertEqual(r.g.d0.digest(row['current_blueprint']),row['current_digest'])
            self.assertEqual(r.g.d0.digest(row['legacy_blueprint']),row['legacy_digest'])
            self.assertNotEqual(row['current_digest'],row['legacy_digest'])
            self.assertEqual(set(identity.differences(row['legacy_blueprint'],row['current_blueprint'])),{'preparation_hash'})

if __name__=='__main__':unittest.main()
