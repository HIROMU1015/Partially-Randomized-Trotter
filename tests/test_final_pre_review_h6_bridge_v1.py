"""Tiny artificial G and mock native builder only."""
import copy
from fractions import Fraction
import importlib.util
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

spec=importlib.util.spec_from_file_location('bridge_fixtures',Path(__file__).with_name('final_pre_review_fixtures_v1.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
b=f.bridge


class NativeBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h6-bridge-artificial-',dir=os.environ['TMPDIR'])
        self.base=Path(self.tmp.name);self.grant,self.record,self.fragments=f.h6_setup(self.base/'source',self.base/'output')
    def tearDown(self):self.tmp.cleanup()
    def prepare(self,prefix=5,q=1,**kwargs):
        self.prepared=f.h6_prepared(prefix)
        return b.prepare(self.grant,prefix=prefix,q=q,binding=b.binding_for(self.grant,prefix,q),
            load_fragments=lambda _:self.fragments,load_prepared=lambda _:self.prepared,
            saved_record=self.record,scalar_input_error_upper=kwargs.get('scalar_input_error_upper',0.),
            rotation_roundoff_upper=kwargs.get('rotation_roundoff_upper',1e-12))

    def test_three_prefixes_tail_and_signed_full_fock_identity(self):
        for (prefix,q),expected in zip(b.PILOT,(-14.,-6.,-4.)):
            request=self.prepare(prefix,q)
            self.assertEqual(request['scalar_receipt']['C'],expected)
            self.assertEqual(request['overlay']['scalar'],3+expected)
            self.assertEqual(request['overlay']['discarded_fragment_ids'],[str(i) for i in range(prefix,19)])
        self.assertEqual(b.exact_identity(((1,1j),(-1j,2)),-.5),Fraction(-2))

    def test_exact_stored_scalar_roundoff_enclosure(self):
        self.fragments[6]['lambda']=.3
        request=self.prepare();receipt=request['scalar_receipt']
        exact=Fraction(int(receipt['exact_stored_scalar_numerator']),int(receipt['exact_stored_scalar_denominator']))
        self.assertGreaterEqual(Fraction(receipt['arithmetic_error_upper']),abs(Fraction(receipt['C'])-exact))
        self.assertFalse(receipt['new_scientific_certificate'])

    def test_rotation_roundoff_recipe_explicit_empirical_not_certificate(self):
        self.grant.contract['rotation_roundoff_policy']='EMPIRICAL_64_EPS_PHASE_SCALE_FLOOR_1E12'
        self.grant.binding_digest=f.d0.digest((self.grant.contract,self.grant.authorization))
        request=self.prepare(rotation_roundoff_upper=None)
        self.assertGreaterEqual(request['scalar_receipt']['rotation_roundoff_upper'],1e-12)
        self.assertTrue(request['scalar_receipt']['rotation_error_is_empirical'])
        self.assertFalse(request['scalar_receipt']['new_scientific_certificate'])

    def test_full_reference_unchanged_and_b0_two_signals_rotate(self):
        old=copy.deepcopy(self.record);request=self.prepare()
        self.assertEqual(request['signal']['reference'],old['reference'])
        z=b.p.rotate_saved_signal(complex(.7,.2),total_time=.8,tail_identity=-14)
        self.assertAlmostEqual(request['signal']['signals']['corrected']['real'],z.real)
        self.assertAlmostEqual(request['signal']['signals']['corrected']['imag'],z.imag)
        self.assertEqual(self.record,old)

    def test_rotated_signal_plus_unrotated_reference_allowance(self):
        request=self.prepare(scalar_input_error_upper=.001,rotation_roundoff_upper=1e-8)
        theta=-.8*request['scalar_receipt']['C'];u=request['signal']['allowance']
        self.assertGreaterEqual(u['real'],abs(math.cos(theta))*.001+abs(math.sin(theta))*.002+.003)
        self.assertGreaterEqual(u['imag'],abs(math.sin(theta))*.001+abs(math.cos(theta))*.002+.004)
        self.assertEqual(u['kind'],'EMPIRICAL');self.assertFalse(u['certified'])

    def test_separate_reference_error_receipt_required(self):
        self.record['error_components'].pop('full_H_reference')
        with self.assertRaises(KeyError):self.prepare()

    def test_conservative_saved_u_split_is_explicit_and_more_than_copy(self):
        self.record['allowance'].update(real=.001,imag=.002,kind='EMPIRICAL',certified=False)
        self.record.pop('error_components')
        self.grant.contract['error_component_policy']='SAVED_COMBINED_EMPIRICAL_U_UPPER_FOR_EACH_CONTRIBUTION'
        self.grant.binding_digest=f.d0.digest((self.grant.contract,self.grant.authorization))
        before=copy.deepcopy(self.record);request=self.prepare()
        self.assertGreater(request['signal']['allowance']['real'],.001)
        self.assertGreater(request['signal']['allowance']['imag'],.002)
        self.assertEqual(self.record,before)
        self.assertFalse(request['signal']['allowance']['certified'])

    def test_signed_nonhermitian_nonfinite_rejected(self):
        for field,value in [('G',((1,2),(1,2))),('G',((1,float('nan')),(float('nan'),2))),('lambda',float('inf'))]:
            grant,record,fragments=f.h6_setup(self.base/('source_'+str(len(str(value)))),self.base/('output_'+str(len(str(value)))))
            fragments[5][field]=value
            with self.assertRaises(ValueError):
                b.prepare(grant,prefix=5,q=1,binding=b.binding_for(grant,5,1),load_fragments=lambda _:fragments,
                    load_prepared=lambda _:f.h6_prepared(5),saved_record=record,
                    scalar_input_error_upper=0,rotation_roundoff_upper=1e-12)

    def test_unapproved_context_never_calls_real_loaders(self):
        loader=Mock(side_effect=AssertionError('MUST NOT READ HAMILTONIAN'))
        with self.assertRaisesRegex(b.g.Stop,'VALIDATED_GRANT_REQUIRED'):
            b.prepare(None,prefix=5,q=1,binding={},load_fragments=loader,load_prepared=loader,
                saved_record={},scalar_input_error_upper=0,rotation_roundoff_upper=0)
        self.assertFalse(loader.called)
        with self.assertRaisesRegex(b.g.Stop,'VALIDATED_GRANT_REQUIRED'):
            b.prepare_from_snapshot(None,prefix=5,q=1,input_mapping_loader=loader)
        self.assertFalse(loader.called)

    def test_source_input_blueprint_mismatch_before_load_or_build(self):
        loader=Mock()
        binding=b.binding_for(self.grant,5,1);binding['input']='0'*64
        with self.assertRaisesRegex(b.g.Stop,'SOURCE_INPUT_BINDING'):
            b.prepare(self.grant,prefix=5,q=1,binding=binding,load_fragments=loader,load_prepared=loader,
                saved_record=self.record,scalar_input_error_upper=0,rotation_roundoff_upper=0)
        self.assertFalse(loader.called)
        (self.base/'source/synthetic_source.txt').write_text('wrong source')
        with self.assertRaises(ValueError):self.prepare()

    def test_scalar_double_overlay_and_extraction_twice_rejected(self):
        self.prepare()
        with self.assertRaisesRegex(b.g.Stop,'EXTRACTION_ALREADY_RESERVED'):self.prepare()

    def test_native_relative_phase_x_y_and_original_blocks_not_mutated(self):
        request=self.prepare();original=copy.deepcopy(self.prepared)
        def evolution(blocks,**kwargs):
            blocks[0]['internal_phase']=99 # supplied copy protects the B0 original.
            return SimpleNamespace(angle=-kwargs['T']*kwargs['scalar'],kwargs=kwargs)
        provider=SimpleNamespace(build_deterministic_native=Mock(side_effect=evolution),
            build_native_hadamard_wrapper=Mock(side_effect=lambda e,axis,**_:dict(axis=axis,angle=e.angle)))
        result=b.build(self.grant,request,provider)
        self.assertEqual(self.prepared,original)
        self.assertEqual(provider.build_deterministic_native.call_count,1)
        self.assertEqual(provider.build_deterministic_native.call_args.kwargs['scalar'],-11)
        self.assertEqual(result['axes'],{'cosine':'real','sine':'imag'})
        self.assertEqual(result['wrappers']['sine']['angle'],8.8)
        self.assertEqual(provider.build_native_hadamard_wrapper.call_count,2)
        self.assertIsNone(result['matched_B0']['cost_delta'])
        with self.assertRaisesRegex(b.g.Stop,'REQUEST_REUSE'):b.build(self.grant,request,provider)

    def test_request_scalar_and_prepared_gate_mutation_rejected(self):
        request=self.prepare();request['overlay']['scalar']+=1;provider=Mock()
        with self.assertRaises(b.g.Stop):b.build(self.grant,request,provider)
        self.assertFalse(provider.build_deterministic_native.called)
        request['overlay']['scalar']-=1;self.prepared.deterministic_blocks[0]['internal_phase']+=1
        with self.assertRaisesRegex(b.g.Stop,'PREPARED_B0_CHANGED'):b.build(self.grant,request,provider)

    def test_already_overlaid_preparation_refused(self):
        prepared=f.h6_prepared(5);prepared.extracted_identity_coefficient=1
        with self.assertRaisesRegex(b.g.Stop,'UNOVERLAID'):
            b.prepare(self.grant,prefix=5,q=1,binding=b.binding_for(self.grant,5,1),
                load_fragments=lambda _:self.fragments,load_prepared=lambda _:prepared,
                saved_record=self.record,scalar_input_error_upper=0,rotation_roundoff_upper=0)

    def test_nonpilot_prefix_boundary_and_scope_refused(self):
        for prefix,q in [(0,1),(19,1),(5,8)]:
            with self.assertRaisesRegex(b.g.Stop,'PILOT_SOURCE_INPUT_BINDING'):
                b.prepare(self.grant,prefix=prefix,q=q,binding={},load_fragments=Mock(),load_prepared=Mock(),
                    saved_record={},scalar_input_error_upper=0,rotation_roundoff_upper=0)

    def test_ordinary_and_directional_native_scalar_argument_same(self):
        for i,control in enumerate(('ordinary','symmetric_directional')):
            grant,record,fragments=f.h6_setup(self.base/('source_control'+str(i)),self.base/('out_control'+str(i)),control)
            request=b.prepare(grant,prefix=5,q=1,binding=b.binding_for(grant,5,1),load_fragments=lambda _:fragments,
                load_prepared=lambda _:f.h6_prepared(5),saved_record=record,scalar_input_error_upper=0,rotation_roundoff_upper=0)
            provider=SimpleNamespace(build_deterministic_native=Mock(return_value='evolution'),
                build_native_hadamard_wrapper=Mock(return_value='wrapper'))
            b.build(grant,request,provider)
            self.assertEqual(provider.build_deterministic_native.call_args.kwargs['scalar'],-11)
            self.assertEqual(provider.build_deterministic_native.call_args.kwargs['control_policy'],control)

    def test_missing_matched_baseline_not_zero_or_auto_build(self):
        self.assertEqual(b.matched_delta({},None),{'status':'MATCHED_B0_NOT_ACQUIRED','deltas':None})
        base=dict(input='i',compiler='c',axis='cosine',control='ordinary',T=.8,prefix=5,q=1,
            **{k:10 for k in ('RZ','CX','RZ_depth','CX_depth','depth','size')})
        changed=dict(base,RZ=12);self.assertEqual(b.matched_delta(changed,base)['deltas']['RZ'],2)
        changed['compiler']='bad'
        with self.assertRaises(b.g.Stop):b.matched_delta(changed,base)


if __name__=='__main__':unittest.main()
