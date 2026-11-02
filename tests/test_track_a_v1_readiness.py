"""Artificial contract, schedule, selection and failure tests; no H6 input."""
import cmath
from copy import deepcopy
from fractions import Fraction
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


c = load('_v1_test_contract', ROOT/'src/trotterlib/track_a_v1_contract.py')
m = load('_v1_test_schedule', ROOT/'src/trotterlib/native_df_gate_model.py')
r = load('_v1_test_runner', ROOT/'scripts/track_a_v1_runner.py')


def mul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)] for i in range(2)]


I = [[1., 0.], [0., 1.]]
X = [[0., 1.], [1., 0.]]
Z = [[1., 0.], [0., -1.]]


def pauli_exp(p, t):
    return [[math.cos(t)*I[i][j]-1j*math.sin(t)*p[i][j] for j in range(2)] for i in range(2)]


def product_formula(q, formula, T=.3):
    out = I
    for s in m.pf_schedule(2, T, q, formula):
        out = mul(pauli_exp((X, Z)[s.block], s.time), out)
    return out


def error(q, formula):
    target = [[math.cos(.3*math.sqrt(2))*I[i][j] -
        1j*math.sin(.3*math.sqrt(2))*(X[i][j]+Z[i][j])/math.sqrt(2) for j in range(2)] for i in range(2)]
    a = product_formula(q, formula)
    return math.sqrt(sum(abs(a[i][j]-target[i][j])**2 for i in range(2) for j in range(2)))


class V1Tests(unittest.TestCase):
    def setUp(self):
        self.stratum = dict(method='B0', formula='2nd', axis='cosine', control='symmetric_directional',
                            compiler_hash='a'*64, profile_hash='b'*64, phase_policy='scalar_on_ancilla')
        self.anchors = [dict(id=f'{p}/{q}', prefix=p, q=q, RZ=7+3*q+19*p*q,
            stratum=self.stratum, status='MEASURED_COMPLETE') for p, q in ((5,1),(10,1),(5,2),(15,4))]

    def test_registry_preserves_104_existing_and_adds_104(self):
        reg=c.registrations()
        self.assertEqual((len(reg['existing']),len(reg['additional']),len(reg['reused_requested'])),(104,104,8))
        ids=[x['id'] for k in ('existing','additional') for x in reg[k]]
        self.assertEqual(len(ids),len(set(ids)))

    def test_both_S4_families_and_all_q(self):
        added=c.registrations()['additional']
        self.assertEqual({(x['method'],x['prefix'],x['q']) for x in added if x['formula']=='4th'},
                         {(f,p,q) for f in ('B0','B0I') for p in range(11,16) for q in c.QS})

    def test_signal_work_deduplicates_B0I_and_old_p10(self):
        p=c.plan(); self.assertEqual(p['demand']['new_B0_signals'],52)
        self.assertEqual(p['demand']['analytic_B0I_overlays'],52)
        self.assertEqual(p['demand']['exact_truncated_reference_pairs'],9)
        self.assertTrue(all(x['prefix']!=10 or x['formula']=='4th' for x in p['registrations']['additional']))

    def test_PR_diagnostics_never_enter_formal_candidates(self):
        reg=c.registrations()
        self.assertEqual(len(reg['diagnostics']),8)
        self.assertFalse(any(x['method']=='B2' and x['prefix']<5 for x in reg['existing']+reg['additional']))

    def test_S2_time_order_and_sum(self):
        stages=m.pf_schedule(3,.8,2,'2nd')
        self.assertEqual([s.block for s in stages[:6]],[0,1,2,2,1,0])
        for block in range(3): self.assertAlmostEqual(sum(s.time for s in stages if s.block==block),.8)

    def test_S4_negative_piece_and_total_time(self):
        stages=m.pf_schedule(3,.8,4,'4th')
        self.assertTrue(any(s.time<0 for s in stages))
        for block in range(3): self.assertAlmostEqual(sum(s.time for s in stages if s.block==block),.8)

    def test_noncommuting_S2_second_order_convergence(self):
        self.assertGreater(error(2,'2nd')/error(4,'2nd'),3.8)
        self.assertLess(error(2,'2nd')/error(4,'2nd'),4.2)

    def test_noncommuting_S4_fourth_order_convergence(self):
        self.assertGreater(error(2,'4th')/error(4,'4th'),15.)
        self.assertLess(error(2,'4th')/error(4,'4th'),17.)

    def test_scalar_relative_phase_is_once_and_reference_fixed(self):
        z=.2+.4j; phase=cmath.exp(-1j*.8*.31)
        forward=product_formula(4,'4th')
        corrected=phase*forward[0][0]
        branch_overlap=phase*forward[0][0]
        self.assertEqual(corrected,branch_overlap)
        record=dict(cell={'method':'B0','q':4},signals={k:dict(real=z.real,imag=z.imag) for k in ('corrected','exact_truncated')},
            reference=dict(real=.6,imag=-.1), allowance=dict(real=1e-12,imag=1e-12,log_B_margin=0.))
        legacy=load('_v1_overlay_test',ROOT/'scripts/h6_fixed_delta_plan_v1.py')
        out=legacy.rotate_record(record,.8,.31)
        self.assertEqual(out['reference'],record['reference'])
        self.assertAlmostEqual(out['signals']['corrected']['imag'],(phase*z).imag)

    def test_no_cancellation_diagnostic_not_certificate(self):
        usual, envelope=c.signal_envelopes({'error_decomposition':dict(total_signed=dict(real=.01,imag=.0),
            signed_parts={'discard':dict(real=.11,imag=-.1),'PF':dict(real=-.10,imag=.1)})})
        self.assertAlmostEqual(usual['real'],.01); self.assertAlmostEqual(envelope['real'],.21)
        self.assertEqual(c.plan()['conservative']['status'],'NO_CERTIFIED_BIAS_BOUND_IN_SAVED_CONTRACT')

    def test_missing_signed_parts_unresolved(self):
        self.assertIsNone(c.signal_envelopes({'error_decomposition':dict(total_signed=dict(real=.01,imag=0.))})[1])

    def test_nonclosing_signed_parts_stop(self):
        with self.assertRaisesRegex(ValueError,'DO_NOT_CLOSE'):
            c.signal_envelopes({'error_decomposition':dict(total_signed=dict(real=1.,imag=0.),
                signed_parts={'PF':dict(real=0.,imag=0.)})})

    def test_affine_fit_uses_saved_values(self):
        fit=c.affine_fit(self.anchors,stratum=self.stratum)
        self.assertEqual(fit['coefficients'],[7,3,19])
        self.assertEqual(c.predicted(fit,dict(prefix=8,q=4),stratum=self.stratum),627)

    def test_fit_rejects_singular_design(self):
        with self.assertRaisesRegex(ValueError,'UNDERDETERMINED'):
            c.affine_fit(self.anchors[:1]*3,stratum=self.stratum)

    def test_no_cross_formula_control_compiler_or_phase(self):
        fit=c.affine_fit(self.anchors,stratum=self.stratum)
        for key in ('formula','control','compiler_hash','profile_hash','phase_policy','method','axis'):
            bad=dict(self.stratum,**{key:'OTHER'})
            with self.assertRaises(ValueError):c.predicted(fit,dict(prefix=8,q=4),stratum=bad)

    def test_incomplete_anchor_rejected(self):
        self.anchors[0]['status']='RESERVED_UNKNOWN'
        with self.assertRaises(ValueError):c.affine_fit(self.anchors,stratum=self.stratum)

    def validation_rows(self):
        return [dict(cell_id=x['cell']['id'],axis=x['axis'],status='MEASURED_COMPLETE',
                     model_sha256='a'*64,identity_verified=True,predicted=100,measured=100)
                for x in c.validation_axes()]

    def test_four_validation_axes_pass(self):
        self.assertEqual(len(c.validation_axes()),4)
        c.validate_predictions(self.validation_rows(),expected_axes=c.validation_axes(),frozen_model_sha256='a'*64)

    def test_exactly_two_percent_stops(self):
        rows=self.validation_rows();rows[0]['predicted']=102
        with self.assertRaisesRegex(ValueError,'GE_2_PERCENT'):
            c.validate_predictions(rows,expected_axes=c.validation_axes(),frozen_model_sha256='a'*64)

    def test_missing_validation_axis_stops(self):
        with self.assertRaises(ValueError):
            c.validate_predictions(self.validation_rows()[:-1],expected_axes=c.validation_axes(),frozen_model_sha256='a'*64)

    def test_refitted_model_identity_stops(self):
        with self.assertRaises(ValueError):
            c.validate_predictions(self.validation_rows(),expected_axes=c.validation_axes(),frozen_model_sha256='b'*64)

    def test_selection_tiebreak_and_missing_cost(self):
        rows=[dict(cell_id=x,G=Fraction(3),eligible=True,cost_status='MEASURED_COMPLETE') for x in ('B','A')]
        self.assertEqual(c.best(rows)['cell_id'],'A')
        rows[0]['G']=None
        with self.assertRaises(ValueError):c.best(rows)

    def test_unknown_eligibility_cannot_be_silently_dropped(self):
        with self.assertRaises(ValueError):c.best([dict(eligible=None)])

    def test_only_complete_matching_success_reused(self):
        good=dict(status='COMPLETE_MANDATORY_STOP',exit_code=0,all_reaped=True,first_stop=None,identity={'source':'a'})
        self.assertTrue(c.complete_reuse(good,expected={'source':'a'}))
        for key,value in (('status','PARTIAL'),('exit_code',1),('all_reaped',False),('identity',{'source':'b'})):
            with self.assertRaises(ValueError):c.complete_reuse(dict(good,**{key:value}),expected={'source':'a'})

    def test_unapproved_grant_rejected_before_any_callback(self):
        with self.assertRaisesRegex(ValueError,'GRANT_REQUIRED'):
            r.verify_grant({},dict(approved_by_user=False),config_sha256='a',source='b',fresh={})

    def test_legacy_shot_rule_and_boundary_guard(self):
        record=dict(cell={'id':'FABRICATED'},error_decomposition={'total_signed':dict(real=0.,imag=0.)},
            log_B=0.,allowance=dict(real=0.,imag=0.,log_B_margin=0.,kind='EMPIRICAL',certified=False))
        row=r.accounting.accounting(record,.01)
        expected=math.ceil(2/(.01/math.sqrt(2))**2*math.log(80))
        self.assertEqual(row['N_total'],2*expected)
        record['allowance']['real']=.01
        self.assertFalse(r.accounting.accounting(record,.01)['cost_acquisition_eligible'])

    def test_plan_has_no_execution_authority(self):
        p=c.plan();self.assertFalse(p['approved']);self.assertFalse(p['runtime_authorization'])
        self.assertEqual(p['allowed_cpus'],[]);self.assertIsNone(p['absolute_launch_command'])

    def test_signal_only_selection_never_exceeds_four_axes(self):
        ids={x['id'] for x in c.registrations()['additional']}
        self.assertEqual(c.select_validation_axes(ids),c.validation_axes())
        self.assertEqual(c.select_validation_axes(set()),[])
        b0={x for x in ids if x.startswith('B0_')}
        self.assertEqual(len(c.select_validation_axes(b0)),2)
        with self.assertRaises(ValueError):c.select_validation_axes({'UNKNOWN'})

    def test_frozen_S4_hypothesis_keeps_wrapper_once(self):
        costs=load('_v1_cost_freezer_test',ROOT/'scripts/track_a_v1_costs.py')
        strata=[]
        for method in ('B0','B0I'):
            for axis in c.AXES:
                s=dict(self.stratum,method=method,axis=axis)
                rows=[dict(x,stratum=s) for x in self.anchors]
                strata.append(dict(rows=rows,stratum=s))
        frozen=costs.freeze_models(strata,contract=c,
            eligible_ids={x['id'] for x in c.registrations()['additional']},
            transfer_receipts=[dict(verified=True,receipt_sha256='f'*64)])
        self.assertEqual(frozen['predictions'][0]['predicted'],7+3*3+3*19*11)
        self.assertFalse(frozen['fitted_after_validation'])

    def test_signal_adapter_checks_permit_before_imports(self):
        adapter=load('_v1_signal_no_permission_test',ROOT/'scripts/track_a_v1_signal.py')
        class Denied:
            def require(self,scope):raise ValueError('DENIED_BEFORE_IMPORT')
        from unittest.mock import patch
        with patch.object(adapter.importlib,'import_module',side_effect=AssertionError('science imported')):
            with self.assertRaisesRegex(ValueError,'DENIED_BEFORE_IMPORT'):
                adapter.evaluate({},Denied(),c,None)

    def test_compile_adapter_checks_permit_before_imports(self):
        adapter=load('_v1_compile_no_permission_test',ROOT/'scripts/track_a_v1_compile.py')
        class Denied:
            def require(self,scope):raise ValueError('DENIED_BEFORE_IMPORT')
        from unittest.mock import patch
        with patch.object(adapter.importlib,'import_module',side_effect=AssertionError('science imported')):
            with self.assertRaisesRegex(ValueError,'DENIED_BEFORE_IMPORT'):
                adapter.evaluate({},Denied(),c,None)


if __name__=='__main__':unittest.main()
