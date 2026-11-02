"""Artificial signals only: reconstruction, reuse, immutable accounting and STOP."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src/trotterlib'))
import h6_noncancellation_stage_a as c


def record(cid='B0I_p5_q1_2nd'):
    return dict(cell=dict(id=cid,method='B0I',prefix=15 if cid==c.REUSED else 5,
                         q=4 if cid==c.REUSED else 1,formula='2nd',T=.8),
        signals=dict(corrected=dict(real=.5+1/32768,imag=.25-1/32768),
                     exact_truncated=dict(real=.5+1/8192,imag=.25+1/16384)),
        reference=dict(real=.5,imag=.25),
        error_decomposition=dict(total_signed=dict(real=1/32768,imag=-1/32768)),
        allowance=dict(real=1e-12,imag=2e-12,log_B_margin=1e-12,kind='EMPIRICAL',certified=False),
        log_B=0.)


def permit():
    return c.Permit(dict(kind='ARTIFICIAL'),{},synthetic=True)


class StageATests(unittest.TestCase):
    def test_new_grant_and_exact_stage_scope_required(self):
        with self.assertRaises(ValueError):c.Permit({}, {})
        with self.assertRaises(ValueError):c.Permit(dict(kind='REAL'),{},synthetic=True)
        contract=dict(scope='H6_NONCANCELLATION_STAGE_A_12x10',candidate_ids=list(c.IDS),
                      call_caps=c.CAPS,SOURCE='source')
        with self.assertRaisesRegex(ValueError,'GRANT'):c.Permit(contract,{})
        grant=dict(scientific_execution_authorized=True,SOURCE='source',retry=False,
                   stage_B_authorized=False,allowed_scope=contract['scope'],call_caps=c.CAPS)
        c.Permit(contract,grant)
        grant['stage_B_authorized']=True
        with self.assertRaisesRegex(ValueError,'STAGE_B'):c.Permit(contract,grant)

    def test_saved_component_differences_leave_reference_unrotated(self):
        r=record();before=deepcopy(r);p=c.parts_from_saved(r)
        self.assertEqual(p['discard'],dict(real=1/8192,imag=1/16384))
        self.assertEqual(p['PF'],dict(real=-3/32768,imag=-3/32768))
        self.assertEqual(r,before)

    def test_axis_absolute_sum_not_total_bias_or_complex_norm(self):
        out=c.analyse([record()],{},permit())
        e=out['components'][0]['noncancellation_envelope']
        self.assertEqual(e,dict(real=7/32768,imag=5/32768))
        self.assertNotEqual(e['real'],abs(record()['error_decomposition']['total_signed']['real']))

    def test_public_component_reuse_does_not_construct_again(self):
        r=record(c.REUSED);recovery=dict(candidate_id=c.REUSED,T=.8,signed_parts_verbatim=c.parts_from_saved(r))
        p=permit()
        with mock.patch.object(c,'parts_from_saved',side_effect=AssertionError('RECONSTRUCT_REUSED')):
            out=c.analyse([r],recovery,p)
        self.assertEqual(p.completed['saved_component_construction'],0)
        self.assertEqual(p.completed['literal_component_copy'],1)
        self.assertEqual(out['components'][0]['signed_parts'],recovery['signed_parts_verbatim'])

    def test_wrong_recovery_candidate_refused_before_accounting(self):
        p=permit()
        with self.assertRaisesRegex(ValueError,'RECOVERY'):
            c.analyse([record(c.REUSED)],dict(candidate_id='OTHER',T=.8),p)
        self.assertEqual(p.reserved['endpoint_accounting'],0)

    def test_closure_failure_stops_before_any_endpoint(self):
        r=record();r['error_decomposition']['total_signed']['real']=.1;p=permit()
        with self.assertRaisesRegex(ValueError,'CLOSE'):c.analyse([r],{},p)
        self.assertEqual(p.reserved['endpoint_accounting'],0)
        self.assertEqual(p.completed['closure_and_envelope'],0)

    def test_original_inputs_allowance_logB_unchanged(self):
        records=[record()];before=deepcopy(records);out=c.analyse(records,{},permit())
        self.assertEqual(records,before)
        self.assertEqual(out['components'][0]['allowance'],before[0]['allowance'])
        self.assertEqual(out['components'][0]['log_B'],0.)
        self.assertTrue(all(r['accounting']['alpha_axis']==.025 for r in out['endpoints']))

    def test_ten_endpoints_and_original_accounting_are_exact(self):
        r=record();p=permit();out=c.analyse([r],{},p)
        view=deepcopy(r);view['error_decomposition']['total_signed']=out['components'][0]['noncancellation_envelope']
        self.assertEqual(len(out['endpoints']),10)
        for row in out['endpoints']:
            expected=c.accounting_source.accounting(view,row['epsilon'],row['allowance_factor'])
            self.assertEqual(row['accounting'],expected)
        self.assertEqual(p.completed['endpoint_accounting'],10)
        self.assertEqual(p.completed['validation_endpoint'],10)

    def test_ineligible_is_not_failed_or_zero_imputed(self):
        r=record();r['signals']['exact_truncated']['real']=.625
        out=c.analyse([r],{},permit())
        self.assertTrue(all(x['eligible'] is False for x in out['endpoints']))
        self.assertTrue(all(x['shots'] is None and x['N_total'] is None for x in out['endpoints']))

    def test_allowance_guard_unknown_is_saved_and_stops_no_retry(self):
        r=record();r['allowance'].update(real=.001,imag=.001);emitted=[];p=permit()
        with self.assertRaisesRegex(ValueError,'UNRESOLVED_ENDPOINT'):
            c.analyse([r],{},p,lambda kind,value:emitted.append((kind,value)))
        self.assertEqual(emitted[-1][0],'endpoint')
        self.assertIsNone(emitted[-1][1]['eligible'])
        self.assertEqual(p.completed['endpoint_accounting'],1)
        self.assertEqual(p.reserved['validation_endpoint'],0)

    def test_nonfinite_saved_value_refused(self):
        for value in (True,float('nan'),float('inf')):
            r=record();r['signals']['corrected']['real']=value
            with self.assertRaises(ValueError):c.parts_from_saved(r)

    def test_candidate_duplicate_and_physical_change_refused(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):c.analyse([record(),record()],{},permit())
        r=record();r['cell']['T']=1.6
        with self.assertRaisesRegex(ValueError,'PHYSICAL'):c.analyse([r],{},permit())

    def test_budget_cannot_retry_or_call_forbidden_stage(self):
        p=permit();p.reserved['endpoint_accounting']=c.CAPS['endpoint_accounting']
        with self.assertRaisesRegex(ValueError,'CALL_BUDGET'):p.call('endpoint_accounting',lambda:None)
        with self.assertRaisesRegex(ValueError,'CALL_BUDGET'):p.call('weighted_cost',lambda:None)

    def test_real_inputs_never_read_by_artificial_suite(self):
        with mock.patch.object(c.accounting_source,'weighted_metric',side_effect=AssertionError('COST_FORBIDDEN')),\
             mock.patch.object(c.accounting_source,'family_selection',side_effect=AssertionError('RANK_FORBIDDEN')):
            out=c.analyse([record()],{},permit())
        self.assertFalse(out['summary']['stage_B_executed'])
        self.assertEqual(out['summary']['closure_passed'],1)


if __name__=='__main__':unittest.main()
