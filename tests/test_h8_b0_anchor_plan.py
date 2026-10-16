"""Artificial acquisition/receipt guards; no native objects, arrays or fit."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from h8_b0_anchor_plan import (SCOPE, PROFILE, COMPILER, cases, axes,
                               design_proof, validate, validate_terminal)


def plan():
    root=Path(__file__).resolve().parents[1]
    return json.loads((root/'artifacts/resource_applicability/track_a_h8_b0_anchor_readiness/2026-10-16/acquisition_plan_UNAPPROVED_v1.json').read_text())


def terminal(p):
    source='SYNTHETIC_NEW_ACQUISITION_SOURCE'
    cs={c['id']:c for c in cases()}
    rows=[]
    for axis in axes():
        row=deepcopy(axis)
        row.update(SOURCE=source,scope=SCOPE,compiler_hash=COMPILER,profile_sha256=PROFILE,
            control='symmetric_directional',scalar_policy=p['scalar_policy'],
            preparation_binding=deepcopy(p['binding']['prefix_preparations'][str(cs[row['case_id']]['prefix'])]),
            native_exit_code=0,reaped=True,fingerprint_before='synthetic-wrapper',
            fingerprint_after='synthetic-wrapper',measured_RZ=100,max_control_error=0.)
        rows.append(row)
    return dict(scope=SCOPE,SOURCE=source,status='COMPLETE_MANDATORY_STOP',
        next_stage_authorized=False,axes=rows,actual_calls=deepcopy(p['call_caps']))


class AnchorPlanTests(unittest.TestCase):
    def test_fixed_four_coordinates_and_eight_axes(self):
        p=plan();r=validate(p)
        self.assertEqual([(c['prefix'],c['q'],c['role']) for c in p['cases']],
            [(14,1,'TRAIN'),(21,1,'TRAIN'),(14,8,'TRAIN'),(21,8,'HOLDOUT')])
        self.assertEqual((r['training_axes'],r['independent_validation_axes']),(6,2))
        self.assertFalse(r['executable'])

    def test_design_uses_coordinates_only(self):
        proof=design_proof()
        self.assertEqual(proof['determinant'],-49)
        self.assertFalse(proof['measured_RZ_used'])
        self.assertIsNone(proof['fitted_coefficients'])

    def test_no_six_axis_saturated_only_design(self):
        p=plan();p['cases'].pop();p['axes']=p['axes'][:-2]
        with self.assertRaises(ValueError):validate(p)

    def test_no_posthoc_role_reassignment_or_grid_change(self):
        for field,value in [('role','TRAIN'),('prefix',20),('q',4),('formula','4th'),('method','B0I')]:
            p=plan();p['cases'][-1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate(p)

    def test_no_fit_or_following_phase_during_acquisition(self):
        changes=[('call_caps','cost_fit',1),('call_caps','cost_reaccount',1),
            ('call_caps','compile',9),('later_B1','automatic_start',True),
            ('later_B1','holdout_in_fit',True),('later_B1','H6_transfer',True),
            ('B2','automatic_start',True),('C','automatic_start',True),
            ('authorization','approved',True),('implementation','scientific_launcher_available',True)]
        for section,key,value in changes:
            p=plan();p[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate(p)

    def test_resource_and_stop_contract_cannot_expand(self):
        for section,key,value in [('resources','max_compile_workers',8),
                ('resources','threads_per_role',2),('resources','aggregate_RSS_bytes',0),
                ('acquisition_STOP','retry',True),('acquisition_STOP','extra_compile',True),
                ('acquisition_STOP','all_pass_still_mandatory_stop',False)]:
            p=plan();p[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate(p)

    def test_artificial_complete_terminal(self):
        p=plan();r=validate_terminal(p,terminal(p))
        self.assertEqual(r['status'],'PASS_TERMINAL_METADATA_ONLY')
        self.assertFalse(r['next_phase_started'])

    def test_missing_duplicate_extra_axis_rejected(self):
        for kind in ('missing','duplicate','extra'):
            p=plan();r=terminal(p)
            if kind=='missing':r['axes'].pop()
            elif kind=='duplicate':r['axes'][-1]=deepcopy(r['axes'][0])
            else:r['axes'].append(deepcopy(r['axes'][0]))
            with self.subTest(kind=kind),self.assertRaises(ValueError):validate_terminal(p,r)

    def test_runtime_basis_scalar_fingerprint_exit_control_guards(self):
        changes=[('profile_sha256','old'),('compiler_hash','old'),('SOURCE','old'),
            ('scope','H8_NATIVE_B0I_FULL_FAMILY12_V1'),('scalar_policy','ADD_TAIL_IDENTITY'),
            ('control','ordinary'),('native_exit_code',1),('reaped',False),
            ('fingerprint_after','mutated'),('role','HOLDOUT'),('measured_RZ',0),
            ('max_control_error',1e-8),('max_control_error',float('nan'))]
        for key,value in changes:
            p=plan();r=terminal(p);r['axes'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_terminal(p,r)
        p=plan();r=terminal(p);r['axes'][0]['preparation_binding']['ordered_block_description_sha256']='wrong'
        with self.assertRaises(ValueError):validate_terminal(p,r)

    def test_no_old_scope_source_or_fit_in_terminal(self):
        changes=[('scope','H8_STRENGTHENED_PHASE_A_SIGNAL_V1'),
            ('SOURCE','cb8a56c6b2a772827275f9200b02638030f33f72'),
            ('SOURCE','c31ef424175c02379eafebe8fc6c7548c1d9aabe'),
            ('next_stage_authorized',True)]
        for key,value in changes:
            p=plan();r=terminal(p);r[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_terminal(p,r)
        for key,value in [('compile',9),('compile',7),('cost_fit',1),('S4_compile',1)]:
            p=plan();r=terminal(p);r['actual_calls'][key]=value
            with self.subTest(call=key),self.assertRaises(ValueError):validate_terminal(p,r)


if __name__=='__main__':unittest.main()
