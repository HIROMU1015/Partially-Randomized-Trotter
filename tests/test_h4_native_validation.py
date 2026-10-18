"""Artificial contracts, scalar/bias arithmetic, scope and owned-worker checks."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h4_native_contract as c


def record(id_='B0_p3_q1_2nd', method='B0', prefix=3, q=1, formula='2nd'):
    r=dict(cell=c.cell(method,prefix,q,formula),signals=dict(corrected=dict(real=.8,imag=.1),
        exact_truncated=dict(real=.82,imag=.13)),reference=dict(real=.9,imag=.15),log_B=0.,b=1.,
        error_decomposition=dict(signed_parts=dict(discard=dict(real=-.08,imag=-.02),PF=dict(real=-.02,imag=-.03)),
            total_signed=dict(real=-.1,imag=-.05)),allowance=dict(real=1e-11,imag=1e-11,log_B_margin=1e-11,
            kind='EMPIRICAL',certified=False),precision_rows={},noncancellation_precision_rows={})
    r['cell']['id']=id_
    return r


class Contract(unittest.TestCase):
    def test_fixed_target_and_no_other_campaign(self):
        p=c.plan();self.assertIn('H4_1Angstrom',p['target'])
        for k in ('new_SCF_DF_state','new_full_H_reference','geometry_campaign','rare_pilot','new_PR_cost_sampling'):
            self.assertFalse(p[k])

    def test_D_exact_grid_and_compile_upper(self):
        p=c.plan()['D'];self.assertEqual(len(p['cells']),96)
        self.assertEqual(len({x['id'] for x in p['cells']}),96)
        self.assertEqual({x['method'] for x in p['cells']},{'B0','B0I'})
        self.assertEqual(p['maximum_compile'],2*p['maximum_direct_configurations'])
        self.assertIsNone(p['proxy_error_threshold']);self.assertFalse(p['proxy_is_cost_model'])

    def test_A_exact_neighbors_without_old_duplicate(self):
        cells=c.plan()['A']['cells'];self.assertEqual(len({x['id'] for x in cells}),16)
        old={c.cell('B2',3,q,r=2 if q==8 else 4,K=k)['id'] for q in (1,2,4,8) for k in (2,4)}
        self.assertFalse(old & {x['id'] for x in cells})
        self.assertEqual({x['q'] for x in cells},{1,4,8})

    def test_noncancellation_axis_specific_and_no_mutation(self):
        r=record();r['error_decomposition']['signed_parts']['PF']['real']=.06
        r['error_decomposition']['total_signed']['real']=-.02
        before=deepcopy(r);v=c.noncancel(r)
        self.assertAlmostEqual(v['error_decomposition']['total_signed']['real'],.14)
        self.assertAlmostEqual(v['error_decomposition']['total_signed']['imag'],.05)
        self.assertEqual(r,before)

    def test_missing_decomposition_refused(self):
        r=record();r['error_decomposition']['signed_parts']={}
        with self.assertRaisesRegex(ValueError,'MISSING'):c.noncancel(r)

    def test_component_mismatch_refused(self):
        r=record();r['error_decomposition']['total_signed']['real']=1.
        with self.assertRaisesRegex(ValueError,'CLOSURE'):c.noncancel(r)

    def test_nonfinite_refused(self):
        r=record();r['error_decomposition']['signed_parts']['PF']['real']=math.nan
        with self.assertRaisesRegex(ValueError,'NONFINITE'):c.noncancel(r)

    def test_grant_refused_before_real_inputs(self):
        for stage in ('D','A','N'):
            with self.assertRaisesRegex(ValueError,'APPROVAL'):
                c.verify({},dict(stage=stage,approved_by_user=False),stage)

    def test_cross_stage_grant_refused(self):
        with self.assertRaisesRegex(ValueError,'STAGE'):c.verify({},dict(stage='D'),'A')

    def test_selector_is_fixed_union_and_exact_lexical_tie(self):
        def ready(id_):
            r=record(id_);r['cell']['prefix']=3
            for field in ('precision_rows','noncancellation_precision_rows'):
                r[field]={str(e):{'1.0':dict(cost_acquisition_eligible=True,
                    axes=dict(real=dict(shots=10),imag=dict(shots=20)))} for e in c.EPS}
            return r
        costs={'B1_p12_q1_2nd':dict(cosine=13,sine=26)}
        ids,why=c.select_direct([ready('z'),ready('a')],costs)
        self.assertEqual(ids,['a']);self.assertEqual(len(why),10)
        self.assertEqual(Fraction(why[0]['screening_score']),200)

    def test_ineligible_selector_not_zero_filled(self):
        r=record()
        for field in ('precision_rows','noncancellation_precision_rows'):
            r[field]={str(e):{'1.0':dict(cost_acquisition_eligible=False)} for e in c.EPS}
        ids,why=c.select_direct([r],{})
        self.assertEqual(ids,[]);self.assertTrue(all(x['cell_id'] is None for x in why))


class ArtificialScience(unittest.TestCase):
    def test_symmetric_control_scalar_S4_against_occupation_oracle(self):
        import numpy as np
        import h4_native_science as s
        from scipy.linalg import expm
        from trotterlib.df_hamiltonian import DFHamiltonian
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare_discard
        from trottertracks.resource_applicability.ax2b_independent_reference import occupation_df_matrix
        m=s.provider()
        ham=DFHamiltonian(.17,np.array([[.2,.04],[.04,-.3]]),np.array([.4,-.2]),
            (np.array([[.6,.1],[.1,-.2]]),np.array([[.3,.07],[.07,.5]])),{})
        prep=_prepare_discard(ham,1)
        state=np.array([1.,1j,.2,.3j]);state=state/np.linalg.norm(state)
        one=np.asarray(occupation_df_matrix(0.,ham.one_body,[],[],list(range(4))),dtype=complex)
        frag=np.asarray(occupation_df_matrix(0.,np.zeros_like(ham.one_body),ham.lambdas[:1],ham.g_matrices[:1],list(range(4))),dtype=complex)
        cell=c.cell('B0',1,2,'4th');v=state.copy();scalar=.17+.123
        for _ in range(2):
            for i,t in m.primitive_time_schedule(cell,T=.8)['ordinary_one_outer_step']:
                v=expm(-1j*t*(one if i==0 else frag))@v
            v*=np.exp(-1j*scalar*.8/2)
        expected=m.complex_record(complex(np.vdot(state,v)))
        qstate=state[[0,2,1,3]]  # Occupation order -> Qiskit little-endian bridge.
        evolution=m.build_deterministic_native(prep.deterministic_blocks,num_system_qubits=2,T=.8,q=2,
            formula='4th',scalar=scalar,control_policy='symmetric_directional',max_instructions=1000000)
        with tempfile.TemporaryDirectory() as tmp:
            s.native_control(evolution,m,qstate,expected,c.Writer(tmp))
            self.assertEqual(c.read(Path(tmp)/'control.json')['status'],'PASS')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'SCALE_AGREEMENT|CONTROL_OR_SCALAR'):
                s.native_control(evolution,m,qstate,dict(expected,real=expected['real']+.01),c.Writer(tmp))

    def test_numerical_accounting_reuses_axis_rule(self):
        import h4_native_science as s
        r=s.decorate(record(),s.accounting_plan())
        self.assertFalse(r['precision_rows']['0.05']['1.0']['cost_acquisition_eligible'])
        self.assertIsNone(r['strict_bias_certificate'])
        self.assertEqual(set(r['precision_rows']),{str(x) for x in c.EPS})

    def test_scalar_overlay_leaves_reference_and_rotates_components(self):
        import h4_native_science as s
        import numpy as np
        from types import SimpleNamespace
        base=record(prefix=0);base['cell']['prefix']=0
        # Make decomposition mathematically consistent with the artificial endpoints.
        base['error_decomposition']['signed_parts']=dict(discard=dict(real=-.08,imag=-.02),PF=dict(real=-.02,imag=-.03))
        ham=SimpleNamespace(g_matrices=(np.diag([.3,.6]),),lambdas=(.4,),constant=.17)
        out=s.overlay(base,ham,s.accounting_plan())
        C=out['scalar_receipt']['C'];z=complex(.8,.1)*np.exp(-1j*.8*C)
        self.assertAlmostEqual(out['signals']['corrected']['real'],z.real)
        self.assertAlmostEqual(out['signals']['corrected']['imag'],z.imag)
        self.assertEqual(out['reference'],base['reference'])
        self.assertEqual(out['cell']['method'],'B0I')
        self.assertGreater(out['allowance']['real'],base['allowance']['real'])
        self.assertFalse(out['allowance']['certified'])

    def test_missing_cost_remains_unresolved(self):
        import h4_native_science as s
        r=record();p=dict(epsilons=[.05])
        a=dict(cost_acquisition_eligible=True,N_total=30,axes=dict(real=dict(shots=10),imag=dict(shots=20)))
        for field in ('precision_rows','noncancellation_precision_rows'):
            r[field]={'0.05':{str(f):a for f in c.FACTORS}}
        out=s.account([r],{},p,scope='artificial')
        self.assertTrue(all(x['G_RZ_point'] is None for x in out['rows']))
        self.assertTrue(all(x['PR_reduction'] is None for x in out['comparisons']))

    def test_paired_weighted_cost_exact_and_SE(self):
        import h4_native_science as s
        r=record();p=dict(epsilons=[.05])
        a=dict(cost_acquisition_eligible=True,N_total=5,axes=dict(real=dict(shots=2),imag=dict(shots=3)))
        for field in ('precision_rows','noncancellation_precision_rows'):
            r[field]={'0.05':{str(f):a for f in c.FACTORS}}
        out=s.account([r],{r['cell']['id']:[dict(cosine=10,sine=20),dict(cosine=30,sine=40)]},p,scope='artificial')
        for row in out['rows']:
            self.assertEqual(row['G_RZ_exact'],'130');self.assertAlmostEqual(row['G_RZ_SE'],50.)

    def test_owned_parallel_workers_reaped(self):
        import os
        import h8_key_compile_parallel as pool
        cpus=sorted(os.sched_getaffinity(0))[:2]
        with tempfile.TemporaryDirectory() as tmp:
            receipt=pool.map_owned([dict(id='one'),dict(id='two')],lambda task,path:dict(id=task['id']),
                cpus=cpus,limits=dict(max_workers=len(cpus)),directory=Path(tmp)/'workers',deadline=math.inf)
            self.assertEqual(receipt['status'],'PASS');self.assertTrue(receipt['all_reaped'])
            self.assertTrue(all(x['exit_code']==0 and x['reaped'] for x in receipt['workers']))


if __name__=='__main__':unittest.main()
