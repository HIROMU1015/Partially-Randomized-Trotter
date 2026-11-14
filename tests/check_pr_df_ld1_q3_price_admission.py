"""Artificial receipt/law fixtures only; no physical kernel or compiler."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import admit_pr_df_ld1_q3_prices as A


def fixtures():
    candidate=dict(id='synthetic', q=1, R_outer=2, K=2, prefix=1, formula='2nd', T=.1, delta=.1)
    cell=dict(id='cell', cohort='development', seeds=[[11]])
    public=dict(index=0, C=[17,19], S=[13,15], event_digest='event-order-hash', compiler='compiler',
        fingerprints=['X-fingerprint','Y-fingerprint'], seeds=[11], laws=[dict(steps=2,orders=[0,2])],
        compiled=dict(cosine=dict(rz_count=17),sine=dict(rz_count=19)))
    original=dict(public, candidate_id='synthetic',cell_id='cell',cohort='development',conditional=False,
                  source=dict(execution_source='fixed'), full_wrapper_statevector_validation=False)
    receipt=dict(compiler='compiler',fingerprint='X-fingerprint',metrics=dict(rz_count=17))
    return candidate, cell, public, original, receipt


class AdmissionTests(unittest.TestCase):
    def bind(self, data, axis='X'):
        c, cell, p, o, receipt=data
        return A.path_binding({}, p, o, receipt, axis, cell, c, 'fixed', 'compiler')

    def test_same_path_axis_preserves_pair_without_residual_or_mean(self):
        data=fixtures();x=self.bind(data)
        self.assertEqual((x['C'],x['S']),(17,13))
        y=deepcopy(data);y[-1].update(fingerprint='Y-fingerprint',metrics=dict(rz_count=19))
        self.assertEqual(self.bind(y,'Y')['C'],19)
        self.assertNotIn('D',x);self.assertNotIn('mean',x)

    def test_axis_receipt_cannot_be_borrowed_even_equal_prices(self):
        data=fixtures();data[2]['C'][1]=17;data[3]['C'][1]=17
        with self.assertRaisesRegex(ValueError,'Axis fingerprint'):
            self.bind(data,'Y')

    def test_physical_source_cohort_and_conditional_reject(self):
        for key,value in [('candidate_id','LD2'),('cohort','other'),('conditional',True)]:
            data=fixtures();data[3][key]=value
            with self.assertRaises(ValueError):self.bind(data)
        data=fixtures();data[3]['source']['execution_source']='other'
        with self.assertRaisesRegex(ValueError,'SOURCE'):self.bind(data)

    def test_event_digest_law_seed_compiler_mismatch(self):
        for kind in ('digest','R','K','seed','compiler'):
            data=fixtures()
            if kind=='digest':data[3]['event_digest']='reordered'
            elif kind=='R':data[2]['laws'][0]['steps']=4
            elif kind=='K':data[2]['laws'][0]['orders']=[0,2,4]
            elif kind=='seed':data[2]['seeds']=[12]
            else:data[-1]['compiler']='different'
            with self.assertRaises(ValueError):self.bind(data)

    def structure(self):
        physical=fixtures()[0];law=dict(configuration_id='new-config',candidate_id='new-canonical',
            definition={'support':'K2'},kappa={'exact':'fixed'},event_mapping='fixed-grid')
        binding=dict(system='synthetic',hamiltonian='H',state='psi',snapshot='snapshot',rank=1,rank_policy='saved',geometry='linear',basis='test')
        profile=dict(wrapper='full',P=0)
        structure=dict(signed_weights=[1.],allocation=[2],scalar_policy='once')
        c=dict(physical,source_candidate_id=physical['id'])
        c['T']=dict(zip(('numerator','denominator'),(.1).as_integer_ratio()));c['delta']=deepcopy(c['T'])
        value=dict(candidate_id=physical['id'],m=0,axis='cosine',q_PF=1,R_outer=2,
            quantity='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT',expectation='IDEAL_FINITE_K_SOURCE_LAW',status='K0_PARITY_PASS_K1_NUMERIC_ENCLOSURE',
            law_binding=dict(law,configuration_id='old-config',candidate_id='old-canonical'),
            identity=dict(structure=structure,binding=binding,profile=profile,candidate=c))
        row=dict(m=0,axis='X',structure=dict(physical=structure))
        return row,physical,value,law,profile,binding

    def test_carrier_bridge_preserves_actual_law(self):
        data=self.structure();A.structural_binding(*data)
        data[3]['kappa']={'exact':'different'}
        with self.assertRaisesRegex(ValueError,'law/grid/kappa'):A.structural_binding(*data)

    def test_structure_exact_time_and_wrapper_mismatch(self):
        data=self.structure();data[2]['identity']['candidate']['T']={'numerator':1,'denominator':10}
        with self.assertRaisesRegex(ValueError,'binary64'):A.structural_binding(*data)
        data=self.structure();data[2]['axis']='sine'
        with self.assertRaisesRegex(ValueError,'axis'):A.structural_binding(*data)

    def test_entrypoint_has_no_scientific_calls(self):
        tree=ast.parse(Path(A.__file__).read_text())
        forbidden={'sample_requests','make_rte_config','transpile','structural_pair','fit','predict','cost','evaluate','run_K0','run_K1'}
        actual={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(actual&forbidden)

    def test_replay_is_reference_not_new_independent_sample(self):
        replay=A.path_role('P2','replay');fresh=A.path_role('P2','fresh')
        self.assertTrue(replay['replay']);self.assertFalse(replay['counts_as_nonreplay_path'])
        self.assertTrue(fresh['counts_as_nonreplay_path'])
        self.assertFalse(fresh['new_science_acquisition'])
        self.assertIsNone(fresh['statistical_IID_guarantee'])


if __name__=='__main__':unittest.main()
