"""Published metadata and synthetic packets only; no A/C/D scientific calls."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import rpe_diagnostic_intake as M
from trottertracks import rpe_d_evidence_intake as D
from trottertracks import pr_rpe_rmse_evidence as E
from trottertracks.bv2_mean_price import V2MeanPriceLookup
from check_pr_rpe_rmse_controller import spec,data

B,p=M.B,M.p


def mock_m1(previous,adapter):
    src=dict(commit='1'*40,path='scripts/run_h4_pr_m1_rmse.py',saved_carrier=deepcopy(M.M1_CARRIER),scope='H4_M1_SAVED_SCALAR_ONLY')
    packet=deepcopy(previous)
    for key in ('pilot_scope','connection_provenance','schema'):
        packet.pop(key,None)
    packet.update(source=src,adapter_source=deepcopy(adapter),preserved_S0_packet=dict(commit=M.S0.A_RELEASE,path=M.S0.ADIR+'V2_N_HANDOFF.json'),
        B_S0_only_validator_applicable=False,diagnostic_scope=dict(newly_reanalyzed_rounds=[1],preserved_S0_rounds=[0,6],
            other_rounds_NOT_EVALUATED=True,whole_schedule_RMSE_qualified=False))
    packet['rmse_evidence']['source']=deepcopy(src)
    for r in packet['rounds']:
        r['diagnostic_source']=deepcopy(src if r['m']==1 else previous['source'] if r['m'] in (0,6) else None)
    packet['rounds'][1].update(nominal_N_X=10**50+3,nominal_N_Y=10**50+9,model_status='UNQUALIFIED_MODEL_ASSUMPTIONS')
    return packet


def structural(config,stage='K1'):
    return dict(configuration_id=config['id'],source={'fixture':'SYNTHETIC_ONLY'},kernel_source={'fixture':'MOCK_KERNEL'},
        stage=stage,K0_status='PASS',quantity=M.STRUCTURAL,qualified_for_candidate=False,mean_compiled_C=None,mean_C_minus_S=None,
        rows=[dict(m=m,axis=axis,identity=deepcopy(config['rounds'][m]['identity']),
            law_binding=deepcopy(config['rounds'][m]['law_binding']),physical_candidate_id=config['rounds'][m]['physical_candidate']['id'],
            profile=deepcopy(config['spec']['profile']),source={'fixture':'SYNTHETIC_ONLY'},
            mean_S=7 if stage=='K1' else None,law_context={'ideal_sampler_distance_qualified':False},
            checks={'transition_reward_parity':'MOCK_PASS'},uncertainty=dict(systematic=None,covariance=None,total_CI=None))
            for m in (0,6) for axis in p.AXES])


class NewEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store=p.SavedArtifacts(ROOT)
        old,_=M.S0.load_published(ROOT);cls.config=old['config'];cls.previous=old['packet'];cls.original=old['original'];cls.cplan=old['c_plan']
        cls.preflight=cls.store.read(dict(commit=M.A_PREPARATION,path=M.A_PREFLIGHT))
        cls.adapter=cls.preflight['source']
        cls.cresult=cls.store.read(dict(commit=M.C_DIAGNOSTIC,path=M.C_DIR+'run02/RESULT.json'))
        cls.checks=cls.store.read(dict(commit=M.C_DIAGNOSTIC,path=M.C_DIR+'CHECKS.json'))
        cls.reduction=cls.store.read(dict(commit=M.C_DIAGNOSTIC,path=M.C_DIR+'REDUCTION.json'))
        cls.cov=cls.store.read(dict(commit=D.COVERAGE,path=D.COVERAGE_JSON))
        cls.examples=cls.store.read(dict(commit=D.PHYSICAL,path=D.EXAMPLE_JSON))['packets']

    def validate_m1(self,packet):
        return M.validate_m1_packet(self.config,packet,self.previous,numerical_source=packet['source'],adapter_source=self.adapter)

    def test_01_new_m1_scope_keeps_exact_nominal_and_old_S0_meaning(self):
        packet=mock_m1(self.previous,self.adapter)
        self.assertTrue(self.validate_m1(packet))
        self.assertEqual(packet['rounds'][1]['nominal_N_X'],10**50+3)
        with self.assertRaises(ValueError):M.H.validate_A_v2_packet(self.config,packet,self.original)
        self.assertTrue(M.H.validate_A_v2_packet(self.config,self.previous,self.original))
        with patch.object(B,'cost',side_effect=AssertionError('No real G')):
            plan=E.plan([dict(configuration=self.config,N_packet=packet)],price_policy='qualified_expectation')
        self.assertIsNone(plan['G']);self.assertIsNone(plan['winner']);self.assertFalse(plan['comparison_complete'])
        self.assertEqual(packet['rounds'][0]['nominal_N_X'],64);self.assertEqual(packet['rounds'][6]['nominal_N_Y'],38)
        self.assertTrue(all(r['N_X'] is None and r['N_Y'] is None for r in packet['rounds']))

    def test_02_m1_scope_identity_law_source_and_previous_source_rejected(self):
        mutations=[lambda p:p['diagnostic_scope'].update(newly_reanalyzed_rounds=[0,1,6]),
            lambda p:p['rounds'][1]['identity']['binding'].update(state='OTHER'),
            lambda p:p['rounds'][1]['law_binding'].update(kappa=None),
            lambda p:p['rounds'][1].update(source_candidate_id='OTHER'),
            lambda p:p['preserved_S0_packet'].update(commit='2'*40),
            lambda p:p['adapter_source'].update(module='OTHER'),
            lambda p:p['source'].update(scope='EXTENDED_SCOPE'),
            lambda p:p['rounds'][6].update(diagnostic_source={'source':'WRONG'})]
        for change in mutations:
            packet=mock_m1(self.previous,self.adapter);change(packet)
            with self.assertRaises(ValueError):self.validate_m1(packet)

    def test_03_sufficient_N_global_bound_old_HP_or_other_round_cannot_be_promoted(self):
        mutations=[lambda p:p['rounds'][1].update(N_X=1),lambda p:p['rmse_evidence'].update(rmse_bound_ha=0),
            lambda p:p['rounds'][2].update(nominal_N_Y=1),lambda p:p['rounds'][0].update(nominal_N_X=65),
            lambda p:p.update(status='RMSE_SCHEDULE_EVALUATED'),lambda p:p['rounds'][1].update(model_status='CONDITIONAL_ROUND_PASS'),
            lambda p:p['rounds'][1].update(nominal_N_X=float(10**50))]
        for change in mutations:
            packet=mock_m1(self.previous,self.adapter);change(packet)
            with self.assertRaises(ValueError):self.validate_m1(packet)

    def test_04_C_published_reduction_preserves_UNKNOWN_and_no_means(self):
        result=M.accept_c_published(self.config,self.cresult,self.checks,self.reduction,self.cplan)
        self.assertEqual(result['state_equivalence']['status'],'UNKNOWN')
        self.assertIsNone(result['physical_transition_reward_parity'])
        self.assertFalse(result['K0_received']);self.assertFalse(result['K1_received'])
        self.assertIsNone(result['mean_S']);self.assertIsNone(result['mean_compiled_C'])
        changed=deepcopy(self.cresult);changed['laws'][1]['R_outer']=4
        with self.assertRaises(ValueError):M.accept_c_published(self.config,changed,self.checks,self.reduction,self.cplan)
        changed=deepcopy(self.cresult);changed['preparation']['H']='OTHER_H'
        with self.assertRaises(ValueError):M.accept_c_published(self.config,changed,self.checks,self.reduction,self.cplan)

    def test_05_K0_K1_structural_only_never_enters_compiled_lookup(self):
        for stage in ('K0','K1'):
            packet=structural(self.config,stage);result=M.accept_structural_packet(self.config,packet)
            self.assertEqual(result['compiled_price_records'],[]);self.assertIsNone(result['mean_compiled_C'])
            poison=dict(packet['rows'][0],quantity=M.STRUCTURAL,mean={'rz_count':7},cohort='MOCK')
            with self.assertRaisesRegex(ValueError,'STRUCTURAL_OR_LOCAL'):
                V2MeanPriceLookup(self.config,[poison],integration=M.H.I)
        missing=structural(self.config);missing['K0_status']='UNKNOWN'
        with self.assertRaises(ValueError):M.accept_structural_packet(self.config,missing)

    def test_06_C_structural_SOURCE_law_profile_and_C_minus_S_rejected(self):
        for change in (lambda v:v['rows'][0]['law_binding'].update(kappa=None),
                lambda v:v['rows'][0]['profile'].update(compiler='OTHER'),
                lambda v:v['rows'][0].update(source={'fixture':'OTHER_SOURCE'}),
                lambda v:v.update(mean_compiled_C=7),lambda v:v.update(quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE'),
                lambda v:v.update(qualified_for_candidate=True)):
            packet=structural(self.config);change(packet)
            with self.assertRaises(ValueError):M.accept_structural_packet(self.config,packet)

    def test_07_D_published_artificial_requests_get_canonical_ids_without_alias(self):
        for name,entry in self.examples.items():
            bound=D.accept_example(entry);self.assertTrue(D.validate_binding(bound))
            self.assertFalse(bound['scientific_conditions_adopted']);self.assertIsNone(bound['G'])
            if name=='FULL_PF_4th':
                self.assertEqual(bound['identity_map'][0]['D_formula_id'],'YOSHIDA4_STANDARD_H46_M5_PLAN_BINARY64')
                self.assertEqual(bound['identity_map'][0]['B_formula_id'],'PF_STANDARD_YOSHIDA4_BINARY64_V1')
            for row in bound['requests']['rounds']:
                for c in row['C']:
                    self.assertEqual(c['identity'],row['A']['identity']);self.assertEqual(c['physical_candidate_id'],row['A']['physical_candidate_id'])

    def test_08_D_coefficient_order_SOURCE_and_physical_map_mutation_rejected(self):
        entry=deepcopy(self.examples['FULL_PF_4th']);entry['requests']['rounds'][0]['A']['physical_round']['signed_weights'].reverse()
        # The 3-piece sequence is symmetric: mutate a coefficient, not just reverse it.
        entry['requests']['rounds'][0]['A']['physical_round']['signed_weights'][0]=0
        with self.assertRaises(ValueError):D.accept_example(entry)
        entry=deepcopy(self.examples['FULL_PF_4th']);entry['spec']['law']['source']['commit']='3'*40
        with self.assertRaises(ValueError):D.accept_example(entry)
        bound=D.accept_example(self.examples['FULL_PF_4th']);bound['identity_map'][0]['V2_candidate_id']='RELABELLED'
        with self.assertRaises(ValueError):D.validate_binding(bound)

    def test_09_D_kappa_free_q_R_grouping_and_component_law_retained(self):
        bound=D.accept_example(self.examples['FULL_H_RTE_explicit_layout'])
        self.assertIsNone(bound['v2_configuration']['spec']['kappa'])
        self.assertEqual(bound['canonical_parameterization']['kind'],B.EXPLICIT_LAYOUT)
        self.assertEqual(bound['original_parameterization']['kind'],D.D.LAYOUT_KIND)
        layout=bound['v2_configuration']['rounds'][3]['mapping']['physical_layout']
        self.assertFalse(layout['priced_equivalence_claim']);self.assertEqual(layout['grouping'],'S2_OUTER_Q_TIMES_R')
        self.assertEqual(layout['component_law'],bound['v2_configuration']['spec']['law'])
        other=D.bind_physical(bound['physical']['spec'],layout_by_round={3:dict(q=1,R_outer=256)})
        self.assertNotEqual(other['v2_configuration']['id'],bound['v2_configuration']['id'])
        self.assertNotEqual(other['identity_map'][3]['physical_candidate']['id'],bound['identity_map'][3]['physical_candidate']['id'])

    def test_10_D_saved_136_counts_do_not_adopt_a_menu_or_upgrade_old_PASS(self):
        out=D.accept_saved_coverage(self.cov)
        self.assertEqual(out['totals']['candidates'],136)
        self.assertEqual(out['totals']['signal'],{'EXACT_METADATA_MATCH':134,'CONDITIONAL_REUSE':2})
        self.assertEqual(out['totals']['X'],{'EXACT_METADATA_MATCH':56,'CONDITIONAL_REUSE':17,'MISSING_IN_CHECKED_SCOPE':63})
        self.assertFalse(out['rmse_policy_qualified']);self.assertFalse(out['candidates_adopted_as_menu'])
        self.assertTrue(all(r['B_V2_configuration_id'] is None for r in out['rows']))
        for change in (lambda c:c['rows'][0]['physical_candidate'].update(method='B0'),
                lambda c:c['rows'][0]['prices'][0].update(kind='MODEL_PREDICTION'),
                lambda c:c['rows'][0].update(formula_id='m5_ALIAS'),
                lambda c:c.update(rmse_policy_qualified=True)):
            cov=deepcopy(self.cov);change(cov)
            with self.assertRaises(ValueError):D.accept_saved_coverage(cov)

    def test_11_D_join_reuses_only_an_explicit_supplied_schedule_and_full_binding(self):
        bound=D.accept_example(self.examples['FULL_PF_4th'])
        with self.assertRaisesRegex(ValueError,'COMPILER_PROFILE'):
            D.join_saved_coverage(bound,self.cov)  # Published example uses MOCK_COMPILER.
        supplied=deepcopy(bound['physical']['spec']);supplied['profile']=deepcopy(p.PROFILE)
        bound=D.bind_physical(supplied)  # Explicit artificial metadata fixture only.
        rows=D.join_saved_coverage(bound,self.cov)
        self.assertEqual(len(rows),4)
        self.assertEqual(rows[0]['group'],'OUTSIDE_SAVED_GRID_ALL_MISSING')
        bad=deepcopy(bound['physical']['spec']);bad['binding']['state']='OTHER'
        changed=D.bind_physical(bad)
        with self.assertRaises(ValueError):D.join_saved_coverage(changed,self.cov)
        with self.assertRaises(ValueError):D.join_saved_coverage(D.accept_example(self.examples['FULL_H_RTE_explicit_layout']),self.cov)

    def test_12_no_inventory_reanalysis_scientific_provider_cost_or_compile(self):
        source=D.published_module('coverage')
        with patch.object(source,'load_carriers',side_effect=AssertionError('No private/raw reads')), \
             patch.object(source,'coverage',side_effect=AssertionError('No repeat inventory')), \
             patch.object(B,'cost',side_effect=AssertionError('No real G')):
            D.accept_saved_coverage(self.cov)
            supplied=deepcopy(self.examples['FULL_PF_4th']['spec']);supplied['profile']=deepcopy(p.PROFILE)
            bound=D.bind_physical(supplied);D.join_saved_coverage(bound,self.cov)
            E.plan([dict(configuration=self.config,N_packet=self.previous)],price_policy='qualified_expectation')
        forbidden={'qiskit','pyscf','h46_long_time_signal','h46_long_time_native','h46_full_h_pure_rte','h4_pr_m1_rmse','h4_pr_physical_closure'}
        self.assertFalse(forbidden & {k.split('.')[-1] for k in sys.modules})

    def test_13_orchestrator_accepts_only_explicit_mock_publications_and_never_costs(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import intake_rpe_new_evidence as CLI
        packet=mock_m1(self.previous,self.adapter)
        ar=dict(commit='4'*40,path='MOCK_RESULT.json');ap=dict(commit='4'*40,path='MOCK_PACKET.json');cp=dict(commit='5'*40,path='MOCK_C.json')
        result=dict(science_source=packet['source']['commit'],configuration_id=self.config['id'],status='M1_NOMINAL_COMPLETE_UNQUALIFIED',
            model_record=dict(nominal_N_X=packet['rounds'][1]['nominal_N_X'],nominal_N_Y=packet['rounds'][1]['nominal_N_Y'],
                status=packet['rounds'][1]['model_status'],identity=dict(candidate=self.config['rounds'][1]['physical_candidate'])))
        read=p.SavedArtifacts.read
        def supplied(store,ref):
            for key,value in ((ar,result),(ap,packet),(cp,structural(self.config))):
                if ref==key:return deepcopy(value)
            return read(store,ref)
        with patch.object(p.SavedArtifacts,'read',supplied),patch.object(B,'cost',side_effect=AssertionError('No real G')):
            out=CLI.build(a_result_ref=ar,a_packet_ref=ap,c_structural_ref=cp)
        self.assertEqual(out['PUBLISHED_EVIDENCE']['A']['new_nominal_rounds'],[1])
        self.assertEqual(out['PUBLISHED_EVIDENCE']['C']['new_kernel_evidence']['compiled_price_records'],[])
        self.assertIsNone(out['PUBLISHED_EVIDENCE']['qualified_mean_compiled_C'])
        self.assertTrue(out['PUBLISHED_EVIDENCE']['C']['new_kernel_evidence']['complete_four_cell_values'])
        self.assertFalse(out['METHOD_GAPS']['FULL_PF']['scientific_menu_automatically_adopted'])


if __name__=='__main__':unittest.main(verbosity=2)
