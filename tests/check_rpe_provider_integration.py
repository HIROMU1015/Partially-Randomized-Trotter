"""B integration: immutable metadata and fully mocked numerical leaves only."""
from contextlib import contextmanager,ExitStack
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import rpe_provider_integration as b
from trottertracks import h6_df19_price_provider as h6
from trottertracks import h4_df10_price_provider as h4
from check_pf_pr_rpe_pipeline import fake_shot
from check_pr_r_dependent_n import pair, kernel
p=b.p;r=b.r


@contextmanager
def passive():
    with ExitStack() as stack:
        for obj,name in [(b.nmodel,'analyze'),(p,'evaluate'),(r,'evaluate_menu'),(p.accounting,'aggregate_supplied_axes'),
                         (h6,'_affine'),(h6.H6AffinePriceProvider,'predict_affine'),(h4.H4PriceProvider,'predict_path')]:
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('Metadata only')))
        yield


def mock_N(menu,row):
    n=fake_shot(menu,row);R=row['candidate']['R_outer']
    n['plan'].update(N_X={4:20,8:5,16:3,32:2}.get(R,3),N_Y={4:20,8:5,16:3,32:2}.get(R,3))
    n.update(law_binding=r.request_context(menu,row)['law_binding'],N_comparison_policy=b.N_POLICY,
        diagnostics=dict(synthetic_only=True,allowance={'policy':b.N_POLICY}))
    return n


def mock_C(menu,rows):
    prices=[]
    for row in rows:
        for axis in p.AXES:
            pr=p.arm(row['candidate'])=='PR'
            prices.append(dict(**p.measurement(row,axis),**r.request_context(menu,row),axis=axis,cohort='MOCK_ONLY',
                classification='MODEL_PREDICTION' if pr else 'DIRECT_COMPILED_DETERMINISTIC',
                expectation='QUALIFIED_EXPECTATION' if pr else 'DETERMINISTIC',
                mean={'rz_count':row['candidate']['R_outer'] if pr else 1},n_paths=999,
                source={'fixture':'MOCK_PRICE'},normalization_is_price_multiplier=False,peak_qubits=None,
                model=None if not pr else dict(source={'fixture':'MOCK'},validation_source={'fixture':'MOCK'},
                    coverage='MOCK_R_DOMAIN',qualified_for_candidate=True,q_variable='q_PF',q=row['q_PF'],
                    domain=deepcopy(row['candidate']),binding=deepcopy(menu['binding']),compiler=menu['profile']['compiler']),
                uncertainty={'joint_covariance':None,'total_CI':None,'shared_calibration_group':'MOCK_ONLY'}))
    return prices


class ProviderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with passive():cls.inputs=b.saved.load_inputs(ROOT,p.SavedArtifacts(ROOT))

    def packet(self,system='H4',mapping=None):
        return b.prepare_menu(ROOT,system,mapping or {10:[4,8,16],12:[4,16,32]},inputs=self.inputs)

    def test_fixed_A_C_sources_preserved_and_local_factories_work(self):
        for commit,names in [(b.A_RELEASE,['src/trottertracks/pr_r_dependent_n.py','scripts/plan_pr_r_dependent_n.py']),
                             (b.C_RELEASE,['src/trottertracks/h6_df19_price_provider.py','src/trottertracks/rpe_nc_coverage.py','scripts/report_rpe_nc_coverage.py'])]:
            for name in names:self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT))
        self.assertIs(p.h6_saved_price_provider(p.SavedArtifacts(ROOT).read).pipeline,p)
        self.assertEqual(Path(p.__file__).resolve(),ROOT/'src/trottertracks/pf_pr_rpe_pipeline.py')

    def test_bound_loaders_are_lazy_and_preserve_system_specific_state_metadata(self):
        from trottertracks import h4_df10_rpe_batch_provider as h4loader, h6_rpe_batch_provider as h6loader
        for system,loader in [('H4',h4loader),('H6',h6loader)]:
            base=b.nmodel.saved_task(ROOT,system)
            task=b.nmodel.with_round_R(base,{12:8 if system=='H4' else 32},{'fixture':'MOCK'})['task']
            ctx=SimpleNamespace()
            metadata={'state_energy':-1.,'state_residual':0.} if system=='H4' else {'solver_energy':-1.,'solver_residual':0.}
            with patch.object(loader,'load_saved_context',return_value=(ctx,metadata)) as loaded:
                factory=b.bound_context_factory(ROOT,system);loaded.assert_not_called()
                if system=='H4':
                    with patch.object(loader,'prepare',return_value={'task':base}):result,state=factory(task)
                else:result,state=factory(task)
            loaded.assert_called_once();self.assertIs(result,ctx)
            self.assertEqual(ctx.scientific_binding,task['binding'])
            self.assertEqual(state,{'state_energy':-1.,'state_residual':0.})

    def test_saved_all13_N_uses_H4_corrected_integer_handoff_and_no_old_R_substitution(self):
        with passive():
            for system,mapping in [('H4',{12:[8,16]}),('H6',{12:[32,64]})]:
                packet=self.packet(system,mapping);menu=packet['menu']
                self.assertEqual(len(packet['shots']),25);self.assertEqual(len(packet['excluded_old_R_N']),1)
                self.assertTrue(all(n['identity']['candidate']['id'] in [v['candidate']['id'] for v in menu['candidates']] for n in packet['shots']))
                self.assertIsNone(packet['total_resources'])
                if system=='H6':self.assertEqual(menu['candidates'][:8],packet['base']['candidates'][:8])
        hand=p.SavedArtifacts(ROOT).read(b.saved.H4_PUBLISHED_N)
        for n in self.inputs['H4']['shots']:
            old=next(v for v in hand['shots'] if p.key(v['identity'])==p.key(n['identity']))
            self.assertEqual(n['plan'],old['plan']);self.assertIs(type(n['plan']['N_X']),int)
        m10=next(n for n in hand['shots'] if n['identity']['m']==10 and n['identity']['candidate']['method']=='PR')
        self.assertEqual(str(m10['plan']['N_X']),'1070910265913026585642')

    def test_allowance_policy_gap_is_metadata_and_blocks_unsupported_comparison(self):
        with passive():
            packet=self.packet('H4',{12:[4,8]});menu=packet['menu']
            out=r.coverage(menu,packet['shots'],packet['prices'],cohort_selection={})
        old=next(v for v in out['rows'] if v['m']==12 and v['arm']=='PR' and v['candidate']['R_outer']==4)
        new=next(v for v in out['rows'] if v['m']==12 and v['arm']=='PR' and v['candidate']['R_outer']==8)
        self.assertEqual(old['N_status'],'SAVED_CONDITIONAL')
        self.assertIn('MISSING_MATCHED_N_ALLOWANCE_POLICY',old['reasons'])
        self.assertIn('MISSING_N',new['reasons'])

    def test_H6_saved_price_domain_forecast_cohort_and_unknowns_remain_distinct(self):
        with passive():
            packet=self.packet('H6',{6:[16,32]});menu=packet['menu']
            selections={row['candidate']['id']:dict(kind='model_prediction',**(dict(price_unit='cohort_mean',
                cohort='H6_PRICE_V2_20261110_Q4_INDEPENDENT') if p.arm(row['candidate'])=='PR' else {})) for row in menu['candidates']}
            provider=b.FixedSavedPrices(p.SavedArtifacts(ROOT).read,'H6',selections=selections)
            rows=[v for v in menu['candidates'] if v['m'] in (6,7)]
            records=provider(menu,rows)
        changed=[v for v in records if v['identity']['candidate']['R_outer']==32]
        self.assertTrue(changed);self.assertTrue(all(v['mean']['rz_count'] is None for v in changed))
        independent=[v for v in records if v['identity']['m']==6 and v['identity']['candidate']['R_outer']==16]
        self.assertTrue(all(v['model']['domain']==v['identity']['candidate'] for v in independent))
        self.assertTrue(all(v['expectation']=='FINITE_COHORT_ONLY' for v in independent))
        self.assertTrue(all(v['provider_quote']['validation']=='UNVALIDATED_EXTRAPOLATION' and v['mean']['rz_count'] is None for v in records if v['identity']['m']==7))
        for v in records:
            self.assertIsNone(v['uncertainty']['joint_covariance']);self.assertIsNone(v['uncertainty']['total_CI'])
            self.assertFalse(v['provider_quote']['full_RPE_price_adopted'])

    def test_RBatch_multiple_alternatives_reaches_exact_mock_cost_and_R_selection(self):
        io,backend=kernel();packet=self.packet();menu=packet['menu']
        changed=[row for row in menu['candidates'] if row['m'] in (10,12) and p.arm(row['candidate'])=='PR' and row['candidate']['R_outer']!=4]
        shots=[mock_N(menu,row) for row in menu['candidates'] if row not in changed]
        ctx=SimpleNamespace(scientific_binding=menu['binding'],half_steps={});factory=Mock(return_value=(ctx,{}))
        refs=Mock(side_effect=lambda ctx,T:dict(binding=deepcopy(menu['binding']),T=T,source={'fixture':'MOCK_REFERENCE'},evidence=pair()))
        provider=b.RMenuBatch(menu,packet['base'],{'fixture':'MOCK_N'},factory,refs)
        prices=Mock(side_effect=mock_C)
        huge=(1<<1200)+7
        def analyzed(task,row,*args,**kwargs):
            value=mock_N(menu,next(v for v in changed if p.key(v)==p.key(row)))
            if row['candidate']['R_outer']==32:
                value['plan'].update(N_X=huge,N_Y=huge)
                value['diagnostics'].update(N_X_decimal=str(huge),N_Y_decimal=str(huge))
            return value
        with (patch.object(backend,'half_step',side_effect=lambda ctx,k:(k,None)),
              patch.object(io,'worker_resources',side_effect=lambda ctx,n:dict(half_step_workers=max(1,n))),
              patch.object(b.nmodel,'check_normalizers'),patch.object(backend,'evolve',return_value={}) as action,
              patch.object(b.nmodel,'analyze',side_effect=analyzed)):
            result=b.run_missing(menu,shots,[],n_provider=provider,c_provider=prices)
        factory.assert_called_once();self.assertEqual(refs.call_count,2);self.assertEqual(action.call_count,4);prices.assert_called_once()
        self.assertEqual(len(provider.last_run['groups']),2);self.assertIsNone(result['totals'])
        value=next(v for v in result['shots'] if v['identity']['candidate']['R_outer']==32)
        self.assertEqual(value['plan']['N_X'],huge)
        self.assertEqual(json.loads(json.dumps(value))['plan']['N_X'],huge)
        select={(v['identity']['candidate']['id'],v['axis']):v['cohort'] for v in result['prices']}
        never=Mock(side_effect=AssertionError('Reuse supplied N/C on one-entry evaluation'))
        out=b.run_evaluation(menu,result['shots'],result['prices'],cohort_selection=select,n_provider=never,c_provider=never)
        never.assert_not_called()
        self.assertTrue(out['menu_complete']);self.assertFalse(out['full_RPE_qualified'])
        self.assertEqual([v['R_outer'] for v in out['selections'] if v['arm']=='PR' and v['m'] in (10,12)],[8,16])
        self.assertFalse(out['rpe_success_qualified']);self.assertIsNone(out['uncertainty']['cross_candidate_covariance'])

    def test_fail_stops_remaining_groups_and_C_and_preserves_no_retry(self):
        io,backend=kernel();packet=self.packet('H4',{12:[4,8,16]});menu=packet['menu']
        changed=[v for v in menu['candidates'] if v['m']==12 and v['candidate']['R_outer'] in (8,16)]
        shots=[mock_N(menu,v) for v in menu['candidates'] if v not in changed]
        ctx=SimpleNamespace(scientific_binding=menu['binding'],half_steps={})
        provider=b.RMenuBatch(menu,packet['base'],{'fixture':'MOCK_FAIL'},lambda task:(ctx,{}),
            lambda ctx,T:dict(binding=deepcopy(menu['binding']),T=T,source={'fixture':'MOCK_REF'},evidence=pair()))
        never=Mock(side_effect=AssertionError('No C or retry after N failure'))
        with (patch.object(backend,'half_step',side_effect=lambda ctx,k:(k,None)),
              patch.object(io,'worker_resources',side_effect=lambda ctx,n:dict(half_step_workers=max(1,n))),
              patch.object(b.nmodel,'check_normalizers'),patch.object(backend,'evolve',side_effect=ValueError('MOCK_PHASE_GATE')) as action):
            out=b.run_missing(menu,shots,[],n_provider=provider,c_provider=never)
        action.assert_called_once();never.assert_not_called();self.assertEqual(out['status'],'STOP_UNQUALIFIED_N')
        self.assertTrue(any(n['plan']['status']=='NOT_EXECUTED_AFTER_N_FAILURE' for n in out['shots']))
        b.run_missing(menu,out['shots'],[],n_provider=never,c_provider=never);never.assert_not_called()
        self.assertFalse(provider.last_run['automatic_retry'])

    def test_future_large_N_packet_and_wrong_law_or_fixed_rows(self):
        packet=self.packet('H6',{12:[16,32]});menu=packet['menu']
        huge=(1<<1200)+7;n=mock_N(menu,menu['candidates'][-1]);n['plan'].update(N_X=huge,N_Y=huge)
        self.assertEqual(json.loads(json.dumps(n))['plan']['N_X'],huge)
        factory=Mock(side_effect=AssertionError('Preflight first'))
        provider=b.RMenuBatch(menu,packet['base'],{'fixture':'MOCK'},factory)
        with self.assertRaisesRegex(ValueError,'saved PF'):provider(menu,[menu['candidates'][0]])
        bad=deepcopy(menu);bad['sampling_law']['sampling']='FOREIGN'
        with self.assertRaisesRegex(ValueError,'law binding'):b.RMenuBatch(bad,packet['base'],{'fixture':'MOCK'},factory)
        factory.assert_not_called()
        prices=mock_C(menu,menu['candidates']);shots=[mock_N(menu,v) for v in menu['candidates']]
        self.assertTrue(all(v['eligible'] for v in r.coverage(menu,shots,prices,cohort_selection={
            (v['identity']['candidate']['id'],v['axis']):v['cohort'] for v in prices})['rows']))


if __name__=='__main__':unittest.main(verbosity=2)
