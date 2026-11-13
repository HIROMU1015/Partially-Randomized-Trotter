"""Artificial whole-task integration. No real H4 input/price/risk qualification."""
import ast
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from check_baseline_signal_pipeline import fixture as baseline_fixture
from trottertracks import rpe_optimization_pipeline as I
from trottertracks import baseline_signal_numerics as N
from trottertracks.r_price_phase1_science import functions

ROOT=Path(__file__).resolve().parents[1]
API=I.providers(str(ROOT)); B,S,W=API.B,API.S,API.W
MOCK=dict(fixture='ARTIFICIAL_INTEGRATION_ONLY')
DATA_SCOPE=dict(m=B,deepcopy=deepcopy)
functions(ROOT,I.B_SOURCE,'tests/check_pr_rpe_rmse_controller.py',('data',),DATA_SCOPE)


def toy(method='FULL',M=1):
    old,ctx,reader=baseline_fixture('FULL' if method=='PR' else method,M=M,prefix=2 if method in ('B0','B0I') else 10)
    spec=deepcopy(old['configuration']['spec'])
    # A's physical recipes are binary64. Preserve that original B input type;
    # decimal rational 1/10 is a different canonical time, not an alias for .1.
    spec['h']=.1
    if method!='FULL_H_RTE':spec['delta_pf']=.05
    spec['rmse_policy']=dict(family='PR_PAPER_RMSE',id='ARTIFICIAL_COMMON_RISK',source=MOCK,
        target_rmse_ha=dict(numerator=1,denominator=10),scientific_status='PROPOSED_NOT_ADOPTED',
        error_contract=dict(fixture='ARTIFICIAL_ONLY',decoder=S.DECODER,energy_prior_ha=[-4,4]))
    if method=='PR':
        spec.update(method='PR',L_D=1,finite_K=2,kappa=dict(kind='PAPER_TOTAL_ROTATIONS',value=1,definition_source=MOCK),
            event_mapping=dict(kind=B.DRAFT_MAPPER,source=MOCK),law=dict(model='ideal_finite_RTE_law',source=MOCK))
        spec['split']=dict(L_D=1,source=MOCK,lambda_R=.5,lambda_source=MOCK,
            one_body_policy='DETERMINISTIC',deterministic_scope='PREFIX',randomized_scope='TAIL_AFTER_PREFIX')
    layouts={m:dict(q=1,R_outer=2) for m in range(M+1)} if method=='FULL_H_RTE' else None
    config=I.C.configure(ROOT,spec,layouts=layouts,api=API)
    target=dict(old['target'],energy_ha=0.,uncertainty_ha=None)
    references=[dict(m=r['m'],T=r['T'],binding=spec['binding'],source=MOCK) for r in config['rounds']]
    physical=None
    if method=='PR':
        physical={}
        for r in config['rounds']:
            c=dict(system='H4',method='PR',formula=spec['formula'],prefix=1,T=float(B.a.exact_scalar(r['T'])),
                q=r['q_PF'],delta=float(B.a.exact_scalar(r['actual_delta_pf'])),R_outer=r['R_outer'],K=2)
            c['id']=B.p.candidate_id(c);physical[r['m']]=c
    request=I.plan(ROOT,config,target=target,signal_source=MOCK,pr_physical=physical,references=references)
    if method=='PR':
        pctx=API.A.make_context(ctx.matrices,ctx.state,ctx.constant,
            {1:dict(scalar=.004,lambda_R=.5,source=MOCK)},binding=spec['binding'],source=MOCK)
        def factory(cfg):return pctx
        def reference(job):
            c=job['physical_context']['physical_candidate']
            desc=dict(m=job['m'],T=c['T'],binding=spec['binding'],source=MOCK)
            value=reader(desc)
            return dict(binding=spec['binding'],T=c['T'],source=MOCK,
                evidence={k:value[k] for k in ('reference','independent_reference','signal_difference')})
    else:
        factory=lambda:ctx;reference=reader
    return request,factory,reference


def run_toy(method='FULL',M=1):
    req,factory,ref=toy(method,M)
    with tempfile.TemporaryDirectory() as td:
        packet=I.produce_signals(ROOT,req,Path(td)/'new',context_factory=factory,reference_provider=ref)
    return req,packet


def fixture(method='FULL',M=1,qualified=True):
    req,packet=run_toy(method,M)
    # Test-only external qualification. This code is never part of a provider.
    # Physical state/H and all N/C/risk values are explicitly fictional.
    if qualified:
        packet=deepcopy(packet)
        for mean in packet['means']:
            mean['qualification']='QUALIFIED';mean['uncertainty']['fixture_qualification']=MOCK
    config=req['configuration']
    contract=dict(id='ARTIFICIAL_STATISTICAL_CONTRACT',source=MOCK,claim_level='SYNTHETIC_ONLY',
        simulation_policy=dict(kind=S.IID,source=MOCK,target_law=config['spec']['law']['model']),
        allow_conditional_unqualified=True,
        summary_options=dict(unresolved_loss=dict(kind='UNDEFINED',source=MOCK),iid_repetitions=True))
    vectors=((1,5),(8,1)) if M==1 else (tuple([2]*(M+1)),tuple([3]*(M+1)))
    allocations=[dict(allocation_id=key,source=MOCK,shot_policy='USER_SUPPLIED',
        shots=[dict(m=m,N_X=n,N_Y=n) for m,n in enumerate(v)]) for key,v in zip(('few_shots','low_G'),vectors)]
    evaluations=[]
    for allocation in allocations:
        n,prices,cohorts=DATA_SCOPE['data'](config)
        for row,shot in zip(n['rounds'],allocation['shots']):row.update(N_X=shot['N_X'],N_Y=shot['N_Y'])
        n['rmse_evidence']['statistical_contract']=deepcopy(contract)
        if method in ('B0','B0I'):
            n['rmse_evidence'].update(truncation_status='QUALIFIED',covers_truncation_error=True,
                target_hamiltonian=deepcopy(config['spec']['binding']['hamiltonian']),
                retained_hamiltonian=deepcopy(config['spec']['split']['retained_hamiltonian']))
        evaluations.append(dict(configuration_id=config['id'],allocation=deepcopy(allocation),
            statistical_contract=deepcopy(contract),source=n['source'],n_packet=n,signal_input=I.signal_input(packet)))
    extension=dict(lowering_id='ARTIFICIAL_LOWERING',source=MOCK)
    coverage=dict(id='ARTIFICIAL_BOX',source=MOCK,configuration_id=config['id'],simultaneous=True,scope='ALL_SELECTED_ROUNDS_AXES')
    bindings=[]
    for price in prices:
        m=price['identity']['m'];v=1 if m==0 else 100
        price.update(quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE',mean=dict(rz_count=v),
            law_context=dict(N_law=config['spec']['law']['model'],price_law=config['spec']['law']['model'],
                ideal_sampler_distance_qualified=True,fixture=MOCK))
        bindings.append(dict(identity=deepcopy(price['identity']),law_binding=deepcopy(price['law_binding']),
            axis=price['axis'],cohort=price['cohort'],price_source=deepcopy(price['source']),source=MOCK,
            physical_context=I.price_context(req,m),profile_extension=deepcopy(extension),mean_interval=dict(
                quantity='FULL_WRAPPER_MEAN_COMPILED_RZ_INTERVAL',lower=v,upper=v,source=MOCK,
                qualification='QUALIFIED',evidence_kind='SIMULTANEOUS_MEAN_INTERVAL',domain=deepcopy(price['identity']['candidate']),
                coverage=deepcopy(coverage))))
    opts=dict(statistical_contract=contract,cohorts=cohorts,price_extension=extension,price_bindings=bindings,interval_contract=coverage)
    return req,packet,allocations,evaluations,prices,opts


def compare(values):return I.compare(ROOT,*values[:5],**values[5])


class IntegrationTests(unittest.TestCase):
    def test_01_all13_rounds_from_all_four_real_providers_on_artificial_matrices(self):
        identities=[]
        with patch.object(N,'load_shared_h4',side_effect=AssertionError('no molecular data')):
            for method in ('PR','FULL','B0I','FULL_H_RTE'):
                req,packet=run_toy(method,12);identities.append(req['configuration']['id'])
                self.assertEqual(len(packet['means']),13)
                prices=I.price_requests(ROOT,req)
                self.assertEqual(len(prices),26)
                self.assertTrue(all(p['mean'] is None and p['profile_extension'] is None for p in prices))
                self.assertTrue(all(p['law_context']['ideal_sampler_distance_qualified'] is None for p in prices))
                self.assertTrue(all(v['raw'] is not None for v in packet['means']),packet['producer_result'].get('errors'))
                self.assertTrue(all(v['qualification']=='UNQUALIFIED' for v in packet['means']))
                allocations=[dict(allocation_id='only',source=MOCK,shot_policy='USER_SUPPLIED',
                    shots=[dict(m=m,N_X=2,N_Y=3) for m in range(13)])]
                entries=W.prepare(req['configuration'],packet['means'],allocations,target=req['target'],
                    statistical_contract=dict(id='MOCK',source=MOCK,claim_level='SYNTHETIC_ONLY'))
                self.assertEqual([r['mean']['raw'] for r in entries[0]['prepared']['rows']],[m['raw'] for m in packet['means']])
                for mean,row in zip(packet['means'],req['configuration']['rounds']):
                    self.assertEqual(mean['identity'],row['identity']);self.assertEqual(mean['law_binding'],row['law_binding'])
                self.assertIsNone(packet['producer_result'].get('G'))
        self.assertEqual(len(set(identities)),4)

    def test_02_qualified_fictional_N_prices_existing_exact_G_and_outer_select(self):
        results=[]
        for method in ('PR','FULL','B0I','FULL_H_RTE'):
            values=fixture(method);out=compare(values);results.append(out)
            costs=[B.a.exact_scalar(r['G']) for r in out['comparison']['allocations']]
            self.assertEqual(costs,[1002,216]);self.assertEqual(out['comparison']['point_minimum']['allocation_id'],'low_G')
            self.assertEqual(out['comparison']['strict_interval_winner'],'low_G')
            self.assertEqual([sum(s['N_X']+s['N_Y'] for s in a['shots']) for a in values[2]],[12,18])
            self.assertEqual(out['comparison']['outer_evaluation']['configuration_id'],values[0]['configuration']['id'])
        selected=I.select(ROOT,results)
        self.assertTrue(selected['fair_method_comparison_complete']);self.assertTrue(selected['menu_complete'])
        self.assertEqual(B.a.exact_scalar(selected['winner']['cost']),216)
        self.assertFalse(selected['scientific_optimum_adopted']);self.assertFalse(selected['cross_method_advantage_qualified'])

    def test_03_existing_decoder_diagnostic_consumes_raw_and_never_qualifies_N(self):
        values=fixture('PR',qualified=False);out=compare(values)
        entry=out['shot_entries'][0]
        with patch.object(B.p,'run_missing',side_effect=AssertionError('no provider fallback')):
            d=W.diagnose(entry,repetitions=7,seed=12,source=MOCK,batch_decoder=API.F.decode_sampled_counts)
        actual=d['diagnostic']['simulation']['input_raw_means']
        self.assertEqual([r['raw'] for r in actual],[m['raw'] for m in values[1]['means']])
        self.assertEqual(len(d['diagnostic']['simulation']['trials']),7)
        self.assertIsNone(d['n_packet']);self.assertIsNone(d['G'])
        self.assertIsNone(out['comparison']['outer_evaluation'])

    def test_04_point_minimum_is_not_interval_dominance(self):
        values=fixture()
        for side in values[5]['price_bindings']:side['mean_interval'].update(lower=0,upper=1000)
        result=compare(values)['comparison']
        self.assertEqual(result['point_minimum']['allocation_id'],'low_G');self.assertIsNone(result['strict_interval_winner'])
        self.assertFalse(I.select(ROOT,[compare(values)])['strict_cost_dominance_qualified'])

    def test_05_missing_unqualified_and_nominal_never_reach_cost(self):
        changes=(lambda v:v[1]['means'][0].update(raw=None),
            lambda v:[m.update(qualification='UNQUALIFIED') for m in v[1]['means']],
            lambda v:[e['n_packet']['rmse_evidence'].update(status='UNQUALIFIED') for e in v[3]],
            lambda v:v[4].pop(),lambda v:v[4][0].update(expectation='FINITE_COHORT_ONLY'))
        for change in changes:
            v=fixture();change(v)
            for e in v[3]:e['signal_input']=I.signal_input(v[1])
            with patch.object(B,'cost',side_effect=AssertionError('no ineligible G')):out=compare(v)
            self.assertIsNone(out['comparison']['outer_evaluation'])
        v=fixture()
        for a,e in zip(v[2],v[3]):a['shot_policy']='NOMINAL';e['allocation']=deepcopy(a)
        with patch.object(B,'cost',side_effect=AssertionError('nominal')):out=compare(v)
        self.assertIsNone(out['comparison']['point_minimum']['cost'])

    def test_06_source_signal_law_physical_and_lowering_mismatches(self):
        for change in (lambda v:v[1].update(source={'fixture':'FOREIGN'}),
            lambda v:v[1]['means'][0]['provider_binding']['physical_context']['physical_candidate'].update(id='FOREIGN'),
            lambda v:v[1]['means'][0].update(source={'fixture':'FOREIGN'}),
            lambda v:v[3][0]['signal_input']['means'][0]['raw'].update(real=0),
            lambda v:v[5]['price_bindings'][0]['physical_context']['physical_candidate'].update(id='FOREIGN'),
            lambda v:v[5]['price_bindings'][0]['physical_context']['profile'].update(compiler='FOREIGN')):
            v=fixture();change(v)
            with self.assertRaises(ValueError):compare(v)
        for field,value in (('law_binding',{}),('price_source',{'fixture':'FOREIGN'}),('profile_extension',{'lowering_id':'OLD'})):
            v=fixture();v[5]['price_bindings'][0][field]=value
            with patch.object(B,'cost',side_effect=AssertionError('unbound D input')):out=compare(v)
            self.assertIsNone(out['comparison']['outer_evaluation'])
        v=fixture();v[4][0]['identity']['profile']['compiler']='FOREIGN'
        self.assertIsNone(compare(v)['comparison']['point_minimum']['cost'])

    def test_07_missing_connections_preserve_pending_scope(self):
        v=fixture();del v[3][0]['signal_input']
        out=compare(v);self.assertIn('few_shots',out['comparison']['pending_allocation_ids'])
        self.assertIsNone(out['comparison']['outer_evaluation'])
        v=fixture();del v[5]['price_bindings'][0]['physical_context']
        out=compare(v);self.assertIn('D_PHYSICAL_PRICE_BINDING_MISSING',out['blockers'])
        self.assertIsNone(out['comparison']['point_minimum']['cost'])
        selected=I.select(ROOT,[out]);self.assertFalse(selected['menu_complete'])
        v=list(fixture());v[4]=I.price_requests(ROOT,v[0])
        with patch.object(B,'cost',side_effect=AssertionError('not-evaluated price requests')):out=compare(v)
        self.assertIsNone(out['comparison']['point_minimum']['cost'])
        self.assertTrue(all(p['mean'] is None for p in out['supplied_prices']))

    def test_08_N_allocation_binding_and_synthetic_scientific_separation(self):
        v=fixture();v[3][0]['n_packet']['rounds'][0]['N_X']=999
        with self.assertRaises(ValueError):compare(v)
        a=compare(fixture('FULL'));b=compare(fixture('PR'))
        b['supplied_evaluations'][0]['n_packet']['evidence_kind']='SCIENTIFIC'
        with self.assertRaisesRegex(ValueError,'Synthetic and scientific'):I.select(ROOT,[a,b])

    def test_09_PR_failure_and_changed_source_not_repaired(self):
        req,ctx,ref=toy('PR')
        def failure(*args,**kwargs):raise API.A.NumericalFailure('ARTIFICIAL_FAIL',dict(residual=3.))
        with tempfile.TemporaryDirectory() as td,patch.object(API.A,'_macros',side_effect=failure):
            out=Path(td)/'fresh';packet=I.produce_signals(ROOT,req,out,context_factory=ctx,reference_provider=ref)
            self.assertEqual(packet['producer_result']['records'][0]['numerical']['residual'],3.)
            self.assertTrue((out/'result.json').exists());self.assertEqual(packet['producer_result']['not_started_rounds'],[1])
            self.assertTrue(all(m['raw'] is None for m in packet['means']))
        req['signal_source']={'fixture':'FOREIGN'}
        with self.assertRaises(ValueError):I._validate(ROOT,req)

    def test_10_passive_import_and_pinned_API(self):
        code="from trottertracks import rpe_optimization_pipeline as i;import sys;i.providers('.');assert not any(v in sys.modules for v in ('numpy','scipy','qiskit','pyscf'))"
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True,capture_output=True)
        self.assertEqual(API.W.B.__name__,API.B.__name__)
        self.assertEqual(API.A.RECIPE,'DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1')


# Two unchanged B metadata regressions; reuse source bodies/fixtures, without
# copying the entire B branch or running its molecular/scientific entries.
old=ast.parse(I.C.blob(ROOT,I.B_SOURCE,'tests/check_pr_rpe_rmse_controller.py'))
cls=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='ControllerTests')
names={'test_shared_kappa_exact_schedule_integer_rounding','test_signed_schedule_and_existing_allocation_are_reused'}
cls.name='PinnedBMetadataTests';cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
nodes=[n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='spec']+[cls]
scope=dict(m=B,unittest=unittest,deepcopy=deepcopy)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'pinned_B_metadata_regressions','exec'),scope)
PinnedBMetadataTests=scope['PinnedBMetadataTests']


if __name__=='__main__':unittest.main(verbosity=2)
