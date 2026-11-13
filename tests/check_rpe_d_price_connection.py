"""D metadata + fictional path/mean/N only. No preparation/sampling/compile."""
from copy import deepcopy
import ast
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import check_rpe_optimization_pipeline as T
from trottertracks import rpe_optimization_pipeline as I
from trottertracks import rpe_d_price_connection as P

ROOT=T.ROOT;API=I.providers(str(ROOT));D=API.Price
SOURCE=dict(fixture='ARTIFICIAL_D_EXECUTION_ONLY')
INPUT=dict(fixture='ARTIFICIAL_INPUT_SOURCE_ONLY')
A_MENU='a9abf8de75bf3432fcd714e868e5c14c8884b8f8'


def pending(req,axes=('X','Y'),**kwargs):
    return I.price_requests(ROOT,req,input_source=INPUT,execution_source=SOURCE,axes=axes,**kwargs)


def path(d,*,axes=None,cohort='ARTIFICIAL_NORMAL',index=0,conditional=False):
    pc=d['physical_context'];prefix=pc['physical_candidate']['prefix']
    records=[dict(axis=axis,compiled_quantity='FULL_WRAPPER_PATH_COMPILED_RZ_COUNT',
        structural_quantity='FULL_WRAPPER_PATH_STRUCTURAL_RZ_COUNT',C=51,S=45,D=6,
        metrics=dict(rz_count=51),fingerprint='ARTIFICIAL_FINGERPRINT_'+axis,cache_hit=False,cache_reference=None)
        for axis in (d['axes'] if axes is None else axes)]
    return dict(quantity=P.PATH,request=deepcopy(d),resolved_context=dict(preparation_hash='ART_PREP_'+str(prefix),
        tail_hash='ART_TAIL_'+str(prefix),lambda_R=pc['split']['lambda_R'],scalar=.07,
        deterministic_fragments=list(range(prefix)),randomized_fragments=list(range(prefix,pc['binding']['rank']))),
        resolved_context_key='ARTIFICIAL_RESOLVED_KEY',records=records,cohort=cohort,index=index,conditional=conditional,
        seeds=[],laws=[],event_digest='ARTIFICIAL_EVENT_DIGEST',new_compiles=0,cache_hits=0,
        mean_compiled_price=None,confidence_interval=None,qualified_expectation=False,
        source=dict(execution=deepcopy(pc['execution_source']),input=deepcopy(pc['input_source']),private_result='ARTIFICIAL_NO_FILE'))


def fixture():
    req,signals,alloc,ev,prices,opts=T.fixture('PR')
    rows=pending(req);lookup={(r['m'],r['axis']):r['d_request'] for r in rows}
    bindings=deepcopy(opts['price_bindings'])
    for c,b in zip(prices,bindings):
        d=lookup[c['identity']['m'],c['axis']]
        c['price_profile']=deepcopy(d['physical_context']['price_profile'])
        b.update(profile_extension=P.extension(d),d_request_id=d['request_id'],d_context_key=d['context_key'])
    mapped=P.bind_results(ROOT,req,rows,prices=prices,bindings=bindings,interval_contract=opts['interval_contract'])
    return req,signals,alloc,ev,rows,prices,bindings,opts,mapped


def compare(f,mapped=None,policy='qualified_expectation'):
    m=mapped or f[8]
    return I.compare(ROOT,f[0],f[1],f[2],f[3],m['prices'],statistical_contract=f[7]['statistical_contract'],
        cohorts=m['cohorts'],price_extension=m['price_extension'],price_bindings=m['price_bindings'],
        interval_contract=m['interval_contract'],price_policy=policy)


class DConnectionTests(unittest.TestCase):
    def test_01_D_A_B_metadata_bridge_and_no_scientific_entry(self):
        req,_,_=T.toy('PR',2)
        with (patch.object(D,'NativeRuntime',side_effect=AssertionError('no preparation')),
              patch.object(D,'Adapter',side_effect=AssertionError('no price acquisition'))):
            rows=pending(req);self.assertEqual(len(rows),6)
        for row in rows:
            d=row['d_request'];m=row['m']
            direct=D.from_b(req['configuration'],m,req['pr_physical'][m],input_source=INPUT,execution_source=SOURCE)
            self.assertEqual(d,direct);self.assertEqual(row['law_binding'],req['configuration']['rounds'][m]['law_binding'])
            self.assertIsNone(row['mean']['rz_count']);self.assertIsNone(row['law_context']['ideal_sampler_distance_qualified'])
            self.assertEqual(row['price_profile']['lowering_id'],P.extension(d)['lowering_id'])

    def test_02_published_A_fixed_schedule_is_metadata_only(self):
        ids=[];pairs=[]
        for prefix in (1,2):
            loc='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD'+str(prefix)+'.json'
            cfg=json.loads(I.C.blob(ROOT,A_MENU,loc));before=deepcopy(cfg)
            target=dict(quantity='ORIGINAL_DF_H_TARGET_ENERGY',binding=cfg['spec']['binding'],energy_shift_ha=0,
                source=dict(commit=A_MENU,path=loc))
            req=I.plan(ROOT,cfg,target=target,signal_source=dict(commit=I.A_SOURCE),
                pr_physical={r['m']:r['physical_candidate'] for r in cfg['rounds']})
            with patch.object(D,'NativeRuntime',side_effect=AssertionError('no physical suitability analysis')):
                rows=pending(req)
            self.assertEqual(len(rows),26);self.assertEqual(before,cfg)
            self.assertTrue(all(r['d_request']['physical_context']['physical_candidate']['prefix']==prefix for r in rows))
            ids.append(rows[0]['d_request']['context_key']);pairs.append((req,rows))
        self.assertNotEqual(*ids)
        with self.assertRaises(ValueError):P.bind_results(ROOT,pairs[0][0],pairs[1][1])

    def test_03_path_only_partial_axes_and_forced_roles_stay_pending(self):
        f=fixture();d=f[4][0]['d_request']
        paths=[path(d,axes=['X']),path(d,index=1,conditional=True,cohort='ARTIFICIAL_FORCED')]
        mapped=P.bind_results(ROOT,f[0],f[4],paths=paths)
        self.assertEqual(mapped['paths'],paths)
        self.assertTrue(all(c['mean']['rz_count'] is None for c in mapped['prices']))
        with patch.object(T.B,'cost',side_effect=AssertionError('path is not E[C]')):out=compare(f,mapped)
        self.assertIsNone(out['comparison']['point_minimum']['cost'])
        self.assertIsNone(out['comparison']['outer_evaluation'])
        self.assertFalse(mapped['numerical_mean_or_interval_computation'])

    def test_04_only_external_artificial_mean_and_N_enable_existing_G(self):
        f=fixture();out=compare(f)['comparison']
        self.assertEqual([T.B.a.exact_scalar(v['G']) for v in out['allocations']],[1002,216])
        self.assertEqual(out['point_minimum']['allocation_id'],'low_G');self.assertEqual(out['strict_interval_winner'],'low_G')
        self.assertEqual([sum(r['N_X']+r['N_Y'] for r in a['shots']) for a in f[2]],[12,18])
        for b in f[6]:b['mean_interval'].update(lower=0,upper=1000)
        mapped=P.bind_results(ROOT,f[0],f[4],prices=f[5],bindings=f[6],interval_contract=f[7]['interval_contract'])
        out=compare(f,mapped)['comparison']
        self.assertEqual(out['point_minimum']['allocation_id'],'low_G');self.assertIsNone(out['strict_interval_winner'])

    def test_05_single_axis_never_supplies_the_other_or_symmetric_price(self):
        f=fixture();only=[c for c in f[5] if c['axis']=='X'];bs=[b for b in f[6] if b['axis']=='X']
        mapped=P.bind_results(ROOT,f[0],f[4],prices=only,bindings=bs)
        self.assertTrue(all(p['mean']['rz_count'] is None for p in mapped['prices'] if p['axis']=='Y'))
        self.assertIsNone(compare(f,mapped)['comparison']['point_minimum']['cost'])
        # Explicit X/Y prices differ; C must not substitute the observed X value.
        for c in f[5]:c['mean']['rz_count']=1 if c['axis']=='X' else 3
        for b in f[6]:b['mean_interval']=None
        mapped=P.bind_results(ROOT,f[0],f[4],prices=f[5],bindings=f[6])
        out=compare(f,mapped)['comparison'];self.assertEqual([T.B.a.exact_scalar(v['G']) for v in out['allocations']],[24,36])
        req=pending(f[0],('Y',));self.assertTrue(all(r['axis']=='Y' for r in req))

    def test_06_mixed_identity_preparation_source_law_profile_and_lowering_rejected(self):
        for change in (lambda p:p.update(mean_compiled_price=51),
            lambda p:p['source'].update(input={'fixture':'FOREIGN'}),
            lambda p:p['resolved_context'].update(lambda_R=.7),
            lambda p:p['records'][0].update(axis='Z')):
            f=fixture();p=path(f[4][0]['d_request']);change(p)
            with self.assertRaises(ValueError):P.bind_results(ROOT,f[0],f[4],paths=[p])
        for change in (lambda c,b:c['price_profile'].update(compiler='FOREIGN'),
            lambda c,b:c['price_profile'].update(lowering_id='OLD'),
            lambda c,b:b.update(price_source={'fixture':'FOREIGN'}),
            lambda c,b:b.update(d_context_key='FOREIGN'),lambda c,b:c.update(law_binding={})):
            f=fixture();change(f[5][0],f[6][0])
            with self.assertRaises(ValueError):P.bind_results(ROOT,f[0],f[4],prices=f[5],bindings=f[6])
        f=fixture();p=path(f[4][0]['d_request']);q=path(f[4][2]['d_request']);q['resolved_context']['tail_hash']='FOREIGN'
        with self.assertRaises(ValueError):P.bind_results(ROOT,f[0],f[4],paths=[p,q])
        f=fixture();p=path(f[4][0]['d_request'],conditional=True,cohort=f[5][0]['cohort'])
        with self.assertRaises(ValueError):P.bind_results(ROOT,f[0],f[4],paths=[p],prices=f[5],bindings=f[6])

    def test_07_missing_binding_unqualified_failed_and_cohort_are_not_promoted(self):
        f=fixture();bs=deepcopy(f[6]);del bs[0]['d_context_key']
        mapped=P.bind_results(ROOT,f[0],f[4],prices=f[5],bindings=bs)
        self.assertTrue(mapped['unmapped']);self.assertIsNone(compare(f,mapped)['comparison']['point_minimum']['cost'])
        for status in ('UNQUALIFIED','NUMERICAL_UNSTABLE'):
            f=fixture();f[5][0]['price_status']=status
            mapped=P.bind_results(ROOT,f[0],f[4],prices=f[5],bindings=f[6])
            self.assertIsNone(compare(f,mapped)['comparison']['point_minimum']['cost'])
        f=fixture()
        for c in f[5]:c['expectation']='FINITE_COHORT_ONLY';c['n_paths']=2;c['law_context']['price_law']='ARTIFICIAL_SAMPLER'
        mapped=P.bind_results(ROOT,f[0],f[4],prices=f[5],bindings=f[6])
        self.assertIsNone(compare(f,mapped)['comparison']['point_minimum']['cost'])
        out=compare(f,mapped,policy='finite_cohort_scenario')['comparison']
        self.assertEqual(out['allocations'][0]['G_status'],'FINITE_COHORT_CONDITIONAL_SCENARIO')
        self.assertIsNone(out['strict_interval_winner'])

    def test_08_outside_domain_no_rounding_duplicate_or_structural_fallback(self):
        for method in ('FULL','B0I','FULL_H_RTE'):
            req,_,_=T.toy(method)
            rows=pending(req);self.assertTrue(all(r['price_status']=='OUTSIDE_DOMAIN' for r in rows))
            self.assertTrue(all(r['mean'] is None for r in rows))
        f=fixture();prices=deepcopy(f[5]);prices[0]['quantity']='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT'
        with self.assertRaises(ValueError):P.bind_results(ROOT,f[0],f[4],prices=prices,bindings=f[6])
        p=path(f[4][0]['d_request'])
        with self.assertRaises(ValueError):P.bind_results(ROOT,f[0],f[4],paths=[p,p])

    def test_09_expected_scalar_roundtrip_uses_D_ids_and_keeps_B_identity(self):
        req,_,_=T.toy('PR',2);before=deepcopy(req);legacy=pending(req)
        with (patch.object(D,'NativeRuntime',side_effect=AssertionError('no preparation')),
              patch.object(D,'Adapter',side_effect=AssertionError('no price acquisition'))):
            rows=pending(req,expected_scalar=.07)
            mapped=P.bind_results(ROOT,req,rows)
        for row,old in zip(rows,legacy):
            d=row['d_request'];m=row['m']
            direct=D.from_b(req['configuration'],m,req['pr_physical'][m],input_source=INPUT,
                execution_source=SOURCE,expected_scalar=.07)
            self.assertEqual(d,direct);self.assertEqual(d['physical_context']['expected_scalar'],.07)
            self.assertNotEqual(d['request_id'],old['d_request']['request_id'])
            self.assertNotEqual(d['context_key'],old['d_request']['context_key'])
            self.assertEqual(d['identity'],old['d_request']['identity'])
            self.assertEqual(d['law_binding'],old['d_request']['law_binding'])
        self.assertEqual(req,before)
        self.assertTrue(all(c['mean']['rz_count'] is None for c in mapped['prices']))
        self.assertEqual(I.D_ADAPTER_SOURCE,'748be6875b5007707d864e16aa7a4fc4250f3714')

    def test_10_mixed_scalar_or_specified_unspecified_schedule_rejected(self):
        req,_,_=T.toy('PR',1)
        for other in (.08,math.nextafter(.07,math.inf),None):
            rows=pending(req,expected_scalar=.07);alternate=pending(req,expected_scalar=other)
            mixed=[r if r['m']==0 else alternate[i] for i,r in enumerate(rows)]
            with self.assertRaisesRegex(ValueError,'Mixed.*scalar'):
                P.bind_results(ROOT,req,mixed)
        zero=pending(req,expected_scalar=0.)
        self.assertTrue(all(r['d_request']['physical_context']['expected_scalar']==0. for r in zero))
        P.bind_results(ROOT,req,zero)
        absent=pending(req)
        with self.assertRaisesRegex(ValueError,'Mixed.*scalar'):
            P.bind_results(ROOT,req,[r if r['m']==0 else absent[i] for i,r in enumerate(zero)])

    def test_11_resolved_scalar_requires_binary64_equality_without_tolerance(self):
        req,_,_=T.toy('PR',1);rows=pending(req,expected_scalar=.07)
        good=path(rows[0]['d_request']);mapped=P.bind_results(ROOT,req,rows,paths=[good])
        self.assertEqual(mapped['paths'],[good])
        self.assertTrue(all(p['mean']['rz_count'] is None for p in mapped['prices']))
        for value in (.08,math.nextafter(.07,math.inf)):
            bad=deepcopy(good);bad['resolved_context']['scalar']=value
            with self.assertRaisesRegex(ValueError,'Resolved scalar differs'):
                P.bind_results(ROOT,req,rows,paths=[bad])
        rows=pending(req,expected_scalar=0.);bad=path(rows[0]['d_request'])
        bad['resolved_context']['scalar']=-0.
        with self.assertRaisesRegex(ValueError,'Resolved scalar differs'):
            P.bind_results(ROOT,req,rows,paths=[bad])

    def test_12_omitted_scalar_requests_equal_legacy_D_constructor(self):
        # Reuse the old constructor only; it has no scientific entry point.
        old='0c6e16342b3c804fa0bd89a4028d6b5bea7b0322'
        tree=ast.parse(I.C.blob(ROOT,old,'src/trottertracks/candidate_price_adapter.py'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='from_b')
        scope=dict(D.__dict__)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),old+':from_b','exec'),scope)
        req,_,_=T.toy('PR',2);rows=pending(req)
        for r in rows:
            d=r['d_request'];m=r['m']
            legacy=scope['from_b'](req['configuration'],m,req['pr_physical'][m],input_source=INPUT,
                execution_source=SOURCE,axes=d['axes'])
            self.assertEqual(d,legacy);self.assertNotIn('expected_scalar',d['physical_context'])
        self.assertEqual(rows,pending(req,expected_scalar=None))
        # The original general-pending route remains distinct from D requests.
        self.assertTrue(all('d_request' not in r for r in I.price_requests(ROOT,req)))
        with self.assertRaisesRegex(ValueError,'SOURCE'):
            I.price_requests(ROOT,req,expected_scalar=.07)

    def test_13_scalar_specific_external_bindings_and_request_integrity(self):
        f=fixture();req=f[0];rows=pending(req,expected_scalar=.07)
        with self.assertRaisesRegex(ValueError,'binding differs'):
            P.bind_results(ROOT,req,rows,prices=f[5],bindings=f[6])
        lookup={(r['m'],r['axis']):r['d_request'] for r in rows};bindings=deepcopy(f[6])
        for b in bindings:
            d=lookup[b['identity']['m'],b['axis']]
            b.update(d_request_id=d['request_id'],d_context_key=d['context_key'])
        mapped=P.bind_results(ROOT,req,rows,prices=f[5],bindings=bindings)
        self.assertEqual([p['mean'] for p in mapped['prices']],[p['mean'] for p in f[5]])
        altered=deepcopy(rows);altered[0]['d_request']['physical_context']['expected_scalar']=.08
        with self.assertRaisesRegex(ValueError,'mutated'):
            P.bind_results(ROOT,req,altered)

    def test_14_corrected_D_total_scalar_provenance_is_forwarded_without_reading_inputs(self):
        latest='e827367629ca6c2920d9d908fc1631314399db86'
        # The newer acquisition observer is outside this passive interface.
        trees=[ast.parse(I.C.blob(ROOT,s,'src/trottertracks/candidate_price_adapter.py'))
               for s in (I.D_ADAPTER_SOURCE,latest)]
        for name in ('from_b','from_a_job','check_request'):
            fns=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
            self.assertEqual(ast.dump(fns[0]),ast.dump(fns[1]))
        inp=dict(INPUT,scalar_binding=dict(
            hamiltonian_constant=dict(quantity='HAMILTONIAN_CONSTANT',value=.5,source=INPUT),
            tail_identity=dict(quantity='TAIL_EXTRACTED_IDENTITY_COEFFICIENT',value=.125,source=INPUT),
            expected_total=dict(quantity='EXPLICIT_TOTAL_SCALAR',value=.625,
                rule='BINARY64_HAMILTONIAN_CONSTANT_PLUS_TAIL_IDENTITY_ONCE')))
        req,_,_=T.toy('PR',1);before=deepcopy(inp)
        with (patch.object(D,'NativeRuntime',side_effect=AssertionError('no preparation/constant reader')),
              patch.object(D,'Adapter',side_effect=AssertionError('no acquisition'))):
            rows=I.price_requests(ROOT,req,input_source=inp,execution_source=SOURCE,expected_scalar=.625)
            d=rows[0]['d_request']
            self.assertEqual(d['physical_context']['input_source'],inp)
            self.assertEqual(d,D.from_b(req['configuration'],0,req['pr_physical'][0],
                input_source=inp,execution_source=SOURCE,expected_scalar=.625))
            p=path(d);p['resolved_context']['scalar']=.625
            mapped=P.bind_results(ROOT,req,rows,paths=[p])
            self.assertTrue(all(c['mean']['rz_count'] is None for c in mapped['prices']))
            p['resolved_context']['scalar']=.125
            with self.assertRaisesRegex(ValueError,'Resolved scalar differs'):
                P.bind_results(ROOT,req,rows,paths=[p])
        self.assertEqual(inp,before)


if __name__=='__main__':unittest.main(verbosity=2)
