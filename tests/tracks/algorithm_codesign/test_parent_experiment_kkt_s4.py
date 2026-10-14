"""Independent analytic/off-domain KKT and immutable-control preparation tests."""
import copy
from decimal import Decimal,localcontext
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from trottertracks.algorithm_codesign.parent_experiment import design as dd
from trottertracks.algorithm_codesign.parent_experiment import supplement as ss
from trottertracks.algorithm_codesign.parent_experiment.runner import final_status,COMPLETE,INCONCLUSIVE
from trottertracks.algorithm_codesign.parent_allocation import RootAllocationGenerator
from trottertracks.algorithm_codesign.return_aggregation import root_interval

POLICY=dict(iterations=128,root_bits=256,share_bits=64)
PREC=dict(root_bits=128,probability_bits=80,rho=F(1,10**10),eta=F(1,10**10))

def decimal(v):return Decimal(v.numerator)/Decimal(v.denominator)

class IndependentKKT(unittest.TestCase):
    def witness(self,a,b,C,z,w,lam):
        # Inputs and expected optimum come from integer Pythagorean triples,
        # not the implementation's multiplier or bisection formula.
        self.assertEqual(sum(z),a)
        for bi,zi,wi,ci in zip(b,z,w,C):
            hyp=root_interval(zi*zi+bi*bi,256)
            self.assertEqual(hyp.lo,hyp.hi)
            self.assertEqual(wi*wi,ci);self.assertEqual(wi*zi/hyp.lo,lam)
        got=dd.kkt_candidate(a,b,C,**POLICY)
        self.assertEqual(sum(got['allocation']),a);self.assertEqual(got['simplex_residual'],0)
        self.assertFalse(got['exact_KKT_optimum']);self.assertLessEqual(got['iterations_performed'],128)
        lo,hi=got['lambda_squared_interval'];self.assertLessEqual(lo,lam*lam);self.assertGreaterEqual(hi,lam*lam)
        self.assertLess(hi,min(C));self.assertLess(got['lambda_interval'][1]**2,min(C))
        self.assertLessEqual(got['lambda_interval'][0],lam);self.assertGreaterEqual(got['lambda_interval'][1],lam)
        for actual,expected,bounds in zip(got['allocation'],z,got['ideal_z_intervals']):
            self.assertLessEqual(bounds[0],expected);self.assertGreaterEqual(bounds[1],expected)
            self.assertLessEqual(abs(actual-expected),got['L_infinity_distance_upper'])
            self.assertLess(abs(actual-expected),F(1,10**15))
        with localcontext() as ctx:
            ctx.prec=80
            slopes=[decimal(ci).sqrt()*decimal(zi)/(decimal(zi*zi+bi*bi).sqrt()) for zi,bi,ci in zip(got['allocation'],b,C)]
            for slope in slopes:self.assertLess(abs(slope-decimal(lam)),Decimal('1e-15'))
        return got
    def test_review_exact_unequal_prices(self):
        r=self.witness(F(7),(F(4),F(3)),(F(1),F(9,16)),(F(3),F(4)),(F(1),F(3,4)),F(3,5))
        self.assertIn('sqrt(raw_event_T_C_i)',r['objective'])
        self.assertNotEqual(F(1)*3/5,F(9,16)*4/5) # Raw-C stationary condition is different.
    def test_second_independent_pythagorean_pair(self):
        self.witness(F(13),(F(12),F(15)),(F(1),F(85,104)**2),(F(5),F(8)),(F(1),F(85,104)),F(5,13))
    def test_three_child_independent_witness(self):
        self.witness(F(16),(F(4),F(12),F(15)),(F(1),F(39,25)**2,F(51,40)**2),
            (F(3),F(5),F(8)),(F(1),F(39,25),F(51,40)),F(3,5))
    def test_raw_price_scaling_preserves_allocation(self):
        a=dd.kkt_candidate(7,[4,3],[1,F(9,16)],**POLICY)
        b=dd.kkt_candidate(7,[4,3],[9,F(81,16)],**POLICY)
        self.assertEqual(a['allocation'],b['allocation'])
        self.assertLessEqual(b['lambda_interval'][0],F(9,5));self.assertGreaterEqual(b['lambda_interval'][1],F(9,5))
    def test_exact_squared_multiplier_hit(self):
        r=dd.kkt_candidate(2,[1,1],[4,4],**POLICY)
        self.assertEqual(r['lambda_squared_interval'],(F(2),F(2)))
        self.assertEqual(r['allocation'],(F(1),F(1)))
    def test_single_child(self):
        r=dd.kkt_candidate(2,[3],[4],**POLICY);self.assertEqual(r['allocation'],(F(2),))
    def test_invalid_positive_domain_and_float(self):
        for a,b,C in [(0,[1],[1]),(1,[0],[1]),(1,[1],[0]),(1,[1],[-1])]:
            r=dd.kkt_candidate(a,b,C,**POLICY);self.assertIsNone(r['allocation']);self.assertEqual(r['status'],'MATHEMATICALLY_NOT_APPLICABLE')
        with self.assertRaises(TypeError):dd.kkt_candidate(1,[1],[.75],**POLICY)
    def test_exhaustion_without_precision_retry(self):
        with self.assertRaises(ArithmeticError):dd.kkt_candidate(7,[4,3],[1,F(9,16)],iterations=1,root_bits=1,share_bits=2)
    def test_fixed_rounding_simplex_and_tie(self):
        r=dd.kkt_candidate(1,[1,1,1],[4,4,4],iterations=128,root_bits=256,share_bits=2)
        self.assertEqual(r['allocation'],(F(1,2),F(1,4),F(1,4)))
        self.assertGreaterEqual(r['L_infinity_distance_upper'],F(1,6))
        self.assertEqual(sum(r['allocation']),1)
    def test_deterministic_unequal_price_output(self):
        self.assertEqual(dd.kkt_candidate(7,[4,3],[1,F(9,16)],**POLICY),dd.kkt_candidate(7,[4,3],[1,F(9,16)],**POLICY))
    def test_common_vertices_independent_of_kkt(self):
        outputs=[]
        for z in [(F(3),F(4)),(F(4),F(3))]:
            with patch.object(dd,'kkt_candidate',return_value={'allocation':z}):
                outputs.append(dd.candidates(7,[4,3],[1,F(9,16)],POLICY))
        self.assertEqual(outputs[0]['candidates'][:3],outputs[1]['candidates'][:3])
        self.assertNotEqual(outputs[0]['candidates'][3:],outputs[1]['candidates'][3:])
    def test_equal_price_dedup_without_replacement(self):
        r=dd.candidates(2,[1,1],[4,4],POLICY);self.assertEqual(len(r['candidates']),3)
        self.assertEqual([v['label'] for v in r['dropped']],['kkt','halfway']);self.assertFalse(r['replacement_candidates'])

class SupplementalControls(unittest.TestCase):
    def test_fixed_scope_only_four(self):
        c={'supplement':{'targets':list(ss.TARGETS),'inherited_laws':50}}
        self.assertEqual(ss.scope(c),ss.TARGETS)
        for change in [list(ss.TARGETS)+[('config_00','ordinary','common')],list(reversed(ss.TARGETS)),[('config_04','ordinary','vertex_0')]+list(ss.TARGETS)[1:]]:
            with self.assertRaises(PermissionError):ss.scope({'supplement':{'targets':change,'inherited_laws':50}})
    def test_comparison_view_does_not_recalculate(self):
        row={'id':'artificial','ledger':{'eligibility':'ELIGIBLE','N_per_axis':123,'prep_readout_slope':246,'expected_native_intercept':{'T':'3/7','CX':'5/9','1Q':'11/13'}}}
        original=copy.deepcopy(row)
        with patch.object(ss,'ledger',side_effect=AssertionError('no inherited ledger')):
            view=ss.comparison_view(row)
        self.assertEqual(row,original);self.assertEqual(view['ledger']['N_per_axis'],123)
        self.assertEqual(view['ledger']['expected_native_intercept']['T'],F(3,7))
    def complete(self):
        return {'status':COMPLETE,'configurations':4,'laws':20,'new_synthesis_calls':12,'new_root_matrix_checks':24,
                'old_guards':0,'retries':0,'inherited_rows':50,'scope_slots':4,'supplemental_scope_complete':True,'omitted_configurations':0}
    def contract(self):return {'supplement':{},'caps':{'configurations':4,'laws':20,'new_synthesis_keys':12}}
    def test_full_supplement_terminal_and_failures(self):
        c=self.contract();i=self.complete();self.assertEqual(final_status(0,i,{},None,c),COMPLETE)
        for k,v in [('inherited_rows',49),('scope_slots',5),('old_guards',1),('new_root_matrix_checks',23),('configurations','4'),('omitted_configurations',-1),('supplemental_scope_complete',False)]:
            self.assertEqual(final_status(0,dict(i,**{k:v}),{},None,c),INCONCLUSIVE)
        for code,reason in [(1,None),(0,'CPU cap')]:self.assertEqual(final_status(code,i,{},reason,c),INCONCLUSIVE)
    def test_all_duplicates_normal_no_candidate_not_negative(self):
        i=dict(self.complete(),configurations=0,laws=0,new_synthesis_calls=0,new_root_matrix_checks=0,omitted_configurations=4)
        self.assertEqual(final_status(0,i,{},None,self.contract()),COMPLETE)
    def test_new_runner_imports_supplement_not_full_pipeline(self):
        import inspect
        from trottertracks.algorithm_codesign.parent_experiment import runner
        self.assertIn('from .supplement import run_pipeline',inspect.getsource(runner.ledger_job))
        self.assertNotIn('from .pipeline import run_pipeline',inspect.getsource(runner.ledger_job))

class SupplementalWorkflow(unittest.TestCase):
    def fixture(self):
        from test_parent_experiment import tool,config
        from trottertracks.algorithm_codesign.parent_allocation_native import recheck_entry
        import hashlib
        p=(F(1,7),F(2,7),F(4,7));x=F(1,3);t=tool();c=config()
        c['input']['p']=list(map(str,p));c.update(precision={k:str(v) if isinstance(v,F) else v for k,v in PREC.items()},
            proposals=['artificial matched policy'],confidence='artificial selected task',alpha_complex='1/20',bias='same artificial certificate',primary='same T intercept',
            candidates={'KKT':dict(iterations=128,root_bits=128,share_bits=32)},primitive_error='1/1000000',matrix_tolerance=1e-10,
            acquisition={'tool_identity':{'path':'tool.json','sha256':'2'*64},'runtime_executable_sha256':'3'*64,'options':t['synthesizer_options'],'interval_dps':50})
        c['caps'].update(configurations=4,laws=20,new_synthesis_keys=12,new_root_events=12,new_root_matrix_checks=24,
            cached_metadata_keys=36,old_unique_guards=36,all_unique_primitive_guards=48,event_bindings=1092,events_per_configuration=273,workers=2)
        c['supplement']={'targets':list(ss.TARGETS),'inherited_laws':50,'v3_source':'1'*40,'v3_authorization':'2'*40,'v3_result_directory':'old',
            'saved_v3_inputs':{k:{'path':k} for k in ['map','freeze','cache','manifest','terminal','audit','old_contract']}}
        def ir(e):
            return ([('R',e['child'],str(e['ratio']),e.get('rotation_sign',1))] if e.get('ratio') is not None else [('S',0)])+[('T',0)]*e['child']
        def entry(ratio):
            exp=ss.expected(ratio,t,c);seq='THT'
            raw={'key':exp['key'],'spec':exp['spec'],'sequence':seq,'sequence_sha256':hashlib.sha256(seq.encode()).hexdigest(),
                 'complete':True,'up_to_phase':False,'T_count':2,'Tdagger_count':0,'one_qubit_count':3,'global_W_count':0,'operator_error_upper':'1/100000000'}
            return recheck_entry(raw,exp,guard=lambda *a,**kw:F(1,10**8))
        source={};cache={}
        for arm,make in [('ordinary',RootAllocationGenerator.ordinary),('closed_P5_full',RootAllocationGenerator.closed_p5)]:
            gen=make(p,x,**PREC);items=[]
            for i,e in enumerate(gen.reference_events()):
                native=ir(e)
                for gate in native:
                    if gate[0]=='R':cache[str(F(gate[2]))]=entry(str(F(gate[2])))
                items.append({'index':i,'event':e,'native_ir':native})
            source[arm]=items
        for items in source.values():
            for item in items:item['native_cost']=ss.native_accounting(item['native_ir'],cache)
        records={};manifest=[];rows=[]
        for ci in range(15):
            for proposal in ['canonical','analytic_IS','regularized_1/16','regularized_1','regularized_16']:
                if proposal=='analytic_IS' and ci in [2,3,8,9,14]:continue
                ri=len(rows);arm='ordinary' if ci<6 else 'closed_P5_full' if ci<12 else ['partial_return_tail','closed_P3_tail','matched_CTS'][ci-12]
                label=['common','vertex_0','vertex_1','vertex_2','kkt','halfway'][ci%6] if ci<12 else 'common'
                row={'id':f'config_{ci:02d}:'+proposal,'arm':arm,'candidate':label,'proposal':proposal,
                    'ledger':{'eligibility':'ELIGIBLE','N_per_axis':50,'prep_readout_slope':100,'expected_native_intercept':{'T':str(5000+ci),'CX':'200','1Q':'9000'}}}
                rows.append(row);records[f'old/rows/row_{ri:02d}.json']=row;manifest.append({'path':f'rows/row_{ri:02d}.json','bytes':1,'sha256':'a'*64})
        for arm,cfg in [('ordinary','config_00'),('closed_P5_full','config_06')]:
            path='events/'+cfg+'.json';records['old/'+path]=source[arm];manifest.append({'path':path,'bytes':1,'sha256':'a'*64})
        documents={'map':{'rows':rows},'freeze':{'scale':'17/2','baseline_T':[{'arm':'fixed artificial','canonical_expected_T':'17/2'}],
            'candidates':[{'arm':arm,'proxy_T':[i['native_cost']['T'] for i in items if not i['event']['raw_word']]} for arm,items in source.items()]},
            'cache':cache,'manifest':{'files':manifest},'terminal':{'status':'PARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW','source_commit':'1'*40,'execution_HEAD':'2'*40},
            'audit':{'status':'SAVED_OUTPUT_PROVENANCE_AUDIT_PASS','checks_pass':918},'old_contract':copy.deepcopy(c)}
        documents.update(records)
        return c,documents,t,entry,ir
    def exercise(self,fail=False):
        import hashlib
        c,documents,t,entry,ir=self.fixture();calls=[]
        class Writer:
            def __init__(self):self.records={}
            def write(self,path,value):
                self.records[path]=copy.deepcopy(value);return {'path':path,'bytes':1,'sha256':hashlib.sha256(path.encode()).hexdigest()}
        w=Writer();original=copy.deepcopy(documents)
        def acquisition(root,c,identity,inventory,writer,guard):
            self.assertIn('candidate_freeze.json',w.records);self.assertIn('inventory.json',w.records)
            self.assertTrue(w.records['candidate_freeze.json']['frozen_before_new_price'])
            self.assertEqual(w.records['candidate_freeze.json']['scale'],F(17,2))
            self.assertTrue(all(cell['candidate'] in ['kkt','halfway'] for cell in w.records['candidate_freeze.json']['configurations']))
            calls.append(1)
            if fail:raise RuntimeError('artificial synthesis failure; no retry')
            return {k['ratio']:entry(k['ratio']) for k in inventory['keys'] if k['acquisition']=='NEW_ONCE'},[]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'tool.json').write_text('{}')
            with patch.object(ss,'read_bound',side_effect=lambda root,e:copy.deepcopy(documents[e['path']])),\
                 patch.object(ss,'tool_binding',return_value=t),patch.object(ss,'native_ir_g9',side_effect=ir),\
                 patch.object(ss,'acquire',side_effect=acquisition),patch.object(ss,'normalize_saved',side_effect=lambda saved,*a,**k:saved),\
                 patch.object(ss,'root_validation',return_value={'mock':True}) as matrix,\
                 patch.object(ss,'native_accounting',wraps=ss.native_accounting) as price,patch.object(ss,'ledger',wraps=ss.ledger) as budget,\
                 patch('trottertracks.algorithm_codesign.parent_experiment.pipeline.recheck_entry',side_effect=AssertionError('no old guard')) as oldguard,\
                 patch('trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch.verify_runtime'),\
                 patch('trottertracks.algorithm_codesign.rte_reallocation.numeric.configure'),\
                 patch('trottertracks.algorithm_codesign.rte_development.numeric.NativeReference'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'artificial synthesis'):ss.run_pipeline(tmp,c,{},w,Mock())
                    result=None
                else:result=ss.run_pipeline(tmp,c,{},w,Mock())
                self.assertEqual(oldguard.call_count,0)
                if not fail:
                    self.assertEqual(matrix.call_count,12);self.assertEqual(price.call_count,12);self.assertEqual(budget.call_count,20)
        self.assertEqual(documents,original);self.assertEqual(len(calls),1)
        return result,w
    def test_four_internal_configs_twenty_laws_fifty_inherited(self):
        r,w=self.exercise();self.assertEqual(r['configurations'],4);self.assertEqual(r['laws'],20)
        self.assertEqual(r['inherited_rows'],50);self.assertEqual(r['old_guards'],0);self.assertEqual(r['new_root_matrix_checks'],24)
        m=w.records['resource_map.json'];self.assertFalse(m['inherited_ledger_or_native_recomputed']);self.assertFalse(m['scientific_adoption'])
        self.assertEqual(len(w.records['inherited_rows.json']),50)
        self.assertTrue(all(row['id'].split(':')[0] in {t[0] for t in ss.TARGETS} for row in m['rows']))
        self.assertTrue(all(item['row']['id'].split(':')[0] not in {t[0] for t in ss.TARGETS} for item in w.records['inherited_rows.json']))
        self.assertEqual(w.records['saved_certificate_reuse.json']['operator_guard_reruns'],0)
    def test_partial_acquisition_does_not_emit_final_map(self):
        r,w=self.exercise(fail=True);self.assertIsNone(r);self.assertNotIn('resource_map.json',w.records)
    def test_saved_semantic_mismatch_stops_before_constructor(self):
        c,documents,t,entry,ir=self.fixture();documents['old_contract']['alpha_axis']='1/99'
        with patch.object(ss,'read_bound',side_effect=lambda root,e:copy.deepcopy(documents[e['path']])),patch.object(ss,'RootAllocationGenerator') as gen:
            with self.assertRaises(PermissionError):ss.saved_controls('unused',c)
            self.assertEqual(gen.mock_calls,[])
    def test_inherited_row_map_mismatch(self):
        c,documents,t,entry,ir=self.fixture();documents['old/rows/row_00.json']=dict(documents['old/rows/row_00.json'],id='tampered')
        with patch.object(ss,'read_bound',side_effect=lambda root,e:copy.deepcopy(documents[e['path']])):
            with self.assertRaises(PermissionError):ss.saved_controls('unused',c)

if __name__=='__main__':unittest.main()
