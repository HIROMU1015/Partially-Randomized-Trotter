"""Synthetic-only C0.5: no molecular archive, probabilities, Qiskit or native imports."""
from fractions import Fraction
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from h8_rare_c05 import sampling as s
from h8_rare_c05.accounting import Budget,Permit,IdentityCache,IDENTITY_FIELDS,atomic,read,check_permit,begin_approved,sample_statistics,paired_pilot_summary
from h8_rare_c05.adapter import Adapter,run_registered
from h8_rare_c05.parallel import map_fork
from h8_rare_c05.provider import OriginalProvider
from h8_rare_c05.resources import Admission


class ToyEvent:
    def __init__(self,k,indices):self.k=k;self.indices=indices
    def to_dict(self):return dict(order=self.k,indices=self.indices,phase='MOCK_SIGN',identity_phase='MOCK')
class ToyLaw:
    orders=(0,2,4);order_probabilities=(.5,.3,.2)
    def to_dict(self):return dict(orders=self.orders,order_probabilities=self.order_probabilities,normalization='MOCK_UNCHANGED')
class ToyProvider:
    mock=True
    def __init__(self,fail_axis=None):
        self.fail_axis=fail_axis
        tail=NS(components=[NS(probability=.6),NS(probability=.4)],tail_id='TOY',tail_hash='TOY')
        prep=NS(preparation_hash='TOY_PREP',num_system_qubits=3,rte_preparation=NS(symbolic_tail=tail,component_specs=('MOCK',),resolve_event_basis_definitions=lambda e:None))
        self.context=NS(preparations={7:prep,8:prep},binding={'compiler':'MOCK_C'});self.evolutions=0
    def api(self):
        return NS(make_event=lambda indices,cs,law,i:ToyEvent(law.orders[i],indices),validate_components=lambda cs:math.fsum(x.probability for x in cs),sequence=lambda **kw:NS(**kw),step=lambda **kw:NS(**kw),rng=lambda x:random.Random(x))
    def law(self,*args):return NS(mock=True),ToyLaw()
    def evolve(self,requests):
        self.evolutions+=1
        return {'events':self.event_values(requests)}
    def wrapper(self,evolution,axis):return dict(evolution=evolution,axis=axis,measurement=True,scalar='ONCE_MOCK',control='symmetric_directional')
    def fingerprint(self,wrapper):return s.digest(wrapper)
    def compile(self,wrapper,fp):
        if wrapper['axis']==self.fail_axis:raise RuntimeError('INJECTED_TOY_COMPILE_FAILURE')
        assert self.fingerprint(wrapper)==fp
        return dict(RZ=11,CX=7,depth=15,RZ_depth=8,CX_depth=4,size=20)
    def event_values(self,requests):return [[e.to_dict() for e in r.rte_occurrence.events] for r in requests]
    def validate(self,evolution,requests,directory):atomic(Path(directory)/'control_mock.json',{'mock_only':True,'scalar':'ONCE','probes':4,'oracle':1})
    def final_gate(self):pass

def task(q=2,st='S2_SINGLE',rep=0):
    cid='B2_p8_q2_2nd_r4_K4' if q==2 else 'B2_p7_q16_2nd_r4_K4'
    return dict(id='toy_'+str(q)+'_'+st+'_'+str(rep),candidate=cid,q=q,r=4,K=4,stratum=st,phase='variance_load_pilot',replica=rep)
def contract(t):
    identity={k:'MOCK_'+k for k in IDENTITY_FIELDS}
    return dict(condition='MOCK_ONLY',candidate=t['candidate'],phase=t['phase'],numeric_policy=s.POLICY,target_preparation_sha256='TOY_PREP',original_binding={'compiler':'MOCK_C'},original_law_sha256=s.digest(ToyLaw().to_dict()),cache_binding=identity,validation_task_ids=[t['id']],strata=[t['stratum']])
def budget(root,n=1,q=2):
    return Budget(root,dict(law_formation=1,count_law_setup=3,component_table_setup=1,pilot_accounting=1,independent_numeric_validation=1,trajectory=n,occurrence=q*n,RTE_event=q*4*n,native_evolution=n,wrapper_build=2*n,wrapper_compile=2*n,control_probe=4*n,event_oracle=n,validation_wrapper_build=2*n))
CAP=lambda:dict(cpu=4,memory=10**12,io=4)
DEMAND=lambda t,c:dict(cpu=2,memory=1024**2,io=1,axis_width=2)

def log_receipt(receipt):
    destination=os.environ.get('H8_C05_MOCK_RECEIPT_FILE')
    if not destination:return
    def sanitized(r):
        return dict(status=r['status'],all_reaped=r['all_reaped'],retry=r['retry'],resume=r['resume'],
            workers=[{k:h[k] for k in ['id','exit_code','reaped']} for h in r['workers']],
            adopted_descendants=[{k:h[k] for k in ['exit_code','reaped']} for h in r['adopted_descendants']])
    value=sanitized(receipt);value['axes']=[]
    for h in receipt['workers']:
        path=Path(h['directory'])/'axes/native_receipt.json'
        if path.exists():value['axes'].append(sanitized(read(path)))
    path=Path(destination);rows=read(path) if path.exists() else []
    rows.append(dict(mock_only=True,receipt=value));atomic(path,rows)


class Numerical(unittest.TestCase):
    def test_count_law_independent_word_enumeration(self):
        for R in (2,3,4):
            for st in s.STRATA:
                p=(Fraction(1,2),Fraction(1,3),Fraction(1,6));raw={}
                for word in itertools.product((0,2,4),repeat=R):
                    n2,n4=word.count(2),word.count(4)
                    if s.in_stratum(n2,n4,st):
                        v=math.prod(p[(0,2,4).index(k)] for k in word);raw[n2,n4]=raw.get((n2,n4),0)+v
                cells=s.count_weights(R,p,st);total=sum(w for _,w in cells)
                self.assertEqual({c:Fraction(w,total) for c,w in cells},{c:v/sum(raw.values()) for c,v in raw.items()})
    def test_underflow_positive_support_retained(self):
        cells=dict(s.count_weights(3,(Fraction(1,2),Fraction(1,2)-Fraction(1,10**400),Fraction(1,10**400)),'S4_PRESENT'))
        self.assertGreater(cells[0,3],0)
    def test_multi_full_counts(self):self.assertEqual({n2 for (n2,n4),w in s.count_weights(6,(.5,.3,.2),'S2_MULTI')},set(range(2,7)))
    def test_s4_mixed_and_multiple(self):
        cells=dict(s.count_weights(4,(.5,.3,.2),'S4_PRESENT'));self.assertIn((2,2),cells);self.assertIn((0,4),cells)
    def test_float_rational_exact(self):self.assertEqual(s.weights([.1,.2]),(1,2))
    def test_scale_invariance(self):self.assertEqual(s.weights([Fraction(1,3),Fraction(2,3)]),s.weights([1,2]))
    def test_empty_support_refused(self):
        with self.assertRaises(ValueError):s.count_weights(2,(1,0,0),'S4_PRESENT')
    def test_negative_refused(self):
        with self.assertRaises(ValueError):s.weights([1,-1])
    def test_zero_total_refused(self):
        with self.assertRaises(ValueError):s.weights([0,0])
    def test_integer_rejection_boundary(self):
        class Bits:
            v=iter([7,6,2])
            def getrandbits(self,k):return next(self.v)
        self.assertEqual(s.below(6,Bits()),2)
    def test_one_category_no_random_draw(self):self.assertEqual(s.below(1,None),0)
    def test_positions_not_pinned(self):
        seen={s.order_word(8,(.5,.3,.2),'S2_SINGLE',random.Random(i)).index(2) for i in range(200)}
        self.assertEqual(seen,set(range(8)))
    def test_pcg_raw_bits_without_numpy(self):
        raw=NS(random_raw=lambda:2**64-1);self.assertEqual(s.PCG64Bits(raw).getrandbits(70),2**70-1)

class Registry(unittest.TestCase):
    def test_order_independent_registry(self):
        ts=[task(2,rep=0),task(16,rep=1)];self.assertEqual(s.register_streams(ts,'S','B','E'),s.register_streams(ts[::-1],'S','B','E'))
    def test_strata_phase_domains(self):
        ts=[task(2,st,rep=i) for i,st in enumerate(s.STRATA)];r=s.register_streams(ts,'S','B','E');self.assertEqual(len({v for x in r.values() for v in x.values()}),9)
        prod=dict(ts[0],phase='production');self.assertNotEqual(s.register_streams([ts[0]],'S','B','E'),s.register_streams([prod],'S','B','E'))
    def test_old_collision_refused(self):
        t=task();v=next(iter(s.register_streams([t],'S','B','E')[t['id']].values()))
        with self.assertRaises(ValueError):s.register_streams([t],'S','B','E',[v])
    def test_derived_collision_refused(self):
        with patch.object(s,'digest',lambda x:'0'*64):
            with self.assertRaises(ValueError):s.register_streams([task()],'S','B','E')
    def test_duplicate_task_refused(self):
        with self.assertRaises(ValueError):s.register_streams([task(),task()],'S','B','E')

class NativeBridge(unittest.TestCase):
    def test_q2_q16_all_strata_request_policy(self):
        for q in (2,16):
            for st in s.STRATA:
                t=task(q,st);p=ToyProvider();streams=s.register_streams([t],'MOCK','MOCK','MOCK')[t['id']]
                requests,orders=s.requests(p.context.preparations[8 if q==2 else 7],t,ToyLaw(),NS(),streams,p.api())
                self.assertEqual(len(requests),q);self.assertEqual(len(orders),q*4)
                self.assertTrue(s.in_stratum(orders.count(2),orders.count(4),st))
                for r in requests:
                    self.assertEqual(len(r.rte_occurrence.events),4);self.assertTrue(r.controlled)
                    self.assertTrue(r.rte_occurrence.identity_as_relative_ancilla_phase)
                    for e in r.rte_occurrence.events:self.assertEqual(len(e.indices),e.k+1)
    def test_q_mismatch_refused(self):
        p=ToyProvider();t=task();t['q']=4
        with self.assertRaises(ValueError):s.requests(p.context.preparations[8],t,ToyLaw(),None,{},p.api())
    def test_prefix_hash_refused(self):
        t=task();c=contract(t);c['target_preparation_sha256']='WRONG'
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):Adapter(ToyProvider(),Permit('MOCK','','MOCK_ONLY'),c,budget(d))
    def test_forged_science_permit_refused(self):
        with self.assertRaises(ValueError):OriginalProvider(NS(),Permit('SCIENCE','','X'))
    def test_unapproved_contract_before_grant_read(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'proposed.json';atomic(p,dict(approved=False,runtime_authorization=False))
            with self.assertRaises(ValueError):begin_approved(p,Path(d)/'ABSENT_GRANT',Path(d)/'ABSENT_MARKER',Path(d)/'ABSENT_RECEIPT',[])
            self.assertFalse((Path(d)/'ABSENT_MARKER').exists())
    def test_full_paired_path_parallel_q2_and_q16(self):
        for q in (2,16):
            with tempfile.TemporaryDirectory() as d:
                t=task(q);p=ToyProvider();b=budget(Path(d)/'budget',q=q);a=Adapter(p,Permit('MOCK','','MOCK_ONLY'),contract(t),b)
                a.initialize_law(0);streams=s.register_streams([t],'MOCK','MOCK','MOCK');r=run_registered(a,[t],streams,Path(d),CAP,DEMAND)
                log_receipt(r)
                self.assertEqual(r['status'],'PASS');self.assertTrue(r['all_reaped'])
                v=r['workers'][0]['result'];self.assertEqual(v['axes']['cosine']['event_digest'],v['axes']['sine']['event_digest'])
                self.assertEqual(b.used()['native_evolution'],1);self.assertEqual(b.used()['wrapper_compile'],2)
                self.assertTrue(v['scientific_mock'])
    def test_compile_failure_partial_STOP_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            t=task();b=budget(Path(d)/'budget');a=Adapter(ToyProvider('sine'),Permit('MOCK','','MOCK_ONLY'),contract(t),b);a.initialize_law(0)
            r=run_registered(a,[t],s.register_streams([t],'M','B','E'),Path(d),CAP,DEMAND)
            log_receipt(r)
            self.assertEqual(r['status'],'STOP');self.assertTrue(r['all_reaped']);self.assertFalse(r['retry'])
            self.assertTrue(list((Path(d)/'tasks').glob('*/trajectory.json')))

class Transactions(unittest.TestCase):
    def test_budget_reservation_nonrefundable(self):
        with tempfile.TemporaryDirectory() as d:
            b=Budget(d,{'compile':2});b.take('compile',1,'X')
            with self.assertRaises(FileExistsError):b.take('compile',1,'X')
            self.assertEqual(b.used(),{'compile':1})
    def test_budget_overrun_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):Budget(d,{'compile':0}).take('compile',1,'X')
    def test_budget_changed_refused(self):
        with tempfile.TemporaryDirectory() as d:
            Budget(d,{'compile':2}).take('compile',1,'X')
            with self.assertRaises(ValueError):Budget(d,{'compile':3}).take('compile',1,'Y')
    def test_cache_hit_and_changed_axis_profile(self):
        with tempfile.TemporaryDirectory() as d:
            c=IdentityCache(d);key={k:'MOCK' for k in IDENTITY_FIELDS};key.update(axis='cosine',measurement=True)
            self.assertFalse(c.get_or_compute(key,lambda:dict(RZ=1))[1]);self.assertTrue(c.get_or_compute(key,lambda:None)[1])
            self.assertFalse(c.get_or_compute(dict(key,axis='sine'),lambda:dict(RZ=2))[1])
            self.assertFalse(c.get_or_compute(dict(key,compiler='OTHER'),lambda:dict(RZ=2))[1])
    def test_corrupt_cache_refused(self):
        with tempfile.TemporaryDirectory() as d:
            c=IdentityCache(d);key={k:'MOCK' for k in IDENTITY_FIELDS};key.update(axis='cosine',measurement=True)
            c.get_or_compute(key,lambda:dict(RZ=1));f=Path(d)/s.digest(key)/'complete.json';r=read(f);r['identity']['compiler']='BAD';atomic(f,r)
            with self.assertRaises(ValueError):c.get_or_compute(key,lambda:dict(RZ=1))
    def test_incomplete_cache_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            c=IdentityCache(d);key={k:'MOCK' for k in IDENTITY_FIELDS};key.update(axis='cosine',measurement=True)
            def fail():raise RuntimeError('MOCK')
            with self.assertRaises(RuntimeError):c.get_or_compute(key,fail)
            with self.assertRaises(ValueError):c.get_or_compute(key,lambda:dict(RZ=1))

class Scheduling(unittest.TestCase):
    def test_dynamic_parallel_and_sorted_results(self):
        with tempfile.TemporaryDirectory() as d:
            r=map_fork([{'id':str(i)} for i in range(4)],lambda t,w,a:{'pid':os.getpid()},Path(d)/'pool',CAP,DEMAND)
            log_receipt(r)
            self.assertEqual(r['status'],'PASS');self.assertTrue(r['all_reaped']);self.assertEqual([h['id'] for h in r['workers']],['0','1','2','3'])
    def test_temporary_busy_resource_defers_without_worker_stop(self):
        with tempfile.TemporaryDirectory() as d:
            calls=[]
            def cap():calls.append(1);return dict(cpu=0 if len(calls)==1 else 4,memory=10**12,io=4)
            r=map_fork([{'id':'a'}],lambda *x:{'ok':True},Path(d)/'pool',cap,DEMAND)
            log_receipt(r)
            self.assertEqual(r['status'],'PASS');self.assertGreaterEqual(len(calls),2)
    def test_hard_parent_failure_adopted_grandchild_reaped(self):
        def callback(t,w,a):
            pid=os.fork()
            if pid==0:
                while True:time.sleep(1)
            os._exit(7)
        with tempfile.TemporaryDirectory() as d:
            r=map_fork([{'id':'a'}],callback,Path(d)/'pool',CAP,DEMAND)
            log_receipt(r)
            self.assertEqual(r['status'],'STOP');self.assertTrue(r['all_reaped']);self.assertTrue(r['adopted_descendants'])
            self.assertTrue(all(x['reaped'] for x in r['adopted_descendants']))
    def test_cancel_only_own_job(self):
        with tempfile.TemporaryDirectory() as d:
            r=map_fork([{'id':'a'}],lambda *x:{},Path(d)/'pool',CAP,DEMAND,cancel=lambda:True)
            log_receipt(r)
            self.assertEqual(r['status'],'STOP');self.assertEqual(r['workers'],[])
    def test_missing_memory_estimate_refused(self):
        with self.assertRaises(ValueError):Admission(None,1,1)
    def test_multithread_parent_refused(self):
        with patch('h8_rare_c05.parallel.single_thread',lambda:False):
            with self.assertRaises(ValueError):map_fork([],None,'unused',CAP,DEMAND)


class AdditionalChecks(unittest.TestCase):
    def test_parent_count_table_reuse(self):
        self.assertIs(s.count_category(8,(.5,.3,.2),'S2_SINGLE'),s.count_category(8,(.5,.3,.2),'S2_SINGLE'))
    def test_cdf_boundary_matches_linear_integer_choice(self):
        class Bits:
            def __init__(self,x):self.x=x
            def getrandbits(self,k):return self.x
        table=s.IntegerCategorical((('a',2),('b',1),('c',5)))
        self.assertEqual([table.draw(Bits(i)) for i in range(8)],['a','a','b','c','c','c','c','c'])
    def test_nonpositive_pcg_bits_refused(self):
        with self.assertRaises(ValueError):s.PCG64Bits(None).getrandbits(-1)
    def test_negative_replica_refused(self):
        with self.assertRaises(ValueError):s.register_streams([dict(task(),replica=-1)],'S','B','E')
    def test_phase_mixing_refused_before_fork(self):
        with tempfile.TemporaryDirectory() as d:
            t=task();a=Adapter(ToyProvider(),Permit('MOCK','','MOCK_ONLY'),contract(t),budget(Path(d)/'b'))
            t2=dict(t,phase='production')
            with self.assertRaises(ValueError):run_registered(a,[t2],{t2['id']:{}},d,CAP,DEMAND)
    def test_duplicate_summaries_refused(self):
        row=dict(candidate='MOCK',phase='pilot',task_id='x',stratum='S2_SINGLE',event_digest='E',axes={a:{'event_digest':'E','metrics':{'RZ':2}} for a in ('cosine','sine')})
        with self.assertRaises(ValueError):paired_pilot_summary([row,row],{'cosine':3,'sine':4},'MOCK','pilot')
    def test_empty_unknown_and_singleton_not_zero_filled(self):
        self.assertEqual(sample_statistics([]),dict(n=0,mean=None,variance=None,SE=None))
        self.assertIsNone(sample_statistics([5])['variance']);self.assertIsNone(sample_statistics([5])['SE'])
    def test_variance_independent_raw_moments(self):
        values=[2,5,8,13];n=len(values);st=sample_statistics(values)
        raw=(Fraction(sum(v*v for v in values))-Fraction(sum(values)**2,n))/(n-1)
        self.assertEqual(Fraction(st['variance']),raw)
    def test_paired_statistics_includes_axis_covariance(self):
        rows=[]
        for i,(a,b) in enumerate([(2,5),(5,2)]):
            rows.append(dict(candidate='MOCK',phase='pilot',task_id=str(i),stratum='S2_SINGLE',event_digest='E',axes={
                'cosine':{'event_digest':'E','metrics':{'RZ':a}},'sine':{'event_digest':'E','metrics':{'RZ':b}}}))
        result=paired_pilot_summary(rows,{'cosine':1,'sine':1},'MOCK','pilot')
        self.assertEqual(result['groups']['S2_SINGLE']['variance'],'0')
        self.assertIsNone(result['groups']['S2_MULTI']['mean']);self.assertFalse(result['population_ranking_guarantee'])
    def test_concurrent_budget_exact_reservations(self):
        with tempfile.TemporaryDirectory() as d:
            b=Budget(Path(d)/'b',{'compile':4})
            r=map_fork([{'id':str(i)} for i in range(4)],lambda t,w,a:(b.take('compile',1,t['id']) or {'ok':True}),Path(d)/'pool',CAP,DEMAND)
            log_receipt(r)
            self.assertEqual(r['status'],'PASS');self.assertEqual(b.used(),{'compile':4})
    def test_final_gate_failure_writes_STOP(self):
        class BadFinal(ToyProvider):
            def final_gate(self):raise RuntimeError('TOY_FINAL_DRIFT')
        with tempfile.TemporaryDirectory() as d:
            t=task();a=Adapter(BadFinal(),Permit('MOCK','','MOCK_ONLY'),contract(t),budget(Path(d)/'b'));a.initialize_law(0)
            with self.assertRaises(RuntimeError):run_registered(a,[t],s.register_streams([t],'M','B','E'),d,CAP,DEMAND)
            self.assertTrue((Path(d)/'STOP.json').exists())
    def test_cache_missing_identity_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):IdentityCache(d).get_or_compute({'axis':'cosine'},lambda:None)


class ClosureGuards(unittest.TestCase):
    def test_no_scientific_modules_loaded(self):
        self.assertNotIn('qiskit',sys.modules);self.assertNotIn('trotterlib.rte',sys.modules);self.assertNotIn('numpy',sys.modules)
    def test_unregistered_layer_refused(self):
        with tempfile.TemporaryDirectory() as d:
            t=task();a=Adapter(ToyProvider(),Permit('MOCK','','MOCK_ONLY'),contract(t),budget(Path(d)/'b'))
            other=dict(t,stratum='S4_PRESENT')
            with self.assertRaises(ValueError):run_registered(a,[other],{other['id']:{}},d,CAP,DEMAND)
    def test_unsealed_cache_refused_before_native(self):
        with tempfile.TemporaryDirectory() as d:
            t=task()
            with self.assertRaises(ValueError):Adapter(ToyProvider(),Permit('MOCK','','MOCK_ONLY'),contract(t),budget(Path(d)/'b'),IdentityCache(Path(d)/'cache'))


class PilotAccountingIntegration(unittest.TestCase):
    def test_mock_fixed_pilot_accounting_independent_route(self):
        with tempfile.TemporaryDirectory() as d:
            ts=[task(rep=i) for i in range(2)];c=contract(ts[0]);c.update(pilot_numeric_accounting=True,shots={'cosine':3,'sine':4})
            b=budget(Path(d)/'b',n=2);a=Adapter(ToyProvider(),Permit('MOCK','','MOCK_ONLY'),c,b);a.initialize_law(0)
            r=run_registered(a,ts,s.register_streams(ts,'MOCK','B','E'),d,CAP,DEMAND);log_receipt(r)
            self.assertEqual(r['status'],'PASS');self.assertEqual(read(Path(d)/'pilot_summary.json')['groups']['S2_SINGLE']['n'],2)
            self.assertEqual(b.used()['independent_numeric_validation'],1)


class LawDiagnostic(unittest.TestCase):
    def test_saved_vs_canonical_toy_not_certificate(self):
        d=s.numerical_law_diagnostic((.5,.3,.2),('0.5','0.3','0.2'))
        self.assertTrue(d['engineering_gate_passed']);self.assertFalse(d['certificate']);self.assertFalse(d['compiled_cost_bias_bounded'])
    def test_large_numeric_model_difference_rejected(self):
        d=s.numerical_law_diagnostic((.5,.3,.2),('0.6','0.2','0.2'))
        self.assertFalse(d['engineering_gate_passed'])

if __name__=='__main__':unittest.main(verbosity=2)
