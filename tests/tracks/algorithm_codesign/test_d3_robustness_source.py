"""D3 off-domain scalars, mocked jobs and storage guards only."""
import ast
import copy
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_robustness import model, launch, acquisition, pipeline, runner
from trottertracks.algorithm_codesign.rte_development.evidence import atomic_json, EvidenceWriter, verify_chunks, digest, dumps
from trottertracks.algorithm_codesign.rte_development.accounting import whole_budget as legacy_budget
from trottertracks.algorithm_codesign.rte_development.launch import Guard, consume_marker

C=json.loads((ROOT/launch.CONTRACT).read_text())
S='1'*40;A='2'*40


def budget_input():
    return dict(M2_upper=F(3),range_upper=F(2),acceptance_upper=F(3,4),
                norm_or_envelope_upper=F(3,2))


def authority():
    return dict(status='APPROVED_FOR_ONE_D3_RUN',science_execution_authorized=True,
                source_commit=S,contract_sha256='c'*64,design_contract_sha256=launch.DESIGN_SHA,
                runs=1,retries=0,mandatory_STOP=True,
                explicit_execution_instruction='Invented test instruction, not an execution authorization.')


def binding(auth=None, **kwargs):
    args=dict(a=authority() if auth is None else auth,contract_sha='c'*64,
              source=S,head=A,parents=[S],changed=[C['execution']['authorization_path']],
              dirty=False,e=C['execution'])
    args.update(kwargs)
    return launch.validate_binding(**args)


def fake_saved(ratio='2/1009',eps='1/10000000'):
    from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
    angle=Angle('atan',F(ratio)); seq='THTH'
    return dict(key=angle.key+':epsilon:'+eps,angle_key=angle.key,epsilon=eps,
                sequence=seq,sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),
                T_count=2,Tdagger_count=0,one_qubit_count=4,global_W_count=0,
                strict_operator_error_upper='0',error_pass=True)


def fake_inventory():
    entries=[]
    for i in range(36):
        ratio=str(F(i+1,1009))
        spec=dict(angle_kind='atan',ratio=ratio,scale='1',epsilon='1/10000000',
                  options=C['acquisition']['options'],
                  tool_identity_sha256=C['acquisition']['tool_identity']['sha256'],
                  relative_controlled_phase='strict; signed inverse from actual adjoint')
        entries.append(dict(ratio=ratio,new_key=hashlib.sha256(dumps(spec).encode()).hexdigest(),
                            new_spec=spec,new_price=None,synthesis_calls_now=0))
    return dict(keys=entries,used_unique_keys=36)


class FixedContractTests(unittest.TestCase):
    def test_exact_design_projection(self):
        self.assertIs(launch.validate_contract(ROOT,C),C)

    def test_native_budget_cannot_receive_new_precision_credit(self):
        c=copy.deepcopy(C);c['error_accounting']['primitive_error_account']='1/10000000'
        with self.assertRaises(PermissionError):launch.validate_contract(ROOT,c)
        with self.assertRaises(ValueError):model.error_budget(F(1,13),1,'G0',c)

    def test_scientific_conditions_cannot_change(self):
        for field,value in [('steps',[1,2,3]),('degree',7),('runs',2)]:
            c=copy.deepcopy(C);c[field]=value
            with self.subTest(field=field),self.assertRaises(PermissionError):launch.validate_contract(ROOT,c)

    def test_resource_cap_or_seed_change_rejected(self):
        for section,key,value in [('caps','worker_RSS_MiB',1024),('caps','total_wall_seconds',1200)]:
            c=copy.deepcopy(C);c[section][key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):launch.validate_contract(ROOT,c)
        c=copy.deepcopy(C);c['acquisition']['options']['seed']=3
        with self.assertRaises(PermissionError):launch.validate_contract(ROOT,c)

    def test_old_namespace_rejected(self):
        c=copy.deepcopy(C);c['execution']['result_directory']='artifacts/old'
        with self.assertRaises(PermissionError):launch.validate_contract(ROOT,c)

    def test_missing_contract_field_rejected(self):
        c=copy.deepcopy(C);del c['error_accounting']
        with self.assertRaises(PermissionError):launch.validate_contract(ROOT,c)

    def test_pending_gate_before_runtime_input_or_git(self):
        with patch.object(launch,'git',side_effect=AssertionError('git accessed')), \
             patch.object(runner,'runtime_preflight',side_effect=AssertionError('runtime accessed')), \
             patch.object(pipeline,'run',side_effect=AssertionError('science accessed')):
            with self.assertRaises(PermissionError):runner.execute(ROOT,S)
        self.assertFalse((ROOT/C['execution']['result_directory']).exists())


class BindingTests(unittest.TestCase):
    def test_valid_fake_binding(self):
        binding()

    def test_pending_or_wrong_status(self):
        for key,value in [('science_execution_authorized',False),('status','PENDING')]:
            a=authority();a[key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):binding(a)

    def test_bool_is_not_runs_or_retry_integer(self):
        for key,value in [('runs',True),('retries',False),('runs',2),('retries',1)]:
            a=authority();a[key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):binding(a)

    def test_instruction_required(self):
        a=authority();a['explicit_execution_instruction']=None
        with self.assertRaises(PermissionError):binding(a)

    def test_source_or_contract_or_design_mismatch(self):
        for key,value in [('source_commit','3'*40),('contract_sha256','d'*64),('design_contract_sha256','d'*64)]:
            a=authority();a[key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):binding(a)

    def test_head_is_source_rejected(self):
        with self.assertRaises(PermissionError):binding(head=S)

    def test_child_parent_exactly_one(self):
        for parents in ([],['3'*40],[S,'3'*40]):
            with self.subTest(parents=parents),self.assertRaises(PermissionError):binding(parents=parents)

    def test_dirty_or_code_change_rejected(self):
        with self.assertRaises(PermissionError):binding(dirty=True)
        with self.assertRaises(PermissionError):binding(changed=[C['execution']['authorization_path'],'src/change.py'])

    def test_authority_change_required(self):
        with self.assertRaises(PermissionError):binding(changed=[C['execution']['optional_receipt_path']])


class RationalAccountingTests(unittest.TestCase):
    def test_legacy_budget_off_domain_exact(self):
        old=dict(rho='1/1000000000000',primitive_error='1/1000000',
                 epsilon_axis='1/200',alpha_axis='49/24000',resource_failure_per_row='1/12000',
                 resource_tail_t=10,caps={'shot_cap_per_axis':100000000})
        for r in (1,2):
            new=model.whole_budget(budget_input(),F(1,13),r,'G0',C)
            model.equal_budget(new,legacy_budget(budget_input(),F(1,13),r,old))

    def test_integral_policy_factor_and_native_account(self):
        a=model.error_budget(F(1,13),2,'G0',C);b=model.error_budget(F(1,13),2,'G1',C)
        self.assertEqual(a['Taylor_step_upper'],3*b['Taylor_step_upper'])
        self.assertEqual(a['rounded_native_step_upper'],b['rounded_native_step_upper'])
        self.assertGreater(b['remaining'],a['remaining'])

    def test_price_precision_does_not_change_budget(self):
        a=model.whole_budget(budget_input(),F(1,13),2,'G1',C)
        c=copy.deepcopy(C);c['error_accounting']['primitive_error_new_acquisition']='1/100000000'
        b=model.whole_budget(budget_input(),F(1,13),2,'G1',c)
        self.assertEqual(a,b)

    def test_budget_exhaustion_no_rescue(self):
        with self.assertRaises(ArithmeticError):model.error_budget(F(1),1,'G0',C)

    def test_shot_cap_no_clipping(self):
        c=copy.deepcopy(C);c['caps']['shot_cap_per_axis']=1
        with self.assertRaises(ArithmeticError):model.whole_budget(budget_input(),F(1,13),2,'G1',c)

    def test_unknown_guarantee_or_r_rejected(self):
        for r,g in ((3,'G0'),(True,'G0'),(1,'G2')):
            with self.subTest(r=r,g=g),self.assertRaises(ValueError):model.error_budget(F(1,13),r,g,C)

    def test_nonunitary_product_implementation_not_r_times_error(self):
        a=model.error_budget(F(1,13),2,'G1',C)
        self.assertGreater(a['implementation_product_upper'],2*a['rounded_native_step_upper'])

    def test_zero_fill_and_boundary_saving(self):
        b=model.whole_budget(budget_input(),F(1,13),2,'G0',C)
        row=dict(steps=2,arm='invented',x='1/13',production_step_bounds=budget_input(),workspace_beyond_system=1)
        boundary=dict(acceptance_squared='1/4',cancellation_saving_per_whole_trial={'T':'1','CX':'0','1Q':'2'})
        value=model.row_account(row,b,dict(T=4,CX=2,**{'1Q':8}),dict(T=9,CX=3,**{'1Q':18}),F(1,2),boundary)
        self.assertEqual(value['per_whole_trial_native']['T'],3)
        self.assertEqual(value['prep_readout_slope'],F(b['N_per_axis'],2))
        self.assertEqual(value['expected_native_intercept']['T'],6*b['N_per_axis'])

    def test_missing_boundary_or_negative_cost_rejected(self):
        b=model.whole_budget(budget_input(),F(1,13),2,'G0',C)
        row=dict(steps=2,arm='invented',x='1/13',production_step_bounds=budget_input(),workspace_beyond_system=1)
        with self.assertRaises(ArithmeticError):model.row_account(row,b,dict(T=1,CX=1,**{'1Q':1}),dict(T=2,CX=2,**{'1Q':2}),F(1,2))
        boundary=dict(acceptance_squared='1/4',cancellation_saving_per_whole_trial=dict(T=3,CX=0,**{'1Q':0}))
        with self.assertRaises(ArithmeticError):model.row_account(row,b,dict(T=1,CX=1,**{'1Q':1}),dict(T=2,CX=2,**{'1Q':2}),F(1,2),boundary)

    def test_dyadic_weight_mismatch_rejected(self):
        event=dict(event=dict(proposal='1/2',coefficient='1/3',weight='1'),native_ir=[['H',0]],native_cost=dict(T=0,CX=0,**{'1Q':1}))
        with self.assertRaises(ArithmeticError):model.event_prices([event],{'P0':{},'P1':{}})

    def test_signed_rotation_prices_shared(self):
        e=dict(event=dict(proposal='1/2',coefficient='1/2',weight='1'),native_ir=[['R',0,'2/1009',1],['R',0,'2/1009',-1]],native_cost=dict(T=4,CX=0,**{'1Q':8}))
        p={'P0':{'2/1009':dict(T_count=2,one_qubit_count=4)},'P1':{'2/1009':dict(T_count=3,one_qubit_count=7)}}
        z,prices,worst=model.event_prices([e],p)
        self.assertEqual(z,F(1,2));self.assertEqual(prices['P1']['T'],3)
        self.assertEqual(worst['P1']['1Q'],14)

    def test_factorial_effect_and_shared_budget(self):
        b=dict(N_per_axis=9);rows={}
        for p,price in zip(('G0_P0','G1_P0','G0_P1','G1_P1'),(10,7,20,14)):
            rows[p]=[dict(row_index=0,budget=b,expected_native_intercept={k:F(price) for k in model.METRICS},prep_readout_slope=F(3))]
        out=model.factorial_effects(rows)
        self.assertEqual(out[0]['guarantee_only'],-3)
        self.assertEqual(out[0]['price_only_G0'],10)
        self.assertEqual(out[0]['price_only_G1'],7)
        self.assertEqual(out[0]['interaction'],-3)
        rows['G0_P1'][0]['budget']=dict(N_per_axis=8)
        with self.assertRaises(ArithmeticError):model.factorial_effects(rows)

    def test_legacy_anchor_mismatch_detected(self):
        a=model.whole_budget(budget_input(),F(1,13),2,'G0',C);b=copy.deepcopy(a);b['N_per_axis']+=1
        with self.assertRaises(ArithmeticError):model.equal_budget(a,b)


class InventoryAndPhaseTests(unittest.TestCase):
    def test_all_invented_keys_complete(self):
        self.assertEqual(len(acquisition.validate_inventory(fake_inventory(),C)),36)

    def test_partial_or_duplicate_key_rejected(self):
        v=fake_inventory();v['keys'].pop()
        with self.assertRaises(PermissionError):acquisition.validate_inventory(v,C)
        v=fake_inventory();v['keys'][-1]=v['keys'][0]
        with self.assertRaises(PermissionError):acquisition.validate_inventory(v,C)

    def test_tampered_precision_or_seed_or_price_rejected(self):
        for key,value in [('epsilon','1/1000000'),('scale','2')]:
            v=fake_inventory();v['keys'][0]['new_spec'][key]=value
            with self.subTest(key=key),self.assertRaises(PermissionError):acquisition.validate_inventory(v,C)
        v=fake_inventory();v['keys'][0]['new_price']=1
        with self.assertRaises(PermissionError):acquisition.validate_inventory(v,C)

    def test_key_order_rejected(self):
        v=fake_inventory();v['keys'].reverse()
        with self.assertRaises(PermissionError):acquisition.validate_inventory(v,C)

    def test_worker_result_binding_and_one_call(self):
        job=dict(key='invented',ratio='2/1009',parent_pid=100)
        result=dict(**job,status='KEY_COMPLETE',synthesis_calls=1,retries=0,saved=fake_saved())
        self.assertEqual(acquisition.validate_result(result,job)['T_count'],2)
        for field,value in [('key','other'),('synthesis_calls',2),('retries',1)]:
            bad=dict(result);bad[field]=value
            with self.subTest(field=field),self.assertRaises((PermissionError,ArithmeticError)):acquisition.validate_result(bad,job)

    def test_missing_key_is_never_free(self):
        e=dict(event=dict(proposal='1',coefficient='1',weight='1'),native_ir=[['R',0,'2/1009',1]],native_cost=dict(T=2,CX=0,**{'1Q':4}))
        with self.assertRaises(KeyError):model.event_prices([e],{'P0':{},'P1':{}})

    def test_saved_error_or_sequence_tamper_rejected(self):
        for field,value in [('error_pass',False),('strict_operator_error_upper','1/1000'),('T_count',3),('epsilon','1/1000000')]:
            row=fake_saved();row[field]=value
            with self.subTest(field=field),self.assertRaises(PermissionError):acquisition.check_saved(row,'2/1009','1/10000000')

    def test_global_phase_guard_off_domain(self):
        from trottertracks.algorithm_codesign.rte_reallocation.native import Angle
        from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure,strict_guard
        configure(100)
        self.assertEqual(strict_guard('',Angle('pi',F(0))),0)
        self.assertGreater(strict_guard('W',Angle('pi',F(0))),F(1,2))

    def test_certificate_binds_old_IR(self):
        cert=dict(all_rows_no_R_cancellation=True,rows=[dict(row=1,source_sha256='a',no_R_can_be_cancelled=True,R_containing_intersections=0)])
        pipeline.check_certificate(cert,1,'a')
        for index,h in ((2,'a'),(1,'b')):
            with self.subTest(index=index),self.assertRaises(PermissionError):pipeline.check_certificate(cert,index,h)


class StorageAndFinalStatusTests(unittest.TestCase):
    def test_exclusive_marker_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'new';marker,_=consume_marker(d,{'invented':True})
            original=marker.read_bytes()
            with self.assertRaises(FileExistsError):consume_marker(d,{'invented':False})
            self.assertEqual(marker.read_bytes(),original)

    def test_existing_namespace_even_without_marker_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'new';d.mkdir()
            with self.assertRaises(FileExistsError):consume_marker(d,{'invented':True})

    def test_receipt_restart_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'started.json';atomic_json(p,dict(invented=True))
            with self.assertRaises(FileExistsError):atomic_json(p,dict(invented=True))

    def test_lossless_UTF8_chunks_and_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            writer=EvidenceWriter(t,131072,chunk_bytes=41)
            identity=writer.write('invented.json',{'text':'位相Δ'*100})
            self.assertTrue(verify_chunks(t,identity))
            p=Path(t)/identity['chunks'][0]['path'];p.write_bytes(b'bad')
            with self.assertRaises(ArithmeticError):verify_chunks(t,identity)

    def test_output_cap_failure_preserves_partial(self):
        with tempfile.TemporaryDirectory() as t:
            writer=EvidenceWriter(t,33000)
            with self.assertRaises(RuntimeError):writer.write('invented.json',{'text':'a'*2000})
            self.assertTrue((Path(t)/'invented.json.partial').exists())

    def test_complete_requires_exit_zero_full_coverage_and_identity(self):
        identity={'source':'invented'}
        inner=dict(status=launch.COMPLETE,**identity,rows=48,new_synthesis_calls=36,registered_pair_evaluations=0,retries=0)
        self.assertEqual(runner.final_status(0,inner,None,identity),launch.COMPLETE)
        for code,reason in ((1,None),(0,'cap')):
            self.assertEqual(runner.final_status(code,inner,reason,identity),launch.INCONCLUSIVE)
        for key,value in [('rows',47),('new_synthesis_calls',35),('source','wrong'),('registered_pair_evaluations',1)]:
            bad=dict(inner);bad[key]=value
            with self.subTest(key=key):self.assertEqual(runner.final_status(0,bad,None,identity),launch.INCONCLUSIVE)

    def test_missing_or_prefix_terminal_inconclusive(self):
        self.assertEqual(runner.final_status(-9,None,None,{}),launch.INCONCLUSIVE)
        self.assertEqual(runner.final_status(0,dict(status=launch.INCONCLUSIVE),None,{}),launch.INCONCLUSIVE)

    def test_periodic_guard_conflict_rejected_and_restored(self):
        old=signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM,lambda *_:None)
            with self.assertRaises(RuntimeError):
                with Guard(C['caps']):pass
        finally:signal.signal(signal.SIGALRM,old)
        with Guard(C['caps']) as g:g.check()
        self.assertEqual(signal.getsignal(signal.SIGALRM),old)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0)


class ResourceAndStaticScopeTests(unittest.TestCase):
    def setUp(self):
        self.snap=dict(CPU=0,RSS=100,processes=[dict(RSS=100,own_CPU=0,completed_child_CPU=0,key=True)])

    def test_CPU_guards(self):
        s=copy.deepcopy(self.snap);s['CPU']=1801
        self.assertEqual(runner.resource_reason(s,C['caps'],0,[]),'total CPU cap')
        s=copy.deepcopy(self.snap);s['processes'][0]['own_CPU']=21
        self.assertEqual(runner.resource_reason(s,C['caps'],0,[]),'per-key CPU cap')

    def test_RSS_guards(self):
        s=copy.deepcopy(self.snap);s['RSS']=1537*1048576
        self.assertEqual(runner.resource_reason(s,C['caps'],0,[]),'aggregate RSS cap')
        s=copy.deepcopy(self.snap);s['processes'][0]['RSS']=513*1048576
        self.assertEqual(runner.resource_reason(s,C['caps'],0,[]),'process RSS cap')

    def test_output_log_worker_guard(self):
        self.assertEqual(runner.resource_reason(self.snap,C['caps'],16777216,[]),'output reserve cap')
        self.assertEqual(runner.resource_reason(self.snap,C['caps'],0,[65537]),'log cap')
        s=copy.deepcopy(self.snap);s['processes']*=3
        self.assertEqual(runner.resource_reason(s,C['caps'],0,[]),'worker count cap')

    def test_no_wall_cap(self):
        self.assertIsNone(C['caps']['total_wall_seconds'])
        self.assertIsNone(C['caps']['phase_wall_seconds'])
        self.assertIsNone(C['caps']['per_key_wall_seconds'])
        self.assertIsNone(runner.resource_reason(self.snap,C['caps'],0,[]))

    def test_scientific_forbidden_calls_absent(self):
        forbidden={'aggregate_pairs','transpile','native_ir','collect_pending','reference_events',
                   'finite_target','exact_target','sample','matrix_power','solve'}
        folder=ROOT/'src/trottertracks/algorithm_codesign/rte_robustness'
        for p in folder.glob('*.py'):
            tree=ast.parse(p.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else None
                    self.assertNotIn(name,forbidden,str(p))


class SavedPipelineOffDomainTests(unittest.TestCase):
    def test_compact_envelope_keeps_minimum_and_tie(self):
        def row(r,arm,a,b):
            return dict(steps=r,arm=arm,expected_native_intercept={k:F(a) for k in model.METRICS},
                        prep_readout_slope=F(b),workspace_beyond_system=1)
        rows=[row(1,'partial_return_tail',1,3),row(2,'closed_P5_full',5,1)]
        out=pipeline.comparisons(ROOT,{'invented':rows})
        curve=out['invented']['all12']['T']
        self.assertEqual(curve['minimum_switch_endpoint_ties'][0]['h'],2)
        self.assertEqual(len(curve['minimum_switch_endpoint_ties'][0]['minimum_arms']),2)
        self.assertTrue(curve['all_affine_inputs_retained_in_resource_rows'])

    def test_four_ledgers_reproduce_invented_r1_and_r2(self):
        caches={'P0':{'2/1009':dict(T_count=2,one_qubit_count=4)},
                'P1':{'2/1009':dict(T_count=3,one_qubit_count=7)}}
        event=dict(event=dict(proposal='1/2',coefficient='1/2',weight='1'),
                   native_ir=[['R',0,'2/1009',1],['CX',1,0],['R',0,'2/1009',-1],['CX',1,0]],
                   native_cost=dict(T=4,CX=2,**{'1Q':8}))
        from trottertracks.algorithm_codesign.g8_bounds import log_upper
        log=log_upper(2/F(C['error_accounting']['alpha_axis']))
        for r in (1,2):
            budget=model.whole_budget(budget_input(),F(1,13),r,'G0',C,log)
            price=dict(T=F(2),CX=F(1),**{'1Q':F(4)});worst=dict(T=4,CX=2,**{'1Q':8})
            row=dict(steps=r,x='1/13',arm='invented',workspace_beyond_system=1,
                     production_step_bounds=budget_input(),reference_law_diagnostic_only={'acceptance':'1/2'},
                     per_step_trial_native=price,worst_single_step_event_native=worst,budget=budget)
            additive=model.row_account(row,budget,price,worst,F(1,2)) if r==1 else None
            if r==2:
                boundary=dict(arm='invented',budget_unchanged=budget,acceptance_squared='1/4',
                              additive_per_whole_trial=price,cancellation_saving_per_whole_trial={'T':'0','CX':'1/8','1Q':'1/4'},
                              after_per_whole_trial=dict(T=F(2),CX=F(7,8),**{'1Q':F(15,4)}))
                additive=dict(expected_native_intercept={k:2*budget['N_per_axis']*v for k,v in price.items()},
                              prep_readout_slope=F(budget['N_per_axis'],2),
                              accepted_tail_native={k:budget['accepted_tail_calls']*r*v for k,v in worst.items()},
                              hard_native={k:2*budget['N_per_axis']*r*v for k,v in worst.items()})
                boundary['new_expected_native_intercept']={k:2*budget['N_per_axis']*v for k,v in boundary['after_per_whole_trial'].items()}
            else:boundary=None
            row.update({k:additive[k] for k in ('expected_native_intercept','prep_readout_slope','accepted_tail_native','hard_native')})
            out=pipeline.evaluate_saved_row(row,[event],boundary,caches,C,log)
            self.assertEqual(set(out),{'G0_P0','G1_P0','G0_P1','G1_P1'})
            self.assertEqual(out['G0_P0']['budget'],out['G0_P1']['budget'])
            self.assertEqual(out['G1_P0']['budget'],out['G1_P1']['budget'])
            self.assertGreater(out['G0_P1']['expected_native_intercept']['T'],out['G0_P0']['expected_native_intercept']['T'])

    def test_48_row_orchestration_mock_never_calls_backend_or_pairs(self):
        # Only orchestration is exercised here. No registered source files read.
        c=copy.deepcopy(C);c['input']['tau']='1/13';c['input']['state']='invented'
        inv=fake_inventory();inv['rows']=[dict(row=i,steps=1 if i<6 else 2,arm='invented_'+str(i)) for i in range(12)]
        old={v['ratio']:fake_saved(v['ratio'],'1/1000000') for v in inv['keys']}
        new={v['ratio']:fake_saved(v['ratio']) for v in inv['keys']}
        bindings=[{}]*157
        reads=Mock(side_effect=[inv,{}]+[bindings]*11+[[{}]*163])
        def evaluated(row,*args):
            return {g:dict(steps=row['steps'],arm=row['arm'],budget={'N_per_axis':5},
                           expected_native_intercept={k:F(10) for k in model.METRICS},
                           prep_readout_slope=F(10),workspace_beyond_system=1)
                    for g in ('G0_P0','G1_P0','G0_P1','G1_P1')}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);base=root/pipeline.D1;base.mkdir(parents=True)
            (base/'synthesis_cache.json').write_text(json.dumps(old))
            (base/'rows').mkdir();(root/pipeline.D2/'rows').mkdir(parents=True)
            for i,entry in enumerate(inv['rows']):
                row=dict(steps=entry['steps'],arm=entry['arm'],degree=5,
                         x=str(F(1,13)/entry['steps']),state='invented',
                         events=dict(path='invented',sha256='0'))
                (base/f'rows/row_{i:02d}.json').write_text(json.dumps(row))
                if i>=6:(root/pipeline.D2/f'rows/row_{i:02d}.json').write_text('{}')
            writer=Mock(directory=root/'output');guard=Mock()
            with patch.object(pipeline,'read_identity',reads),patch.object(pipeline,'check_saved'), \
                 patch.object(pipeline,'acquire',return_value=(new,[])) as acq, \
                 patch.object(pipeline,'check_certificate'),patch.object(pipeline,'evaluate_saved_row',side_effect=evaluated), \
                 patch.object(pipeline,'comparisons',return_value={}):
                coverage=pipeline.run(root,c,{},writer,guard)
            self.assertEqual(coverage['rows'],48);self.assertEqual(coverage['event_metadata_reads'],1890)
            self.assertEqual(acq.call_count,1)
            self.assertEqual(sum(str(args.args[0]).startswith('rows/') for args in writer.write.call_args_list),48)


class WorkerAndJobTests(unittest.TestCase):
    def test_fast_child_final_log_cap_overrides_complete(self):
        c=copy.deepcopy(C);c['caps']['per_key_log_bytes']=8
        identity=dict(source_commit=S,execution_HEAD=A)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d=root/c['execution']['result_directory']
            def child(*args,**kwargs):
                kwargs['stdout'].write(b'0123456789');kwargs['stdout'].flush()
                inner=dict(**identity,status=launch.COMPLETE,rows=48,new_synthesis_calls=36,
                           registered_pair_evaluations=0,retries=0)
                atomic_json(d/'inner_terminal.json',inner)
                return Mock(pid=12345,returncode=None)
            usage=Mock(ru_utime=0,ru_stime=0,ru_maxrss=1)
            with patch.object(runner,'verify_launch',return_value=(c,{},A,{})), \
                 patch.object(runner,'runtime_preflight',return_value={}), \
                 patch.object(runner,'execution_identity',return_value=identity), \
                 patch.object(runner.subprocess,'Popen',side_effect=child), \
                 patch.object(runner.os,'wait4',return_value=(12345,0,usage)), \
                 patch.object(runner,'check_identities'),patch.object(runner,'check_execution_identity'):
                self.assertEqual(runner.execute(root,S),1)
            terminal=json.loads((d/'terminal.json').read_text())
            self.assertEqual(terminal['status'],launch.INCONCLUSIVE)
            self.assertEqual(terminal['technical_reason'],'log cap')
            self.assertTrue((d/'STOP.json').exists())

    def owned_fixture(self,t):
        root=Path(t); e=C['execution']
        def write(p,value):
            p=root/p;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(json.dumps(value))
        write(launch.CONTRACT,C)
        p=root/launch.DESIGN;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((ROOT/launch.DESIGN).read_bytes())
        write(e['source_manifest'],{'critical_sha256':{}})
        auth=authority();auth['contract_sha256']=digest(root/launch.CONTRACT)
        auth['inventory_sha256']=C['acquisition']['key_inventory']['sha256']
        write(e['authorization_path'],auth)
        identity=launch.execution_identity(root,C,auth,A,S)
        directory=root/e['result_directory']
        marker,_=consume_marker(directory,dict(**identity,runs=1,retries=0,mandatory_STOP=True))
        job=dict(**identity,marker_sha256=digest(marker),parent_pid=os.getppid())
        write(e['result_directory']+'/ledger_job.json',job)
        return root,directory,job

    @staticmethod
    def fake_git(root,*args):
        if args==('rev-parse','HEAD'):return A
        if args==('show','-s','--format=%P',A):return S
        if 'diff' in args:return C['execution']['authorization_path']
        raise AssertionError('unexpected Git call')

    def test_owned_job_marker_and_parent(self):
        with tempfile.TemporaryDirectory() as t:
            root,d,job=self.owned_fixture(t)
            with patch.object(launch,'git',side_effect=self.fake_git):
                c,got,path=launch.verify_job(root,d/'ledger_job.json','ledger')
                self.assertEqual(got,job);self.assertEqual(path,d)
                job['parent_pid']+=1;(d/'ledger_job.json').write_text(json.dumps(job))
                with self.assertRaises(PermissionError):launch.verify_job(root,d/'ledger_job.json','ledger')

    def test_owned_job_tampered_marker_and_terminal(self):
        with tempfile.TemporaryDirectory() as t:
            root,d,job=self.owned_fixture(t)
            with patch.object(launch,'git',side_effect=self.fake_git):
                p=d/'one_shot_consumed.json';original=p.read_bytes();p.write_text('{}')
                with self.assertRaises(PermissionError):launch.verify_job(root,d/'ledger_job.json','ledger')
                p.write_bytes(original);(d/'STOP.json').write_text('{}')
                with self.assertRaises(PermissionError):launch.verify_job(root,d/'ledger_job.json','ledger')

    def test_owned_job_wrong_source_and_manifest_snapshot(self):
        with tempfile.TemporaryDirectory() as t:
            root,d,job=self.owned_fixture(t)
            with patch.object(launch,'git',side_effect=self.fake_git):
                job['source_commit']='3'*40;(d/'ledger_job.json').write_text(json.dumps(job))
                with self.assertRaises(PermissionError):launch.verify_job(root,d/'ledger_job.json','ledger')
                job['source_commit']=S
                launch.check_execution_identity(root,C,job)
                (root/C['execution']['source_manifest']).write_text('{}')
                with self.assertRaises(PermissionError):launch.check_execution_identity(root,C,job)

    def test_wrong_job_namespace_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root,d,job=self.owned_fixture(t)
            with self.assertRaises(PermissionError):launch.verify_job(root,d/'other.json','ledger')

    def test_remote_mismatch_rejects_before_saved_input_check(self):
        with tempfile.TemporaryDirectory() as t:
            root,d,job=self.owned_fixture(t)
            def fake(root,*args):
                if args==('remote','get-url','origin'):return 'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
                if args==('branch','--show-current'):return 'invented-branch'
                if 'ls-remote' in args:return '3'*40+' refs/heads/invented-branch'
                if 'status' in args:return ''
                return self.fake_git(root,*args)
            with patch.object(launch,'git',side_effect=fake), \
                 patch.object(launch,'check_identities',side_effect=AssertionError('input read')):
                with self.assertRaisesRegex(PermissionError,'remote authorization SHA'):
                    launch.verify_launch(root,S)

    def test_mock_worker_one_call_and_restart_rejected(self):
        from trottertracks.algorithm_codesign.rte_robustness import worker
        def synth(*args,**kwargs):
            kwargs['on_synthesis']()
            return fake_saved()
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);p=d/'key_000.json';p.write_text('{}')
            job=dict(ratio='2/1009',key='invented',parent_pid=os.getpid())
            with patch.object(worker,'verify_job',return_value=(C,job,d)), \
                 patch.object(worker.resource,'setrlimit'), \
                 patch('trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch.verify_runtime'), \
                 patch('trottertracks.algorithm_codesign.rte_reallocation.numeric.synthesize',side_effect=synth) as backend:
                self.assertEqual(worker.execute(ROOT,p),0)
                self.assertEqual(backend.call_count,1)
                self.assertTrue(p.with_suffix('.backend_call_consumed.json').exists())
                original=p.with_suffix('.result.json').read_bytes()
                with self.assertRaises(FileExistsError):worker.execute(ROOT,p)
                self.assertEqual(backend.call_count,1)
                self.assertEqual(p.with_suffix('.result.json').read_bytes(),original)

    def test_mock_worker_failure_no_second_call(self):
        from trottertracks.algorithm_codesign.rte_robustness import worker
        def synth(*args,**kwargs):
            kwargs['on_synthesis']()
            raise ArithmeticError('invented failure')
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);p=d/'key_000.json';p.write_text('{}')
            job=dict(ratio='2/1009',key='invented',parent_pid=os.getpid())
            with patch.object(worker,'verify_job',return_value=(C,job,d)), \
                 patch.object(worker.resource,'setrlimit'), \
                 patch('trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch.verify_runtime'), \
                 patch('trottertracks.algorithm_codesign.rte_reallocation.numeric.synthesize',side_effect=synth) as backend:
                self.assertEqual(worker.execute(ROOT,p),1)
                self.assertEqual(backend.call_count,1)
                value=json.loads(p.with_suffix('.result.json').read_text())
                self.assertEqual(value['status'],'KEY_TECHNICAL_INCONCLUSIVE')
                self.assertFalse(value['prefix_scientific_use'])

    def test_process_tree_accounts_finished_child_CPU(self):
        with tempfile.TemporaryDirectory() as t:
            proc=Path(t)
            for pid,ppid,cpu,done,key in ((100,1,2,3,False),(101,100,4,5,True)):
                p=proc/str(pid);p.mkdir()
                fields=['S','0']+['0']*20;fields[1]=str(ppid)
                ticks=os.sysconf('SC_CLK_TCK')
                fields[11]=str(cpu*ticks);fields[13]=str(done*ticks);fields[21]='1'
                (p/'stat').write_text(str(pid)+' (python) '+' '.join(fields))
                (p/'cmdline').write_bytes(b'/run_d3_synthesis_key.py\0' if key else b'python\0')
            value=runner.process_tree(100,proc)
            self.assertEqual(value['CPU'],14)
            self.assertEqual(len(value['processes']),2)
            self.assertEqual(sum(v['key'] for v in value['processes']),1)


if __name__=='__main__':unittest.main()
