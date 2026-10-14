"""Artificial scalar fixtures only. No repository scientific results are loaded."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('d0',ROOT/'scripts/d0_saved_value_reanalysis_v1.py')
d0=importlib.util.module_from_spec(spec);spec.loader.exec_module(d0)


def synthetic_signal(cid='SYNTHETIC_B0',method='B0'):
    return dict(cell=dict(id=cid,method=method),error_decomposition=dict(total_signed=dict(real=0.,imag=0.)),
        allowance=dict(real=1e-11,imag=1e-11,log_B_margin=1e-11,kind='EMPIRICAL',certified=False),log_B=0.,precision_rows={})


def wrapper(cid='SYNTHETIC_B0',replica=0,axis='cosine',phase='exploration',value=1):
    return dict(cell_id=cid,phase=phase,replica=replica,seed=100+replica,axis=axis,
        control='symmetric_directional',event_digest='SYNTHETIC_EVENT_'+str(replica),compiler_hash='SYNTHETIC_COMPILER',
        **{m:value for m in d0.METRICS})


def pair(cos,sin,replica=0):
    return dict(cosine=wrapper(replica=replica,value=cos),sine=wrapper(replica=replica,axis='sine',value=sin))


class ShotRuleTests(unittest.TestCase):
    def test_legacy_order_integer_ceil(self):
        s=synthetic_signal();a=d0.accounting(s,.05)
        h=.05/math.sqrt(2)-1e-11
        expected=math.ceil(2*math.exp(1e-11)**2/h**2*math.log(80))
        self.assertEqual(a['axes']['real']['shots'],expected)
        self.assertEqual(a['N_total'],2*expected)

    def test_signed_bias_absolute(self):
        a=synthetic_signal();b=copy.deepcopy(a)
        a['error_decomposition']['total_signed']['imag']=-.0001
        b['error_decomposition']['total_signed']['imag']=.0001
        self.assertEqual(d0.accounting(a,.001),d0.accounting(b,.001))

    def test_headroom_boundary_three_states(self):
        f=d0._rule.axis_headroom
        self.assertEqual(f(1.,1.,0.)['status'],'INELIGIBLE')
        self.assertEqual(f(1.,.99,.02)['status'],'UNDETERMINED')
        self.assertEqual(f(1.,None,.02)['status'],'UNDETERMINED')
        self.assertEqual(f(1.,.5,.02)['status'],'ELIGIBLE')

    def test_u_ratio_guard(self):
        s=synthetic_signal();s['allowance'].update(real=1e-5,imag=1e-5)
        a=d0.accounting(s,.001)
        self.assertEqual(a['eligibility_under_declared_allowance'],'ELIGIBLE')
        self.assertFalse(a['cost_acquisition_eligible'])

    def test_log_only_preserves_no_integer(self):
        s=synthetic_signal();s['log_B']=20.
        a=d0.accounting(s,.001)
        self.assertIsNone(a['N_total'])
        self.assertEqual(a['axes']['real']['integer_status'],'LOG_DOMAIN_ONLY')

    def test_normalization_binary64_binding(self):
        b=1.1;h=.05/math.sqrt(2)
        a=d0._rule.u_aware_shots(dict(real=0.,imag=0.),dict(real=0.,imag=0.),epsilon=.05,
            log_B_upper=math.log(b),normalization_upper=b,evidence_kind='EMPIRICAL',evidence_ref='SYNTHETIC')
        self.assertEqual(a['axes']['real']['shots'],math.ceil(2*b**2/h**2*math.log(80)))
        with self.assertRaisesRegex(ValueError,'NORMALIZATION_LOG_BINDING'):
            d0._rule.u_aware_shots(dict(real=0.,imag=0.),dict(real=0.,imag=0.),epsilon=.05,
                log_B_upper=math.log(b),normalization_upper=1.2,evidence_kind='EMPIRICAL',evidence_ref='SYNTHETIC')

    def test_nonfinite_bool_and_float_overflow(self):
        for x in (True,float('nan'),float('inf'),10**1000):
            s=synthetic_signal();s['log_B']=x
            with self.assertRaises(ValueError):d0.accounting(s,.05)

    def test_sensitivity_not_new_estimator(self):
        s=synthetic_signal();a=d0.accounting(s,.001,10.)
        h=.001/math.sqrt(2)-10.*1e-11
        self.assertEqual(a['axes']['real']['shots'],math.ceil(2*math.exp(10.*1e-11)**2/h**2*math.log(80)))
        with self.assertRaises(ValueError):d0.accounting(s,.001,2.)

    def test_empirical_not_certificate(self):
        s=synthetic_signal();s['allowance']['certified']=True
        with self.assertRaises(ValueError):d0.accounting(s,.05)


class PairTests(unittest.TestCase):
    def test_paired_covariance_and_no_axis_double_count(self):
        r=d0.weighted_metric([pair(1,10),pair(3,14,1)],dict(cosine=2,sine=3),'RZ')
        self.assertEqual(r['G_point'],40)
        self.assertEqual(r['G_standard_error'],8)
        self.assertEqual(r['paired_samples'],2)
        self.assertEqual(r['wrapper_axes'],4)

    def test_exact_integer_products_above_binary64_safe_integer(self):
        n=2**51
        r=d0.weighted_metric([pair(3,7)],dict(cosine=n,sine=n+1),'CX')
        self.assertEqual(int(r['G_exact']['numerator']),10*n+7)
        self.assertEqual(r['G_exact']['denominator'],'1')
        self.assertIsNone(r['G_standard_error'])

    def test_h4_and_h6_operation_orders(self):
        ps=[pair(1,4),pair(2,6,1),pair(4,8,2)]
        shots=dict(cosine=7,sine=11)
        a=d0.weighted_metric(ps,shots,'depth',point_rule='mean_paired_totals')
        b=d0.weighted_metric(ps,shots,'depth',point_rule='sum_axis_products')
        self.assertEqual(a['G_point'],statistics.mean([51,80,116]))
        self.assertEqual(b['G_point'],7*statistics.mean([1,2,4])+11*statistics.mean([4,6,8]))
        self.assertEqual(a['G_exact'],b['G_exact'])

    def test_missing_and_not_saved_distinct(self):
        r=d0.weighted_metric([],dict(cosine=1,sine=1),'RZ')
        self.assertEqual(r['status'],'MISSING')
        r=d0.mean_only_metric(dict(cosine=None,sine=2),dict(cosine=1,sine=1))
        self.assertEqual(r['status'],'NOT_SAVED')

    def test_means_never_fabricate_paired_se(self):
        r=d0.mean_only_metric(dict(cosine=2,sine=4),dict(cosine=3,sine=5))
        self.assertEqual(r['G_point'],26)
        self.assertIsNone(r['G_standard_error']);self.assertIsNone(r['paired_samples'])

    def test_duplicate_axis(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE_PAIRED_AXIS'):
            d0.paired_rows([wrapper(),wrapper()],cell_id='SYNTHETIC_B0',phase='exploration',compiler_hash='SYNTHETIC_COMPILER')

    def test_incomplete_axis(self):
        with self.assertRaisesRegex(ValueError,'INCOMPLETE_PAIRED_AXES'):
            d0.paired_rows([wrapper()],cell_id='SYNTHETIC_B0',phase='exploration',compiler_hash='SYNTHETIC_COMPILER')

    def test_identity_phase_compiler_mismatch(self):
        for key,value in [('seed',200),('event_digest','WRONG'),('phase','confirmation'),('compiler_hash','OTHER')]:
            ps=pair(1,1);ps['sine'][key]=value
            with self.assertRaises(ValueError):
                d0.paired_rows(list(ps.values()),cell_id='SYNTHETIC_B0',phase='exploration',compiler_hash='SYNTHETIC_COMPILER')

    def test_ordinary_excluded(self):
        ps=pair(1,1);ordinary=wrapper();ordinary['control']='ordinary'
        self.assertEqual(len(d0.paired_rows([*ps.values(),ordinary],cell_id='SYNTHETIC_B0',phase='exploration',compiler_hash='SYNTHETIC_COMPILER')),1)

    def test_integer_shots_and_cost_guard(self):
        for n in (True,1.,0,2**52):
            with self.assertRaises(ValueError):d0.weighted_metric([pair(1,1)],dict(cosine=n,sine=1),'RZ')


class ScopeSelectionTests(unittest.TestCase):
    def test_rz_selected_vs_metric_reselection(self):
        rows=[dict(cell_id='A',method='B2',status='DERIVED_SAVED_SCALAR',metrics={'RZ':{'G_point':5},'CX':{'G_point':20}}),
            dict(cell_id='B',method='B2',status='DERIVED_SAVED_SCALAR',metrics={'RZ':{'G_point':7},'CX':{'G_point':2}})]
        a=d0.family_selection(rows,'CX')['B2']
        self.assertEqual(a['RZ_selected_cell'],'A');self.assertEqual(a['metric_selected_cell'],'B')

    def test_original_equality_gate_and_mismatch(self):
        s=synthetic_signal();a=d0.accounting(s,.05);s['precision_rows']={'.05':{'1.0':a}}
        # Use canonical string keys, exactly like saved native precision rows.
        s['precision_rows']={'0.05':{'1.0':a}}
        cfg=dict(compiler_hashes=['SYNTHETIC_COMPILER'],primary_phase='exploration',point_rule='sum_axis_products')
        records=d0.analyze_scope([s],list(pair(2,3).values()),cfg,[.05],[1.])
        value=records[0]['metrics']['RZ']['G_point']
        old={'rows':[dict(cell_id=s['cell']['id'],epsilon=.05,allowance_factor='1.0',metric='RZ',G_point=value,G_standard_error=None,N_total=a['N_total'])]}
        self.assertEqual(d0.verify_original([s],records,old)['cost_cases'],1)
        old['rows'][0]['G_point']+=1
        with self.assertRaisesRegex(ValueError,'ORIGINAL_MAP_VALUE_MISMATCH'):d0.verify_original([s],records,old)

    def test_confirmation_excludes_exploration_random_samples(self):
        s=synthetic_signal('SYNTHETIC_B2','B2')
        cfg=dict(compiler_hashes=['SYNTHETIC_COMPILER'],primary_phase='confirmation',point_rule='sum_axis_products',confirmation_ids=['SYNTHETIC_B2'])
        wrong=[r for i in range(8) for r in [wrapper('SYNTHETIC_B2',i),wrapper('SYNTHETIC_B2',i,'sine')]]
        with self.assertRaisesRegex(ValueError,'WRAPPER_PHASE'):d0.analyze_scope([s],wrong,cfg,[.05],[1.])
        right=[wrapper('SYNTHETIC_B2',i,axis,'confirmation') for i in range(32) for axis in ('cosine','sine')]
        self.assertEqual(d0.analyze_scope([s],right,cfg,[.05],[1.])[0]['sample_count'],32)

    def test_ineligible_and_numeric_unresolved_separate(self):
        s=synthetic_signal();s['error_decomposition']['total_signed']['real']=.1
        cfg=dict(compiler_hashes=['SYNTHETIC_COMPILER'],primary_phase='exploration',point_rule='sum_axis_products')
        self.assertEqual(d0.analyze_scope([s],[],cfg,[.05],[1.])[0]['status'],'INELIGIBLE')
        s=synthetic_signal();s['log_B']=20.
        self.assertEqual(d0.analyze_scope([s],[],cfg,[.05],[1.])[0]['status'],'NUMERICALLY_UNRESOLVED')


class PreparationTests(unittest.TestCase):
    def test_fixed_crossing_parallel_negative_and_equal(self):
        a=dict(intercept=0,slope=10);b=dict(intercept=5,slope=5)
        self.assertEqual(d0.fixed_crossing(a,b)['P'],dict(numerator='1',denominator='1'))
        self.assertEqual(d0.fixed_crossing(a,dict(intercept=5,slope=10))['status'],'PARALLEL_NO_CROSSING')
        self.assertEqual(d0.fixed_crossing(a,a)['status'],'IDENTICAL_LINES')
        self.assertEqual(d0.fixed_crossing(a,dict(intercept=-1,slope=5))['status'],'NEGATIVE_DOMAIN_EXCLUDED')

    def test_envelope_candidate_switches(self):
        lines=[dict(cell_id='A',intercept=0,slope=10),dict(cell_id='B',intercept=5,slope=5),dict(cell_id='C',intercept=10,slope=1)]
        e=d0.lower_envelope(lines)
        self.assertEqual([r['cell_ids'] for r in e['segments']],[['A'],['B'],['C']])
        self.assertEqual(e['segments'][2]['P_start'],dict(numerator='5',denominator='4'))

    def test_envelope_identical_and_vertex_only_ties(self):
        lines=[dict(cell_id='A',intercept=0,slope=10),dict(cell_id='A2',intercept=0,slope=10),
            dict(cell_id='D',intercept=2,slope=8),dict(cell_id='B',intercept=5,slope=5)]
        e=d0.lower_envelope(lines)
        self.assertEqual(e['segments'][0]['cell_ids'],['A','A2'])
        self.assertEqual(e['vertices'][1]['tied_cells'],['A','A2','B','D'])

    def test_empty_envelope_and_exact_fraction(self):
        self.assertEqual(d0.lower_envelope([])['segments'],[])
        self.assertEqual(d0.rational(d0.Fraction(1,3)),dict(numerator='1',denominator='3'))


class GateTests(unittest.TestCase):
    def test_strict_json(self):
        for data in ('{"x":NaN}','{"x":1,"x":2}'):
            with self.assertRaises(ValueError):d0.strict_json(data)

    def test_explicit_separate_approval_and_one_shot(self):
        with tempfile.TemporaryDirectory(prefix='d0-synthetic-',dir='/home/AbeHiromu/projects/h4-handoff-evidence/20261014/d0-saved-value-preflight-v1') as directory:
            out=Path(directory)/'output'
            c=dict(analysis_source='SYNTHETIC_SOURCE',new_science_calls=0,H8_included=False)
            a=dict(approved_by_user=True,scope='D0_SAVED_SCALAR_ONE_SHOT',contract_sha256='SYNTHETIC_SHA',analysis_source='SYNTHETIC_SOURCE',exclusive_output=str(out),automatic_retry=False)
            d0.execution_gate(c,a,contract_sha='SYNTHETIC_SHA',output=out)
            b=dict(a,approved_by_user=False)
            with self.assertRaises(ValueError):d0.execution_gate(c,b,contract_sha='SYNTHETIC_SHA',output=out)
            out.with_name('output.oneshot').write_text('SYNTHETIC_MARKER')
            with self.assertRaisesRegex(ValueError,'ONE_SHOT'):d0.execution_gate(c,a,contract_sha='SYNTHETIC_SHA',output=out)

    def test_hash_mismatch_and_non_scalar_format(self):
        with tempfile.TemporaryDirectory(prefix='d0-synthetic-',dir='/home/AbeHiromu/projects/h4-handoff-evidence/20261014/d0-saved-value-preflight-v1') as directory:
            p=Path(directory)/'synthetic.json';b=b'{"fixture":"synthetic"}'
            p.write_bytes(b);r=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
            self.assertEqual(d0.bound_input(p,r),b)
            p.write_bytes(b.replace(b'synthetic',b'changedxx'))
            with self.assertRaises(ValueError):d0.bound_input(p,r)
            q=Path(directory)/'synthetic.npz';q.write_bytes(b)
            with self.assertRaises(ValueError):d0.bound_input(q,r)

    def test_synthetic_cli_complete_then_one_shot_refusal(self):
        self.synthetic_cli_case(output_cap=2**20,expect_failure=False)

    def test_synthetic_cli_output_failure_retains_first_stop(self):
        self.synthetic_cli_case(output_cap=10,expect_failure=True)

    def synthetic_cli_case(self,*,output_cap,expect_failure):
        with tempfile.TemporaryDirectory(prefix='d0-synthetic-',dir='/home/AbeHiromu/projects/h4-handoff-evidence/20261014/d0-saved-value-preflight-v1') as directory:
            base=Path(directory);out=base/'output'
            s=synthetic_signal();a=d0.accounting(s,.05);s['precision_rows']={'0.05':{'1.0':a}}
            cfg=dict(signal_alias='signals.json',wrapper_alias='wrappers.csv',old_map_alias='old.json',compiler_hashes=['SYNTHETIC_COMPILER'],primary_phase='exploration',point_rule='sum_axis_products')
            rs=list(pair(2,3).values());keys=list(rs[0])
            csv=(',' .join(keys)+'\n'+'\n'.join(','.join(str(r[k]) for k in keys) for r in rs)+'\n').encode()
            old=dict(rows=[dict(cell_id=s['cell']['id'],epsilon=.05,metric='RZ',allowance_factor='1.0',G_point=2*a['axes']['real']['shots']+3*a['axes']['imag']['shots'],G_standard_error=None,N_total=a['N_total'])])
            blobs={'signals.json':json.dumps(dict(signals=[s])).encode(),'wrappers.csv':csv,'old.json':json.dumps(old).encode()}
            paths={}
            for name,b in blobs.items():(base/name).write_bytes(b);paths[name]=str(base/name)
            c=dict(analysis_source='SYNTHETIC_SOURCE',analysis_source_hashes={},analysis_python=d0.sys.version.split()[0],
                new_science_calls=0,H8_included=False,inputs=[dict(alias=k,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in blobs.items()],
                scopes={d0.SCOPES[1]:cfg},original_epsilons=[.05],epsilons=[.05,.0005],allowance_factors=[1.],
                budget=dict(address_space_bytes=512*2**20,cpu_seconds=60,wall_seconds=60,fresh_available_bytes=1,fresh_available_inodes=1,
                    rss_bytes=512*2**20,output_bytes=output_cap,fresh_memory_available_bytes=1))
            b=json.dumps(c).encode();(base/'contract.json').write_bytes(b)
            # This ephemeral fixture is not a real user grant; no real input is bound.
            a=dict(approved_by_user=True,scope='D0_SAVED_SCALAR_ONE_SHOT',contract_sha256=hashlib.sha256(b).hexdigest(),analysis_source='SYNTHETIC_SOURCE',exclusive_output=str(out),automatic_retry=False)
            (base/'synthetic_authorization.json').write_text(json.dumps(a));(base/'inputs.json').write_text(json.dumps(paths))
            argv=['d0','--contract',str(base/'contract.json'),'--authorization',str(base/'synthetic_authorization.json'),'--inputs',str(base/'inputs.json'),'--output',str(out)]
            snapshot=dict(effective_available_bytes=2**30,host_full_psi_avg10=0.,host_oom_kill=0,cgroups=[])
            with mock.patch.object(d0.sys,'argv',argv),mock.patch.object(d0.resource,'setrlimit'),mock.patch.object(d0.signal,'signal'),mock.patch.object(d0.signal,'setitimer'),mock.patch.object(d0,'fresh_memory_gate',return_value=snapshot),mock.patch.object(d0,'memory_snapshot',return_value=snapshot),mock.patch.dict(d0.os.environ,{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}):
                if expect_failure:
                    with self.assertRaisesRegex(ValueError,'D0_OUTPUT_CAP'):d0.main()
                else:d0.main()
                terminal=json.loads((out/'terminal_v1.json').read_text())
                self.assertEqual(terminal['status'],'FAIL_CLOSED_STOP' if expect_failure else 'COMPLETE_MANDATORY_STOP')
                self.assertEqual(terminal['first_stop_reason'],'ValueError:D0_OUTPUT_CAP' if expect_failure else None)
                self.assertTrue(out.with_name('output.oneshot').exists())
                with self.assertRaisesRegex(ValueError,'ONE_SHOT'):d0.main()

    def test_resource_pressure_oom_and_missing_ancestry(self):
        snap=dict(effective_available_bytes=2**30,host_full_psi_avg10=0.,host_oom_kill=0,
            cgroups=[dict(path='SYNTHETIC',full_psi_avg10=0.,events={'oom':0})])
        budget=dict(fresh_memory_available_bytes=2**29)
        d0.check_memory_snapshot(snap,budget,snap)
        for change in (dict(effective_available_bytes=1),dict(host_full_psi_avg10=1.),dict(host_oom_kill=1),
            dict(cgroups=[]),dict(cgroups=[dict(path='SYNTHETIC',full_psi_avg10=.001,events={'oom':0})]),
            dict(cgroups=[dict(path='SYNTHETIC',full_psi_avg10=0.,events={'oom':1})])):
            new=dict(snap,**change)
            with self.assertRaises(ValueError):d0.check_memory_snapshot(new,budget,snap)


if __name__=='__main__':
    unittest.main(verbosity=2)
