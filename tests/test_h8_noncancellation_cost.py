"""Manufactured endpoints, shots, costs, approvals and two minimal child cases.

Only registered candidate coordinates are read. No saved H8 numerical input is
opened. TMPDIR must point into the user's home for all artificial files.
"""
from copy import deepcopy
import csv
import gzip
from fractions import Fraction as F
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import h8_noncancellation_cost_contract as r
import run_h8_noncancellation_cost as runner
import test_h8_noncancellation_diagnostic as old
import test_h8_phase_c as phase_fixture
c=r.c;n=c.n
REGISTRY=dict(old.REGISTRY,confirmation_ids=list(old.UNION))


def csv_bytes(rows):
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return out.getvalue().encode()


def projections(raw):
    rows=[]
    for x in raw['accounting_rows']:
        cid=x['cell_id'];group='old32' if cid in old.OLD else 'additional152' if cid in old.CELLS else 'PR_confirmation_union6'
        a=x['accounting'];shots=x['shots']
        item=dict(candidate_id=cid,method=x['method'],set=group,epsilon=x['epsilon'],allowance_factor=x['factor'],
            noncancellation_eligibility=x['eligibility'],
            shots_real=a['axes']['real']['shots'] if a['axes']['real']['shots'] is not None else '',
            shots_imag=a['axes']['imag']['shots'] if a['axes']['imag']['shots'] is not None else '',
            noncancellation_N_total=x['N_total'] if shots else '',
            bias_kind=x['bias_kind'],rigorous_bias_guarantee='')
        for axis in ('real','imag'):
            item['headroom_'+axis]=a['axes'][axis]['headroom']
            item['u_over_headroom_'+axis]=a['axes'][axis]['u_over_headroom'] if shots else ''
        rows.append(item)
    axes=[dict(candidate_id=x['cell_id'],axis=x['axis'],status=x['status'],
        usual_signed_bias_abs_total=x['usual_signed_bias'],empirical_noncancellation_bias=x['empirical_noncancellation_bias'])
        for x in raw['axis_envelopes']]
    checks=[dict(candidate_id=cid,check=x['name'],passed=str(x['passed']),signed_difference=x['signed_difference'])
            for cid,group in raw['gates']['PR_component_checks'].items() for x in group['checks']]
    return dict(csv=csv_bytes(rows),axis_csv=csv_bytes(axes),PR_csv=csv_bytes(checks))


def fixture():
    data=old.base()
    raw=n.evaluate(data,old.REGISTRY,old.PERMIT,stages=('NONCANCELLATION_SHOTS',))
    receipts=[dict(role='artificial_role_'+str(i),bytes=10,sha256='a'*64) for i in range(22)]
    raw.update(SOURCE=c.SHOT_SOURCE,plan_sha256=c.SHOT_PLAN,grant_sha256=c.SHOT_GRANT,
               post_execution_binding_verified=True,input_receipts=deepcopy(receipts))
    proof=dict(raw=raw,binding=dict(SOURCE=c.SHOT_SOURCE,plan_digest=c.SHOT_PLAN,
        input_manifest_sha256=c.SHOT_INPUT,post_execution_binding_verified=True,one_shot_lock_retained=True,
        input_receipts=deepcopy(receipts),grant=dict(digest=c.SHOT_GRANT,authorized_stages=['NONCANCELLATION_SHOTS'],
            cost_stage_authorized=False,one_shot=True,automatic_retry=False)),
        summary=dict(SOURCE=c.SHOT_SOURCE,plan_digest=c.SHOT_PLAN,grant_digest=c.SHOT_GRANT,status=c.SHOT_STATUS,
            stages_executed=['NONCANCELLATION_SHOTS'],rows=1900,axes=380,unresolved_axes=0,
            gates=dict(saved_usual_rows_reproduced=1900),cost_stage=dict(status='NOT_EXECUTED')),
        terminal=dict(SOURCE=c.SHOT_SOURCE,status=c.SHOT_STATUS,stages_executed=['NONCANCELLATION_SHOTS'],
            first_stop=None,mandatory_STOP=True,automatic_retry=False,next_stage_authorized=False,all_children_reaped=True),
        native=dict(native_exit_code=0,reaped=True,pid_absent_after_wait=True,children_started_by_runner=0),
        **projections(raw))
    data.update(completed_shots=proof,shot_manifest=dict(files=receipts),
                B2=phase_fixture.b2_proof(old.MODEL_SHA))
    cells,union,records=n.bind_domain(data,old.REGISTRY)
    usual=n.usual_index(data['saved_usual'],[*cells,*union]);usual_rows=[]
    for (cid,e,f),a in usual.items():
        ok,shots=n.eligibility(a)
        usual_rows.append(dict(cell_id=cid,method=records[cid]['cell']['method'],epsilon=e,factor=f,
            eligible=ok,shots=shots,G_exact=None,G_standard_error=None))
    measured=c.phase_c.b2_gate(data['B2'])
    cost=n.costs(data,cells,union,records,usual_rows,usual,{k:v['RZ'] for k,v in measured.items()},
                 old.OLD,list(cells),lambda:None)
    data.update(usual_result=dict(status='COMPLETE_MANDATORY_STOP',bias_kind='USUAL_SIGNED_SAVED',
        noncancellation_diagnostic_used=False,model_sha256=old.MODEL_SHA,
        B2_binding=dict(status=c.phase_c.B2_STATUS),comparisons=cost['comparisons']),usual_rows=usual_rows)
    return data


class CostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patches=[patch.object(n,'MODEL_SHA',old.MODEL_SHA),patch.object(c,'MODEL_SHA',old.MODEL_SHA),
                     patch.object(c.phase_c,'MODEL_SHA',old.MODEL_SHA)]
        for p in cls.patches:p.start()
        cls.original=fixture()
    @classmethod
    def tearDownClass(cls):
        for p in reversed(cls.patches):p.stop()
    def setUp(self):self.data=deepcopy(self.original)
    def run_cost(self):
        calls=[];rows=[]
        result=c.evaluate(self.data,REGISTRY,r.Permit(synthetic=True),take=calls.append,
                          emit=lambda kind,row:rows.append(row))
        return result,calls,rows
    def test_single_cost_stage_no_shot_or_usual_reaccount(self):
        with patch.object(n,'evaluate',side_effect=AssertionError('SHOT_REEXECUTION')),\
             patch.object(n,'frozen_accounting',side_effect=AssertionError('NEW_SHOTS')),\
             patch.object(n.d0,'accounting',side_effect=AssertionError('NEW_SHOTS')):
            result,calls,rows=self.run_cost()
        self.assertEqual((len(rows),calls.count('reaccounting_pass'),calls.count('comparison')),(1900,1,20))
        self.assertEqual(result['stages_executed'],['COST_COMPARISON'])
        self.assertEqual(result['shots_recalculated'],0)
        self.assertTrue(result['gates']['PR_saved_paired_metric_reproduction']['executed'])
        self.assertTrue(all(x['bias_kind']==c.BIAS for x in rows))
    def test_B2_four_axis_sources_measurement_overrides(self):
        _,_,rows=self.run_cost()
        for cid,axis,source,_ in c.phase_c.B2_AXES:
            row=next(x for x in rows if x['cell_id']==cid and x['eligible'])
            self.assertEqual(row['axis_costs'][axis]['final_B2_axis_evidence']['SOURCE'],source)
            self.assertTrue(row['axis_costs'][axis]['measured'])
    def test_partial_STOP_wrong_SOURCE_model_fingerprint_native_and_threshold_reject(self):
        mutations=[lambda b:b['binding']['axes'].pop(),lambda b:b['binding'].update(status='STOP'),
            lambda b:b['binding']['axes'][3].update(SOURCE=c.phase_c.B2_AXES[2][2]),
            lambda b:b['binding'].update(model_sha256='b'*64),
            lambda b:b['binding']['axes'][0].update(fingerprint_after='changed'),
            lambda b:b['receipts']['native_launcher_wait'].update(native_exit_code=1),
            lambda b:b['receipts']['raw_launcher_terminal'].update(all_reaped=False),
            lambda b:b['binding']['final_gate'].update(status='STOP'),
            lambda b:b['binding']['axes'][0]['model_check'].update(exact_relative_error='1/50')]
        for mutate in mutations:
            self.data=deepcopy(self.original);mutate(self.data['B2']);calls=[]
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                c.evaluate(self.data,REGISTRY,r.Permit(synthetic=True),take=calls.append)
            self.assertEqual(calls,[])
    def test_shot_SOURCE_plan_grant_input_native_scope_and_missing_reject(self):
        mutations=[lambda p:p['raw'].update(SOURCE='b'*40),lambda p:p['binding'].update(plan_digest='b'*64),
            lambda p:p['summary'].update(grant_digest='b'*64),lambda p:p['binding'].update(input_manifest_sha256='b'*64),
            lambda p:p['native'].update(reaped=False),lambda p:p['terminal'].update(status='STOP'),
            lambda p:p['raw'].update(stages_executed=['COST_COMPARISON']),lambda p:p['raw']['accounting_rows'].pop(),
            lambda p:p['raw']['accounting_rows'].append(deepcopy(p['raw']['accounting_rows'][0])),
            lambda p:p['raw']['input_receipts'][0].update(sha256='b'*64)]
        for mutate in mutations:
            self.data=deepcopy(self.original);mutate(self.data['completed_shots'])
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.run_cost()
    def test_csv_mismatched_shot_scope_nonfinite_and_missing_reject(self):
        for field,value in [('shots_real','999'),('bias_kind','USUAL_SIGNED_SAVED'),('headroom_real','nan'),('set','other')]:
            self.data=deepcopy(self.original);p=self.data['completed_shots'];rows=c.csv_rows(p['csv'])
            row=next(x for x in rows if x['noncancellation_eligibility']=='ELIGIBLE');row[field]=value;p['csv']=csv_bytes(rows)
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_cost()
    def test_nonfinite_raw_and_UNDETERMINED_reject(self):
        for change in [lambda row:row['accounting']['axes']['real'].update(headroom=float('nan')),
                       lambda row:row.update(eligible=None,eligibility='UNRESOLVED')]:
            self.data=deepcopy(self.original);row=next(x for x in self.data['completed_shots']['raw']['accounting_rows'] if x['eligible'])
            change(row)
            with self.assertRaises(ValueError):self.run_cost()
    def test_PR_exploration_seed_source_compiler_pairing_failures(self):
        for field,value in [('phase','exploration'),('seed',-1),('source','wrong'),('compiler_hash','wrong')]:
            self.data=deepcopy(self.original);row=next(x for x in self.data['wrappers'] if x['phase']=='confirmation')
            row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_cost()
        self.data=deepcopy(self.original)
        self.data['wrappers']=[x for x in self.data['wrappers'] if not(x['cell_id']==old.UNION[0] and x['replica']==31)]
        with self.assertRaises(ValueError):self.run_cost()
    def test_saved_PR_metric_reproduction_required(self):
        target=next(x for x in self.data['PR_saved_metrics'] if x['status']=='DERIVED_SAVED_SCALAR')
        target['metrics']['RZ']['G_standard_error']+=1
        with self.assertRaisesRegex(ValueError,'NOT_REPRODUCED'):self.run_cost()
    def test_new_SE_reweighted_from_pairs_and_exact_det_axis_sum(self):
        for row in self.data['completed_shots']['raw']['accounting_rows']:
            if row['method']=='B2' and row['eligible']:
                row['shots']['real']*=2
                row['N_total']=sum(row['shots'].values())
                row['accounting']['axes']['real']['shots']=row['shots']['real']
                row['accounting']['N_total']=row['N_total']
        self.data['completed_shots'].update(projections(self.data['completed_shots']['raw']))
        _,_,rows=self.run_cost()
        current=next(x for x in rows if x['method']=='B2' and x['eligible'])
        pairs=n.PR_pairs(self.data['wrappers'],old.UNION,self.data['confirmation']['registrations'],
                         self.data['wrapper_binding']['allowed_sources'])[current['cell_id']]
        metric=n.d0.weighted_metric(pairs,{a:current['shots'][n.CHANNEL[a]] for a in n.CHANNEL},'RZ',point_rule='sum_axis_products')
        self.assertEqual(current['G_standard_error'],metric['G_standard_error'])
        normal=next(x for x in self.data['usual_rows'] if c.row_key(x)==c.row_key(current))
        self.assertNotEqual(current['G_standard_error'],normal['G_standard_error'])
        self.assertEqual(F(current['G_exact']),F(int(metric['G_exact']['numerator']),int(metric['G_exact']['denominator'])))
        for row in rows:
            if row['method']!='B2' and row['eligible']:
                self.assertEqual(F(row['G_exact']),sum(row['shots'][n.CHANNEL[a]]*F(x['RZ']) for a,x in row['axis_costs'].items()))
    def test_phase_C_unit_drift_or_missing_eligible_cost_refused(self):
        normal=next(x for x in self.data['usual_rows'] if x['method']!='B2' and x['eligible'])
        normal['axis_costs']['cosine']['RZ']='1'
        with self.assertRaisesRegex(ValueError,'UNIT_COST_DIFFERS'):self.run_cost()
    def test_no_winner_selection_based_only_on_usual_winner(self):
        first=self.data['usual_result']['comparisons'][0]
        usual=first['scopes']['strengthened184']['deterministic']['best']['cell_id']
        for row in self.data['completed_shots']['raw']['accounting_rows']:
            if row['cell_id']==usual:
                row.update(eligible=False,eligibility='INELIGIBLE',shots=None,N_total=None)
        self.data['completed_shots'].update(projections(self.data['completed_shots']['raw']))
        result,_,_=self.run_cost()
        best=result['comparisons'][0]['scopes']['strengthened184']['deterministic']['best']
        self.assertNotEqual(best['cell_id'],usual)
        self.assertTrue(result['comparisons'][0]['difference_from_saved_usual']['strengthened184']['deterministic_winner_changed'])
    def test_scope_changes_and_percentage_points_use_saved_comparison(self):
        before=deepcopy(self.data['usual_result']);result,_,_=self.run_cost()
        self.assertEqual(before,self.data['usual_result'])
        for x in result['comparisons']:
            self.assertEqual(set(x['scopes']),{'old32','strengthened184'})
            for scope,diff in x['difference_from_saved_usual'].items():
                a,b=x['scopes'][scope]['PR_reduction_percent_exact'],diff['usual_saved']['PR_reduction_percent_exact']
                if a is None or b is None:self.assertIsNone(diff['reduction_change_percentage_points_exact'])
                else:self.assertEqual(F(diff['reduction_change_percentage_points_exact']),F(a)-F(b))
    def test_evidence_labels_extrapolation_S4_and_rare_orders(self):
        result,_,rows=self.run_cost()
        units=result['unit_costs'];self.assertTrue(any(not x['measured'] and x['prefix_outside_B1_fit_range'] for x in units))
        self.assertTrue(any('S4_MODEL' in x['status'] for x in units))
        self.assertIsNone(result['rigorous_bias_guarantee']);self.assertFalse(result['full_grid_model_error_guaranteed'])
        self.assertEqual(result['rare_layers'],self.data['completed_shots']['raw']['rare_layers'])
    def test_exact_fraction_tie_lexical_order_and_unresolved_subset(self):
        rows=[dict(cell_id=cid,eligible=True,G_exact=value,cost_status='MODEL_CONDITIONAL_POINT')
              for cid,value in [('z','2/3'),('a','4/6')]]
        result=n.minimum(rows);self.assertEqual(result['best']['cell_id'],'a');self.assertEqual(result['exact_ties'],['a','z'])
        rows.append(dict(cell_id='missing',eligible=None,G_exact=None))
        self.assertEqual(n.minimum(rows)['status'],'UNRESOLVED_FULL_DOMAIN');self.assertIsNone(n.minimum(rows)['best'])
    def test_all_ineligible_is_explicit_null_not_zero_or_best(self):
        result=n.minimum([dict(cell_id='a',eligible=False,G_exact=None)])
        self.assertEqual(result,dict(status='NO_ELIGIBLE_CANDIDATE',best=None))


class ContractTests(unittest.TestCase):
    def authorization(self):
        cfg=dict(SOURCE='a'*40,execution_sealed=True,plan_sha256=c.digest(r.plan()))
        grant=dict(schema='H8_NONCANCELLATION_COST_GRANT_V3',scope=r.SCOPE,SOURCE=cfg['SOURCE'],
            approved=True,runtime_authorization=True,execution_sealed=True,approval_text='ARTIFICIAL ONLY',
            authorized_stages=['COST_COMPARISON'],plan_sha256=cfg['plan_sha256'],config_sha256=c.digest(cfg),
            input_contract_sha256=r.sha(r.ROOT/r.INPUT_CONTRACT),call_caps=r.CAPS,resource_policy=r.RESOURCE_POLICY,
            allowed_cpus=[min(os.sched_getaffinity(0))],one_shot=True,automatic_retry=False,new_shot_calculation=False,
            next_stage_authorized=False)
        return cfg,grant
    def test_only_new_cost_grant_accepts_and_no_shots_grant(self):
        cfg,g=self.authorization();self.assertTrue(r.authorize(cfg,g))
        for field,value in [('approved',False),('runtime_authorization',False),('execution_sealed',False),
                            ('authorized_stages',['NONCANCELLATION_SHOTS','COST_COMPARISON']),('SOURCE','b'*40),
                            ('plan_sha256','b'*64),('input_contract_sha256','b'*64),('automatic_retry',True),
                            ('new_shot_calculation',True),('allowed_cpus',[])]:
            wrong=deepcopy(g);wrong[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):r.authorize(cfg,wrong)
    def test_invalid_grant_before_input_read_or_process_or_write(self):
        cfg,g=self.authorization();g['approved']=False
        with patch.object(r,'read',side_effect=[cfg,g]),patch.object(r,'verify',side_effect=AssertionError('INPUT_READ')),\
             patch.object(runner,'save',side_effect=AssertionError('WRITE')),\
             patch.object(runner.subprocess,'Popen',side_effect=AssertionError('CHILD')):
            with self.assertRaisesRegex(ValueError,'SEALED_GRANT'):runner.launch('ARTIFICIAL','ARTIFICIAL')
    def test_output_and_one_shot_reuse_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cfg=dict(output=str(p/'out'),control=str(p/'control'),lock=str(p/'lock'),protected_files=[],protected_worktrees=[])
            self.assertEqual(len(r.path_set(cfg)),3)
            (p/'lock').touch()
            with self.assertRaises(ValueError):r.path_set(cfg)
            (p/'lock').unlink();(p/'out').mkdir()
            with self.assertRaises(ValueError):r.path_set(cfg)
    def test_cost_output_cannot_be_inside_SOURCE(self):
        cfg=dict(output=str(ROOT/'FORBIDDEN_OUTPUT'),control=str(ROOT/'FORBIDDEN_CONTROL'),lock=str(ROOT/'FORBIDDEN_LOCK'),
                 protected_files=[],protected_worktrees=[])
        with self.assertRaisesRegex(ValueError,'OUTSIDE_SOURCE'):r.path_set(cfg)
    def test_SOURCE_and_input_drift_checked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'source.py').write_text('pass\n');(root/'input.json').write_text('{}')
            spec=dict(role='fixture',bytes=2,sha256=r.sha(root/'input.json'))
            cfg=dict(schema='H8_NONCANCELLATION_COST_CONFIG_V3',SOURCE='a'*40,plan_sha256='fixture',
                environment=r.original.environment(),python=r.PYTHON,python_sha256=r.sha(r.PYTHON),
                source_hashes={'source.py':r.sha(root/'source.py')},inputs=[dict(spec,path=str(root/'input.json'))])
            with patch.object(r,'ROOT',root),patch.object(r,'SOURCES',('source.py',)),\
                 patch.object(r,'plan',return_value={}),patch.object(c,'digest',return_value='fixture'),\
                 patch.object(r,'read',return_value=dict(files=[spec])):
                r.verify(cfg,git=False,protection=False)
                (root/'source.py').write_text('changed\n')
                with self.assertRaisesRegex(ValueError,'SOURCE_BYTES_CHANGED'):r.verify(cfg,git=False,protection=False)
                (root/'source.py').write_text('pass\n');(root/'input.json').write_text('{"x":1}')
                with self.assertRaises(ValueError):r.verify(cfg,git=False,protection=False)
    def test_unrelated_worktree_progress_observational(self):
        cfg=dict(source_hashes={},inputs=[],protected_files=[],protected_worktrees=[dict(path=str(ROOT),HEAD='old',status_sha256='old')])
        report=r.common.protected(cfg)
        self.assertTrue(report['worktrees'][0]['changed']);self.assertFalse(report['worktrees'][0]['enforced'])
    def test_strict_json_nonfinite_duplicate_and_csv_duplicate(self):
        for text in ['{"a":1,"a":2}','{"a":NaN}','{"a":1e999}']:
            with self.assertRaises(ValueError):r.loads(text.encode())
        with self.assertRaises(ValueError):c.csv_rows(b'a,a\n1,2\n')
    def test_saved_loader_keeps_full_registration_without_science(self):
        data=old.base();registry=r.read(r.ROOT/r.REGISTRY)
        manifest=r.read(r.ROOT/r.original.MANIFEST)
        inputs=[dict(role='original:'+x['role'],path='/home/AbeHiromu/ARTIFICIAL/inputs/'+x['role'],
                     bytes=1,sha256='a'*64) for x in manifest['files']]
        for role in ('comparisons_v1.json','accounting_rows_v1.jsonl.gz','accounting_rows_receipt_v1.json'):
            inputs.append(dict(role='usual:'+role,path='/home/AbeHiromu/ARTIFICIAL/'+role,bytes=1,sha256='a'*64))
        decoded=b'{}\n';compressed=gzip.compress(decoded,mtime=0)
        import hashlib
        def read(path):
            if str(path).endswith(r.original.MANIFEST):return manifest
            if str(path).endswith(r.REGISTRY):return registry
            if str(path).endswith('accounting_rows_receipt_v1.json'):
                return dict(original=dict(bytes=len(decoded),sha256=hashlib.sha256(decoded).hexdigest()))
            return {'ARTIFICIAL_COMPARISON':True}
        with patch.object(r,'proof_metadata',return_value=({}, {'ARTIFICIAL':True},{'ARTIFICIAL':True})),\
             patch.object(r.original,'load_saved',return_value=(data,old.REGISTRY,[])),\
             patch.object(r.original,'read_verified',return_value=compressed),patch.object(r,'read',side_effect=read):
            assembled,actual,permit=r.load_saved(dict(inputs=inputs))
        self.assertEqual(actual,registry);self.assertIn('confirmation_cells',actual)
        self.assertEqual(assembled['usual_rows'],[{}]);self.assertFalse(permit.synthetic)
    def test_fresh_gate_staleness_and_digest(self):
        cfg,g=self.authorization();gate=dict(SOURCE=cfg['SOURCE'],config_sha256=c.digest(cfg),grant_sha256=c.digest(g),
            allowed_cpus=g['allowed_cpus'],monotonic_seconds=100,
            **{x:True for x in ['source_input_verified','CPU_selection_verified','resources_admission_verified',
                               'quota_verified','protected_jobs_unchanged','launcher_wait_ready']})
        r.fresh_binding(cfg,g,gate,now=110)
        with self.assertRaises(ValueError):r.fresh_binding(cfg,g,gate,now=131)
        gate['grant_sha256']='wrong'
        with self.assertRaises(ValueError):r.fresh_binding(cfg,g,gate,now=110)
    def child_case(self,fail):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=dict(SOURCE='a'*40,python=r.PYTHON,output=str(root/'out'),control=str(root/'control'),
                lock=str(root/'lock'),protected_files=[],protected_worktrees=[])
            real_popen=subprocess.Popen;original_wait=runner.OwnedChild.wait;state=dict(first=True)
            def popen(*args,**kwargs):
                return real_popen([r.PYTHON,'-P','-B','-c','import time;time.sleep(30)' if fail else 'pass'],**kwargs)
            def wait(owned,blocking=False):
                if fail and state['first']:
                    state['first']=False;raise RuntimeError('ARTIFICIAL_WAIT_INTERRUPTED')
                return original_wait(owned,blocking)
            def read(path):
                if str(path)=='config':return cfg
                if str(path)=='grant':return {}
                return dict(status='COMPLETE_MANDATORY_STOP')
            with patch.object(r,'read',side_effect=read),patch.object(r,'authorize',return_value=[min(os.sched_getaffinity(0))]),\
                 patch.object(r,'verify',return_value={}),patch.object(runner,'fresh',return_value={}),\
                 patch.object(runner.subprocess,'Popen',side_effect=popen),patch.object(runner.OwnedChild,'wait',wait):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'ARTIFICIAL_WAIT_INTERRUPTED'):runner.launch('config','grant')
                else:runner.launch('config','grant')
            native=json.loads((root/'control/native_exit.json').read_text())
            self.assertTrue(native['reaped']);self.assertIsNone(r.identity(native['pid']))
            if fail:
                stop=json.loads((root/'NC_COST_STOP.json').read_text())
                self.assertEqual(stop['first_stop'],'ARTIFICIAL_WAIT_INTERRUPTED');self.assertTrue(stop['all_children_reaped'])
                self.assertEqual(stop['cleanup_errors'],[])
            else:self.assertTrue((root/'NC_COST_final_gate.json').exists())
    def test_minimal_artificial_child_normal_exit_native_wait(self):self.child_case(False)
    def test_minimal_artificial_child_interrupted_cleanup_native_wait(self):self.child_case(True)


if __name__=='__main__':unittest.main()
