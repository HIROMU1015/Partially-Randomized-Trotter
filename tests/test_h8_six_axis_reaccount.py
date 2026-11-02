"""Artificial numbers only. No saved H8 accounting or scientific inputs read."""
from copy import deepcopy
from fractions import Fraction as F
import errno
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import h8_six_axis_reaccount_contract as c
import run_h8_six_axis_reaccount as runner
import witness_h8_six_axis_reaccount as witness
n = c.n
TEMP_ROOT = Path('/home/AbeHiromu/projects/h8-six-axis-reaccount/20261017-01')


def fixture():
    old = [dict(id='old_%02d' % i) for i in range(32)]
    extra = [dict(id=cid) for cid in n.REPLACEMENTS]
    extra += [dict(id='extra_%03d' % i) for i in range(149)]
    prs = ['pr_%d' % i for i in range(6)]
    registry = dict(old=old, additional=extra, confirmation_ids=prs)
    measured = {}
    for cid in n.REPLACEMENTS:
        measured[cid] = {a: dict(measured_RZ=5 if a=='cosine' else 7, SOURCE='a'*40,
            compiler_hash=n.COMPILER, profile_sha256=n.PROFILE, result_sha256='b'*64,
            model_check=dict(exact_prediction='1000', PASS=True)) for a in n.AXES}
    rows, accounting = [], []
    for e in n.EPSILONS:
        for f in n.FACTORS:
            for cid in [r['id'] for r in old+extra] + prs:
                row = dict(cell_id=cid, epsilon=e, factor=f, eligible=True, G_exact='400',
                           cost_status='SAVED_MODEL', untouched_bias='ARTIFICIAL')
                if cid in prs:
                    row.update(G_exact=str(150+prs.index(cid)), saved_paired_SE=1.25,
                               paired_samples=32, cost_status='PR_SAVED')
                else:
                    unit = 1000 if cid in measured else 100
                    row['axis_costs'] = {a: dict(RZ=str(unit), measured=False, model_sha256=n.MODEL) for a in n.AXES}
                    row['G_exact'] = str(4*unit)
                rows.append(row)
                accounting.append(dict(cell_id=cid, epsilon=e, factor=f,
                    accounting=dict(cost_acquisition_eligible=True, eligibility_under_declared_allowance='ELIGIBLE',
                        N_total=4, axes=dict(real=dict(shots=1), imag=dict(shots=3)))))
    cmps = []
    for e in n.EPSILONS:
        for f in n.FACTORS:
            candidates = {r['cell_id']: r for r in rows if r['epsilon']==e and r['factor']==f}
            scopes = {}
            for scope, cells in [('old32',old),('strengthened184',old+extra)]:
                det = n.minimum([candidates[x['id']] for x in cells])
                pr = dict(status='CONDITIONAL_POINT_MINIMUM', best=candidates['pr_0'], exact_ties=['pr_0'])
                scopes[scope] = dict(deterministic=det, confirmed_PR=pr,
                                    PR_reduction_percent_exact=n.reduction(pr,det))
            cmps.append(dict(epsilon=e, factor=f, scopes=scopes))
    mode = dict(rows=rows, accounting=accounting, comparisons=cmps)
    return dict(kind='ARTIFICIAL', modes={m: deepcopy(mode) for m in n.MODES}), registry, measured


class ArtificialAccounting(unittest.TestCase):
    def setUp(self): self.data, self.registry, self.measured = fixture()
    def run_fixture(self):
        return n.reaccount(self.data,self.registry,self.measured,n.Permit(synthetic=True))
    def test_both_modes_all_conditions_and_exact_cost(self):
        r=self.run_fixture()
        self.assertEqual((r['saved_rows'],r['comparison_count']), (3800,40))
        for change in r['measured_candidate_changes']:
            self.assertEqual(change['G_before'],'4000')
            self.assertEqual(change['G_after'],'26')  # 1*5 + 3*7, not swapped.
            self.assertEqual(change['saved_shots'],dict(real=1,imag=3))
    def test_old32_pr_se_shots_and_input_unchanged(self):
        before=deepcopy(self.data);r=self.run_fixture();self.assertEqual(before,self.data)
        for item in r['accounting_rows']:
            if item['saved']['cell_id'] not in n.REPLACEMENTS:
                self.assertEqual(item['saved'],item['updated'])
        for row in r['comparisons']:
            self.assertEqual(row['saved_PR']['best']['saved_paired_SE'],1.25)
            if row['scope']=='old32':self.assertEqual(row['before'],row['after'])
    def test_exact_tie_lexical(self):
        best=n.minimum([dict(cell_id='z',eligible=True,G_exact='2/3'),dict(cell_id='a',eligible=True,G_exact='2/3')])
        self.assertEqual(best['best']['cell_id'],'a');self.assertEqual(best['exact_ties'],['a','z'])
    def test_all184_not_only_measured_candidates(self):
        # An unmodified extra candidate can be the best: the selector must not
        # limit the domain to the three newly measured candidates.
        rows=[dict(cell_id=x['id'],eligible=True,G_exact='100') for x in self.registry['old']+self.registry['additional']]
        rows[-1]['G_exact']='25/2'
        rows[32]['G_exact']='26'
        self.assertEqual(n.minimum(rows)['best']['cell_id'],rows[-1]['cell_id'])
    def test_missing_is_not_zero(self):
        with self.assertRaisesRegex(ValueError,'ELIGIBLE_COST_MISSING'):
            n.minimum([dict(cell_id='a',eligible=True,G_exact=None)])
    def test_unresolved_prevents_domain_claim(self):
        r=n.minimum([dict(cell_id='a',eligible=None,G_exact=None),dict(cell_id='b',eligible=True,G_exact='1')])
        self.assertEqual(r['status'],'UNRESOLVED_FULL_DOMAIN');self.assertIsNone(r['best'])
    def test_ineligible_empty_is_distinct(self):
        self.assertEqual(n.minimum([dict(cell_id='a',eligible=False,G_exact=None)])['status'],'NO_ELIGIBLE_CANDIDATE')
    def test_no_real_permit(self):
        with self.assertRaisesRegex(ValueError,'NEW_ONE_SHOT_GRANT'):n.Permit()
        self.data['kind']='SAVED_H8'
        with self.assertRaisesRegex(ValueError,'DATA_SCOPE_PERMIT'):self.run_fixture()
    def test_candidate_formula_axis_exact_joins(self):
        for old,new in [('B0I_p15_q4_4th','B0I_p15_q4_2nd'),('B0_p15_q4_2nd','B0I_p15_q4_2nd')]:
            measured=deepcopy(self.measured);measured[new]=measured.pop(old)
            with self.assertRaisesRegex(ValueError,'EXACT_SIX_OUTSIDE_OLD32'):
                n.reaccount(self.data,self.registry,measured,n.Permit(synthetic=True))
        del self.measured['B0I_p15_q4_4th']['sine']
        with self.assertRaises(KeyError):self.run_fixture()
    def test_saved_factor_string_is_identifier_normalization(self):
        self.assertEqual(n.row_key(dict(cell_id='a',epsilon=.05,factor='10.0')),('a',.05,10.))
    def test_duplicate_missing_and_extra_saved_rows(self):
        self.data['modes'][n.MODES[0]]['rows'].append(deepcopy(self.data['modes'][n.MODES[0]]['rows'][0]))
        with self.assertRaisesRegex(ValueError,'DUPLICATE_KEY'):self.run_fixture()
    def test_wrong_saved_shots_closure_stop(self):
        self.data['modes'][n.MODES[0]]['accounting'][0]['accounting']['axes']['real']['shots']=2
        with self.assertRaisesRegex(ValueError,'SAVED_POSITIVE_INTEGER_SHOTS'):self.run_fixture()
    def test_pr_se_mismatch_stop(self):
        self.data['modes'][n.MODES[0]]['comparisons'][0]['scopes']['old32']['confirmed_PR']['best']=deepcopy(
            self.data['modes'][n.MODES[0]]['comparisons'][0]['scopes']['old32']['confirmed_PR']['best'])
        self.data['modes'][n.MODES[0]]['comparisons'][0]['scopes']['old32']['confirmed_PR']['best']['saved_paired_SE']=9
        with self.assertRaisesRegex(ValueError,'SAVED_PR_SE_BINDING'):self.run_fixture()
    def test_fixed_model_mismatch_stop(self):
        r=next(r for r in self.data['modes'][n.MODES[0]]['rows'] if r['cell_id'] in self.measured)
        r['axis_costs']['cosine']['model_sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'UNCHANGED_FIXED_MODEL_REQUIRED'):self.run_fixture()
    def test_overlapping_measured_mismatch_stop(self):
        r=next(r for r in self.data['modes'][n.MODES[0]]['rows'] if r['cell_id'] in self.measured)
        r['axis_costs']['cosine']['measured']=True
        with self.assertRaisesRegex(ValueError,'OVERLAPPING_MEASUREMENT_CONFLICT'):self.run_fixture()
    def test_old32_membership_mismatch_stop(self):
        self.registry['old'][0]['id']='B0I_p15_q4_4th'
        self.registry['additional'][0]['id']='old_00'
        with self.assertRaisesRegex(ValueError,'EXACT_SIX_OUTSIDE_OLD32'):self.run_fixture()
    def test_negative_reduction_is_preserved(self):
        r=self.run_fixture();self.assertLess(F(r['comparisons'][1]['reduction_after_exact']),0)


class ContractAndNative(unittest.TestCase):
    def test_outer_signal_requests_cooperative_stop_before_reap(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            p=Path(td)
            config=dict(SOURCE='a'*40,_config_path=str(p/'config'),_grant_path=str(p/'grant'),
                _witness_receipt_path=str(p/'native'),_witness_lock=str(p/'marker'),
                _stop_request_path=str(p/'stop'))
            grant={'artificial':True};child=mock.Mock(returncode=None)
            def wait_owned(ch,path,**kw):
                signal.raise_signal(signal.SIGTERM)
                kw['poll']()
                self.assertEqual(c.read(p/'stop')['first_stop'],'OUTER_SIGNAL_'+str(signal.SIGTERM))
                self.assertIsNone(ch.returncode)
                ch.returncode=0
                return dict(native_exit_code=0,reaped=True,cleanup_errors=[])
            with mock.patch.object(c,'authorize'),mock.patch.object(c,'verify'),mock.patch.object(c,'paths'),\
                 mock.patch.object(witness.subprocess,'Popen',return_value=child),\
                 mock.patch.object(runner,'wait_owned',side_effect=wait_owned):
                self.assertEqual(witness.witness(config,grant),0)
            self.assertTrue((p/'marker').exists())
    def test_six_receipt_source_profile_compiler_mismatch_stop(self):
        def evidence():
            rows=[]; values={}; by={}
            status='PASS_ONLY_SIX_REGISTERED_AXES_MANDATORY_STOP'
            for i,(cid,axis) in enumerate(((cid,a) for cid in n.REPLACEMENTS for a in n.AXES),1):
                task=dict(case_id=cid,axis=axis,measured_RZ=n.REPLACEMENTS[cid])
                row=dict(task,SOURCE=n.OLD_SIX_SOURCE if i==1 else n.SIX_SOURCE,
                    compiler_hash=n.COMPILER,profile_sha256=n.PROFILE,model_sha256=n.MODEL,
                    model_check=dict(PASS=True,exact_relative_error='0'),max_control_error=0.,
                    fingerprint_before='artificial',fingerprint_after='artificial',
                    native_exit_code=0,reaped=True,worker_identity=[i,1,1,1],result_sha256='d'*64)
                worker=dict(exit_code=0,reaped=True,identity=row['worker_identity'],result_sha256='d'*64)
                values['six:native_%02d'%i]=dict(all_reaped=True,status='PASS',cleanup_errors=[],workers=[worker])
                values['six:task_%02d'%i]=task
                rows.append(row)
            values.update({'six:six_axis_results.json':dict(SOURCE=n.SIX_SOURCE,selected_axes=rows,status=status,
                full_runtime_final_gate_already_PASS=True,compiler_hash=n.COMPILER,profile_sha256=n.PROFILE,
                model_sha256=n.MODEL,model_refit_performed=False,
                axis_native_proofs=[dict(receipt_sha256='c'*64,worker_result_sha256='d'*64) for _ in rows]),
                'six:raw_result':dict(SOURCE=n.SIX_SOURCE,rows=rows,status=status),
                'six:raw_terminal':dict(status='COMPLETE_MANDATORY_STOP'),
                'six:result_audit_manifest.json':dict(SOURCE=n.SIX_SOURCE,result_sha256='c'*64,
                    terminal_sha256='c'*64,final_gate_sha256='c'*64,launcher_native_exit_sha256='c'*64,
                    public_projection_sha256='c'*64)})
            for key in list(values)+['six:raw_final_gate','six:raw_launcher_exit']:
                by[key]=dict(path=key,sha256='d'*64 if ':task_' in key else 'c'*64)
            return values,by
        for field,error in [('SOURCE','SIX_RESULT_SOURCE_ROWS'),('profile_sha256','SIX_COMPILER_PROFILE_MODEL'),
                            ('compiler_hash','SIX_COMPILER_PROFILE_MODEL'),('receipt','SIX_ORIGINAL_RECEIPT_SHA')]:
            with self.subTest(field=field):
                values,by=evidence()
                if field=='receipt':by['six:raw_result']['sha256']='f'*64
                else:values['six:six_axis_results.json'][field]='f'*64
                with mock.patch.object(c,'verify_inputs',return_value=by),mock.patch.object(c,'read',side_effect=lambda p:values[p]):
                    with self.assertRaisesRegex(ValueError,error):c.proof_metadata({})

    def test_invalid_and_old_grants_rejected(self):
        for g in ({},{'approved':False},{'approved':True,'runtime_authorization':True,'scope':'OLD_USED_GRANT'}):
            with self.assertRaisesRegex(ValueError,'NEW_PUBLISHED_SEALED_ONE_SHOT_GRANT'):c.authorize({},g)
    def test_duplicate_nonfinite_json_reject(self):
        for data in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(ValueError):c.loads(data)
    def test_private_home_paths_and_one_shot_reject(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            p=Path(td);cfg=dict(output=str(p/'output'),control=str(p/'control'),lock=str(p/'lock'),inputs=[])
            c.paths(cfg);(p/'lock').write_text('spent')
            with self.assertRaisesRegex(ValueError,'ONE_SHOT_ALREADY_SPENT'):c.paths(cfg)
            with self.assertRaisesRegex(ValueError,'CANONICAL_HOME_PATH'):c.home('/tmp/bad')
    def test_input_receipt_byte_source_changed_reject(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            p=Path(td)/'input.json';p.write_text('{}')
            spec=dict(role='artificial',bytes=2,sha256=c.sha(p))
            cfg=dict(inputs=[dict(spec,path=str(p))])
            with mock.patch.object(c,'read',return_value=dict(files=[spec])):
                c.verify_inputs(cfg);p.write_text('{ }')
                with self.assertRaisesRegex(ValueError,'INPUT_BYTE_SHA'):c.verify_inputs(cfg)
    def test_native_success_and_pipes_closed(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            child=subprocess.Popen([c.PYTHON,'-P','-B','-c','pass'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            r=runner.wait_owned(child,Path(td)/'native.json')
            self.assertEqual(r['native_exit_code'],0);self.assertTrue(r['reaped'])
            self.assertTrue(child.stdout.closed and child.stderr.closed)
            with self.assertRaisesRegex(ValueError,'ALREADY_REAPED_CHILD'):
                runner.wait_owned(child,Path(td)/'second.json')
    def test_native_failure_exit_preserved(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            child=subprocess.Popen([c.PYTHON,'-P','-B','-c','raise SystemExit(7)'])
            r=runner.wait_owned(child,Path(td)/'native.json')
            self.assertEqual(r['native_exit_code'],7);self.assertTrue(r['reaped'])
    def test_foreign_ownership_rejected_without_signal(self):
        child=mock.Mock(pid=os.getpid(),returncode=None)
        with mock.patch.object(signal,'pidfd_send_signal') as send:
            with self.assertRaisesRegex(ValueError,'DIRECT_CHILD_OWNERSHIP'):
                runner.wait_owned(child,TEMP_ROOT/'never-created.json')
            send.assert_not_called()
    def test_source_head_mismatch_rejected(self):
        cfg=dict(SOURCE='a'*40,plan_sha256=c.digest(c.plan()))
        with mock.patch.object(c.subprocess,'check_output',return_value=b'b'*40):
            with self.assertRaisesRegex(ValueError,'SOURCE_HEAD'):c.verify(cfg)
    def test_interrupt_cleanup_reap_first_reason(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            child=subprocess.Popen([c.PYTHON,'-P','-B','-c','import time;time.sleep(10)'])
            def interrupt():raise KeyboardInterrupt('FIRST_STOP')
            r=runner.wait_owned(child,Path(td)/'native.json',interrupt=interrupt)
            self.assertIn('FIRST_STOP',r['first_stop']);self.assertTrue(r['reaped'])
            with self.assertRaises(ChildProcessError):os.waitpid(child.pid,os.WNOHANG)
    def test_esrch_is_exit_race_reap_still_required(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            child=subprocess.Popen([c.PYTHON,'-P','-B','-c','import time;time.sleep(.03)'])
            def interrupt():raise KeyboardInterrupt('FIRST_STOP')
            with mock.patch.object(signal,'pidfd_send_signal',side_effect=ProcessLookupError(errno.ESRCH,'exited')):
                r=runner.wait_owned(child,Path(td)/'native.json',interrupt=interrupt)
            self.assertEqual(r['cleanup_errors'],[]);self.assertTrue(r['reaped'])
    def test_non_esrch_signal_error_is_not_success(self):
        original=signal.pidfd_send_signal
        def send(fd,sig):
            if sig==signal.SIGTERM:raise PermissionError(errno.EPERM,'fixture')
            return original(fd,sig)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as td:
            child=subprocess.Popen([c.PYTHON,'-P','-B','-c','import time;time.sleep(10)'])
            def interrupt():raise KeyboardInterrupt('FIRST_STOP')
            with mock.patch.object(signal,'pidfd_send_signal',side_effect=send):
                r=runner.wait_owned(child,Path(td)/'native.json',interrupt=interrupt)
            self.assertTrue(r['cleanup_errors']);self.assertTrue(r['reaped'])


if __name__=='__main__': unittest.main()
