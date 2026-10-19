"""Invented sides, fake archives and mock gates only. No saved science input."""
import ast
import copy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'tests'))
import mechanism_a82_factors as m
import run_mechanism_a82_factors as cli
from test_mechanism_a82 import fixture as side_fixture


def fixture():
    old_bundle,core,archive=side_fixture()
    golden=m.equations.explain(old_bundle['comparison'])  # invented values only
    assert golden['status']=='EXPLANATION_COMPLETE'
    for row in archive:
        row['explanation']['status']='EXPLANATORY_INPUT_MISSING' if row['contrast_id']==m.TARGET else 'EXPLANATION_COMPLETE'
        row['explanation']['ratios']={} if row['contrast_id']==m.TARGET else copy.deepcopy(golden['ratios'])
    archive[-1]['explanation']['sides']['PF']=copy.deepcopy(golden['sides']['PF'])
    delta=dict(schema='TRACK_A_A82_PR_EXPLANATION_DELTA_V1',target_side_id=m.TARGET+'.PR',
        candidate_id=m.IDENTITY['PR_id'],epsilon='0.05',new_PR_side=copy.deepcopy(golden['sides']['PR']),
        inherited_PF_side=copy.deepcopy(golden['sides']['PF']),historical_PR_status='EXPLANATORY_INPUT_MISSING',
        old_archives_modified=False,new_comparison_factors_computed=False,next_stage_authorized=False)
    bundle=dict(schema=m.SCHEMA,kind='ARTIFICIAL',artificial=True,target_ids=[m.TARGET],axes=list(m.AXES),
        factor_names=list(m.FACTORS),identity=copy.deepcopy(m.IDENTITY),side_source=m.SIDE_SOURCE,
        prior_result_commit=m.PRIOR_RESULT,definition_source=m.DEFINITION_SOURCE,
        input_result_digests=dict(delta_sha256='INVENTED',old_core_sha256='INVENTED',old_explanations_sha256='INVENTED'))
    return bundle,delta,core,archive,golden


class FactorTests(unittest.TestCase):
    def setUp(self):self.b,self.d,self.c,self.a,self.g=fixture()
    def run_factors(self):return m.analyze_factors(self.b,self.d,self.c,self.a,m.equations.artificial_permit())
    def test_formula_AST_equals_fixed_existing31_definition(self):
        old=ast.parse(inspect.getsource(m.equations.explain)).body[0]
        block=next(s for s in old.body if isinstance(s,ast.If) and isinstance(s.test,ast.Compare))
        new=ast.parse(inspect.getsource(m.saved_side_factors)).body[0]
        self.assertEqual(ast.dump(ast.Module(body=block.body,type_ignores=[])),
                         ast.dump(ast.Module(body=new.body[2:-1],type_ignores=[])))
    def test_invented_factors_equal_original_definition_exactly(self):
        self.assertEqual(self.run_factors()['ratios'],self.g['ratios'])
    def test_one_factor_pass_no_side_core_or_bundle_calls(self):
        with patch.object(m,'saved_side_factors',wraps=m.saved_side_factors) as factor, \
             patch.object(m.equations,'explain_side',side_effect=AssertionError('side forbidden')), \
             patch.object(m.equations,'normalization',side_effect=AssertionError('normalization forbidden')), \
             patch.object(m.equations,'explain',side_effect=AssertionError('pair forbidden')), \
             patch.object(m.equations,'analyze_bundle',side_effect=AssertionError('bundle forbidden')), \
             patch.object(m.equations.original,'paired_costs',side_effect=AssertionError('core forbidden')):
            r=self.run_factors();self.assertEqual(factor.call_count,1)
            self.assertEqual(r['calls'],m.CALLS)
    def test_inputs_not_mutated(self):
        before=copy.deepcopy((self.b,self.d,self.c,self.a));self.run_factors()
        self.assertEqual((self.b,self.d,self.c,self.a),before)
    def test_permission_rejects_real_kind_before_arithmetic(self):
        self.b.update(kind='SAVED_REAL_VALUES',artificial=False)
        with patch.object(m,'saved_side_factors',side_effect=AssertionError('no formula')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_NUMERICAL_PERMISSION'):self.run_factors()
    def test_artificial_flag_required(self):
        self.b['artificial']=False
        with self.assertRaisesRegex(ValueError,'EXPLICIT_NUMERICAL_PERMISSION'):self.run_factors()
    def test_extra_target_or_axis_rejected(self):
        for key,extra in [('target_ids','OTHER'),('axes','OTHER'),('factor_names','OTHER')]:
            b=copy.deepcopy(self.b);b[key].append(extra)
            with self.assertRaisesRegex(ValueError,'EXACT_ONE_COMPARISON'):
                m.analyze_factors(b,self.d,self.c,self.a,m.equations.artificial_permit())
    def test_wrong_scientific_identity_rejected(self):
        for key in self.b['identity']:
            b=copy.deepcopy(self.b);b['identity'][key]='WRONG'
            with self.assertRaisesRegex(ValueError,'FIXED_IDENTITY_LINEAGE'):
                m.validate_saved_lineage(b,self.d,self.c,self.a)
    def test_wrong_source_rejected(self):
        self.b['side_source']='f'*40
        with self.assertRaisesRegex(ValueError,'FIXED_IDENTITY_LINEAGE'):self.run_factors()
    def test_PF_copy_tampering_rejected(self):
        self.d['inherited_PF_side']['token']='TAMPER'
        with self.assertRaisesRegex(ValueError,'PF_SAVED_COPY_CHANGED'):self.run_factors()
    def test_PR_unvalidated_rejected(self):
        self.d['new_PR_side']['status']='UNRESOLVED'
        with self.assertRaisesRegex(ValueError,'SAVED_SIDE_NOT_COMPLETE'):self.run_factors()
    def test_other31_incomplete_rejected_without_evaluation(self):
        self.a[0]['explanation']['status']='UNRESOLVED'
        with self.assertRaisesRegex(ValueError,'OTHER31_SAVED_COMPLETE'):self.run_factors()
    def test_core_status_required_without_arithmetic(self):
        self.c[0]['saved_G_exact_match']=False
        with self.assertRaisesRegex(ValueError,'SAVED_CORE_COMPLETE'):self.run_factors()
    def test_historical_missing_status_kept(self):
        self.a[-1]['explanation']['sides']['PR']['status']='EXPLANATION_COMPLETE'
        with self.assertRaisesRegex(ValueError,'HISTORICAL_MISSING_RETAINED'):self.run_factors()
    def test_duplicate_archive_join_rejected(self):
        self.c[0]['contrast_id']=self.c[1]['contrast_id']
        with self.assertRaisesRegex(ValueError,'UNCHANGED_32_ARCHIVE_JOIN'):self.run_factors()
    def test_missing_normalization_rejected(self):
        del self.d['new_PR_side']['normalization']['log_B_margin']
        with self.assertRaisesRegex(ValueError,'NORMALIZATION_INPUT_MISSING'):self.run_factors()
    def test_missing_rounding_rejected(self):
        self.d['new_PR_side']['axes']['cosine']['log_kappa']=None
        with self.assertRaisesRegex(ValueError,'SAVED_AXIS_INPUT_MISSING'):self.run_factors()
    def test_changed_integer_route_rejected(self):
        self.d['new_PR_side']['axes']['sine']['integer_status']='LOG_DOMAIN_ONLY'
        with self.assertRaisesRegex(ValueError,'SAVED_AXIS_INPUT_MISSING'):self.run_factors()
    def test_nonfinite_output_rejected(self):
        r=self.run_factors();r['ratios']['cosine']['log_factors']['headroom_squared']=float('nan')
        with self.assertRaisesRegex(ValueError,'NONFINITE_OR_UNREPRESENTABLE'):m.validate_result(r)
    def test_unrepresentable_factor_fail_closed(self):
        self.d['new_PR_side']['normalization']['log_B_upper']=1e9
        with self.assertRaisesRegex(ValueError,'NONFINITE_OR_UNREPRESENTABLE'):self.run_factors()
    def test_output_missing_factor_rejected(self):
        r=self.run_factors();del r['ratios']['sine']['factors']['rounding_factor']
        with self.assertRaisesRegex(ValueError,'OUTPUT_FACTOR_SCHEMA'):m.validate_result(r)


class GateTests(unittest.TestCase):
    def test_invalid_grant_rejects_before_contract_or_input(self):
        with patch.object(cli.io,'read',return_value={}) as read,patch.object(cli,'verify_inputs',side_effect=AssertionError('input forbidden')):
            with self.assertRaisesRegex(ValueError,'NUMERICAL_AUTHORIZATION_REQUIRED'):cli.gate('invalid','absent')
            self.assertEqual(read.call_count,1)
    def test_real_CLI_invalid_grant_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'invalid.json').write_text('{}')
            r=subprocess.run([sys.executable,'-I','-B',str(ROOT/'scripts/run_mechanism_a82_factors.py'),
                '--grant',str(p/'invalid.json'),'--contract',str(p/'absent.json')],capture_output=True,text=True)
            self.assertEqual(r.returncode,1);self.assertIn('NUMERICAL_AUTHORIZATION_REQUIRED',r.stderr)
            self.assertEqual({x.name for x in p.iterdir()},{'invalid.json'})
    def test_old_side_grant_scope_not_reusable(self):
        g=dict(approved=True,approved_by_user=True,runtime_authorization=True,one_shot=True,
            schema='TRACK_A_A82_PR_EXPLANATION_DELTA_V1',scope='TRACK_A_A82_OLD_PR_SIDE_ONLY',kind='SAVED_REAL_VALUES')
        with patch.object(cli.io,'read',return_value=g):
            with self.assertRaisesRegex(ValueError,'FACTOR_ONLY_SCOPE'):cli.gate('fake','absent')
    def test_new_scope_requires_factor_approval(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);a=p/'approval.json';cli.io.write_new(a,dict(approved_by_user=True,factor_computation_authorized=False))
            g=dict(approved=True,approved_by_user=True,runtime_authorization=True,one_shot=True,
                schema=m.SCHEMA,scope=m.SCOPE,kind='SAVED_REAL_VALUES',target_ids=[m.TARGET],axes=list(m.AXES),
                retry=False,next_stage_authorized=False,approval_receipt=str(a),approval_receipt_sha256=cli.io.sha(a))
            gp=p/'grant.json';cli.io.write_new(gp,g)
            with self.assertRaisesRegex(ValueError,'NEW_EXPLICIT_FACTOR_APPROVAL_REQUIRED'):cli.gate(gp,p/'absent')
    def test_retry_scope_rejected(self):
        g=dict(approved=True,approved_by_user=True,runtime_authorization=True,one_shot=True,
            schema=m.SCHEMA,scope=m.SCOPE,kind='SAVED_REAL_VALUES',target_ids=[m.TARGET],axes=list(m.AXES),retry=True,next_stage_authorized=False)
        with patch.object(cli.io,'read',return_value=g):
            with self.assertRaisesRegex(ValueError,'NO_RETRY'):cli.gate('fake','absent')
    def test_wrong_SOURCE_closure_rejected(self):
        with self.assertRaisesRegex(ValueError,'FACTOR_SOURCE_CLOSURE'):cli.verify_source(dict(source_files={}))
    def test_actual_SOURCE_mismatch_rejected(self):
        c=dict(source_commit='f'*40,source_files={name:'0'*64 for name in cli.SOURCE_FILES})
        with self.assertRaisesRegex(ValueError,'SOURCE_BYTES_MISMATCH'):cli.verify_source(c)
    def test_input_binding_hash_tamper_rejected_before_read(self):
        with patch.object(cli.io,'sha',return_value='BAD'),patch.object(cli.io,'read',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'INPUT_BINDING_SHA'):
                cli.verify_inputs(dict(input_binding='INVENTED',input_binding_sha256='GOOD'))
    def test_individual_input_byte_tamper_rejected(self):
        b=dict(schema=m.SCHEMA,target_ids=[m.TARGET],delta_sha256='GOOD',old_core_sha256='GOOD',old_explanations_sha256='GOOD',
            input_files=[dict(path='INVENTED_FILE',sha256='GOOD')])
        c=dict(input_binding='INVENTED_BINDING',input_binding_sha256='GOOD',delta_sha256='GOOD',old_core_sha256='GOOD',old_explanations_sha256='GOOD')
        with patch.object(cli.io,'read',return_value=b),patch.object(cli.io,'sha',side_effect=lambda p:'GOOD' if p=='INVENTED_BINDING' else 'BAD'):
            with self.assertRaisesRegex(ValueError,'INPUT_BYTES_MISMATCH'):cli.verify_inputs(c)
    def test_canonical_delta_tamper_rejected(self):
        b=dict(schema=m.SCHEMA,target_ids=[m.TARGET],input_files=[])
        for name in cli.REFERENCES:b[name]=name;b[name+'_sha256']='GOOD'
        c=dict(input_binding='INVENTED_BINDING',input_binding_sha256='GOOD',delta_sha256='GOOD',old_core_sha256='GOOD',old_explanations_sha256='GOOD')
        with patch.object(cli.io,'read',return_value=b),patch.object(cli.io,'sha',side_effect=lambda p:'BAD' if p=='delta' else 'GOOD'):
            with self.assertRaisesRegex(ValueError,'CANONICAL_INPUT_SHA'):cli.verify_inputs(c)
    def test_runtime_profile_tamper_rejected(self):
        with patch.object(cli.io,'read',return_value={}),patch.object(cli.io,'sha',return_value='BAD'):
            with self.assertRaisesRegex(ValueError,'RUNTIME_PROFILE_HASH'):
                cli.io.verify_runtime(dict(runtime_profile='INVENTED',runtime_profile_sha256='GOOD'))
    def test_contract_tamper_rejected_before_source_or_input(self):
        g=dict(approved=True,approved_by_user=True,runtime_authorization=True,one_shot=True,schema=m.SCHEMA,
            scope=m.SCOPE,kind='SAVED_REAL_VALUES',target_ids=[m.TARGET],axes=list(m.AXES),retry=False,
            next_stage_authorized=False,approval_receipt='INVENTED_RECEIPT',approval_receipt_sha256='GOOD',contract_sha256='GOOD')
        a=dict(approved_by_user=True,factor_computation_authorized=True,scope=m.SCOPE,target_ids=[m.TARGET],axes=list(m.AXES),retry=False,contract_sha256='GOOD')
        with patch.object(cli.io,'read',side_effect=[g,a]),patch.object(cli.io,'sha',side_effect=['GOOD','BAD']),patch.object(cli,'verify_source',side_effect=AssertionError('source forbidden')):
            with self.assertRaisesRegex(ValueError,'CONTRACT_SHA'):cli.gate('INVENTED_GRANT','INVENTED_CONTRACT')
    def test_post_marker_or_ledger_tamper_rejected(self):
        with patch.object(cli,'gate',return_value=None),patch.object(cli.io,'sha',return_value='BAD'):
            with self.assertRaisesRegex(ValueError,'POST_CONTROL_SHA_MISMATCH'):
                cli.post_gate('INVENTED_GRANT','INVENTED_CONTRACT',{'INVENTED_MARKER':'GOOD'})
    def test_protected_output_and_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            with self.assertRaisesRegex(ValueError,'PROTECTED_PATH_WRITE_REFUSED'):cli.io.prepare_output(p/'run',p,[p])
            cli.io.prepare_output(p/'run',p,[])
            with self.assertRaisesRegex(ValueError,'ONE_SHOT_OUTPUT_ALREADY_EXISTS'):cli.io.prepare_output(p/'run',p,[])
    def test_symlink_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'real').mkdir();(p/'link').symlink_to(p/'real',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'SYMLINK_OUTPUT_PATH'):cli.io.prepare_output(p/'link/run',p/'link',[])


class NativeArtificialTests(unittest.TestCase):
    def native_case(self,broken=False):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);b,d,c,a,golden=fixture()
            if broken:d['new_PR_side']['axes']['cosine']['headroom']=0
            for name,data in [('delta',d),('old_core',c),('old_explanations',a)]:cli.io.write_new(p/(name+'.json'),data)
            digests={name+'_sha256':cli.io.sha(p/(name+'.json')) for name in ('delta','old_core','old_explanations')}
            b['input_result_digests']=digests;cli.io.write_new(p/'payload.json',b)
            binding=digests|{name:str(p/(name+'.json')) for name in ('delta','old_core','old_explanations','payload')}
            binding.update(input_files=[],protected_paths=[])
            cli.io.write_new(p/'approval.json',{'ARTIFICIAL_MOCK_ONLY':True})
            cli.io.write_new(p/'grant.json',{'approved':False,'ARTIFICIAL_MOCK_ONLY':True})
            cli.io.write_new(p/'contract.json',{'ARTIFICIAL_MOCK_ONLY':True})
            grant=dict(approval_receipt=str(p/'approval.json'))
            contract=dict(output_root=str(p/'run'),allowed_run_parent=str(p),input_binding=str(p/'binding.json'),runtime_profile=str(p/'runtime.json'),source_commit='INVENTED')
            cli.io.write_new(p/'test_gate.json',dict(grant=grant,contract=contract,binding=binding))
            wrapper=p/'mock_worker.py'
            wrapper.write_text('import sys,json\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(ROOT/'scripts'))+')\nimport run_mechanism_a82_factors as r\np=Path('+repr(str(p))+')\nx=json.loads((p/"test_gate.json").read_text())\nr.gate=lambda *args:(x["grant"],x["contract"],x["binding"])\nr.adapter.equations._authorized_production_permit=r.adapter.equations.artificial_permit\nr.worker(p/"grant.json",p/"contract.json",p/"run")\n')
            native_popen=subprocess.Popen
            def artificial_popen(argv,**kwargs):return native_popen([sys.executable,'-I','-B',str(wrapper)],**kwargs)
            with patch.object(cli,'gate',return_value=(grant,contract,binding)),patch.object(cli.subprocess,'Popen',side_effect=artificial_popen) as launch:
                if broken:
                    with self.assertRaisesRegex(ValueError,'NATIVE_WORKER_FAILURE'):cli.launch(p/'grant.json',p/'contract.json')
                    self.assertTrue((p/'run/STOP.json').is_file());self.assertFalse((p/'run/terminal.json').exists())
                else:
                    cli.launch(p/'grant.json',p/'contract.json')
                    r=cli.io.read(p/'run/A82_factor_delta.json');self.assertEqual(r['ratios'],golden['ratios'])
                    self.assertEqual(cli.io.read(p/'run/terminal.json')['status'],'COMPLETE_MANDATORY_STOP')
                    self.assertEqual(cli.io.read(p/'run/ledger.json')['calls'],m.CALLS)
                self.assertEqual(launch.call_count,1)
            wait=cli.io.read(p/'run/native_wait.json');self.assertTrue(wait['reaped'])
            self.assertEqual(wait['native_exit_code'],1 if broken else 0)
            with patch.object(cli,'gate',return_value=(grant,contract,binding)):
                with self.assertRaisesRegex(ValueError,'ONE_SHOT_OUTPUT_ALREADY_EXISTS'):cli.launch(p/'grant.json',p/'contract.json')
    def test_native_artificial_pass_and_exclusive_consumption(self):self.native_case()
    def test_native_artificial_failure_STOP_no_retry(self):self.native_case(broken=True)


if __name__=='__main__':unittest.main()
