"""No scientific SOURCE execution. Fixtures are invented control-plane records."""
import builtins
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h2_common_production_contract as c
import h2_common_production_backend as backend_module
import h2_common_production_diagnostics as diagnostics_module


class FakeBackend:
    """Only records budget entries. Never uses matrices, SOURCE kernels or RNG."""
    def __init__(self,s):self.s=s
    def setup(self):pass
    def prepare(self,prefix):pass
    def diagnose(self,can):
        self.s.take('path_signal',2);self.s.take('reference_rebind');return {'toy':True}
    def native(self,can,diagnostic):
        pcs=can.q*(1 if can.formula=='2nd' else 3)
        for k,n in dict(compose=1,lowering=pcs,wrapper=2,compile=2,transpile=2,simulate=6).items():self.s.take(k,n)
        if can.method=='PR':
            self.s.take('request_factory',pcs);self.s.take('rte_event',32*can.q);self.s.take('occupation_trajectory')
        return [{'synthetic':True}]
    def reuse(self,can,w):return []
    def allowance(self,*a):return {'synthetic':True}
    def account(self,can,*a):return {'candidate':can.id}
    def rank(self,rows):return {'status':'SYNTHETIC_ENGINE_TRACE','science_GO':False}


class ProductionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h2-production-synthetic-')
        self.root=Path(self.tmp.name)
        bind=dict(c.BINDING)  # identity strings only; all payloads/numbers invented
        ref=self.root/'reference.json'
        ref.write_text(json.dumps({'old_reference':{'binding':bind,'T':.8,
            'reference':{'real':.125,'imag':.25},'error':1e-6}}))
        snap=self.root/'dummy.bin';snap.write_bytes(b'ARTIFICIAL_NOT_HAMILTONIAN')
        receipt=self.root/'receipt.json';receipt.write_text('{}')
        self.cfg=dict(schema='h2_common_production_contract_v1',series=c.SERIES,
            branch='DIAGNOSTICS',source_commit='b'*40,physical=copy.deepcopy(c.policy.PHYSICAL),
            compiler=copy.deepcopy(c.policy.COMPILER),native_policy=c.policy.NATIVE_POLICY,
            allowance_policy=c.policy.ALLOWANCE_POLICY,
            reference_policy='REUSE_OLD_FP_SCALAR_AND_COMBINED_EMPIRICAL_DIAGNOSTIC',
            candidates=[asdict(x) for x in c.policy.CANDIDATES],budget=c.counts('DIAGNOSTICS'),
            files=[dict(path=str(p),sha256=c.sha(p),role='SYNTHETIC_ONLY') for p in (ref,snap,receipt)],
            source_files=[],acquisition_root=str(self.root/'acquisition'),provider_root=str(self.root/'provider'),
            inputs=dict(snapshot=str(snap),snapshot_receipt=str(receipt),old_reference=str(ref)),binding=bind,
            runtime=dict(python_sha256=c.sha(Path(sys.executable).resolve()),qiskit='1.3.0',numpy='1.26.4',loaded_closure_complete=True),
            run_root=str(self.root/'run'),reuse_witnesses=[],retry=False,mandatory_STOP=True)
        self.source_patch=patch.object(c,'source_check');self.source_patch.start()
        self.reference_patch=patch.object(c,'reference_bytes_check');self.reference_patch.start()

    def tearDown(self):self.source_patch.stop();self.reference_patch.stop();self.tmp.cleanup()

    def approval_grant(self,cfg=None):
        cfg=cfg or self.cfg
        a=dict(issuer='USER_EXPLICIT',execution_authorized=True,contract_sha256=c.policy.digest(cfg),
               policies_adopted=[cfg[k] for k in ('native_policy','allowance_policy','reference_policy')])
        g=dict(id='SYNTHETIC_TEST_GRANT',contract_sha256=c.policy.digest(cfg),approval_sha256=c.policy.digest(a),
               run_root=cfg['run_root'],budget=cfg['budget'],state='UNUSED',retry=False)
        return a,g

    def test_preflight_pass_is_not_go_and_has_no_output(self):
        self.assertFalse(c.preflight(self.cfg)['science_GO']);self.assertFalse(Path(self.cfg['run_root']).exists())

    def test_exact_counts_all_branches(self):
        self.assertEqual(c.counts('DIAGNOSTICS')['path_signal'],8)
        x=c.counts('COMPARE_REUSE_PR');self.assertEqual(x['compile'],4);self.assertEqual(x['simulate'],12)
        self.assertEqual(x['request_factory'],0);self.assertEqual(x['lowering'],10)
        x=c.counts('PR_TWO_NATIVE_CONTROLS');self.assertEqual(x['request_factory'],10);self.assertEqual(x['rte_event'],192)
        self.assertEqual(x['diagnose'],0)

    def test_unknown_branch(self):
        with self.assertRaises(c.Stop):c.counts('FULL_GRID')

    def test_unapproved_execute_cannot_construct_backend(self):
        with patch.object(backend_module,'Backend') as factory:
            a,g=self.approval_grant();a['execution_authorized']=False
            with self.assertRaises(c.Stop):c.execute(self.cfg,a,g)
            factory.assert_not_called()

    def test_adoption_required(self):
        a,g=self.approval_grant();a['policies_adopted']=[]
        with self.assertRaises(c.Stop):c.authorize(self.cfg,a,g)

    def test_invalid_used_or_scope_grant(self):
        for key,value in [('state','USED'),('run_root','/different'),('contract_sha256','e'*64),('retry',True),('budget',{})]:
            a,g=self.approval_grant();g[key]=value
            with self.subTest(key=key):
                with self.assertRaises(c.Stop):c.authorize(self.cfg,a,g)

    def test_approval_digest_bound(self):
        a,g=self.approval_grant();g['approval_sha256']='e'*64
        with self.assertRaises(c.Stop):c.authorize(self.cfg,a,g)

    def test_null_or_wrong_budget(self):
        for budget in [None,{},dict(c.counts('DIAGNOSTICS'),compile=1),dict(c.counts('DIAGNOSTICS'),setup=True)]:
            cfg=copy.deepcopy(self.cfg);cfg['budget']=budget
            with self.assertRaises(c.Stop):c.preflight(cfg)

    def test_fixed_candidates(self):
        for key,value in [('id','H4'),('q',8),('prefix',0)]:
            cfg=copy.deepcopy(self.cfg);cfg['candidates'][0][key]=value
            with self.assertRaises(c.Stop):c.preflight(cfg)

    def test_source_and_runtime_contract_rejections(self):
        for key,value in [('source_commit',None),('physical',{}),('compiler',{}),('reference_policy','NEW_REFERENCE'),('mandatory_STOP',False),('retry',True)]:
            cfg=copy.deepcopy(self.cfg);cfg[key]=value
            with self.assertRaises(c.Stop):c.preflight(cfg)
        for field,value in [('python_sha256','e'*64),('loaded_closure_complete',False),('qiskit','2.0')]:
            cfg=copy.deepcopy(self.cfg);cfg['runtime'][field]=value
            with self.assertRaises(c.Stop):c.preflight(cfg)

    def test_file_mutation_and_duplicate_seal(self):
        cfg=copy.deepcopy(self.cfg);Path(cfg['inputs']['snapshot']).write_bytes(b'changed')
        with self.assertRaises(c.Stop):c.preflight(cfg)
        cfg=copy.deepcopy(self.cfg);cfg['files'].append(cfg['files'][0])
        with self.assertRaises(c.Stop):c.preflight(cfg)

    def test_reference_binding_and_uncertainty(self):
        original=Path(self.cfg['inputs']['old_reference']).read_text()
        for mutation in [lambda r:r['binding'].update(state='e'*64),lambda r:r.update(error=None),lambda r:r.update(error=float('nan'))]:
            cfg=copy.deepcopy(self.cfg);p=Path(cfg['inputs']['old_reference']);r=json.loads(original);mutation(r['old_reference'])
            p.write_text(json.dumps(r));cfg['files'][0]['sha256']=c.sha(p)
            with self.assertRaises((c.Stop,TypeError)):c.preflight(cfg)

    def test_compare_needs_two_witnesses(self):
        cfg=copy.deepcopy(self.cfg);cfg.update(branch='COMPARE_REUSE_PR',budget=c.counts('COMPARE_REUSE_PR'))
        with self.assertRaises(c.Stop):c.preflight(cfg)

    def test_three_mock_traces_use_exact_budget(self):
        for branch in c.BRANCHES:
            cfg=copy.deepcopy(self.cfg);cfg.update(branch=branch,budget=c.counts(branch),run_root=str(self.root/branch))
            if branch=='COMPARE_REUSE_PR':
                cfg['reuse_witnesses']=[cfg['inputs']['old_reference']]*2
            if branch=='PR_TWO_NATIVE_CONTROLS':
                p=self.root/'seeds.json';p.write_text(json.dumps({x.id:{str(o)+':'+str(j):7 for o in range(x.q) for j in range(1 if x.formula=='2nd' else 3)} for x in c.policy.CANDIDATES[2:]}))
                cfg['inputs']['control_seeds']=str(p);cfg['files'].append(dict(path=str(p),sha256=c.sha(p),role='synthetic_seeds'))
            a,g=self.approval_grant(cfg)
            with patch.object(c.Session,'loaded_closure'),patch.object(c,'validate_reuse_preflight'):
                c.execute(cfg,a,g,backend_factory=FakeBackend)
            result=c.read(Path(cfg['run_root'])/'completion.json')
            self.assertEqual(result['used'],c.counts(branch));self.assertTrue(result['mandatory_STOP'])

    def test_one_shot_reuse_refused(self):
        a,g=self.approval_grant();s=c.Session(self.cfg,a,g)
        with self.assertRaises(c.Stop):c.Session(self.cfg,a,g)
        self.assertTrue((s.root/'CONSUMED.json').exists())

    def test_reservation_is_nonrefundable_on_failure(self):
        a,g=self.approval_grant();s=c.Session(self.cfg,a,g)
        def fail():raise RuntimeError('synthetic_failure')
        with self.assertRaises(RuntimeError):s.call('setup',fail)
        self.assertEqual(s.used['setup'],1)
        self.assertTrue(c.read(s.root/'ledger/000001.json')['nonrefundable'])
        with self.assertRaises(c.Stop):s.call('setup',lambda:None)

    def test_marker_mutation_stop(self):
        a,g=self.approval_grant();s=c.Session(self.cfg,a,g);(s.root/'CONSUMED.json').write_text('{}')
        with self.assertRaises(c.Stop):s.verify()

    def test_partial_failure_has_stop_no_completion(self):
        class Failing(FakeBackend):
            def diagnose(self,can):raise RuntimeError('synthetic_partial')
        a,g=self.approval_grant()
        with self.assertRaises(RuntimeError):c.execute(self.cfg,a,g,backend_factory=Failing)
        run=Path(self.cfg['run_root']);self.assertEqual(c.read(run/'STOP.json')['status'],'STOP_PARTIAL_NOT_PASS')
        self.assertFalse((run/'completion.json').exists())

    def test_incomplete_budget_cannot_pass(self):
        class Incomplete(FakeBackend):
            def diagnose(self,can):return {}
        a,g=self.approval_grant()
        with patch.object(c.Session,'loaded_closure'):
            with self.assertRaises(c.Stop):c.execute(self.cfg,a,g,backend_factory=Incomplete)
        self.assertFalse((Path(self.cfg['run_root'])/'completion.json').exists())

    def test_backend_rejects_no_session_before_import(self):
        with self.assertRaises(c.Stop):backend_module.Backend(None)

    def test_import_guard_denies_unsealed_module_before_loader(self):
        p=self.root/'synthetic_unsealed.py';p.write_text('raise RuntimeError("MUST_NOT_EXECUTE")')
        guard=c.ImportSeal(self.cfg)
        with self.assertRaisesRegex(c.Stop,'IMPORT_NOT_SEALED'):guard.find_spec('synthetic_unsealed',[str(self.root)])

    def test_no_science_imported_in_tests(self):
        self.assertFalse(any(k.split('.')[0] in {'numpy','scipy','qiskit','trotterlib','trottertracks'} for k in sys.modules))

    def test_cli_execute_requires_external_files_before_contract_read(self):
        r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_h2_common_production.py'),'--contract','/absent','--execute'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('APPROVAL_AND_GRANT_REQUIRED',r.stderr)
        self.assertNotIn('No such file',r.stderr)

    def test_old_mock_cli_still_refuses(self):
        r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/prepare_h2_common_native.py'),'--plan','/absent','--execute'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('SCIENCE_DISPATCH_UNBOUND',r.stderr)

    def test_production_module_imports_are_lazy(self):
        self.assertNotIn('numpy',vars(backend_module));self.assertNotIn('scipy',vars(diagnostics_module))

    def test_unknown_contract_fields(self):
        cfg=copy.deepcopy(self.cfg);cfg['authorized']=True
        with self.assertRaises(c.Stop):c.preflight(cfg)
        self.reference_patch.stop()
        try:
            with self.assertRaisesRegex(c.Stop,'ORIGINAL_REFERENCE_PROJECTION_BYTES'):c.reference_bytes_check(self.cfg['inputs']['old_reference'])
        finally:self.reference_patch.start()

    def test_exact_witness_fields_and_invalid_hash(self):
        can=c.policy.CANDIDATES[2];current={k:'a'*64 for k in c.REUSE_FIELDS}
        witness=dict(candidate=can.id,saved=dict(current),proposed=dict(current),all_identities_equal=True,
            tasks=[dict(sample=i,cosine=7,sine=9,receipt_sha256='c'*64,accepted=True) for i in range(33)],
            correctness_scope='SAVED_COHORT_ACCEPTANCE_WITH_DECLARED_VECTOR_GATE_SCOPE',proof={})
        actual=c.saved_cost_tasks(can);self.assertEqual(len(actual),33)
        with patch.object(c,'saved_cost_tasks',return_value=witness['tasks']):
            self.assertEqual(len(c.reuse_costs(witness,current,can)),33)
            for key in c.REUSE_FIELDS:
                r=copy.deepcopy(witness);r['saved'][key]='b'*64
                with self.assertRaises(c.Stop):c.reuse_costs(r,current,can)

    def test_missing_norm_and_wrong_closure_unit_still_stop(self):
        self.assertTrue(callable(c.policy.AdaptiveInputs.validate))
        o=c.policy.Observation(None,'RAW_VECTOR','B_RAW_MINUS_CORRECTED_SIGNAL','synthetic')
        with self.assertRaises(c.Stop):o.validate('CORRECTED_SIGNAL','B_RAW_MINUS_CORRECTED_SIGNAL')

    def test_ast_native_connection_and_scalar(self):
        import ast
        tree=ast.parse((ROOT/'scripts/h2_common_production_backend.py').read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='native')
        text=ast.unparse(fn)
        self.assertIn("candidate.method == 'PR'",text)
        self.assertIn('self.ctx.ham.constant + (self.ctx.tail_identity',text)
        self.assertIn('self.native_kernel.compose',text)
        self.assertIn('include_measurement=True',text)
        self.assertIn('self.guard.transpile',text)
        self.assertIn("self.s.call('lowering'",text)

    def test_ast_diagnostics_cover_internal_stages(self):
        text=(ROOT/'scripts/h2_common_production_diagnostics.py').read_text()
        for token in ['tail_term_', 'tail_sum_', 'det_basis_in', 'det_diagonal',
                      'raw_log_scale', 'backend.reference',
                      "work['tail']+=columns", 'CORRECTED_SIGNAL']:
            self.assertIn(token,text)

    def test_schedule_uses_acquisition_module_not_absent_provider_path(self):
        text=(ROOT/'scripts/h2_common_production_backend.py').read_text()
        self.assertIn('self.plan=self.native_kernel.c.core',text)
        self.assertNotIn("import_module('trotterlib.h46_m5_plan')",text)

    def test_direct_source_loader_also_checks_seal(self):
        p=self.root/'unsealed_direct.py';p.write_text('raise RuntimeError("MUST_NOT_RUN")')
        guard=c.ImportSeal(self.cfg);guard.install()
        try:
            spec=importlib.util.spec_from_file_location('unsealed_direct',p)
            module=importlib.util.module_from_spec(spec)
            with self.assertRaisesRegex(c.Stop,'DIRECT_SOURCE_IMPORT_NOT_SEALED'):spec.loader.exec_module(module)
        finally:guard.remove()

    def test_real_source_check_refuses_empty_closure(self):
        self.source_patch.stop()
        try:
            with self.assertRaisesRegex(c.Stop,'SOURCE_CLOSURE'):c.preflight(self.cfg)
        finally:self.source_patch.start()

    def test_loaded_closure_rejects_incomplete_manifest(self):
        a,g=self.approval_grant();s=c.Session(self.cfg,a,g)
        with self.assertRaisesRegex(c.Stop,'UNSEALED_LOADED_MODULE'):s.loaded_closure()

    def test_production_result_ranking_never_sets_go(self):
        b=object.__new__(backend_module.Backend)
        rows=[dict(candidate=x.id,eligible=False) for x in c.policy.CANDIDATES]
        self.assertFalse(b.rank(rows)['science_GO'])

    def test_failure_never_refunds_bootstrap(self):
        class FailingInit:
            def __init__(self,s):raise RuntimeError('synthetic_import_failure')
        a,g=self.approval_grant()
        with self.assertRaises(RuntimeError):c.execute(self.cfg,a,g,backend_factory=FailingInit)
        self.assertEqual(c.read(Path(self.cfg['run_root'])/'STOP.json')['used']['bootstrap'],1)


if __name__=='__main__':unittest.main(verbosity=2)
