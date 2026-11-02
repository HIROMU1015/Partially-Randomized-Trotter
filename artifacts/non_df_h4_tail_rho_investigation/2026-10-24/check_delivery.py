"""Finalize this static investigation; no saved-H4 numerical evaluator."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def validate_draft(d):
    for k in ('execution_authorized','grant_valid','dispatch_enabled','N_G_authorized','formal_certificate_accepted'):
        assert d[k] is False
    for k in ('SOURCE','runtime_manifest','grant_id'):assert d[k] is None
    assert d['not_a_production_contract'] and d['mandatory_STOP'] and not d['automatic_retry']
    assert len(d['designs'])==2
    for design in d['designs']:
        b=design['future_scientific_budget_max']
        assert all(type(x) is int and x>=0 for x in b.values())
        assert b['sampling']==b['compile']==b['formal_certificate_acceptance']==0


def artificial():
    r=load('delivery_existing_rules',ROOT/'src/trotterlib/h4_transfer_rules.py')
    tree=ast.parse((ROOT/'src/trotterlib/df_rte_qiskit.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='estimate_df_rte_structural_size_upper_bound')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    env=dict(require_integer_count=lambda x,**_:x)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[])),'<existing counter>','exec'),env)
    class Tests(unittest.TestCase):
        def test_counter_uses_only_invented_components(self):
            toy=SimpleNamespace(diagonal_pauli_support=(0,1),basis_change_operations=('a','b','c'))
            self.assertEqual(env['estimate_df_rte_structural_size_upper_bound']([toy],maximum_taylor_order=4,event_count=2,controlled=True),79)
            self.assertIsNone(read(OUT/'evidence_map.json')['cost_bound']['compiled_RZ_upper'])
        def test_existing_phase_sensitive_rules(self):
            for kind in ('CRZ','RZZ','CONTROLLED_RZZ','PHASE'):
                self.assertEqual(r.controlled_diagonal_identity(kind)['implementation_choice_in_saved_H4'],'NOT_PROVED')
            with self.assertRaises(ValueError):r.diagonal_identity('one_body',phase_sign=-1)
        def test_transfer_preserves_both_branches(self):
            x=r.measurement_transfer(normalized_ray=True,zero_exact=False)
            self.assertEqual(x['expectation_factor'],'1')
            self.assertIn('eta0',x['measurement_bound_symbolic'])
            self.assertFalse(x['numerical_B_scaled_bound_evaluated'])
        def test_execution_flags_and_grant_cannot_be_added(self):
            d=read(OUT/'next_contracts_DRAFT.json');validate_draft(d)
            for k in ('execution_authorized','grant_valid','dispatch_enabled','N_G_authorized'):
                bad=deepcopy(d);bad[k]=True
                with self.assertRaises(AssertionError):validate_draft(bad)
            bad=deepcopy(d);bad['grant_id']='fake'
            with self.assertRaises(AssertionError):validate_draft(bad)
        def test_no_future_sampling_budget(self):
            bad=read(OUT/'next_contracts_DRAFT.json');bad['designs'][0]['future_scientific_budget_max']['sampling']=1
            with self.assertRaises(AssertionError):validate_draft(bad)
        def test_unknown_evidence_not_promoted(self):
            e=read(OUT/'evidence_map.json')
            self.assertIsNone(e['current_actual_rho']);self.assertIsNone(e['cost_bound']['population_cost_upper'])
            self.assertEqual(e['old_C_L'],'COMBINATION_BLOCKED')
            self.assertEqual(e['old_compiler'],'MISSING_EVIDENCE')
            self.assertFalse(e['formal_certificate_accepted'])
    return unittest.defaultTestLoader.loadTestsFromTestCase(Tests)


def main():
    helper=load('delivery_saved_checker',OUT/'check_static.py')
    additional=helper.audit()  # Existing C/L token identities and514 price joins.
    (OUT/'static_audit.json').write_text(json.dumps(additional,ensure_ascii=False,indent=2)+'\n')
    protected=read(OUT/'preservation_before.json')
    for p,h in protected.items():assert sha(ROOT/p)==h,p
    remote=read(OUT/'github_verification.json')
    for item in remote['files']:
        b=(ROOT/item['path']).read_bytes()
        assert hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest()==item['sha']
        assert subprocess.check_output(['git','show',item['ref']+':'+item['path']],cwd=ROOT)==b
    seal=read(ROOT/'artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/seal.json')
    suffixes=('preset_passmanagers/level1.py','preset_passmanagers/builtin_plugins.py',
        'optimization/optimize_1q_decomposition.py','optimization/inverse_cancellation.py',
        'basis/basis_translator.py','synthesis/high_level_synthesis.py','synthesis/unitary_synthesis.py',
        'circuit/add_control.py','standard_gates/rz.py','standard_gates/rzz.py',
        'standard_gates/xx_plus_yy.py','standard_gates/equivalence_library.py','qiskit/_accelerate.abi3.so')
    runtime={p:h for p,h in seal['legacy_runtime_files'].items() if any(p.endswith(s) for s in suffixes)}
    assert len(runtime)==13
    for p,h in runtime.items():assert sha(Path(p))==h
    validate_draft(read(OUT/'next_contracts_DRAFT.json'))
    assert len(read(OUT/'evidence_map.json')['rho_edges'])==12
    with (OUT/'artificial_tests_final.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(artificial())
    assert result.wasSuccessful()
    assert not any(x.startswith(('qiskit','numpy','flint','scipy','pyscf','openfermion')) for x in sys.modules)
    for p,h in protected.items():assert sha(ROOT/p)==h,p
    report=dict(status='PASS_STATIC_ONLY',base_commit=helper.BASE,
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        remote_files_verified=len(remote['files']),additional_receipt_rows_git_checked=additional['github_current_full_byte_checks'],
        runtime_source_files_and_binary_SHA_verified=len(runtime),runtime_files=runtime,
        preserved_files=len(protected),preservation_manifest_SHA256=sha(OUT/'preservation_before.json'),
        old_paired_joins=additional['old_paired_join_counts'],support_descriptor_rows=324,rho_scope_rows=12,
        nonexecution_designs=2,artificial_tests=result.testsRun,artificial_failures=0,
        current_checker_SHA256=sha(Path(__file__)),reused_static_checker_SHA256=sha(OUT/'check_static.py'),
        evidence_map_SHA256=sha(OUT/'evidence_map.json'),draft_SHA256=sha(OUT/'next_contracts_DRAFT.json'),
        new_H4_numeric_evaluations=0,new_scientific_cost_or_probability_values=0,new_N_G_accounting=0,
        new_sampling=0,new_compile=0,scientific_grants_issued=0,
        old_C_L='COMBINATION_BLOCKED',old_compiler='MISSING_EVIDENCE',actual_rho=None,
        old_native='UNKNOWN_UNCHANGED',mandatory_STOP=True)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='runtime_files'}))


if __name__=='__main__':main()
