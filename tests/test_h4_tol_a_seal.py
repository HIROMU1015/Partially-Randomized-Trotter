"""Artificial A50 seal checks. No real molecular or saved-cost arithmetic."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h4_tol_main_contract as c
import seal_h4_tol_a as a


def fixture(directory):
    draft = {name:'synthetic' for name in a.IMMUTABLE}
    draft.update(stage='A', SOURCE=a.SOURCE, binding=deepcopy(c.BINDING), plan=c.plan('A'), inputs=[],
        approved_by_user=False, runtime_authorization=False, next_stage_authorized=False,
        automatic_retry=False, actual_scientific_calls=0, worker_cpus=[], fresh_resources=dict(admitted=False),
        output=str(directory/'unused/output'), lock=str(directory/'unused/ONE_SHOT'))
    return draft


class ASeal(unittest.TestCase):
    def test_unchanged_A_scope_and_approval_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft = fixture(Path(tmp)); a.check_scope(deepcopy(draft), draft)

    def test_D_or_P_cannot_be_sealed(self):
        for stage in ('D','P','S_STATS'):
            with tempfile.TemporaryDirectory() as tmp:
                draft=fixture(Path(tmp))
                with self.subTest(stage=stage), self.assertRaisesRegex(ValueError,'A_ONLY'):
                    a.check_scope(dict(draft,stage=stage), draft)

    def test_candidate_n_or_budget_change_rejected(self):
        for key in ('cells','calls'):
            with tempfile.TemporaryDirectory() as tmp:
                draft=fixture(Path(tmp)); contract=deepcopy(draft)
                if key=='cells': contract['plan']['cells'].pop()
                else: contract['plan']['calls']['new_exploration_n']=9
                with self.subTest(key=key), self.assertRaisesRegex(ValueError,'FIXED_A_SOURCE_PLAN'):
                    a.check_scope(contract,draft)

    def test_rebound_source_inputs_and_runtime_rejected(self):
        for key in ('SOURCE','runtime_hashes','provider_hashes','FP_evidence','inputs'):
            with tempfile.TemporaryDirectory() as tmp:
                draft=fixture(Path(tmp));draft['inputs']=[dict(role='old',sha256='x')]
                contract=deepcopy(draft); contract[key]='changed' if key!='inputs' else []
                with self.subTest(key=key), self.assertRaises(ValueError):a.check_scope(contract,draft)

    def test_preparation_cannot_activate_science_retry_or_resources(self):
        for key in ('approved_by_user','runtime_authorization','automatic_retry','next_stage_authorized','worker_cpus'):
            with tempfile.TemporaryDirectory() as tmp:
                draft=fixture(Path(tmp)); contract=dict(draft,**{key:True if key!='worker_cpus' else [0]})
                with self.subTest(key=key), self.assertRaises(ValueError):a.check_scope(contract,draft)

    def test_existing_namespace_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            draft=fixture(Path(tmp)); Path(draft['output']).parent.mkdir()
            with self.assertRaisesRegex(ValueError,'UNUSED_A_NAMESPACE'):a.check_scope(draft,draft)

    def test_seed_metadata_decoders_and_conditional_domain(self):
        task=dict(seed=123,cell=dict(q=2),step_seeds=[456])
        base=set(c.old.seed_hierarchy(123,2))|{456}
        self.assertEqual(a.identifiers(dict(tasks=[task]),'tasks'),base)
        self.assertEqual(a.identifiers(dict(plan=dict(registrations=[dict(task=task)])),'fixed_delta'),base)
        self.assertEqual(a.identifiers(dict(tasks={'x':dict(master=123,step=456)}),'flat_registry'),{123,456})
        for mode,domain in [('H6_CONDITIONAL','H6_DBA'),('H8_CONDITIONAL','H8_AB')]:
            expected=base|{int(c.digest([domain+'_ORDER',123])[:16],16)}|{
                int(c.digest([domain+'_COMPONENT',123,i])[:16],16) for i in range(2)}
            self.assertEqual(a.identifiers(dict(tasks=[task]),mode),expected)

    def test_invalid_seed_identifier_rejected(self):
        for value in (-1,2**64,True,'123'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'UINT64'):
                a.identifiers(dict(values=[value]),'values')

    def registry_fixture(self,tmp):
        # The source is artificial here; no production seeds are allocated in tests.
        with patch.object(a,'SOURCE','a'*40):registry,values=a.new_registry()
        c.save(tmp/'registry',registry);c.save(tmp/'excluded',dict(values=[]))
        receipt=dict(status='COMPLETE_COLLISION_ZERO',all_registered_scopes_covered=True,
            new_seed_values_digest=c.digest(sorted(values)),old_explicit_exclusions_sha256=c.sha(tmp/'excluded'),
            geometry_checks=[dict(collisions=[])])
        c.save(tmp/'receipt',receipt)
        return dict(SOURCE='a'*40,plan=c.plan('A'),seed_registry=str(tmp/'registry'),
            old_seed_exclusions=str(tmp/'excluded'),seed_inventory_receipt=str(tmp/'receipt')),registry,values

    def test_exact_392_registry_and_positive_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            contract,registry,values=self.registry_fixture(Path(tmp));c.validate_seed_registry(contract)
            self.assertEqual(len(registry['tasks']),392)
            self.assertEqual(len({x['cell']['id'] for x in registry['tasks']}),49)
            self.assertNotIn(c.FP_PR,{x['cell']['id'] for x in registry['tasks']})
            self.assertEqual(len(values),2760)

    def test_exclusion_collision_never_redraws(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);contract,registry,values=self.registry_fixture(tmp)
            c.save(tmp/'excluded',dict(values=[registry['tasks'][0]['seed']]))
            with self.assertRaisesRegex(ValueError,'SEED_COLLISION'):c.validate_seed_registry(contract)
            self.assertEqual(c.read(tmp/'registry'),registry)

    def test_missing_task_rebound_seed_and_bad_receipt_rejected(self):
        for mutation in ('missing','seed','receipt','geometry'):
            with tempfile.TemporaryDirectory() as tmp:
                tmp=Path(tmp);contract,registry,values=self.registry_fixture(tmp)
                if mutation=='missing':registry['tasks'].pop();c.save(tmp/'registry',registry)
                elif mutation=='seed':registry['tasks'][0]['seed']+=1;c.save(tmp/'registry',registry)
                else:
                    receipt=c.read(tmp/'receipt')
                    if mutation=='receipt':receipt['new_seed_values_digest']='wrong'
                    else:receipt['geometry_checks'][0]['collisions']=[1]
                    c.save(tmp/'receipt',receipt)
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):c.validate_seed_registry(contract)

    def test_snapshot_copy_detects_changed_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);path=tmp/'origin';path.write_text('first');store=a.Store(tmp)
            copy=store.copy('synthetic',path);expected=c.sha(copy);path.write_text('second')
            self.assertNotEqual(c.sha(path),expected);self.assertEqual(copy.read_text(),'first')
            with self.assertRaisesRegex(ValueError,'REGISTERED_METADATA_CHANGED'):store.copy('synthetic',path,expected)

    def test_direct_witness_invalid_grants_refuse_without_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);contract=fixture(tmp);c.save(tmp/'contract',contract)
            c.save(tmp/'grant',c.invalid_grant('A',contract));rows=a.refuse(tmp/'contract',tmp/'grant',tmp)
            self.assertEqual(len(rows),2);self.assertTrue(all(r['refused_before_science'] for r in rows))
            self.assertFalse(Path(contract['output']).parent.exists())

    def test_parallel_and_terminal_plan_preserves_one_shot(self):
        p=a.parallel_and_completion_plan()
        self.assertIsNone(p['agent_imposed_CPU_RAM_worker_wall_caps'])
        self.assertTrue(p['PSI_diagnostic_only']);self.assertFalse(p['automatic_retry'])
        self.assertFalse(p['next_stage_authorized']);self.assertFalse(p['scientific_accounting_repeated_by_auditor'])


if __name__=='__main__':unittest.main()
