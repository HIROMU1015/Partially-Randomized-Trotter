"""Artificial metadata/receipts only; never sample or compile molecular science."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h6_pr_confirmation as h
import h6_pr_confirmation_seal as s
import prepare_h6_pr_confirmation as old
import prepare_h6_pr_confirmation_seal as prep


class ArtificialFinalSeal(unittest.TestCase):
    def test_memory_admission_counts_coordinators_driver_and_descendants(self):
        self.assertEqual([s.process_slots(n) for n in (1,2,5,63)], [3,4,9,96])
        for bad in (None,True,0,-1):
            with self.assertRaises(ValueError):s.process_slots(bad)

    def test_final_N_and_exact_per_scope_and_total_budget(self):
        self.assertEqual(s.FINAL_N, dict(H6_CONFIRM_P4_Q2=48, H6_CONFIRM_P4_Q8=104))
        values = [h.calls_for(t, s.FINAL_N[t['scope']]) for t in h.TARGETS]
        self.assertEqual({k:sum(v[k] for v in values) for k in values[0]}, dict(
            trajectory=152, occurrence=928, RTE_event=3712, native_evolution=152,
            wrapper_compile=304, control_probe=8, independent_event_oracle=2))

    def test_explicit_and_higher_PF_seed_readers(self):
        self.assertEqual(s.seed_values(dict(values=[1,2,2]), 'values'), {1,2})
        self.assertEqual(s.seed_values(dict(all_master_and_step_values=[3,4]),
                                      'all_master_and_step_values'), {3,4})
        self.assertEqual(s.seed_values(dict(tasks={'a':{'0:0':5, '1:0':6}}), 'flat_registry'), {5,6})
        for bad in [True, -1, 2**64, '7']:
            with self.assertRaisesRegex(ValueError, 'UINT64'):
                s.seed_values(dict(values=[bad]), 'values')
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_SEED_METADATA_MODE'):
            s.seed_values({}, 'unknown')

    def test_frozen_step_hierarchy_and_stopped_reservations_retained(self):
        tasks = [dict(seed=123, cell=dict(q=2)), dict(seed=None, cell=dict(q=8))]
        expected = set(h.shared.hierarchy(123, 2)) | {7}
        self.assertEqual(s.seed_values(dict(tasks=tasks, excluded=[7]), 'tasks'), expected)
        value = dict(plan=dict(registrations=[dict(task=t) for t in tasks]))
        self.assertEqual(s.seed_values(value, 'fixed_delta'), expected-{7})
        tasks = [dict(seed=234, group=dict(cell=dict(q=1)))]
        self.assertEqual(s.seed_values(dict(tasks=tasks), 'tasks'), {234})

    def fixture(self, directory):
        path = Path(directory)
        h.c.save(path/'old', dict(values=[11,12,13]))
        sources = {campaign:[(path/'old', 'values')] for campaign in s.REQUIRED_CAMPAIGNS}
        inventory, excluded, origins = prep.collect_inventory(path, sources)
        self.assertEqual(excluded, {11,12,13})
        selection = dict(schema='H6_FINAL_SELECTION_V1', n_by_scope=s.FINAL_N,
            targets=list(h.TARGETS), scientific_execution_authorized=False)
        h.c.save(path/'selection', selection)
        inventory_path = path/'seed_inventory_PRIVATE.json'
        inputs = [old.byte_row(inventory_path,'inventory'), old.byte_row(path/'selection','selection')]
        plan = dict(SOURCE='a'*40, n_by_scope=s.FINAL_N, source_hashes={'fake':'b'*64},
            design_digest='c'*64, seed_inventory_sha256=h.c.sha(inventory_path),
            final_selection_sha256=h.c.sha(path/'selection'), immutable_inputs=deepcopy(inputs))
        h.c.save(path/'plan', plan)
        inputs.append(old.byte_row(path/'plan','plan'))
        spec = dict(SOURCE='a'*40, n_by_scope=s.FINAL_N, source_hashes=plan['source_hashes'],
            design_digest='c'*64, seed_plan_digest=h.c.digest(plan), inputs=inputs,
            seed_inventory=str(inventory_path), seed_plan=str(path/'plan'), final_selection=str(path/'selection'))
        registry = dict(excluded=[11,12,13], seed_inventory_sha256=h.c.sha(inventory_path))
        return path, inventory, origins, spec, registry

    def verify(self, spec, registry):
        read = h.c.read
        def own_read(path):
            if Path(path) == h.ROOT/h.PLAN: return dict(targets=list(h.TARGETS))
            return read(path)
        with patch.object(h.c,'read',side_effect=own_read):
            return s.verify_final_binding(spec,registry)

    def test_complete_campaign_inventory_and_unchanged_origin_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path, inventory, origins, spec, registry = self.fixture(directory)
            self.verify(spec,registry)
            h.c.verify_byte_rows(origins)
            (path/'old').write_text('changed original metadata')
            # Launch binds private snapshots; observations of other worktrees
            # are not frozen. Historical origin changes are still detectable.
            self.verify(spec,registry)
            with self.assertRaisesRegex(ValueError,'INPUT_BYTES_CHANGED'):
                h.c.verify_byte_rows(origins)

    def test_missing_later_campaign_or_false_completeness_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, inventory, _, _, _ = self.fixture(directory)
            for campaign in s.REQUIRED_CAMPAIGNS:
                changed = deepcopy(inventory); del changed['campaigns'][campaign]
                with self.assertRaisesRegex(ValueError,'COMPLETE_CAMPAIGN'):
                    s.inventory_values(changed)
            changed = deepcopy(inventory);changed['known_registered_campaigns_complete']=False
            with self.assertRaisesRegex(ValueError,'COMPLETE_CAMPAIGN'):
                s.inventory_values(changed)

    def test_exclusion_omission_and_modified_seed_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, inventory, _, spec, registry = self.fixture(directory)
            with self.assertRaisesRegex(ValueError,'EXACT_COMPLETE_EXCLUSIONS'):
                self.verify(spec, dict(registry,excluded=[11,12]))
            row = inventory['campaigns'][s.REQUIRED_CAMPAIGNS[0]][0]
            Path(row['path']).write_text('changed')
            with self.assertRaisesRegex(ValueError,'INPUT_BYTES_CHANGED'):
                self.verify(spec,registry)

    def test_seed_plan_source_input_and_scope_N_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, spec, registry = self.fixture(directory)
            for key,value in [('SOURCE','d'*40), ('seed_plan_digest','e'*64),
                              ('n_by_scope',{scope:48 for scope in h.SCOPES})]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    self.verify(dict(spec,**{key:value}),registry)
            with self.assertRaisesRegex(ValueError,'UNCOVERED_FINAL_METADATA'):
                self.verify(dict(spec,inputs=spec['inputs'][:-1]),registry)

    def test_final_seal_cannot_change_adopted_N_by_updating_digest(self):
        t = h.TARGETS[1]
        spec = dict(schema='H6_CONFIRMATION_EXECUTION_V1', technical_sealed=True,
            target=t, scope=t['scope'], cohort=h.COHORT, n=48,
            n_by_scope={scope:48 for scope in h.SCOPES}, calls=h.calls_for(t,48))
        spec['technical_digest']=h.shared.technical_digest(spec)
        with self.assertRaisesRegex(ValueError,'FINAL_ADOPTED_H6_N'):
            h.sealed(spec)

    def test_final_identifiers_unique_and_old_collision_has_no_redraw(self):
        # Artificial SOURCE/digest; the real sealed registry is never read.
        tasks=h.derive_registry(h.TARGETS,'a'*40,'b'*64,[],s.FINAL_N)
        values=[v for t in tasks for v in h.shared.hierarchy(t['seed'],t['cell']['q'])]
        self.assertEqual(len(tasks),152);self.assertEqual(len(set(values)),1080)
        with self.assertRaisesRegex(ValueError,'COLLISION_NO_REDRAW'):
            h.derive_registry(h.TARGETS,'a'*40,'b'*64,[values[-1]],s.FINAL_N)

    def test_completion_exit_failure_rejected_without_reading_result(self):
        with patch.object(h.c,'read',side_effect=AssertionError('NO_READ')):
            with self.assertRaisesRegex(ValueError,'DRIVER_NATIVE_EXIT0'):
                s.audit_completion({},dict(exit_code=1,reaped=True))

    def completion_fixture(self, directory):
        path=Path(directory);scope=h.TARGETS[0]['scope']
        tasks=[dict(id='fake_'+str(i),scope=scope) for i in range(2)]
        pairs=[dict(artificial=i) for i in range(2)]
        calls=dict(trajectory=2,wrapper_compile=4)
        spec=dict(output=str(path),SOURCE='a'*40,scope=scope,n=2,calls=calls,registry=str(path/'registry'))
        h.c.save(path/'registry',dict(tasks=tasks))
        result=dict(status='COMPLETE_MANDATORY_STOP',SOURCE='a'*40,scope=scope,calls=calls,
            old_samples_pooled=False,automatic_retry=False,next_stage_authorized=False,paired_costs=pairs)
        h.c.save(path/'result.json',result)
        h.c.save(path/'terminal.json',dict(status=result['status'],SOURCE=spec['SOURCE'],
                                         result_sha256=h.c.sha(path/'result.json')))
        handles=[dict(task_id=t['id']) for t in tasks]
        h.c.save(path/'native_receipt.json',dict(all_reaped=True,
            maps=[dict(all_reaped=True,cleanup_errors=[],workers=handles)]))
        budget=h.c.Budget(path,calls)
        for kind,count in calls.items():budget.take(kind,count)
        def accept(task,handle,spec):return pairs[int(task['id'][-1])],{}
        return path,spec,accept

    def test_completion_requires_native_receipts_and_exact_parent_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            path,spec,accept=self.completion_fixture(directory)
            with patch.object(h,'accept',side_effect=accept):
                audit=s.audit_completion(spec,dict(exit_code=0,reaped=True))
                self.assertFalse(audit['fresh_statistics_recomputed'])
                (path/'wrapper_compile_4_reserved.json').unlink()
                with self.assertRaisesRegex(ValueError,'PARENT_LEDGER'):
                    s.audit_completion(spec,dict(exit_code=0,reaped=True))

    def test_missing_worker_and_failed_or_unreaped_map_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path,spec,accept=self.completion_fixture(directory)
            original=h.c.read(path/'native_receipt.json')
            for key,value in [('all_reaped',False),('cleanup_errors',['failure']),('workers',[])]:
                changed=deepcopy(original);changed['maps'][0][key]=value
                (path/'native_receipt.json').unlink();h.c.save(path/'native_receipt.json',changed)
                with patch.object(h,'accept',side_effect=accept),self.assertRaises(ValueError):
                    s.audit_completion(spec,dict(exit_code=0,reaped=True))


if __name__ == '__main__':unittest.main()
