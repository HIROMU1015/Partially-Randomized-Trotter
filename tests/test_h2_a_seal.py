"""Artificial metadata fixtures only; no molecular arrays or scientific imports."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h2_panels_contract as c
import prepare_h2_a_seal as seal
import test_h2_panels as inherited


class DependencyMetadata(unittest.TestCase):
    def setUp(self):
        self.result = dict(stage='D', SOURCE=seal.D_SOURCE, status='COMPLETE_MANDATORY_STOP', binding=c.BINDING)
        self.done = dict(stage='D', SOURCE=seal.D_SOURCE, status='COMPLETE_MANDATORY_STOP',
                         result_sha256='toy-hash', native_exit_code=0, same_parent_reaped=True, wait4_complete=True)
        self.native = dict(exit_code=0, same_parent_reaped=True, wait4_complete=True)
        self.accepted = dict(result_sha256='toy-hash', accepted_by_user=True, verbatim_user_message='ARTIFICIAL')

    def check(self):
        seal.dependency_payload('D', self.result, self.done, self.native, self.done, self.accepted, 'toy-hash')

    def test_completed_recovery_source_accepted(self): self.check()

    def test_old_interrupted_D_is_not_a_dependency(self):
        self.result['status'] = 'STOP_NATIVE_COMPLETION_UNCONFIRMED'
        with self.assertRaisesRegex(ValueError, 'COMPLETE_SAME_BINDING'): self.check()

    def test_wrong_sector_rejected(self):
        self.result['binding'] = dict(c.BINDING, sector_dimension=6)
        with self.assertRaisesRegex(ValueError, 'SAME_BINDING'): self.check()

    def test_wrong_result_hash_rejected(self):
        self.accepted['result_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'DEPENDENCY_HASH'): self.check()

    def test_missing_native_reap_rejected(self):
        self.native['wait4_complete'] = False
        with self.assertRaisesRegex(ValueError, 'NATIVE_EXIT_WAIT4'): self.check()

    def test_wrong_scientific_source_rejected(self):
        self.result['SOURCE'] = seal.SCIENCE_SOURCE
        with self.assertRaisesRegex(ValueError, 'DEPENDENCY_SOURCE'): self.check()

    def test_saved_pf_is_copy_only(self):
        rows = [dict(cell_id='toy', epsilon=e, eligible=True, cohort='DETERMINISTIC', n=1,
                     G_RZ_exact=str(i+2)) for i,e in enumerate(c.EPS)]
        result = dict(accounting=dict(rows=rows, winners={str(r['epsilon']):r for r in rows}))
        saved = dict(binding=c.BINDING, D_result_sha256='toy-hash', winners=deepcopy(result['accounting']['winners']))
        seal.pf_payload(saved, result, 'toy-hash')
        saved['winners']['0.05']['G_RZ_exact'] = '0'
        with self.assertRaisesRegex(ValueError, 'COUNTERPART_COPY'): seal.pf_payload(saved, result, 'toy-hash')

    def test_missing_pf_endpoint_rejected(self):
        result = dict(accounting=dict(rows=[], winners={}))
        saved = dict(binding=c.BINDING, D_result_sha256='toy-hash', winners={})
        with self.assertRaisesRegex(ValueError, 'FIVE_SAVED'): seal.pf_payload(saved, result, 'toy-hash')


class OriginalPendingSeal(unittest.TestCase):
    def setUp(self):
        self.original = dict(SOURCE=seal.SCIENCE_SOURCE, plan=c.plan('A'), calls=c.budget('A',90),
            primitive_probes=90, status='SEALED_PENDING_APPROVAL', inputs=[dict(role='toy')],
            pending=['ACCEPTED_D_RESULT_SHA'], dependencies={}, worker_cpus=[], resource_receipt=None)
        self.spec = deepcopy(self.original); self.spec['pending'] = list(seal.PENDING)

    def test_unchanged_original_with_resolved_dependencies(self):
        self.spec['inputs'].append(dict(role='D_result')); self.spec['dependencies'] = {'D':'toy','FP':'toy'}
        seal.unchanged_science(self.spec, self.original)

    def test_grid_change_rejected(self):
        self.spec['plan']['cells'].pop()
        with self.assertRaisesRegex(ValueError, 'ORIGINAL_A_SEAL_CHANGED'): seal.unchanged_science(self.spec,self.original)

    def test_budget_change_rejected(self):
        self.spec['calls']['wrapper_compile'] += 2
        with self.assertRaisesRegex(ValueError, 'ORIGINAL_A_SEAL_CHANGED'): seal.unchanged_science(self.spec,self.original)

    def test_original_input_removal_rejected(self):
        self.spec['inputs'] = []
        with self.assertRaisesRegex(ValueError, 'INPUT_ROWS_PRESERVED'): seal.unchanged_science(self.spec,self.original)

    def test_seal_cannot_activate(self):
        self.spec['status'] = 'SEALED_FOR_EXECUTION'
        with self.assertRaises(ValueError): seal.unchanged_science(self.spec,self.original)

    def test_n8_no_FP_pooling_and_fixed_maxima(self):
        plan = c.plan('A'); calls = c.budget('A',90)
        self.assertEqual((len(plan['cells']),plan['n'],calls['trajectory'],calls['wrapper_compile']),(66,8,528,1056))
        self.assertEqual((calls['base_signal'],calls['new_signed_shot_endpoints'],calls['primitive_probe']),(64,320,90))
        self.assertFalse(plan['FP_PR_cost_pooled_with_A'])
        self.assertEqual((calls['occurrence'],calls['RTE_event'],calls['control_probe']),(3504,16032,264))


class SeedMetadata(unittest.TestCase):
    def setUp(self):
        parent = c.PRIVATE/'artificial_a_seal'; parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent); self.root = Path(self.temp.name)
        self.store = self.root/'original'; self.store.mkdir(); self.dest = self.root/'final'; self.dest.mkdir()
        (self.root/'FP_activation01').mkdir()
        self.patches = [patch.object(c,'STORE',self.store), patch.object(c,'PRIVATE',self.root)]
        for p in self.patches: p.start()
        a = c.seed_tasks('A',seal.SCIENCE_SOURCE); fp = c.seed_tasks('FP',seal.SCIENCE_SOURCE)
        c.save(self.store/'A_seeds.json',dict(tasks=a)); c.save(self.store/'FP_seeds.json',dict(tasks=fp))
        self.new = c.seed_values(a); self.fp = c.seed_values(fp)
        c.save(self.store/'old_seed_values.json',dict(values=[1,2]))
        origin = self.root/'external_seed_metadata.json'; c.save(origin,dict(artificial=True))
        r = dict(role='toy_campaign',origin=str(origin),snapshot=str(origin),sha256=c.sha(origin))
        c.save(self.root/'FP_activation01/seed_origins.json',[r])
        c.save(self.store/'seed_origins.json',[dict(role='geometry',origin=str(origin),path=str(origin),sha256=c.sha(origin))])
        check = self.store/'A_check.json'; c.save(check,dict(new_values_digest=c.digest(sorted(self.new)),
            joint_FP_A_values_digest=c.digest(sorted(self.new|self.fp)),inherited_inventory_cutoff='ARTIFICIAL',
            geometry_checks=[dict(collisions=[],old_seed_slots_checked=7737216)]))
        self.spec = dict(seed_registry=str(self.store/'A_seeds.json'),seed_check=str(check))

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def test_keep_original_registry_and_exclude_FP_values(self):
        before = c.sha(self.spec['seed_registry']); receipt = seal.seed_check(self.spec,self.dest)
        self.assertEqual(c.sha(self.spec['seed_registry']),before)
        self.assertEqual((receipt['master_seeds'],receipt['new_seed_slots'],receipt['exclusion_values']),(528,3888,6))
        self.assertFalse(receipt['seed_redrawn']); self.assertEqual(receipt['new_RTE_sampling'],0)
        self.assertFalse(self.new & set(c.read(self.dest/'fresh_seed_exclusions.json')['values']))

    def test_collision_STOPS_without_redraw(self):
        (self.store/'old_seed_values.json').write_text(json.dumps(dict(values=[next(iter(self.new))])))
        with self.assertRaisesRegex(ValueError,'SEED_COLLISION'): seal.seed_check(self.spec,self.dest)
        self.assertFalse((self.dest/'seed_refresh.json').exists())

    def test_changed_registered_metadata_needs_new_seal(self):
        (self.root/'external_seed_metadata.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'SEED_SNAPSHOT_CHANGED'): seal.seed_check(self.spec,self.dest)

    def test_geometry_proof_for_wrong_seed_set_rejected(self):
        p=Path(self.spec['seed_check']); value=c.read(p);value['joint_FP_A_values_digest']='wrong';p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'GEOMETRY_PROOF'): seal.seed_check(self.spec,self.dest)


class ExistingGateCompatibility(unittest.TestCase):
    # Reuse the existing disposable valid gate fixture without inheriting its tests.
    setUp = inherited.SealedMetadataTests.setUp
    tearDown = inherited.SealedMetadataTests.tearDown
    fixture = inherited.SealedMetadataTests.fixture

    def test_existing_A_gate_accepts_recovery_D_SOURCE(self):
        spec,grant=self.fixture('A');path=Path(spec['dependencies']['D']['result'])
        result=c.read(path);result['SOURCE']=seal.D_SOURCE;path.write_text(json.dumps(result))
        h=c.sha(path);dep=spec['dependencies']['D'];dep['sha256']=h
        for key in ('acceptance','completion_audit'):
            p=Path(dep[key]);v=c.read(p);v['result_sha256']=h;p.write_text(json.dumps(v));dep[key+'_sha256']=c.sha(p)
        grant['spec_digest']=c.digest(spec);c.authorize(spec,grant,interpreter=False)

    def test_pending_A_still_needs_separate_explicit_approval(self):
        spec,grant=self.fixture('A')
        with self.assertRaisesRegex(ValueError,'SEPARATE_EXPLICIT_STAGE_APPROVAL_REQUIRED'):
            c.authorize(spec,c.invalid_grant('A'),interpreter=False)

    def test_invalid_A_driver_worker_adapter_without_packages(self):
        spec=self.root/'invalid_spec.json';grant=self.root/'invalid_grant.json'
        c.save(spec,dict(stage='A'));c.save(grant,c.invalid_grant('A'))
        entry=Path(__file__).resolve().parents[1]/'scripts/run_h2_panels.py'
        for flags in ([],['--worker']):
            r=subprocess.run([sys.executable,'-B','-S',str(entry),'--spec',str(spec),'--grant',str(grant),*flags],
                             capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('SEPARATE_EXPLICIT_STAGE_APPROVAL_REQUIRED',r.stderr)
            self.assertNotIn('ModuleNotFoundError',r.stderr)
        import h2_panels_science
        with self.assertRaisesRegex(ValueError,'SEPARATE_EXPLICIT_STAGE_APPROVAL_REQUIRED'):
            h2_panels_science.execute(dict(stage='A'),c.invalid_grant('A'),self.root/'uncreated')
        self.assertFalse((self.root/'uncreated').exists())


if __name__ == '__main__':
    unittest.main()
