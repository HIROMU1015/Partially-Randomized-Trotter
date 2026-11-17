"""Ordinary executor factory defaults and operational-only profile changes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_rpe_method_optimizer as CLI
from trottertracks import rpe_optimizer_concurrent as C
from trottertracks import rpe_plan_prepare as P
from trottertracks import rpe_plan_readiness as R


class AutoOptionsTests(unittest.TestCase):
    def test_normal_factory_defaults_auto_keeps_explicit_and_persisted_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            spec = dict(max_new_q3_per_method={'ARTIFICIAL': 0}, science_approved=False,
                        executor=dict(scheduler='concurrent', distributed={}))
            def invoke(value):
                with patch.object(C, 'DistributedExecutor') as constructor:
                    CLI.make_executor(ROOT, run, {}, {}, {}, value)
                    return constructor.call_args.kwargs
            self.assertEqual(invoke(spec)['max_concurrent'], 'auto')
            explicit = deepcopy(spec); explicit['executor']['max_concurrent_jobs'] = 2
            self.assertEqual(invoke(explicit)['max_concurrent'], 2)
            path = run / 'distributed/ALLOCATION_SETTINGS.json';path.parent.mkdir()
            path.write_text(json.dumps(dict(mode='fixed', max_concurrent_jobs=3)))
            self.assertEqual(invoke(spec)['max_concurrent'], 3)
            self.assertEqual(invoke(explicit)['max_concurrent'], 3)

    def test_non_distributed_default_remains_one(self):
        spec = dict(max_new_q3_per_method={'ARTIFICIAL': 0}, science_approved=False,
                    executor=dict(scheduler='concurrent'))
        with tempfile.TemporaryDirectory() as temporary, patch.object(C, 'ConcurrentExecutor') as constructor:
            CLI.make_executor(ROOT, temporary, {}, {}, {}, spec)
            self.assertEqual(constructor.call_args.kwargs['max_concurrent'], 1)
        spec['executor']['max_concurrent_jobs'] = 'auto'
        with self.assertRaisesRegex(ValueError, 'distributed'):
            CLI.make_executor(ROOT, '/tmp/not-used-auto-option', {}, {}, {}, spec)

    def test_new_prepared_profiles_auto_slots_leave_scientific_core_unchanged(self):
        from trottertracks.rpe_host_profiles import profile_core
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);(root / 'prepared/profiles').mkdir(parents=True)
            profile = dict(host_id='host', root='fixed-root', interpreter='fixed-python',
                           dp_workers=3, route_views={'fixed': 'fixed-view'})
            (root / 'prepared/profiles/host.json').write_text(json.dumps(profile))
            config = root / 'EVIDENCE.json'
            config.write_text(json.dumps(dict(slot_policy=dict(mode='auto', initial_slots=1))))
            _, _, profiles = P.host_profile_inputs({'output': str(root / 'prepared')}, root / 'out', ROOT, config)
            self.assertEqual(profiles['host']['slot_policy']['mode'], 'auto')
            self.assertEqual(profiles['host']['slots'], 1)
            self.assertEqual(profile_core(profiles['host']), profile_core(profile))

    def test_timed_profile_pack_keeps_scientific_inputs_and_marks_fresh_check(self):
        import prepare_rpe_auto_profiles as T
        base = ROOT / 'docs/research/rpe_q3_host_eligibility/optional_wsl/NORMAL_ENTRY/worker_profiles'
        layout = json.loads((ROOT / 'docs/research/rpe_q3_auto_allocation/HOST_LAYOUT.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'new-pack'
            rows = T.prepare(base, layout, '870d7c153', output, 'ARTIFICIAL_METADATA')
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row['status'], 'TIMED_ADAPTER_PROFILE_REQUIRES_FRESH_DRY_CHECK')
                old = json.loads((base / (row['host_id'] + '.json')).read_text())
                new = json.loads((output / 'worker_profiles' / (row['host_id'] + '.json')).read_text())
                self.assertEqual(old['root'], new['root'])
                self.assertEqual(old['route_views'], new['route_views'])
                self.assertEqual(new['slot_policy']['initial_slots'], 1)
                before = [old] + [r['profile'] for r in old.get('route_profiles', [])]
                after = [new] + [r['profile'] for r in new.get('route_profiles', [])]
                for a, b in zip(before, after):
                    for key in ('interpreter', 'route_views', 'dp_workers', 'prefix_cache_directory', 'runtime_checks'):
                        self.assertEqual(a.get(key), b.get(key))
                    self.assertEqual(b['adapter_root'], row['transport_root'])
                    self.assertIn('src/trottertracks/rpe_q3_timing.py', b['adapter_manifest']['files'])

    def test_operational_slots_do_not_invalidate_existing_dry_cache(self):
        base = ROOT / 'docs/research/rpe_q3_host_eligibility/optional_wsl/NORMAL_ENTRY'
        profile = json.loads((base / 'worker_profiles/server2.json').read_text())
        assignment_path = ROOT / 'docs/research/rpe_distributed_h4_plan_20261117/assignments/FULL_PF.json'
        assignment = json.loads(assignment_path.read_text())
        if not Path(profile['root']).is_dir():
            self.skipTest('Existing fixed computation root not available')
        first = R.dry_check_fingerprint(profile['root'], assignment, profile)
        changed = deepcopy(profile)
        changed.update(slots=7, work_dir='/tmp/different-owned-spool', slot_policy=dict(mode='auto', initial_slots=1))
        self.assertEqual(R.dry_check_fingerprint(profile['root'], assignment, changed), first)


if __name__ == '__main__':
    unittest.main()
