"""Saved-evidence policy regressions only; no remote connection or scientific job."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from trottertracks import rpe_host_eligibility as H
from trottertracks.rpe_live_q3_readiness import (
    _HOST_PROBE, _supply_inputs, accepts_probe, controller_input_hashes, q3_input_dependencies,
)
from trottertracks.rpe_q3_coordinator import digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class HostEligibilityTests(unittest.TestCase):
    def test_old_pbs_offer_does_not_prevent_probing_a_new_session(self):
        host = 'pbs-0'
        manager = SimpleNamespace(configured={host}, allocations=[dict(host_id=host, status='F')])
        executor = SimpleNamespace(hosts={host:dict(accepting_new_jobs=False)}, pbs_manager=manager,
                                   _host_admission=lambda host, policy, _: policy)
        self.assertTrue(accepts_probe(executor, host, dict(accepting=True), None))
        self.assertFalse(accepts_probe(executor, host, dict(accepting=False), None))
        manager.allocations[0]['status'] = 'R'
        self.assertFalse(accepts_probe(executor, host, dict(accepting=True), None))

    def test_live_readiness_supplies_saved_input_objects_without_changing_head(self):
        with tempfile.TemporaryDirectory() as directory:
            top=Path(directory);controller=top/'controller';worker=top/'worker'
            controller.mkdir()
            def git(root,*arguments):
                return subprocess.check_output(['git',*arguments],cwd=root,stderr=subprocess.PIPE,text=True).strip()
            git(controller,'init')
            git(controller,'config','user.name','Synthetic fixture')
            git(controller,'config','user.email','fixture@example.invalid')
            (controller/'fixed.txt').write_text('fixed implementation fixture')
            git(controller,'add','.');git(controller,'commit','-m','fixed')
            base=git(controller,'rev-parse','HEAD')
            view=top/'frozen-route';git(top,'clone','--no-local',str(controller),str(view))
            computation=top/'fixed-computation';git(top,'clone','--no-local',str(controller),str(computation))
            input_name = 'input with space.json'
            raw=b'{"fixture":true}\n';(controller/input_name).write_bytes(raw)
            (controller/'unrelated.bin').write_bytes(b'belongs to another request')
            git(controller,'add',input_name,'unrelated.bin');git(controller,'commit','-m','new saved input')
            revision=git(controller,'rev-parse','HEAD')
            modules=worker/'src'/'trottertracks';modules.mkdir(parents=True)
            (modules/'__init__.py').touch()
            (modules/'rpe_host_profiles.py').write_text('select_host_profile=lambda profile, assignment:profile\nprofile_core=lambda profile:profile\n')
            (modules/'rpe_host_eligibility.py').write_text('current_host_fingerprint=lambda assignment, profile:{}\n')
            scripts=worker/'scripts';scripts.mkdir()
            (scripts/'check_rpe_q3_host_readiness.py').write_text('''import json,sys
from pathlib import Path
a=json.loads(Path(sys.argv[sys.argv.index('--assignment')+1]).read_text())
p=Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir()
(p/'READINESS.json').write_text(json.dumps(dict(status='REAL_HOST_DRY_CHECK_PASS',
    scientific_evaluation_calls=0,request_sha256=a['request_sha256'])))
''')
            payload=dict(worker_root=str(worker),profile=dict(root=str(view)),input_anchors=[base],
                assignment=dict(request={'route':{'kind':'fixture'}},request_sha256='fixture',
                    input_hashes=[dict(commit=base,path='fixed.txt',sha256=hashlib.sha256(
                        (controller/'fixed.txt').read_bytes()).hexdigest()),
                        dict(commit=revision,path=input_name,sha256=hashlib.sha256(raw).hexdigest())]))
            executor=SimpleNamespace(root=controller,computation_root=computation,input_hash_cache={})
            request=dict(route=dict(kind='fixture',templates=[dict(commit=base,path='fixed.txt'),
                                                             dict(commit=revision,path=input_name)]))
            self.assertCountEqual(controller_input_hashes(executor,request),payload['assignment']['input_hashes'])
            self.assertEqual(git(computation,'rev-parse','HEAD'),base)
            self.assertFalse((computation/input_name).exists())
            def invoke(data):
                return json.loads(subprocess.check_output([sys.executable,'-B','-c',_HOST_PROBE],
                    input=json.dumps(data),text=True))
            first=invoke(payload)
            self.assertEqual(first['status'],'NEEDS_Q3_INPUT_OBJECTS')
            self.assertEqual(first['repositories'][0]['commits'],[revision])
            result=_supply_inputs(executor,payload,invoke,first)
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['input_transfer']['objects'],3)
            self.assertIn('delivery_check_seconds', result['input_transfer'])
            self.assertNotEqual(subprocess.run(['git','cat-file','-e',revision+':unrelated.bin'],
                                               cwd=view,stderr=subprocess.DEVNULL).returncode,0)
            self.assertEqual(git(view,'show',revision+':'+input_name),raw.decode().strip())
            self.assertEqual(git(view,'rev-parse','HEAD'),base)
            self.assertFalse((view/input_name).exists())
            self.assertEqual(invoke(payload)['status'],'PASS')

            packet_name = 'signal-packet.json'
            (controller/packet_name).write_text(json.dumps(dict(means=[
                dict(source=dict(commit=revision, path='unrelated.bin'))])))
            git(controller, 'add', packet_name)
            git(controller, 'commit', '-m', 'saved signal packet')
            packet_commit = git(controller, 'rev-parse', 'HEAD')
            route_request = dict(route=dict(kind='prdf_d2_common_entry',
                signal_packet=dict(commit=packet_commit, path=packet_name)))
            identity = controller_input_hashes(executor, route_request)
            dependencies = q3_input_dependencies(executor, route_request)
            self.assertEqual([row['path'] for row in identity], [packet_name])
            self.assertEqual([row['path'] for row in dependencies], ['unrelated.bin'])
            prdf_payload = dict(payload, input_dependencies=dependencies,
                assignment=dict(payload['assignment'], request=route_request, input_hashes=identity))
            first = invoke(prdf_payload)
            self.assertEqual(set(first['repositories'][0]['commits']), {revision, packet_commit})
            result = _supply_inputs(executor, prdf_payload, invoke, first)
            self.assertEqual(result['status'], 'PASS')
            self.assertEqual(git(view, 'show', revision+':unrelated.bin'), 'belongs to another request')
            self.assertEqual(git(view, 'rev-parse', 'HEAD'), base)

    def runtime_guard_fixture(self):
        root = Path(__file__).resolve().parents[1]
        base = root / 'docs/research/rpe_q3_host_eligibility/pauli_descriptor_route'
        if not (base / 'server2_runtime_ast_v2/READINESS.json').is_file():
            self.skipTest('Saved actual-runtime guard evidence is unavailable')
        assignment = H._read(base / 'ASSIGNMENT.json')
        profile = H._read(base / 'PROFILE_RUNTIME_AST_V2.json')
        return assignment, profile, base

    def romutail_fixture(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / 'docs/research/rpe_q3_host_eligibility/evidence/romutail_prdf_source_09efb068f'
        commit = '09efb068f242d5d6ff88296df3e50283b2f951bf'
        if not evidence.is_dir() or subprocess.run(['git', 'cat-file', '-e', commit], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            self.skipTest('Pinned RomuTail publication is unavailable')
        target = self.base / 'romutail-evidence'
        shutil.copytree(evidence, target)
        assignment = H._read(target / 'ASSIGNMENT.json')
        profile = H._read(target / 'PROFILE.json')
        observed = H._read(target / 'checks/pass01/READINESS.json')
        profile.update(adapter_root=observed['adapter_source']['root'],
            adapter_manifest=dict(commit=commit, implementation_commit=observed['adapter_source']['commit'],
                files={'scripts/run_rpe_optimizer_q3_job.py': observed['adapter_source']['sha256'],
                       'src/trottertracks/rpe_dp_prefix_reuse.py': '554460755ab6fcd38ad5b9f1a680b543aed7e26906579b26470d73fa659148f2'}))
        descriptor = dict(kind='fixed_git', root=str(target), fixed_commit=commit,
            fixed_prefix='docs/research/rpe_prdf_romutail_source_20261117',
            portability_checks=str(target / 'CHECKS.json'), published_profile=str(target / 'PROFILE.json'))
        return assignment, profile, descriptor, target

    def dedicated_fixture(self):
        profile = deepcopy(self.profile)
        profile['interpreter'] = '/owned/runtime-acceptance/python'
        profile['prefix_cache_directory'] = '/owned/dedicated-cache'
        root = self.base / 'dedicated'
        path = root / 'dry' / 'arbitrary-candidate'
        resolved = H._resolved_request(self.assignment, profile)
        save(path / 'RESOLVED_REQUEST.json', resolved)
        runtime = dict(executable='/owned/runtime/bin/python3', version='3.11.0rc1 (saved build)',
                       constructor='PASS_UNCHANGED_GUARD', guard_modified=False)
        adapter = dict(self.adapter, runtime=runtime)
        save(path / 'adapter/DRY_CHECK.json', adapter)
        observed = dict(self.readiness, python='/usr/bin/python3', python_version='3.12.checker-only',
                        route=resolved['route'], adapter_result=adapter,
                        resolved_request_sha256=H._sha(path / 'RESOLVED_REQUEST.json'),
                        command=[profile['interpreter'], '-B', 'adapter.py', '--mode', 'dry-check'])
        save(path / 'READINESS.json', observed)
        save(root / 'profile.json', profile)
        capability = dict(host_id=self.host, method=self.assignment['method'], job_id=self.assignment['job_id'],
            source_commit=self.assignment['source_manifest']['commit'], request_sha256=self.assignment['request_sha256'],
            route_kind=self.assignment['request']['route']['kind'], route_commit='b' * 40,
            profile_sha256=H._sha(root / 'profile.json'), readiness_sha256=H._sha(path / 'READINESS.json'),
            resolved_request_sha256=observed['resolved_request_sha256'], status=observed['status'],
            source_files_sha256=self.assignment['source_manifest']['files'],
            input_hashes=[dict(row, observed_sha256=row['sha256']) for row in self.assignment['input_hashes']],
            scientific_evaluation_calls=0, queue_operations=0, source_and_ast_guards_modified=False,
            runtime_guard=runtime, interpreter_invocation=profile['interpreter'],
            actual_executable=runtime['executable'], actual_executable_realpath='/owned/runtime/bin/python3.11',
            interpreter_launcher_sha256='1' * 64, actual_binary_sha256='2' * 64,
            version_info=[3, 11, 0, 'candidate', 1],
            route_views=dict(source=dict(path=profile['root'], head='a' * 40, tracked_dirty=[]),
                             route=dict(path='/owned/fixed-route', head='b' * 40, tracked_dirty=[])))
        save(root / 'HOST_METHOD_READINESS.json', capability)
        descriptor = dict(kind='fixed_git', root=str(root), host_method_readiness=str(root / 'HOST_METHOD_READINESS.json'),
                          profile=str(root / 'profile.json'))
        base_profile = dict(self.profile, work_dir='/owned/common-spool', slots=4,
            route_profiles=[dict(route_kind='saved_route_kind', view_commit='b' * 40, profile=profile)])
        return base_profile, descriptor, capability, root

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rpe-host-eligibility-')
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.host = 'arbitrary-host'
        self.profile = dict(host_id=self.host, root='/owned/fixed-source', interpreter='/owned/venv/bin/python',
            route_views={'b' * 40: '/owned/fixed-route'}, dp_workers=3, prefix_cache_directory='/owned/cache')
        request = dict(request_id='arbitrary-job', method='ARBITRARY_METHOD', configuration_id='arbitrary-configuration',
            N_X=[3, 2], N_Y=[4, 2], cost=dict(numerator=9, denominator=1), origin=dict(kind='EXISTING_RULE'),
            route=dict(kind='saved_route_kind', view_commit='b' * 40, view='/controller/view',
                       interpreter='/controller/python', templates=[]), expected=dict(contract='fixed'),
            optimizer_source='a' * 40, execution_profile=dict(dp_workers=3, prefix_checkpoint_rounds=[1]))
        self.assignment = dict(job_id=request['request_id'], method=request['method'], request=request,
            request_sha256=digest(request), source_manifest=dict(commit='a' * 40, files={'src/kernel.py': 'd' * 64}),
            input_hashes=[dict(commit='c' * 40, path='saved/input.json', sha256='e' * 64)])
        self.resolved = H._resolved_request(self.assignment, self.profile)
        self.root = self.base / 'evidence' / self.host
        self.cell = self.root / 'dry' / 'arbitrary-candidate'
        save(self.cell / 'RESOLVED_REQUEST.json', self.resolved)
        self.adapter = dict(status='DRY_CHECK_PASS', mode='dry-check', request_id=request['request_id'], route='saved_route_kind')
        save(self.cell / 'adapter/DRY_CHECK.json', self.adapter)
        self.readiness = dict(status='REAL_HOST_DRY_CHECK_PASS', host_id=self.host, job_id=self.assignment['job_id'],
            source=self.assignment['source_manifest']['commit'], request_sha256=self.assignment['request_sha256'],
            input_hashes=self.assignment['input_hashes'], route=self.resolved['route'],
            scientific_evaluation_calls=0, queue_operations=0,
            resolved_request_sha256=H._sha(self.cell / 'RESOLVED_REQUEST.json'),
            command=[self.profile['interpreter'], '-B', 'adapter.py', '--mode', 'dry-check'],
            adapter_result=self.adapter, python_version='3.12.saved-observation')
        save(self.cell / 'READINESS.json', self.readiness)
        self.facts = dict(python=self.profile['interpreter'], python_version=self.readiness['python_version'],
            source_head=dict(returncode=0, stdout='a' * 40 + '\n'),
            source_files={'src/kernel.py': dict(expected='d' * 64, actual='d' * 64, match=True)},
            route_views={'b' * 40: dict(path='/owned/fixed-route', realpath='/owned/fixed-route',
                head=dict(returncode=0, stdout='b' * 40 + '\n'), tracked_source_diff=dict(returncode=0, stdout=''))})
        save(self.root / 'HOST_FACTS.json', self.facts)
        self.descriptor = dict(kind='received', root=str(self.root),
            manifest=str(self.base / 'TRANSFER_MANIFEST.json'), receipt=str(self.base / 'RETURN_RECEIPT.json'),
            manifest_root=str(self.base / 'evidence'))
        self.receive()

    def receive(self):
        manifest = {path.relative_to(self.base / 'evidence').as_posix(): H._sha(path)
                    for path in self.root.rglob('*') if path.is_file()}
        save(Path(self.descriptor['manifest']), manifest)
        save(Path(self.descriptor['receipt']), dict(transfer_manifest_sha256=H._sha(self.descriptor['manifest']),
            verified_transferred_files=len(manifest)))

    def derive(self, assignment=None, profile=None, descriptor=None):
        policy = H.derive_host_eligibility([assignment or self.assignment], [profile or self.profile],
            {self.host: descriptor or self.descriptor})
        return policy['jobs'][self.assignment['job_id']]

    def test_arbitrary_method_and_candidate_are_eligible_from_bound_evidence(self):
        row = self.derive()
        self.assertEqual(row['eligible_host_ids'], [self.host])
        self.assertEqual(row['hosts'][self.host]['status'], 'PASS')
        self.assertTrue(all(v == 'MATCH' for v in row['hosts'][self.host]['dimensions'].values()))

    def test_same_configuration_with_new_xy_distribution_is_unchecked(self):
        changed = deepcopy(self.assignment)
        changed['request']['N_Y'][0] += 1
        changed['request_sha256'] = digest(changed['request'])
        self.assertEqual(self.derive(changed)['hosts'][self.host]['status'], 'UNCHECKED')

    def test_changed_source_or_input_hash_is_unchecked(self):
        for part in ('source', 'input'):
            changed = deepcopy(self.assignment)
            if part == 'source':
                changed['source_manifest']['files']['src/kernel.py'] = 'f' * 64
            else:
                changed['input_hashes'][0]['sha256'] = 'f' * 64
            self.assertEqual(self.derive(changed)['hosts'][self.host]['status'], 'UNCHECKED')

    def test_changed_route_or_runtime_requirement_is_unchecked(self):
        changed = deepcopy(self.profile)
        changed['route_views']['b' * 40] = '/owned/new-route'
        self.assertEqual(self.derive(profile=changed)['hosts'][self.host]['status'], 'UNCHECKED')
        changed = deepcopy(self.profile)
        changed['runtime'] = dict(version='3.12.changed-in-place')
        self.assertEqual(self.derive(profile=changed)['hosts'][self.host]['status'], 'UNCHECKED')

    def test_bound_failure_remains_fail(self):
        failed = deepcopy(self.readiness)
        failed.update(status='REAL_HOST_DRY_CHECK_MISSING_OR_INVALID_INPUT', error='fixed runtime contract failed')
        save(self.cell / 'READINESS.json', failed)
        self.receive()
        row = self.derive()
        self.assertEqual(row['eligible_host_ids'], [])
        self.assertEqual(row['hosts'][self.host]['status'], 'FAIL')
        self.assertIn('fixed runtime contract failed', row['hosts'][self.host]['reasons'])

    def test_missing_receipt_or_resolved_request_is_unchecked(self):
        Path(self.descriptor['receipt']).unlink()
        self.assertEqual(self.derive()['hosts'][self.host]['status'], 'UNCHECKED')
        self.receive()
        (self.cell / 'RESOLVED_REQUEST.json').unlink()
        self.assertEqual(self.derive()['hosts'][self.host]['status'], 'UNCHECKED')

    def test_tampered_received_readiness_is_unchecked(self):
        changed = deepcopy(self.readiness)
        changed['source'] = 'f' * 40
        save(self.cell / 'READINESS.json', changed)
        self.assertEqual(self.derive()['hosts'][self.host]['status'], 'UNCHECKED')

    def test_local_checker_version_does_not_replace_adapter_runtime(self):
        runtime = dict(executable=self.profile['interpreter'], version='3.11.adapter-actual')
        raw = dict(profile=self.profile, runtime=runtime, source=self.assignment['source_manifest'],
            source_head='a' * 40, input_hashes=self.assignment['input_hashes'],
            request_sha256=self.assignment['request_sha256'],
            views=[dict(path='/owned/fixed-route', head='b' * 40, tracked_changes={})])
        cache = self.base / 'DRY_CACHE.json'
        save(cache, dict(methods={'anything': dict(result_sha256=H._sha(self.cell / 'READINESS.json'), fingerprint=raw)}))
        descriptor = dict(kind='local', root=str(self.root), runtime_cache=str(cache))
        row = self.derive(descriptor=descriptor)
        self.assertEqual(row['hosts'][self.host]['status'], 'PASS')
        self.assertEqual(row['hosts'][self.host]['current_host_fingerprint']['runtime']['version'], '3.11.adapter-actual')
        operational = dict(self.profile, work_dir='/new/owned/spool', slots=12)
        relocated = self.derive(profile=operational, descriptor=descriptor)
        self.assertEqual(relocated['hosts'][self.host]['status'], 'PASS')
        self.assertEqual(row['hosts'][self.host]['current_host_fingerprint_sha256'],
                         relocated['hosts'][self.host]['current_host_fingerprint_sha256'])

    def test_actual_runtime_version_change_at_same_path_changes_poll_fingerprint(self):
        raw = dict(profile=self.profile, runtime=dict(executable=self.profile['interpreter'], version='3.11.first'),
            source=self.assignment['source_manifest'], source_head='a' * 40,
            input_hashes=self.assignment['input_hashes'], request_sha256=self.assignment['request_sha256'],
            views=[dict(path='/owned/fixed-route', head='b' * 40, tracked_changes={})])
        with patch('trottertracks.rpe_plan_readiness.dry_check_fingerprint', return_value=raw):
            first = H.current_host_fingerprint(self.assignment, self.profile)
        changed = deepcopy(raw)
        changed['runtime']['version'] = '3.11.changed'
        with patch('trottertracks.rpe_plan_readiness.dry_check_fingerprint', return_value=changed):
            second = H.current_host_fingerprint(self.assignment, self.profile)
        self.assertNotEqual(digest(first), digest(second))

    def test_missing_or_malformed_evidence_is_unchecked(self):
        row = H.derive_host_eligibility([self.assignment], [self.profile], {})['jobs'][self.assignment['job_id']]
        self.assertEqual(row['hosts'][self.host]['status'], 'UNCHECKED')
        save(self.cell / 'READINESS.json', {'malformed': True})
        self.receive()
        self.assertEqual(self.derive()['hosts'][self.host]['status'], 'UNCHECKED')

    def test_dedicated_launcher_profile_uses_adapter_runtime_and_keeps_common_profile(self):
        base_profile, descriptor, capability, root = self.dedicated_fixture()
        before = deepcopy(base_profile)
        with patch.object(H, '_receipt_check', return_value=dict(status='MATCH', kind='DIRECT_LOCAL_OBSERVATION')):
            row = self.derive(profile=base_profile, descriptor=descriptor)
        cell = row['hosts'][self.host]
        self.assertEqual(cell['status'], 'PASS')
        self.assertEqual(cell['execution_profile']['interpreter'], capability['interpreter_invocation'])
        runtime = cell['current_host_fingerprint']['runtime']
        self.assertEqual(runtime['version'], '3.11.0rc1 (saved build)')
        self.assertEqual(runtime['executable'], capability['actual_executable'])
        self.assertEqual(runtime['launcher_sha256'], '1' * 64)
        self.assertEqual(runtime['actual_binary_sha256'], '2' * 64)
        self.assertEqual(base_profile, before)
        from trottertracks.rpe_host_profiles import select_host_profile
        other = deepcopy(self.assignment)
        other['request']['route']['view_commit'] = 'unrelated-route'
        self.assertEqual(select_host_profile(base_profile, other)['interpreter'], self.profile['interpreter'])

    def test_common_runtime_fallback_or_bad_capability_hash_is_unchecked(self):
        base_profile, descriptor, capability, root = self.dedicated_fixture()
        with patch.object(H, '_receipt_check', return_value=dict(status='MATCH', kind='DIRECT_LOCAL_OBSERVATION')):
            self.assertEqual(self.derive(profile=self.profile, descriptor=descriptor)['hosts'][self.host]['status'], 'UNCHECKED')
            capability['readiness_sha256'] = '0' * 64
            save(root / 'HOST_METHOD_READINESS.json', capability)
            self.assertEqual(self.derive(profile=base_profile, descriptor=descriptor)['hosts'][self.host]['status'], 'UNCHECKED')

    def test_actual_launcher_fingerprint_hashes_launcher_and_actual_binary(self):
        profile = dict(self.profile, interpreter=str(self.base / 'launcher'))
        Path(profile['interpreter']).write_text('#!/bin/sh\nexec python3 "$@"\n')
        raw = dict(profile=profile, runtime=dict(executable=sys.executable, version=sys.version),
            source=self.assignment['source_manifest'], source_head='a' * 40,
            input_hashes=self.assignment['input_hashes'], request_sha256=self.assignment['request_sha256'],
            views=[dict(path='/owned/fixed-route', head='b' * 40, tracked_changes={})])
        with patch('trottertracks.rpe_plan_readiness.dry_check_fingerprint', return_value=raw):
            actual = H.current_host_fingerprint(self.assignment, profile)
        self.assertEqual(actual['runtime']['launcher_sha256'], H._sha(profile['interpreter']))
        self.assertEqual(actual['runtime']['actual_binary_sha256'], H._sha(Path(sys.executable).resolve()))

    def test_published_portability_adapter_keeps_original_source_request_and_resolved_copy(self):
        assignment, profile, descriptor, target = self.romutail_fixture()
        original = H._read(target / 'ORIGINAL_REQUEST.json')
        resolved = H._read(target / 'checks/pass01/RESOLVED_REQUEST.json')
        self.assertEqual(assignment['request'], original)
        self.assertEqual(H._resolved_request(assignment, profile), resolved)
        policy = H.derive_host_eligibility([assignment], [profile], {profile['host_id']: descriptor})
        cell = policy['jobs'][assignment['job_id']]['hosts'][profile['host_id']]
        self.assertEqual(cell['status'], 'PASS', cell['reasons'])
        self.assertEqual(cell['execution_profile']['adapter_manifest']['implementation_commit'],
                         'e37f6544dea3e021d3694264f7dae87add7ad30e')
        self.assertEqual(cell['current_host_fingerprint']['adapter_source']['commit'],
                         '09efb068f242d5d6ff88296df3e50283b2f951bf')
        self.assertEqual(cell['current_host_fingerprint']['source_manifest'], assignment['source_manifest'])

    def test_portability_adapter_mismatch_or_unbound_fallback_is_unchecked(self):
        assignment, profile, descriptor, target = self.romutail_fixture()
        for changed in (deepcopy(profile), H._read(target / 'PROFILE.json')):
            if changed.get('adapter_manifest'):
                changed['adapter_manifest']['files']['scripts/run_rpe_optimizer_q3_job.py'] = '0' * 64
            policy = H.derive_host_eligibility([assignment], [changed], {changed['host_id']: descriptor})
            self.assertEqual(policy['jobs'][assignment['job_id']]['hosts'][changed['host_id']]['status'], 'UNCHECKED')

    def test_tampered_portability_external_mapping_is_unchecked(self):
        assignment, profile, descriptor, target = self.romutail_fixture()
        save(target / 'checks/pass01/ROUTE_VIEWS.json', {'wrong-inner': '/wrong/path'})
        policy = H.derive_host_eligibility([assignment], [profile], {profile['host_id']: descriptor})
        self.assertEqual(policy['jobs'][assignment['job_id']]['hosts'][profile['host_id']]['status'], 'UNCHECKED')

    def test_required_runtime_guard_rejects_older_metadata_only_pass(self):
        assignment, profile, base = self.runtime_guard_fixture()
        descriptor = dict(kind='local', root=str(base / 'server2_dry_check'), runtime_cache=str(base / 'SERVER2_DRY_CACHE.json'))
        policy = H.derive_host_eligibility([assignment], [profile], {profile['host_id']: descriptor})
        cell = policy['jobs'][assignment['job_id']]['hosts'][profile['host_id']]
        self.assertEqual(cell['status'], 'UNCHECKED')
        self.assertTrue(any('precheck' in reason for reason in cell['reasons']))

    def test_saved_actual_runtime_guard_is_bound_to_the_poll_fingerprint(self):
        assignment, profile, base = self.runtime_guard_fixture()
        descriptor = dict(kind='local', root=str(base / 'server2_runtime_ast_v2'), runtime_cache=str(base / 'SERVER2_RUNTIME_AST_V2_DRY_CACHE.json'))
        policy = H.derive_host_eligibility([assignment], [profile], {profile['host_id']: descriptor})
        cell = policy['jobs'][assignment['job_id']]['hosts'][profile['host_id']]
        self.assertEqual(cell['status'], 'PASS', cell['reasons'])
        check = cell['current_host_fingerprint']['runtime_checks'][0]
        self.assertEqual(check['actual_ast_sha256'], check['expected_ast_sha256'])
        self.assertEqual(cell['execution_profile']['runtime_checks'], profile['runtime_checks'])

    def test_resigned_readiness_with_wrong_actual_ast_is_unchecked(self):
        assignment, profile, base = self.runtime_guard_fixture()
        target = self.base / 'wrong-ast'
        shutil.copytree(base / 'server2_runtime_ast_v2', target)
        observed = H._read(target / 'READINESS.json')
        observed['runtime_precheck']['checks'][0]['actual_ast_sha256'] = '0' * 64
        save(target / 'READINESS.json', observed)
        index = H._read(base / 'SERVER2_RUNTIME_AST_V2_DRY_CACHE.json')
        for row in index['methods'].values():
            row['result_sha256'] = H._sha(target / 'READINESS.json')
        cache = self.base / 'wrong-ast-index.json'
        save(cache, index)
        descriptor = dict(kind='local', root=str(target), runtime_cache=str(cache))
        policy = H.derive_host_eligibility([assignment], [profile], {profile['host_id']: descriptor})
        self.assertEqual(policy['jobs'][assignment['job_id']]['hosts'][profile['host_id']]['status'], 'UNCHECKED')

    def test_current_runtime_guard_failure_never_produces_a_claim_fingerprint(self):
        assignment, profile, base = self.runtime_guard_fixture()
        with patch('trottertracks.rpe_host_profiles.check_runtime_profile', return_value=dict(status='FAIL', checks=[])), \
             patch('trottertracks.rpe_plan_readiness.dry_check_fingerprint') as old_metadata:
            with self.assertRaisesRegex(ValueError, 'precheck'):
                H.current_host_fingerprint(assignment, profile)
            old_metadata.assert_not_called()


if __name__ == '__main__':
    unittest.main()
