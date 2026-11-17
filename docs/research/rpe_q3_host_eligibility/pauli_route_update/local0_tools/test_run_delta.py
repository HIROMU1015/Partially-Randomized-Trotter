"""Fake SSH/SCP, Git and checker processes only; no real host or scientific call."""
import base64
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('_changed_cell_delta', Path(__file__).with_name('run_delta.py'))
D = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(D)
SOURCE = '28d976b99cdf8329adcc1b79327e329e947ef774'
ROUTE = '33e324b43690caeaa353ecdb9e3d42f3139198b6'


class FakeSSH:
    def __init__(self, bundle):
        self.bundle = bundle;self.commands = [];self.checkers = [];self.host_payloads = [];self.ssh_aliases = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command));out = b''
        if command[0] == 'scp':
            shutil.copyfile(self.bundle, command[-1])
        elif command[0] == 'ssh':
            self.assert_ssh_argv(command)
            alias = command[4];self.ssh_aliases.append(alias)
            encoded_source, raw = kwargs['input'].split(b'\n', 1)
            assert base64.b64decode(encoded_source) == Path(D.__file__).read_bytes()
            payload = json.loads(raw)
            try:
                if payload['action'] == 'host':
                    self.host_payloads.append(deepcopy(payload));result = D.host_action(payload)
                else:
                    result = D.receive_action(payload)
            except Exception as error:
                result = dict(ok=False, error=str(error))
            out = json.dumps(result).encode()
        elif command[0] == 'git':
            location = Path(command[2]);args = command[3:]
            if args[:2] == ['rev-parse', 'HEAD']:
                out = (location / 'ARTIFICIAL_HEAD').read_bytes()
            elif args[:2] == ['worktree', 'add']:
                target = Path(args[-2]);target.mkdir(parents=True)
                (target / 'ARTIFICIAL_HEAD').write_text(args[-1])
            elif args[0] not in ('cat-file', 'sparse-checkout', 'checkout', 'diff', 'fetch'):
                raise AssertionError('Unexpected Git command: ' + repr(command))
        elif len(command) > 2 and command[1] == '-c':
            out = json.dumps(dict(executable=command[0], version='ARTIFICIAL PYTHON VERSION', platform='ARTIFICIAL')).encode()
        elif len(command) > 2 and command[2].endswith('/scripts/check_rpe_q3_host_readiness.py'):
            self.checkers.append(list(command))
            assignment = json.loads(Path(command[command.index('--assignment') + 1]).read_text())
            profile = json.loads(Path(command[command.index('--profile') + 1]).read_text())
            helper_path = Path(command[2]).parents[1] / D.PROFILE_HELPER
            spec = importlib.util.spec_from_file_location('_artificial_runtime_helper', helper_path)
            helper = importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
            report = helper.check_runtime_profile(profile)
            target = Path(command[command.index('--output') + 1]);target.mkdir(parents=True)
            request = deepcopy(assignment['request']);request['route'].update(interpreter=profile['interpreter'],
                view=profile['route_views'][ROUTE])
            resolved = D.encode(request);(target / 'RESOLVED_REQUEST.json').write_bytes(resolved)
            record = dict(status='REAL_HOST_DRY_CHECK_PASS', fixture='ARTIFICIAL_FAKE_CHECKER',
                host_id=profile['host_id'], job_id=assignment['job_id'], source=SOURCE,
                python=profile['interpreter'], python_version='ARTIFICIAL PYTHON VERSION',
                runtime_precheck=report,
                request_sha256=assignment['request_sha256'], input_hashes=assignment['input_hashes'],
                route=request['route'], command=[command[0], '-B', '/ARTIFICIAL/adapter', '--mode', 'dry-check'],
                resolved_request_sha256=D.sha(resolved), scientific_evaluation_calls=0, queue_operations=0)
            (target / 'READINESS.json').write_bytes(D.encode(record));(target / 'ADAPTER.log').write_bytes(b'')
            (target / 'adapter').mkdir();(target / 'adapter/DRY_CHECK.json').write_bytes(D.encode(dict(status='DRY_CHECK_PASS')))
        else:
            raise AssertionError('Forbidden subprocess: ' + repr(command))
        return subprocess.CompletedProcess(command, 0, stdout=out, stderr=b'')

    @staticmethod
    def assert_ssh_argv(command):
        assert command[:4] == ['ssh', '-T', '-o', 'BatchMode=yes']
        assert command[4] in ('qg1', 'RomuTail', 'qg2')
        assert len(command) == 6 and len(command[5]) < 1000  # Code is stdin, not a Windows-size command argument.


class ChangedCellDeltaTests(unittest.TestCase):
    def configuration(self, root):
        assets = root / 'assets';assets.mkdir()
        (assets / 'checker.py').write_text('# ARTIFICIAL FAKE CHECKER; never executed\n')
        (assets / 'profiles.py').write_text('''def select_host_profile(profile, assignment):
    return profile

def check_runtime_profile(profile):
    status = profile.get('fixture_runtime_status', 'PASS')
    runtime = dict(executable=profile.get('fixture_actual_interpreter', profile['interpreter']),
                   version='ARTIFICIAL PYTHON VERSION', version_info=[3, 11, 0, 'candidate', 1],
                   platform='ARTIFICIAL')
    if runtime['executable'] != profile['interpreter']:
        runtime.update(interpreter_invocation=profile['interpreter'], launcher_sha256='a' * 64,
                       actual_executable_realpath=runtime['executable'], actual_binary_sha256='b' * 64)
    return dict(status=status, runtime=runtime, checks=[dict(kind='FIXED_FUNCTION_AST',
        status=status, expected_ast_sha256='c' * 64,
        actual_ast_sha256=('c' if status == 'PASS' else 'd') * 64)])
''')
        sources = {D.CHECKER: 'assets/checker.py', D.PROFILE_HELPER: 'assets/profiles.py'}
        cells = []
        original = {}
        for host, alias in [('server1', 'qg1'), ('local', 'RomuTail')]:
            repo = root / host;repo.mkdir();scientific = repo / 'fixed28d';scientific.mkdir()
            (scientific / 'ARTIFICIAL_HEAD').write_text(SOURCE)
            hashes = {}
            for i in range(8):
                p = scientific / ('ARTIFICIAL_SOURCE_' + str(i));raw = ('ARTIFICIAL_FIXED_' + str(i)).encode()
                p.write_bytes(raw);hashes[p.name] = D.sha(raw);original[str(p)] = raw
            profile = dict(host_id=host, root=str(scientific), interpreter=str(repo / 'ARTIFICIAL_PYTHON'),
                           route_views={ROUTE: str(repo / 'new-route-view')}, dp_workers=1)
            if host == 'server1':
                profile['fixture_actual_interpreter'] = str(repo / 'ARTIFICIAL_ACTUAL_BINARY')
            request = dict(request_id='ARTIFICIAL_PAULI_JOB', method='PR_PAULI', configuration_id='ARTIFICIAL_CFG',
                route=dict(kind='pauli_b_common_entry', view_commit=ROUTE), N_X=[1], N_Y=[1])
            assignment = dict(job_id=request['request_id'], request=request, method='PR_PAULI', request_sha256='ARTIFICIAL',
                              source_manifest=dict(commit=SOURCE, files=hashes), input_hashes=[])
            p = assets / (host + '.json');p.write_bytes(D.encode(profile))
            a = assets / (host + '-assignment.json');a.write_bytes(D.encode(assignment))
            sources['profiles/' + host + '.json'] = str(p.relative_to(root))
            sources['assignments/' + host + '/PR_PAULI.json'] = str(a.relative_to(root))
            cells.append(dict(host_id=host, ssh_alias=alias, repository=str(repo), transport_python=profile['interpreter'],
                run_directory=str(repo / 'new-owned-delta-run'), profile='profiles/' + host + '.json',
                assignments=['assignments/' + host + '/PR_PAULI.json']))
        config = dict(schema=D.SCHEMA, mode='dry-check', source_files=sources, hosts=cells,
                      **{'return': dict(ssh_alias='qg2', python='python3', directory=str(root / 'server2-new-return'))})
        path = root / 'config.json';path.write_bytes(D.encode(config))
        return path, original

    def test_fake_ssh_runs_only_explicit_two_cells_and_hash_verifies_return(self):
        with tempfile.TemporaryDirectory(prefix='rpe-artificial-delta-') as temporary:
            root = Path(temporary);config, original = self.configuration(root);bundle = root / 'delta.zip'
            packed = D.package(config, bundle);self.assertEqual(packed['changed_cells'], 2)
            transport = FakeSSH(bundle)
            with patch.object(D.subprocess, 'run', side_effect=transport), redirect_stdout(io.StringIO()):
                summary, receipt = D.prepare('qg2:/ARTIFICIAL/delta.zip', packed['bundle_sha256'], root / 'local0-new-run')
                D.return_only(root / 'local0-new-run')  # Retry return only, with no second host dry-check.
            self.assertTrue(receipt['ok']);self.assertEqual(len(transport.checkers), 2)
            self.assertEqual({row['host_id'] for row in summary['rows']}, {'server1', 'local'})
            self.assertEqual({row['method'] for row in summary['rows']}, {'PR_PAULI'})
            self.assertEqual(set(transport.ssh_aliases), {'qg1', 'RomuTail', 'qg2'})
            self.assertTrue(all(len(p['host']['assignments']) == 1 for p in transport.host_payloads))
            for payload in transport.host_payloads:
                self.assertIn(D.CHECKER, payload['files']);self.assertIn(D.PROFILE_HELPER, payload['files'])
            returned = root / 'server2-new-return'
            manifest = json.loads((returned / 'TRANSFER_MANIFEST.json').read_text())
            ack = json.loads((returned / 'RETURN_RECEIPT.json').read_text())
            self.assertEqual(ack['transfer_manifest_sha256'], D.sha((returned / 'TRANSFER_MANIFEST.json').read_bytes()))
            self.assertEqual(ack['verified_transferred_files'], len(manifest))
            self.assertTrue(all(D.sha((returned / path).read_bytes()) == expected for path, expected in manifest.items()))
            facts = json.loads((returned / 'server1/HOST_FACTS.json').read_text())
            runtime = facts['actual_runtime']
            self.assertNotEqual(runtime['executable'], facts['python'])
            self.assertEqual(runtime['interpreter_invocation'], facts['python'])
            self.assertEqual(runtime['launcher_sha256'], 'a' * 64)
            self.assertEqual(runtime['actual_binary_sha256'], 'b' * 64)
            self.assertEqual(facts['runtime_checks'][0]['status'], 'PASS')
            self.assertEqual(json.loads((returned / 'server1/RUNTIME_CHECKS.json').read_text())['runtime'], runtime)
            readiness = json.loads((returned / 'server1/dry/PR_PAULI/READINESS.json').read_text())
            self.assertEqual(readiness['runtime_precheck']['runtime'], runtime)
            self.assertEqual(readiness['runtime_precheck']['checks'], facts['runtime_checks'])
            self.assertFalse(any(method in path for path in manifest for method in ('FULL_PF', 'TRUNCATED_PF', 'PR_DF')))
            for path, raw in original.items():self.assertEqual(Path(path).read_bytes(), raw)
            result = dict(status='PASS', scope='FAKE_SSH_GIT_AND_CHECKER_ONLY', changed_cells=2,
                checker_calls=2, methods=['PR_PAULI'], old_pass_cells_rerun=0, source_files_unchanged=16,
                transfer_files_verified=len(manifest), return_retry_additional_checker_calls=0,
                SSH_aliases=sorted(set(transport.ssh_aliases)), runtime_checks='ARTIFICIAL_AST_ONLY_PASS',
                launcher_and_actual_binary_bound=True, real_SSH_calls=0, scientific_calls=0)
            if os.environ.get('RPE_DELTA_RUNNER_ACCEPTANCE'):
                Path(os.environ['RPE_DELTA_RUNNER_ACCEPTANCE']).write_bytes(D.encode(result))

    def test_existing_different_route_head_is_preserved_without_checker(self):
        with tempfile.TemporaryDirectory(prefix='rpe-artificial-delta-') as temporary:
            root = Path(temporary);config, original = self.configuration(root);bundle = root / 'delta.zip'
            D.package(config, bundle);manifest, files = D.unpack(bundle.read_bytes());host = manifest['hosts'][0]
            view = root / 'server1/new-route-view';view.mkdir();marker = view / 'ARTIFICIAL_HEAD';marker.write_text('WRONG_HEAD')
            needed = [D.CHECKER, D.PROFILE_HELPER, host['profile']] + host['assignments']
            payload = dict(action='host', host=host, files={p: base64.b64encode(files[p]).decode() for p in needed},
                           hashes={p: manifest['files'][p] for p in needed})
            transport = FakeSSH(bundle)
            with patch.object(D.subprocess, 'run', side_effect=transport):
                with self.assertRaisesRegex(ValueError, 'HEAD differs; preserved'):
                    D.host_action(payload)
            self.assertEqual(marker.read_text(), 'WRONG_HEAD');self.assertEqual(transport.checkers, [])
            for path, raw in original.items():self.assertEqual(Path(path).read_bytes(), raw)

    def test_windows_subprocesses_use_no_window_creation_flag(self):
        with patch.object(D.os, 'name', 'nt'), patch.object(D.subprocess, 'CREATE_NO_WINDOW', 0x08000000, create=True), \
             patch.object(D.subprocess, 'run', return_value=subprocess.CompletedProcess(['ssh'], 0, b'', b'')) as call:
            D.run(['ssh'], capture_output=True)
        self.assertEqual(call.call_args.kwargs['creationflags'], 0x08000000)

    def test_runtime_ast_mismatch_preserves_facts_without_checker(self):
        with tempfile.TemporaryDirectory(prefix='rpe-artificial-delta-') as temporary:
            root = Path(temporary);config, _ = self.configuration(root)
            path = root / 'assets/server1.json';profile = json.loads(path.read_text())
            profile['fixture_runtime_status'] = 'FAIL';path.write_bytes(D.encode(profile))
            bundle = root / 'delta.zip';D.package(config, bundle)
            manifest, files = D.unpack(bundle.read_bytes());host = manifest['hosts'][0]
            needed = [D.CHECKER, D.PROFILE_HELPER, host['profile']] + host['assignments']
            payload = dict(action='host', host=host, files={p: base64.b64encode(files[p]).decode() for p in needed},
                           hashes={p: manifest['files'][p] for p in needed})
            transport = FakeSSH(bundle)
            with patch.object(D.subprocess, 'run', side_effect=transport):
                result = D.host_action(payload)
            self.assertEqual(transport.checkers, [])
            self.assertEqual(result['result']['rows'][0]['status'], 'RUNTIME_PRECHECK_FAIL')
            self.assertEqual(result['result']['rows'][0]['checker_started'], False)
            raw = base64.b64decode(result['files']['server1/RUNTIME_CHECKS.json'])
            self.assertEqual(json.loads(raw)['checks'][0]['status'], 'FAIL')
            self.assertIn('server1/HOST_FACTS.json', result['files'])
            self.assertNotIn('server1/dry/PR_PAULI/READINESS.json', result['files'])


if __name__ == '__main__':
    unittest.main()
