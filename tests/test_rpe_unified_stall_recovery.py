"""Technical regression only: no scientific jobs or remote host calls."""
import json
import signal
from pathlib import Path
from unittest.mock import Mock, patch

from trottertracks import rpe_optimizer_concurrent as C


def test_hold_fingerprint_and_retry_classification():
    def decision(reason):
        return dict(request_sha256='REQ', source_manifest_sha256='SRC', eligible_host_ids=[],
                    host_decisions={'worker':dict(status='HOLD',reasons=[reason])})
    first = decision('warning: nesting too deep\nValueError: Fixed Q3 configuration differs')
    second = decision('different traceback\nValueError: Fixed Q3 configuration differs')
    assert C.readiness_hold_signature(first) == C.readiness_hold_signature(second)
    assert C.readiness_retry_seconds(first) > C.LIVE_READINESS_RETRY_SECONDS
    transient = decision('SSH connection timed out')
    assert C.readiness_hold_signature(first) != C.readiness_hold_signature(transient)
    assert C.readiness_retry_seconds(transient) == C.LIVE_READINESS_RETRY_SECONDS
    second['source_manifest_sha256'] = 'NEW'
    assert C.readiness_hold_signature(first) != C.readiness_hold_signature(second)


def test_local_transport_rollover_drains_only_matching_daemon(tmp_path):
    profile = tmp_path/'LOCAL_WORKER.json'
    profile.write_text('{}')
    daemon = tmp_path/'LOCAL_DAEMON.json'
    daemon.write_text(json.dumps(dict(pid=123,profile=str(profile))))
    executor = C.DistributedExecutor.__new__(C.DistributedExecutor)
    executor.worker_script = tmp_path/'new/scripts/run_rpe_q3_worker.py'
    executor.controller_url = 'http://127.0.0.1:8763'
    executor.log = Mock()
    executor._local_daemon_alive = Mock(return_value=False)
    old_args = ['python','/old/scripts/run_rpe_q3_worker.py','--profile',str(profile),'--controller',executor.controller_url]
    with patch.object(Path,'read_bytes',autospec=True,side_effect=lambda p:
            ('\0'.join(old_args)+'\0').encode() if str(p)=='/proc/123/cmdline'
            else p.read_text().encode()), patch.object(C.os,'kill') as kill, \
            patch.object(C.subprocess,'Popen') as start:
        executor._start_local_worker(profile)
        executor._start_local_worker(profile)
        kill.assert_called_once_with(123,signal.SIGTERM)
        start.assert_not_called()
        assert executor.local_worker_pid == 123



def test_relay_version_rollover_waits_for_original_owner(tmp_path):
    import fcntl
    from types import SimpleNamespace
    directory = tmp_path/'distributed/relay'
    directory.mkdir(parents=True)
    path = directory/'CONFIG.json'
    old = tmp_path/'old/scripts/run_rpe_q3_relay.py'
    new = tmp_path/'new/scripts/run_rpe_q3_relay.py'
    args = ['python', '-B', str(old), '--config', str(path), '--run']
    settings = {'workers': []}
    path.write_text(json.dumps(dict(settings, state_directory=str(directory), external_tunnel=True,
                                    controller={'local_port': 8763})))
    (directory/'LAUNCH.json').write_text(json.dumps(dict(pid=123, command=args)))
    executor = C.DistributedExecutor.__new__(C.DistributedExecutor)
    executor.run_dir = tmp_path
    executor.root = tmp_path
    executor.distributed = dict(relay=settings, relay_script=str(new))
    executor.state = SimpleNamespace()
    executor.log = Mock()
    with (directory/'RELAY.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = Path.read_bytes
        with patch.object(Path,'read_bytes',autospec=True,side_effect=lambda p:
                          ('\0'.join(args)+'\0').encode() if str(p)=='/proc/123/cmdline' else original(p)), \
             patch.object(C.subprocess,'Popen') as start:
            executor.start_controller_relay(8763)
            assert (directory/'STOP').is_file()
            assert executor.state.relay_release_rollover
            start.assert_not_called()
    executor.rpc_server = SimpleNamespace(server_address=('127.0.0.1',8763))
    with patch.object(C.subprocess,'Popen',return_value=Mock(pid=456)):
        executor._check_relay_process()
    assert not (directory/'STOP').exists()
    assert not executor.state.relay_release_rollover
    assert json.loads((directory/'LAUNCH.json').read_text())['pid'] == 456
