#!/usr/bin/env python3
"""Install the local-only storage timer and cold-cache restore hook on RomuTail."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HOOK = '''            # Restore T7 delivery caches before the fixed normal preflight.
            if request['action'] in ('start', 'resume'):
                storage_scripts = pathlib.Path(config['state_directory']).parent / 'deployment' / 'romutail-storage-default-20261126' / 'scripts'
                sys.path.insert(0, str(storage_scripts))
                from rpe_romutail_cold_storage import preflight_with_cold_storage
                sys.path.pop(0)
                original_preflight = gateway.preflight
                if request['action'] == 'resume' and resume_archives is not None:
                    if preflight is None and not getattr(original_preflight, 'restores_preparation_archives', False):
                        original_preflight = resume_archives.preflight_with_archives(
                            config['state_directory'], original_preflight, RequestError)
                    if preflight is not None or original_preflight != gateway.normal_preflight:
                        original_preflight.restores_preparation_archives = True
                gateway.preflight = preflight_with_cold_storage(original_preflight, RequestError)
'''


def install_hook(central, deployment):
    gateway = central/'CLIENT_GATEWAY.py'
    raw = gateway.read_text()
    if 'from rpe_romutail_cold_storage import preflight_with_cold_storage' in raw:
        return
    anchor = '            result = gateway.handle(request)\n'
    if raw.count(anchor) != 1:
        raise ValueError('Existing normal client gateway differs; inspect before installing the hook')
    modified = raw.replace(anchor, HOOK+anchor)
    compile(modified, str(gateway), 'exec')
    backup = deployment/'CLIENT_GATEWAY.before-cold-storage.py'
    if not backup.exists():
        backup.write_text(raw)
    with tempfile.NamedTemporaryFile(mode='w',dir=central,prefix='.CLIENT_GATEWAY-',delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(modified)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if gateway.read_text() != raw:
            raise ValueError('Client gateway changed during installation')
        temporary.chmod(gateway.stat().st_mode & 0o777)
        os.replace(temporary, gateway)
    finally:
        temporary.unlink(missing_ok=True)


def install(repository, deployment):
    central=repository/'.rpe-central'
    python=repository/'.venv311/bin/python'
    src=deployment/'scripts'
    source=Path(__file__).resolve().parents[1]
    support=deployment/'src/trottertracks'
    support.mkdir(parents=True,exist_ok=True)
    for name in ('__init__.py','rpe_q3_coordinator.py','rpe_preparation_artifacts.py','rpe_dry_check_recovery.py'):
        original=source/'src/trottertracks'/name
        target=support/name
        if original.resolve() != target.resolve():
            shutil.copyfile(original,target)
    helper=src/'recover_stopped_rpe_run.py'
    if not helper.exists():
        shutil.copyfile(Path(__file__).with_name('recover_stopped_rpe_run.py'),helper)
    install_hook(central, deployment)
    unitdir=Path.home()/'.config/systemd/user'
    unitdir.mkdir(parents=True,exist_ok=True)
    units={
        'rpe-romutail-disk-maintenance.service': f'''[Unit]
Description=RomuTail local Git and cold-storage maintenance

[Service]
Type=oneshot
Environment="PYTHONPATH={deployment/'src'}:{src}"
ExecStart="{python}" -B "{src/'run_romutail_disk_maintenance.py'}" --repository "{repository}" --state-directory "{central/'state'}"
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=6
TimeoutStartSec=infinity
''',
        'rpe-romutail-disk-maintenance.timer': '''[Unit]
Description=Regular proactive RomuTail disk maintenance

[Timer]
OnBootSec=2min
OnUnitInactiveSec=5min
AccuracySec=30s
Unit=rpe-romutail-disk-maintenance.service

[Install]
WantedBy=timers.target
'''}
    for name, content in units.items():
        local=deployment/name
        local.write_text(content)
        shutil.copyfile(local, unitdir/name)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','enable','--now','rpe-romutail-disk-maintenance.timer'],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    args=parser.parse_args()
    repository=args.repository.resolve()
    deployment=repository/'.rpe-central/deployment/romutail-storage-default-20261126'
    install(repository, deployment)

if __name__=='__main__':main()
