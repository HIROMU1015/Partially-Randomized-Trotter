"""Once-only COST22 -> ACCOUNT -> separately committed read-only completion."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import pf_common_run_contract as c


def run(private):
    private = Path(private).resolve()
    c.save(private/'RECOVERY_PIPELINE_ENTERED.json', dict(retry=False, stages=['COST', 'ACCOUNT', 'READ_ONLY_FINAL']))
    for name in ('cost', 'account'):
        c.require(not (private/name.upper()).exists(), 'NEVER_REENTER_STAGE')
        with (private/(name+'_stdout.log')).open('x') as out, (private/(name+'_stderr.log')).open('x') as err:
            result = subprocess.run([sys.executable, '-B', str(c.ROOT/'scripts/h6_pf_recovery_execution.py'),
                name, '--private', str(private)], cwd=c.ROOT, stdout=out, stderr=err)
        c.save(private/(name+'_external_exit.json'), dict(exit_code=result.returncode, child_reaped=True, launched_once=True, retry=False))
        c.require(result.returncode == 0 and c.read(private/name.upper()/'completion.json')['status'] == 'PASS', 'STAGE_FAILED:'+name)
    c.save(private/'RECOVERY_SCIENTIFIC_SUCCESS.json', dict(status='PASS', old_grant_reused=False, new_stages=['COST', 'ACCOUNT']))
    # Only read-only reporting implementation may be supplied while acquisition
    # runs. Scientific SOURCE remains sealed and unchanged.
    ready = private/'READ_ONLY_COMPLETION_READY.json'
    while not ready.exists(): time.sleep(2)
    spec = c.read(ready)
    for e in spec['files']:
        c.require(c.sha(e['path']) == e['sha256'], 'READ_ONLY_SOURCE_CHANGED')
        c.require(Path(e['path']).read_bytes() == c.gitbytes(spec['SOURCE'], e['relative_path']), 'READ_ONLY_COMMITTED_SOURCE')
    with (private/'final_stdout.log').open('x') as out, (private/'final_stderr.log').open('x') as err:
        result = subprocess.run([sys.executable, '-B', str(c.ROOT/'scripts/finish_h6_pf_recovery_results.py'),
            '--private', str(private)], cwd=c.ROOT, stdout=out, stderr=err)
    c.save(private/'final_external_exit.json', dict(exit_code=result.returncode, child_reaped=True, scientific_calls=0))
    c.require(result.returncode == 0, 'READ_ONLY_FINAL_FAILED_NO_SCIENTIFIC_RETRY')
    c.save(private/'RECOVERY_PIPELINE_COMPLETE.json', dict(status='LOCAL_COMMIT_MANDATORY_STOP', retry=False, GitHub_push=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--private', required=True); a = p.parse_args()
    try: run(a.private)
    except BaseException as exc:
        path = Path(a.private)/'RECOVERY_PIPELINE_STOP.json'
        if not path.exists(): c.save(path, dict(status='MANDATORY_STOP', reason=str(exc), retry=False))
        raise
