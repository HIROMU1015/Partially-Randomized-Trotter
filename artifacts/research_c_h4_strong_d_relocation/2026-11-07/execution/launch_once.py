"""Launch the fixed scientific CLI once; record execution and stop own group on failure."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

FIXED = 'e6652ab841aae5ae700286bf1ab249ad4fb85461'
CONTRACT = 'docs/research/research_c_h4_strong_d_relocation_contract_unapproved_20261107.json'


def main():
    root = Path.cwd().resolve()
    record = root/'artifacts/research_c_h4_strong_d_relocation/2026-11-07/execution'
    assert root == Path(__file__).resolve().parents[4]
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() == FIXED
    assert (root/CONTRACT).read_bytes() == subprocess.check_output(['git','show',FIXED+':'+CONTRACT])
    sys.path.insert(0,str(root/'src'))
    from trottertracks import c_h4_strong_d_relocation_execution as runner
    c = runner.preflight(root, CONTRACT)
    def save(name, data):
        with (record/name).open('x') as stream:
            json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')
    memory = next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    source = runner.read(root/c['SOURCE'])
    save('preflight.json',dict(status='PASSIVE_PREFLIGHT_PASS',fixed_commit=FIXED,
        contract_sha256=runner.sha(root/CONTRACT),SOURCE_sha256=c['SOURCE_sha256'],
        source_files=len(source['files']),saved_inputs=len(c['saved_inputs']),
        protected_hashes=dict(source['files'],**c['saved_inputs']),arrays_decoded=0,
        fresh_output_exists=False,packages={k:importlib.metadata.version(k) for k in c['packages']},
        python=sys.version,affinity_cpus=len(os.sched_getaffinity(0)),MemAvailable_bytes=memory,
        load_average=list(os.getloadavg())))
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MPLCONFIGDIR=str(root/'.cache/matplotlib'),TMPDIR=str(root/'.cache/tmp'))
    (root/'.cache/tmp').mkdir(parents=True,exist_ok=True)
    out = root/c['fresh_output']
    started=time.monotonic(); reason=None
    with (record/'stdout.log').open('xb') as stdout, (record/'stderr.log').open('xb') as stderr:
        process=subprocess.Popen([sys.executable,'-B','scripts/run_research_c_h4_strong_d_relocation.py','--execute'],
            cwd=root,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        save('start.json',dict(fixed_commit=FIXED,runner_pid=process.pid,process_group=process.pid,
            contract_sha256=runner.sha(root/CONTRACT),SOURCE_sha256=c['SOURCE_sha256'],
            fresh_output=c['fresh_output'],start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            launches=1,retry=0,resume=0))
        print('FIXED_TWO_CANDIDATE_RUN_STARTED',process.pid,flush=True)
        while process.poll() is None:
            for path in [out/'CDF/failure.json',out/'XDF/failure.json',out/'CDF/STOP.json',out/'XDF/STOP.json',out/'failure.json']:
                if not path.exists():continue
                try:data=runner.read(path)
                except (json.JSONDecodeError,OSError):continue
                if path.name=='failure.json' or data.get('failed'):
                    reason=dict(trigger=str(path.relative_to(root)),record=data)
                    break
            if reason is not None:
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                save('termination.json',dict(action='SIGTERM own dedicated process group after registered failure',
                    process_group=process.pid,reason=reason,retry=0,resume=0))
                break
            time.sleep(.25)
        code=process.wait()
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    save('completion.json',dict(exit_code=code,failed=code!=0 or reason is not None,
        failure_monitor=reason,wall_seconds=time.monotonic()-started,
        children_CPU_seconds=usage.ru_utime+usage.ru_stime,children_peak_RSS_KiB=usage.ru_maxrss,
        end_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        root_STOP_exists=(out/'STOP.json').exists(),launches=1,retry=0,resume=0,mandatory_stop=True))
    print('FIXED_RUN_EXIT',code,'wall',time.monotonic()-started,flush=True)
    return code


if __name__=='__main__':sys.exit(main())
