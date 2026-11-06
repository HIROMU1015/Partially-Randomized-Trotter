"""One permitted continuation, with a spawn-safe entry and no retry."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

CONTRACT='docs/research/research_c_h4_handoff_contract_20261106.json'


def main():
    root=Path.cwd().resolve();assert root==Path(__file__).resolve().parents[4]
    fixed=sys.argv[1]
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==fixed
    assert (root/CONTRACT).read_bytes()==subprocess.check_output(['git','show',fixed+':'+CONTRACT])
    sys.path.insert(0,str(root/'src'))
    from trottertracks import c_h4_relocated_execution as execution
    c=execution.preflight(root,CONTRACT)
    packages={p:importlib.metadata.version(p) for p in c['packages']}
    assert packages==c['packages']
    record=root/'artifacts/research_c_h4_handoff/2026-11-06/execution'
    def save(name,value):
        with (record/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    tick=time.monotonic();cpu=time.process_time();children=resource.getrusage(resource.RUSAGE_CHILDREN)
    save('start.json',dict(fixed_commit=fixed,pid=os.getpid(),contract_sha256=hashlib.sha256((root/CONTRACT).read_bytes()).hexdigest(),
        SOURCE_sha256=c['SOURCE_sha256'],sampling_SOURCE_sha256=c['sampling_SOURCE_sha256'],
        checkpoint_sha256=c['handoff']['checkpoint_sha256'],fresh_output=c['fresh_output'],packages=packages,
        start_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),planned_handoffs=1,retry=0,resume=0))
    failed=None
    try:
        print('SINGLE_PLANNED_HANDOFF_START',flush=True)
        result=execution.execute(root,CONTRACT)
        print(json.dumps(dict(event='CONTINUATION_RETURN',result=result)),flush=True)
    except BaseException as error:
        failed=repr(error);traceback.print_exc();raise
    finally:
        end=resource.getrusage(resource.RUSAGE_CHILDREN)
        save('completion.json',dict(failed=failed,wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            children_CPU_seconds=end.ru_utime+end.ru_stime-children.ru_utime-children.ru_stime,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            end_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),planned_handoffs=1,retry=0,resume=0,mandatory_stop=True))


if __name__=='__main__':main()
