"""Execute one separately authorized H4 native stage and stop."""
import argparse
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import h4_native_contract as c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['D','A','N']);p.add_argument('--config',type=Path);p.add_argument('--grant',type=Path)
    a=p.parse_args()
    if not a.stage: print(__import__('json').dumps(c.plan(),indent=2));return 0
    config,grant=c.read(a.config),c.read(a.grant)
    output=Path(grant['output']); start=time.monotonic()
    c.require(not output.exists() and not Path(grant['lock']).exists(),'ONE_SHOT_ALREADY_USED')
    gate=c.verify(config,grant,a.stage)
    output.mkdir(parents=True);c.save(grant['lock'],dict(stage=a.stage,SOURCE=grant['SOURCE'],plan_digest=grant['plan_digest']))
    c.save(output/'fresh_gate.json',gate)
    os.sched_setaffinity(0,set(grant['worker_cpus']))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    os.environ['QISKIT_PARALLEL']='FALSE';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    scratch=output/'scratch';scratch.mkdir()
    for key,name in [('TMPDIR','tmp'),('NUMBA_CACHE_DIR','numba'),('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache')]:
        path=scratch/name;path.mkdir();os.environ[key]=str(path)
    font=config['font_seed'];raw=Path(font['path']).read_bytes()
    c.require(__import__('hashlib').sha256(raw).hexdigest()==font['sha256'],'FONT_SEED')
    (scratch/'mpl'/font['name']).write_bytes(raw)
    import tempfile
    tempfile.tempdir=os.environ['TMPDIR']
    sys.path.insert(0,str(Path(config['provider_root'])/'src'))
    code=0
    try:
        import h4_native_science as science
        result=science.execute(a.stage,config,grant,output)
        c.save(output/'final_gate.json',c.verify(config,grant,a.stage))
        c.save(output/'result.json',result)
    except BaseException as exc:
        code=2;c.save(output/'STOP.json',dict(reason=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc()))
    c.save(output/'terminal.json',dict(stage=a.stage,SOURCE=grant['SOURCE'],exit_code=code,
        status='COMPLETE_MANDATORY_STOP' if code==0 else 'STOP',wall_seconds=time.monotonic()-start,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,next_stage_authorized=False))
    return code


if __name__=='__main__':raise SystemExit(main())
