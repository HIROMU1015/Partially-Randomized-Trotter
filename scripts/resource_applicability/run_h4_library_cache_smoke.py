"""Bounded dependency-import regression: cold private cache, then guarded reads."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates, library_cache
from trottertracks.resource_applicability.h4_geometry.identity import require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('cold','guarded-driver','guarded-worker'),required=True)
    parser.add_argument('--evidence',required=True)
    args=parser.parse_args()
    evidence=library_cache.private_path(args.evidence)
    require(evidence.is_dir() and sys.flags.safe_path and sys.dont_write_bytecode, 'home test evidence / -P -B')
    require(all(os.environ.get(k)==v for k,v in gates.THREAD_ENV.items()), 'test process thread1')
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(library_cache.CACHE_CAP,library_cache.CACHE_CAP))
    signal.alarm(60)
    cache=evidence/'matplotlib-cache'
    result_path=evidence/(args.mode+'-result.json')
    profile_path=evidence/'library-cache-profile.json'
    if args.mode=='cold':cache.mkdir(mode=0o700)
    import tempfile
    tempfile.tempdir=str(cache)
    os.environ['MPLCONFIGDIR']=str(cache)
    fontconfig_denials=[]
    def guard(event,values):
        if event=='subprocess.Popen':
            # Matplotlib catches OSError and uses its ordinary directory scan.
            # No additional process is needed to reproduce config/cache I/O.
            fontconfig_denials.append(str(values[1]));raise FileNotFoundError('bounded test: no fontconfig subprocess')
        if event in ('os.system','os.posix_spawn','os.exec'):raise RuntimeError('external process forbidden')
        if event=='import' and values[0].split('.')[0]=='cupy':raise RuntimeError('GPU import forbidden')
        if event=='open' and isinstance(values[0],(str,bytes,os.PathLike)):
            name=os.fsdecode(values[0])
            if name.startswith('/tmp/') or name.endswith(('.npz','.npy','.sqlite','.pkl')):
                raise RuntimeError('scientific input/temporary file forbidden')
        if args.mode!='cold':return
        if event not in ('open','os.mkdir','os.remove','os.rename','os.rmdir'):return
        if not values or not isinstance(values[0],(str,bytes,os.PathLike)):return
        writing=event!='open'
        if event=='open':
            mode=values[1] or '';flags=values[2] or 0
            writing=bool((isinstance(mode,str) and any(x in mode for x in 'wax+')) or
                         flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if not writing:return
        for raw in values[:2] if event=='os.rename' else values[:1]:
            path=Path(os.fsdecode(raw)).absolute()
            require(path==Path('/dev/null') or path.is_relative_to(cache) or path in (result_path,profile_path), 'cold fixture write escape')
    sys.addaudithook(guard)
    if args.mode!='cold':
        from trottertracks.resource_applicability.h4_geometry import launch_binding as bind
        profile=json.loads(profile_path.read_text())
        library_cache.configure(profile)
        plan={'output_root':str(evidence/'unused-output'),'control_root':str(evidence/'unused-control')}
        # Runtime hook except the explicitly bounded test result path.
        def runtime_guard(event,values):
            if event=='open' and values and values[0]==str(result_path):return
            bind.write_guard(plan,event,values,worker=args.mode=='guarded-worker',library_cache=profile)
        sys.addaudithook(runtime_guard)
    started=time.monotonic()
    from openfermion import FermionOperator, get_sparse_operator
    import matplotlib.pyplot
    import qiskit
    import matplotlib
    require(Path(matplotlib.get_configdir())==cache and Path(matplotlib.get_cachedir())==cache, 'private Matplotlib routing')
    rows=[]
    for p in sorted(cache.iterdir()):
        actual=library_cache.streaming_sha(p)
        rows.append({'file':p.name,'bytes':actual['bytes'],'sha256':actual['sha256']})
    total=sum(row['bytes'] for row in rows)
    require(1<=len(rows)<=library_cache.MAX_FILES and 0<total<=library_cache.CACHE_CAP, 'bounded fixture bytes/files')
    if args.mode=='cold':
        profile={'schema_version':'h4-library-cache-v1','root':str(cache),'matplotlib_version':metadata.version('matplotlib'),
                 'files':rows,'bytes':total,'scientific_cache':False,'runtime_writes':False}
        with profile_path.open('x') as f:json.dump(profile,f,indent=2);f.write('\n')
    else:
        require(rows==profile['files'], 'guarded cache mutated')
        require(not any('fc-list' in command for command in fontconfig_denials), 'guarded import attempted cache regeneration')
    status=Path('/proc/self/status').read_text()
    peak=int(next(l for l in status.splitlines() if l.startswith('VmHWM:')).split()[1])*1024
    require(peak<=512*2**20, 'dependency smoke RSS512MiB')
    result={'status':'PASS','mode':args.mode,'wall_seconds':time.monotonic()-started,'current_process_peak_RSS_bytes':peak,
            'library_cache_bytes':total,'cache_files':len(rows),'external_subprocess_denials':fontconfig_denials,
            'actual_transpiles':0,'scientific_input_reads':0,'scientific_calls':0,'production_workers':0,
            'own_affinity_changes':0,'maximum_test_processes':1,'venv_or_shared_changes':False,
            'cache_SHA_unchanged':args.mode!='cold','imported':['openfermion','matplotlib.pyplot','qiskit']}
    with result_path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0)
    print(json.dumps(result))


if __name__=='__main__':main()
