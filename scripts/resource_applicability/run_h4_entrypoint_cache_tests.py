"""One bounded metadata process; reproduce forbidden mkdir and memory-only fix."""
import builtins,importlib.util,json,os,resource,signal,sys,tempfile,time,unittest
from importlib import metadata
from pathlib import Path
ROOT=Path(__file__).absolute().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry import gates,library_cache as cache,launch_binding as bind
from trottertracks.resource_applicability.h4_geometry.identity import Stop,require

def main():
    evidence=Path(os.environ['H4_ENTRY_CACHE_TEST_EVIDENCE']);plan=json.loads((evidence.parent/'CACHE_TEST_PLAN_v1.json').read_bytes())
    require(str(ROOT)==plan['checkout'] and sys.executable==plan['python'] and sys.flags.safe_path and sys.dont_write_bytecode,'fixed checkout/Python -P -B')
    require(all(os.environ.get(k)==v for k,v in gates.THREAD_ENV.items()),'internal thread1')
    tempfile.tempdir=str(evidence);resource.setrlimit(resource.RLIMIT_AS,(plan['AS_bytes'],plan['AS_bytes']));resource.setrlimit(resource.RLIMIT_FSIZE,(plan['output_bytes'],plan['output_bytes']));signal.alarm(plan['wall_seconds'])
    denied=[];original=builtins.__import__;scope={'guard':False};write_attempts=[]
    def guarded(name,*args,**kw):
        if name.split('.')[0] in ('qiskit','pyscf','openfermion','openfermionpyscf','trotterlib','cupy'):
            denied.append(name);raise RuntimeError('scientific/compiler/GPU imports forbidden')
        return original(name,*args,**kw)
    builtins.__import__=guarded
    profile=json.loads((evidence.parent/'library_cache_profile_v2.json').read_bytes())
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=os.fsdecode(args[0])
            if path.startswith('/tmp/') or path.endswith(('.npz','.npy','.pkl')):
                denied.append(path);raise RuntimeError('array or /tmp access forbidden')
        if event in ('subprocess.Popen','os.system','os.posix_spawn','os.exec','os.fork'):
            denied.append(event);raise RuntimeError('children forbidden')
        if scope['guard']:
            try:bind.write_guard({},event,args,worker=True,library_cache=profile)
            except Stop:
                write_attempts.append(dict(event=event,path=os.fsdecode(args[0])));raise
    sys.addaudithook(audit)
    spec=importlib.util.spec_from_file_location('h4_entry_cache_limited',ROOT/'tests/tracks/resource_applicability/test_h4_entrypoint_cache.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(module.EntryCacheTests))
    # Configure before first stevedore import; dependency code is unchanged.
    cache.configure(profile)
    from stevedore import _cache
    require(_cache._c._disable_caching is True,'dependency recognizes .disable')
    enabled=evidence/'disk-enabled';enabled.mkdir();instance=_cache.Cache(str(enabled))
    scope['guard']=True
    try:
        try:instance.get_group_all('qiskit.transpiler.init')
        except Stop as error:reproduced=str(error)
        else:raise RuntimeError('expected forbidden entrypoint cache mkdir was not reproduced')
    finally:scope['guard']=False
    expected_write_attempts=list(write_attempts);write_attempts.clear()
    groups=('init','layout','routing','translation','optimization','scheduling');matches={}
    points=metadata.entry_points()
    scope['guard']=True
    try:
        for short in groups:
            group='qiskit.transpiler.'+short
            actual=[(e.name,e.value,e.group) for e in _cache.get_group_all(group)]
            expected=list(dict.fromkeys((e.name,e.value,e.group) for e in points.select(group=group)))
            require(actual==expected,'ordered plugin metadata changed');matches[group]=len(actual)
            require([(e.name,e.value,e.group) for e in _cache.get_group_all(group)]==actual,'in-memory cache repeat changed')
    finally:scope['guard']=False
    require(not write_attempts and not denied,'memory-only policy must not attempt disk write/science imports')
    cache.verify(profile)
    peak=int(next(row.split()[1] for row in Path('/proc/self/status').read_text().splitlines() if row.startswith('VmHWM:')))*1024
    total=sum(p.stat().st_size for p in evidence.rglob('*') if p.is_file())
    require(peak<=plan['RSS_bytes'] and total<=plan['output_bytes'],'RSS/output caps')
    payload=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        wall_seconds=time.monotonic()-started,peak_RSS_bytes=peak,output_bytes_before_result=total,reproduced_forbidden_write=reproduced,
        expected_denied_write_attempts=expected_write_attempts,disabled_policy_write_attempts=write_attempts,
        ordered_plugin_metadata_matches=matches,memory_cache_repeat=True,child_processes=0,scientific_array_reads=0,artificial_builds=0,
        additional_transpiles=0,Qiskit_imports=0,affinity_changes=0,production_launches=0)
    with (evidence/'test_result_v1.json').open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps(payload));signal.alarm(0);return 0 if payload['status']=='PASS' else 1

if __name__=='__main__':sys.exit(main())
