"""One minimal owned child: a GIL-holding C call, no scientific imports.

Plan: driver fixture + existing test/observer process (2 processes maximum),
one thread each; child own affinity; AS512MiB/RSS256MiB/CPU60s/wall60s;
output2MiB; no fork stress, arrays, build, transpile, or shared-setting change.
"""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('fixture_real_runner',ROOT/'scripts/h6_b0i_pilot_runner_v1.py');g=r.g
e=load('fixture_real_external',ROOT/'scripts/h6_external_resource_monitor_v1.py')
f=load('fixture_real_budget',ROOT/'tests/final_pre_review_fixtures_v1.py')

def child(path):
    c=json.loads(path.read_text());os.sched_setaffinity(0,{c['parallel_compile']['driver_cpu']})
    path.with_suffix('.ready').write_text('READY')
    sys.stdin.readline()
    grant=SimpleNamespace(contract=c,contract_sha='b'*64,authorization_sha='c'*64,output=Path(c['exclusive_output']))
    m=e.monitor_class(g)(grant);m.phase='ARTIFICIAL_GIL_HOLDING_C';m.install()
    try:
        # PyDLL retains the GIL. OpenSSL does not return early on SIGALRM,
        # unlike sleep/usleep; the external process must keep observing.
        crypto=ctypes.PyDLL('libcrypto.so.3');fn=crypto.PKCS5_PBKDF2_HMAC_SHA1
        fn.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        fn.restype=ctypes.c_int;buf=ctypes.create_string_buffer(20)
        started=time.monotonic();assert fn(b'a',1,b'b',1,32000000,20,buf)==1
        m.tick(force=True)
        longest=max(x['driver_callback_interval_seconds'] for x in m.history)
        assert longest>5 and m.first_stop is None
        print(json.dumps(dict(status='PASS',C_call_wall_seconds=time.monotonic()-started,
            longest_driver_callback_gap=longest,independent_max_gap=m.history[-1]['external_max_interval_seconds'],
            scientific_imports=[k for k in sys.modules if k.split('.')[0] in ('numpy','qiskit','pyscf','cupy')],
            additional_transpile=0)),flush=True)
    finally:m.close()

def parent(base):
    base.mkdir(mode=0o700);control=base/'control';control.mkdir();output=base/'output';output.mkdir()
    c=dict(SOURCE='a'*40,budget=dict(f.BUDGET,cpu_seconds=60,wall_seconds=60),
        exclusive_output=str(output),control_directory=str(control),log_cap_bytes=2*2**20,
        external_monitor_memory_baseline=g.d0.memory_snapshot(),
        parallel_compile=dict(driver_cpu=min(os.sched_getaffinity(0)),worker_cpus=[]))
    config=base/'config.json';config.write_text(json.dumps(c));(control/'launch_stdout.log').touch()
    (base/'plan.json').write_text(json.dumps(dict(scope='ARTIFICIAL_GIL_ONLY',max_processes=2,threads=1,
        child_AS=512*2**20,child_RSS=256*2**20,child_CPU=60,wall=60,output=2*2**20,
        scientific_operations=0,additional_transpile=0,subagents=0),indent=2))
    process=None;identity=None;started=time.monotonic()
    with (base/'child_stdout.json').open('xb') as log:
        try:
            process=subprocess.Popen([sys.executable,'-P','-B',str(Path(__file__).resolve()),'--child',str(config)],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
            identity=g.identity(process.pid);assert identity is not None
            ready_deadline=time.monotonic()+5
            while not config.with_suffix('.ready').exists():
                assert time.monotonic()<ready_deadline,'FIXTURE_CHILD_STARTUP'
                time.sleep(.01)
            g.marker_path(output).write_text(json.dumps(dict(identity=identity,parent_identity=g.identity())))
            grant=SimpleNamespace(contract=c,contract_sha='b'*64,authorization_sha='c'*64,output=output)
            observer=e.Observer(grant,identity,g.identity(),g)
            process.stdin.write(b'GO\n');process.stdin.close()
            while process.poll() is None:
                assert time.monotonic()-started<60,'REAL_FIXTURE_WALL_CAP'
                observer.step();time.sleep(.25)
            status=process.wait(timeout=5);observer.summary()
            assert status==0,(base/'child_stdout.json').read_text()
        finally:
            if process is not None and process.poll() is None:
                launcher=load('fixture_cleanup_launcher',ROOT/'scripts/h6_b0i_pilot_launcher_v1.py')
                launcher.terminate_owned(process.pid,identity,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    launcher.terminate_owned(process.pid,identity,signal.SIGKILL);process.wait(timeout=5)
    child_result=json.loads((base/'child_stdout.json').read_text());assert child_result['scientific_imports']==[]
    assert g.identity(identity['pid']) is None
    report=dict(status='PASS',child_result=child_result,native_exit_code=status,
        child_reaped=True,original_identity_absent=True,observer=json.loads((control/'external_monitor_summary_v1.json').read_text()),
        wall_seconds=time.monotonic()-started,maximum_processes=2,real_scientific_operations=0,additional_transpile=0)
    (base/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':
    if sys.argv[1]=='--child':child(Path(sys.argv[2]))
    else:
        resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
        resource.setrlimit(resource.RLIMIT_CPU,(120,122))
        resource.setrlimit(resource.RLIMIT_FSIZE,(2*2**20,2*2**20))
        parent(Path(sys.argv[1]))
