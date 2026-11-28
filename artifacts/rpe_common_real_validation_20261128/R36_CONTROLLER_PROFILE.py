"""Profile the ordinary distributed controller in an isolated artificial acceptance.

Run on the actual RomuTail OS; all three workers are loopback fixtures.
No saved scientific REQUEST is launched, no production controller is modified.
"""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
import check_rpe_distributed_q3 as acceptance

def child(directory, mode):
    from trottertracks.rpe_optimizer_concurrent import DistributedExecutor
    done = threading.Event()
    observations={}
    hz=os.sysconf('SC_CLK_TCK')
    def sample():
        while not done.is_set():
            names={x.native_id:x.name for x in threading.enumerate()}
            for task in Path('/proc/self/task').iterdir():
                tid=int(task.name)
                try:
                    raw=(task/'stat').read_text()
                    fields=raw[raw.rfind(')')+2:].split()
                    ticks=int(fields[11])+int(fields[12])
                except (OSError,ValueError,IndexError):
                    continue
                saved=observations.setdefault(tid,dict(first=ticks,last=ticks,name=names.get(tid,'native/unknown')))
                saved['last']=ticks
                if tid in names:saved['name']=names[tid]
            done.wait(.05)
    monitor=threading.Thread(target=sample,name='validation-cpu-sampler',daemon=True)
    monitor.start()
    start=time.perf_counter()
    cpu=time.process_time()
    try:
        with ExitStack() as stack:
            if mode=='individual':
                # Reference to the prior per-host aggregation path on the
                # otherwise identical controller and fixture.
                stack.enter_context(patch.object(DistributedExecutor,'_capacity_feedback_batch',
                    lambda self:{host:self._capacity_feedback(host) for host in self.hosts}))
            result=acceptance.controller(directory)
        elapsed=time.perf_counter()-start
        used=time.process_time()-cpu
    finally:
        done.set();monitor.join()
    rows=[dict(tid=k,name=v['name'],observed_cpu_seconds=(v['last']-v['first'])/hz)
          for k,v in observations.items()]
    acceptance.write(directory/'CONTROLLER_PROFILE.json',dict(mode=mode,wall_seconds=elapsed,
        process_cpu_seconds=used,thread_observation_seconds=.05,
        observed_thread_cpu=rows,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scientific_calls=0))
    print(json.dumps(dict(mode=mode,outcomes=result['outcomes'],wall=elapsed,process_cpu=used)),flush=True)

def run(directory,mode,profiler,blocking=False):
    directory=Path(directory).resolve()
    port=acceptance.free_port()
    acceptance.prepare(directory,port=port,total_slots=3,sleep_seconds=.12)
    spec=json.loads((directory/'ROUND.json').read_text())
    spec['executor']['distributed']['local_worker']['host_id']='simulated-server2'
    acceptance.write(directory/'ROUND.json',spec)
    workers=[]
    with ExitStack() as stack:
        try:
            for host in ('simulated-server1','simulated-local'):
                profile_path=directory/('PROFILE_'+host+'.json')
                acceptance.write(profile_path,acceptance.profile(directory,host))
                log=stack.enter_context((directory/(host+'.log')).open('ab'))
                workers.append(subprocess.Popen([sys.executable,'-B',str(ROOT/'scripts/run_rpe_q3_worker.py'),
                    '--profile',str(profile_path),'--controller','http://127.0.0.1:'+str(port),'--poll-seconds','.02'],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
            with (directory/'PROFILER.log').open('wb') as log:
                start=time.perf_counter()
                subprocess.run([profiler,'record','--gil','--idle','--threads',*([] if blocking else ['--nonblocking']),'--rate','100',
                    '--format','raw','--output',str(directory/'GIL.raw'),'--',sys.executable,'-B',
                    str(Path(__file__).resolve()),'child',str(directory),mode],cwd=ROOT,check=True,
                    stdout=log,stderr=subprocess.STDOUT)
                profile_wall=time.perf_counter()-start
            r=acceptance.check(directory)
            participating=[h for h,c in r['by_host'].items() if c['completed']>0 and c['adopted']>0]
            if len(participating)!=3:raise ValueError('Not all artificial workers contributed')
            sample_count=sum(int(line.rsplit(' ',1)[1]) for line in (directory/'GIL.raw').read_text().splitlines() if line.strip())
            acceptance.write(directory/'GIL_SUMMARY.json',dict(gil_filtered_raw_samples=sample_count,
                configured_rate_hz=100,profiler_wall_seconds=profile_wall,
                nonblocking=not blocking,scope='entire isolated ordinary controller process; includes startup and verification',
                participating_hosts=participating,approximate_fraction_of_nominal_samples=sample_count/(100*profile_wall)))
        finally:
            for worker in workers:
                if worker.poll() is None:worker.terminate()
            for worker in workers:
                try:worker.wait(timeout=15)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()
if __name__=='__main__':
    if sys.argv[1]=='child':child(Path(sys.argv[2]),sys.argv[3])
    else:run(Path(sys.argv[2]),sys.argv[3],sys.argv[4],blocking='blocking' in sys.argv[5:])
