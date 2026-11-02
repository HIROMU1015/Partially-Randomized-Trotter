"""Owned spawn workers over anonymous pipes; no auxiliary resource-tracker process.

    Pickle transport is private to the parent's own child pipes, never a file or
    network input. Only two fixed job names are dispatched after checkout gates.
"""
import io
import gc
import json
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import threading
import time
import traceback

MAX_FRAME=64*2**20
FAILURE_CAP=8192


def bounded_text(value,limit):
    return value.encode('utf-8',errors='replace')[:limit].decode('utf-8',errors='ignore')


def exception_text(exc,phase):
    try:message=str(exc)
    except BaseException:message='<exception message unavailable>'
    # Never inspect locals (which can include large circuit/matrix objects).
    frames=traceback.format_list(traceback.extract_tb(exc.__traceback__,limit=-12))
    return bounded_text(type(exc).__name__+': '+bounded_text(message,1024)+
                        '\nworker_phase='+phase+'\n'+''.join(frames),6144)


def write_frame(stream,value):
    from trottertracks.resource_applicability.h4_geometry.identity import require
    data=pickle.dumps(value,protocol=5)
    require(len(data)<=MAX_FRAME,'owned IPC frame budget')
    stream.write(len(data).to_bytes(8,'big'));stream.write(data);stream.flush()


def read_frame(stream):
    from trottertracks.resource_applicability.h4_geometry.identity import require
    def read_exact(length):
        pieces=[]
        while length:
            data=stream.read(length)
            require(bool(data),'interrupted owned worker pipe STOP')
            pieces.append(data);length-=len(data)
        return b''.join(pieces)
    length=int.from_bytes(read_exact(8),'big')
    require(0<length<=MAX_FRAME,'owned IPC frame budget')
    return pickle.loads(read_exact(length))


def private_dispatch(permit,name,args,options):
    from trottertracks.resource_applicability.h4_geometry.identity import require
    from trottertracks.resource_applicability.h4_geometry import execution
    if name=='_generate_worker':
        require(permit.stage=='input_generation' and len(args)==2 and args[0]==permit,'generation worker gate')
        return execution._generate_worker(*args)
    require(name=='_compile_worker' and permit.stage=='signal_compile' and len(args)==2 and args[1]==options,
            'compile worker gate')
    return execution._compile_worker(*args)


class CappedText(io.StringIO):
    def __init__(self):
        super().__init__();self.bytes=0

    def write(self,value):
        from trottertracks.resource_applicability.h4_geometry.identity import require
        size=len(value.encode());require(self.bytes+size<=8192,'owned worker log budget')
        self.bytes+=size
        return super().write(value)


def owned_worker_main(parent_pid,index=None):
    from contextlib import redirect_stdout,redirect_stderr
    from trottertracks.resource_applicability.h4_geometry.identity import require
    from trottertracks.resource_applicability.h4_geometry.gates import checkout_gate,Permit
    from trottertracks.resource_applicability.h4_geometry.resources import limit_owned_address_space
    incoming,outgoing=sys.stdin.buffer,sys.stdout.buffer
    phase='bootstrap';logs=None;completed_log=''
    try:
        require(os.getppid()==parent_pid,'owned parent handshake')
        # Soft8 protects bootstrap while retaining the inherited ceiling until
        # the private permit has been authorized and its source/profile verified.
        limit_owned_address_space(preserve_hard=True)
        phase='permit_decode';permit=read_frame(incoming)
        require(isinstance(permit,Permit),'owned permit')
        if permit.plan.get('schema_version')=='h4-newhost-plan-v2':
            from trottertracks.resource_applicability.h4_geometry.launch_binding import role_affinity
            phase='worker_binding';role_affinity(permit,'worker',index)
        phase='checkout';_contract,options=checkout_gate(permit)
        cap = permit.plan['caps']['worker_AS_RSS'] if permit.plan.get('schema_version')=='h4-newhost-plan-v2' else 8*2**30
        phase='worker_memory_binding';limit_owned_address_space(cap)
        phase='ready';write_frame(outgoing,{'ready':os.getpid()})
        while True:
            phase='job_decode';job=read_frame(incoming)
            require(isinstance(job,dict) and set(job)=={'name','args'},'owned worker job wire')
            logs=CappedText();phase='dispatch'
            with redirect_stdout(logs),redirect_stderr(logs):
                result=private_dispatch(permit,job['name'],job['args'],options)
            # Drop the previous circuit before blocking on the next IPC frame.
            # Collect before publishing success: a GC failure must not release
            # this worker back to the parent's available queue.
            completed_log=logs.getvalue();del job;logs.close();logs=None
            phase='gc';gc.collect()
            phase='response_encode'
            write_frame(outgoing,{'result':result,'log':completed_log,'error':None})
            result=None;completed_log=''
    except BaseException as exc:
        # Includes bootstrap, unpickle/schema, response encoding and GC failures.
        # Stay alive after reporting: an exited worker would let the observer
        # SIGTERM the driver before its pipe thread has fsynced the original error.
        try:
            write_frame(outgoing,{'result':None,'log':logs.getvalue() if logs else completed_log,
                                 'error':exception_text(exc,phase)})
            while incoming.read(65536):
                pass  # wait for parent STOP/EOF; discard, never dispatch another job
        except (OSError,ValueError):
            pass  # broken parent pipe; independent ownership monitoring stops
        finally:
            if logs is not None:logs.close()


class OwnedPool:
    """Exactly w owned Python spawn workers. No replacement, retry or resume."""
    def __init__(self,workers,permit,monitor,budget):
        from concurrent.futures import ThreadPoolExecutor
        from trottertracks.resource_applicability.h4_geometry.gates import PYTHON
        python=PYTHON
        if permit.plan.get('schema_version')=='h4-newhost-plan-v2':
            from trottertracks.resource_applicability.h4_geometry.launch_binding import reference
            python=reference(Path(permit.source_root),permit.plan['environment_profile'])['python']
        self.monitor,self.budget,self.processes=monitor,budget,[]
        self.counter,self.assignment_lock,self.failure=0,threading.Lock(),None
        self.failure_record=None;self.failure_publication_error=None
        self.available=queue.Queue()
        for index in range(workers):self.available.put(index)
        self.io=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='owned-pipe-io')
        try:
            for index in range(workers):
                argv=[python,'-P','-B',__file__,'--owned-worker',str(os.getpid())]
                if permit.plan.get('schema_version')=='h4-newhost-plan-v2':argv.append(str(index))
                process=subprocess.Popen(argv,
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                    cwd=permit.source_root,env=dict(os.environ),close_fds=True)
                self.processes.append(process);monitor.own_child(process.pid)
            def init(process):
                try:
                    write_frame(process.stdin,permit)
                    response=read_frame(process.stdout)
                    from trottertracks.resource_applicability.h4_geometry.identity import require
                    if isinstance(response,dict) and response.get('error'):
                        raise RuntimeError(response['error'])
                    require(response=={'ready':process.pid},'owned worker handshake')
                except BaseException as exc:
                    self._record_failure(exc,None,process,'worker_handshake')
                    raise
            futures=[self.io.submit(init,p) for p in self.processes]
            for future in futures:
                future.result(timeout=5)
        except BaseException as exc:
            self._record_failure(exc,None,None,'pool_bootstrap')
            self.shutdown(wait=True,cancel_futures=True)
            raise

    def _record_failure(self,exc,serial,process,phase,logs=''):
        """First bounded cause is durable before any own-child cleanup signal."""
        with self.assignment_lock:
            if self.failure is not None:return
            self.failure_publication_error=None
            record={'kind':'owned_worker_first_failure','serial':serial,
                    'worker_pid':process.pid if process is not None else None,
                    'observed_exit_code':process.poll() if process is not None else None,
                    'phase':phase,'monotonic':time.monotonic(),
                    'error':exception_text(exc,phase),'log':bounded_text(logs,512)}
            # JSON escaping can enlarge text; keep the existing sealed 8KiB cap.
            while True:
                data=json.dumps(record,ensure_ascii=False,sort_keys=True).encode()+b'\n'
                if len(data)<=FAILURE_CAP:break
                record['error']=bounded_text(record['error'],len(record['error'].encode())//2)
            self.failure_record=record
            try:self.budget.write('worker-log-first-stop.txt',data)
            except BaseException as publication_error:
                self.failure_publication_error=publication_error
            # Keep the latch lock until reporting: a simultaneous second failure
            # must not signal children before the first cause reaches the observer.
            report={k:record[k] for k in ('serial','worker_pid','phase')}
            report['reason']=bounded_text(record['error'].split('\n',1)[0],1024)
            if self.failure_publication_error is not None:
                report['reason']+='; failure publication: '+bounded_text(str(self.failure_publication_error),256)
            try:self.monitor.report_failure(report)
            except BaseException:
                # Original cause remains latched; terminal EOF is secondary.
                pass
            finally:
                # pulse()/abort() can signal children as soon as they see this
                # field. Publish it only after the durable/reporting steps.
                self.failure=exc

    def submit(self,function,*args):
        from trottertracks.resource_applicability.h4_geometry.identity import require
        require(function.__name__ in ('_generate_worker','_compile_worker'),'closed worker job set')
        with self.assignment_lock:
            require(self.failure is None,'owned pool failed; no additional submit')
            serial=self.counter;self.counter+=1
        return self.io.submit(self._call,serial,function.__name__,args)

    def _call(self,serial,name,args):
        from trottertracks.resource_applicability.h4_geometry.identity import require
        process=None;phase='assignment';logs=''
        try:
            with self.assignment_lock:
                require(self.failure is None,'owned pool failed; no queued dispatch')
            index=self.available.get()
            with self.assignment_lock:
                require(self.failure is None,'owned pool failed; no queued dispatch')
            process=self.processes[index]
            require(process.poll() is None,'owned worker died STOP')
            phase='job_encode'
            write_frame(process.stdin,{'name':name,'args':args})
            phase='worker_response'
            response=read_frame(process.stdout)
            require(isinstance(response,dict) and set(response)=={'result','log','error'},'owned worker response')
            require(type(response['log']) is str and len(response['log'].encode())<=8192 and
                    (response['error'] is None or type(response['error']) is str),'owned worker response types')
            logs=response['log']
            require(response['error'] is None,'owned worker failure STOP: '+str(response['error']))
            if logs:
                phase='worker_log_publish';self.budget.write('worker-log-%06d.txt'%serial,logs.encode())
        except BaseException as exc:
            try:self._record_failure(exc,serial,process,phase,logs)
            finally:self.monitor.stop_children()
            raise
        self.available.put(index)  # refill the worker that actually finished
        return response['result']

    def shutdown(self,wait=True,cancel_futures=True):
        first=None
        try:self.monitor.stop_children()
        except BaseException as exc:first=exc
        for process in self.processes:
            try:
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    # Popen retains this child's PID and start ownership, never a foreign process.
                    process.kill();process.wait(timeout=2)
            except BaseException as exc:
                if first is None:first=exc
            finally:
                for stream in (process.stdin,process.stdout):
                    try:stream.close()
                    except BaseException as exc:
                        if first is None:first=exc
        try:self.io.shutdown(wait=wait,cancel_futures=cancel_futures)
        except BaseException as exc:
            if first is None:first=exc
        if first is not None:raise first


if __name__=='__main__':
    # Direct private worker entry, only started by OwnedPool after launch gates.
    sys.path.insert(0,str(Path(__file__).absolute().parents[3]))
    if len(sys.argv) not in (3,4) or sys.argv[1]!='--owned-worker':
        raise SystemExit('private owned-worker entry only')
    owned_worker_main(int(sys.argv[2]),int(sys.argv[3]) if len(sys.argv)==4 else None)
