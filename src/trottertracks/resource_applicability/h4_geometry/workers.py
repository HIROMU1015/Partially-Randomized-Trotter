"""Owned spawn workers over anonymous pipes; no auxiliary resource-tracker process.

    Pickle transport is private to the parent's own child pipes, never a file or
    network input. Only two fixed job names are dispatched after checkout gates.
"""
import io
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import threading

MAX_FRAME=64*2**20


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


def owned_worker_main(parent_pid):
    from contextlib import redirect_stdout,redirect_stderr
    from trottertracks.resource_applicability.h4_geometry.identity import require
    from trottertracks.resource_applicability.h4_geometry.gates import checkout_gate,Permit
    from trottertracks.resource_applicability.h4_geometry.resources import limit_owned_address_space
    require(os.getppid()==parent_pid,'owned parent handshake')
    limit_owned_address_space()
    incoming,outgoing=sys.stdin.buffer,sys.stdout.buffer
    permit=read_frame(incoming)
    require(isinstance(permit,Permit),'owned permit')
    _contract,options=checkout_gate(permit)
    write_frame(outgoing,{'ready':os.getpid()})
    while True:
        job=read_frame(incoming)
        require(isinstance(job,dict) and set(job)=={'name','args'},'owned worker job wire')
        logs=CappedText()
        try:
            with redirect_stdout(logs),redirect_stderr(logs):
                result=private_dispatch(permit,job['name'],job['args'],options)
            write_frame(outgoing,{'result':result,'log':logs.getvalue(),'error':None})
        except BaseException as exc:
            write_frame(outgoing,{'result':None,'log':logs.getvalue(),'error':type(exc).__name__+': '+str(exc)})
            return  # failed invocation consumed; never another job or restart


class OwnedPool:
    """Exactly w owned Python spawn workers. No replacement, retry or resume."""
    def __init__(self,workers,permit,monitor,budget):
        from concurrent.futures import ThreadPoolExecutor
        from trottertracks.resource_applicability.h4_geometry.gates import PYTHON
        self.monitor,self.budget,self.processes=monitor,budget,[]
        self.counter,self.assignment_lock,self.failure=0,threading.Lock(),None
        self.available=queue.Queue()
        for index in range(workers):self.available.put(index)
        self.io=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='owned-pipe-io')
        try:
            for _ in range(workers):
                process=subprocess.Popen([PYTHON,'-B',__file__,'--owned-worker',str(os.getpid())],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                    cwd=permit.source_root,env=dict(os.environ),close_fds=True)
                self.processes.append(process);monitor.own_child(process.pid)
            def init(process):
                write_frame(process.stdin,permit)
                response=read_frame(process.stdout)
                from trottertracks.resource_applicability.h4_geometry.identity import require
                require(response=={'ready':process.pid},'owned worker handshake')
            futures=[self.io.submit(init,p) for p in self.processes]
            for future in futures:
                future.result(timeout=5)
        except BaseException:
            self.shutdown(wait=True,cancel_futures=True)
            raise

    def submit(self,function,*args):
        from trottertracks.resource_applicability.h4_geometry.identity import require
        require(function.__name__ in ('_generate_worker','_compile_worker'),'closed worker job set')
        with self.assignment_lock:
            require(self.failure is None,'owned pool failed; no additional submit')
            serial=self.counter;self.counter+=1
        return self.io.submit(self._call,serial,function.__name__,args)

    def _call(self,serial,name,args):
        from trottertracks.resource_applicability.h4_geometry.identity import require
        try:
            with self.assignment_lock:
                require(self.failure is None,'owned pool failed; no queued dispatch')
            index=self.available.get()
            with self.assignment_lock:
                require(self.failure is None,'owned pool failed; no queued dispatch')
            process=self.processes[index]
            require(process.poll() is None,'owned worker died STOP')
            write_frame(process.stdin,{'name':name,'args':args})
            response=read_frame(process.stdout)
            require(isinstance(response,dict) and set(response)=={'result','log','error'},'owned worker response')
            if response['log']:
                self.budget.write('worker-log-%06d.txt'%serial,response['log'].encode())
            require(response['error'] is None,'owned worker failure STOP: '+str(response['error']))
        except BaseException as exc:
            with self.assignment_lock:
                if self.failure is None:self.failure=exc
            self.monitor.stop_children()
            raise
        self.available.put(index)  # refill the worker that actually finished
        return response['result']

    def shutdown(self,wait=True,cancel_futures=True):
        self.monitor.stop_children()
        for process in self.processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Popen retains this child's PID and start ownership, never a foreign process.
                process.kill();process.wait(timeout=2)
            for stream in (process.stdin,process.stdout):
                stream.close()
        self.io.shutdown(wait=wait,cancel_futures=cancel_futures)


if __name__=='__main__':
    # Direct private worker entry, only started by OwnedPool after launch gates.
    sys.path.insert(0,str(Path(__file__).absolute().parents[3]))
    if len(sys.argv)!=3 or sys.argv[1]!='--owned-worker':
        raise SystemExit('private owned-worker entry only')
    owned_worker_main(int(sys.argv[2]))
