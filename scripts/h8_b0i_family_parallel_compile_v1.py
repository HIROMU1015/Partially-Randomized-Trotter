"""Two axes per H8 case, inherited validated wrappers, bounded owned Linux children.

No scientific import, input loader, build, QPY/pickle transfer, pool or retry.
Only an already authorized parent may call compile_batch after all A/B pass.
"""
import ctypes
import errno
import json
import importlib.util
import os
from pathlib import Path
import resource
import signal
import time

MAX_MESSAGE=16384
BLOCKED_SIGNALS={signal.SIGALRM,signal.SIGTERM,signal.SIGINT}
CLEANUP_SIGNALS=BLOCKED_SIGNALS|{signal.SIGXCPU}
# Fit both owned-child waits inside the launcher's existing 5s TERM window.
CLEANUP_WAIT_SECONDS=((signal.SIGTERM,1.),(signal.SIGKILL,2.))


class CleanupSignals:
    """Defer Python STOP callbacks while reaping; never resume failed science.

    The independent metadata observer remains active. On failure the process
    keeps these signals blocked through terminal recording and exits; SIGKILL
    and the external launcher's original deadline remain effective.
    """
    def __init__(self,enabled):
        self.enabled=enabled;self.deferred=False

    def defer(self):
        if not self.enabled:return
        self.mask=signal.pthread_sigmask(signal.SIG_BLOCK,CLEANUP_SIGNALS)
        try:self.timer=signal.setitimer(signal.ITIMER_REAL,0)
        except BaseException:
            signal.pthread_sigmask(signal.SIG_SETMASK,self.mask)
            raise
        self.deferred=True

    def finish(self,*,resume):
        if self.deferred and resume:
            signal.setitimer(signal.ITIMER_REAL,*self.timer)
            self.deferred=False
            signal.pthread_sigmask(signal.SIG_SETMASK,self.mask)

    def receipt(self):
        return dict(enabled=self.enabled,retained_until_process_exit=self.deferred,
            TERM_wait_seconds=1.,KILL_wait_seconds=2.,
            independent_observer_unchanged=True,science_during_cleanup=False)


def parent_death_guard(parent_pid):
    """Prevent compile children outliving their numerical driver."""
    if os.getppid()!=parent_pid:raise RuntimeError('COMPILE_PARENT_ALREADY_EXITED')
    libc=ctypes.CDLL(None,use_errno=True)
    libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
    libc.prctl.restype=ctypes.c_int
    if libc.prctl(1,int(signal.SIGKILL),0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if os.getppid()!=parent_pid:raise RuntimeError('COMPILE_PARENT_EXIT_RACE')


def terminate_owned(pid,identity,sig,g):
    current=g.identity(pid)
    if current is None:return
    if identity is None or current!=identity:raise g.Stop('COMPILE_OWNERSHIP_CHANGED')
    try:fd=os.pidfd_open(pid)
    except OSError as error:
        if error.errno==errno.ESRCH:return
        raise
    try:
        current=g.identity(pid)
        if current is None:return
        if current!=identity:raise g.Stop('COMPILE_OWNERSHIP_CHANGED')
        try:signal.pidfd_send_signal(fd,sig)
        except OSError as error:
            if error.errno!=errno.ESRCH:raise
    finally:os.close(fd)


def worker_main(task,api,grant,g,write_fd,parent_pid,old_mask,other_read_fds):
    monitor=None;status=2;compile_entered=False
    try:
        parent_death_guard(parent_pid)
        signal.setitimer(signal.ITIMER_REAL,0)
        for fd in other_read_fds:os.close(fd)
        os.sched_setaffinity(0,{task['cpu']})
        temporary=grant.output/'temp'/('compile_'+task['name']);temporary.mkdir(mode=0o700)
        for key,path in dict(TMPDIR=temporary,MPLCONFIGDIR=temporary/'mpl',XDG_CACHE_HOME=temporary/'cache',
                NUMBA_CACHE_DIR=temporary/'numba').items():
            if path!=temporary:path.mkdir(mode=0o700)
            os.environ[key]=str(path)
        g.seed_font_cache(grant,temporary/'mpl')
        import tempfile
        tempfile.tempdir=str(temporary)
        monitor_factory=g.Monitor
        if grant.contract.get('external_resource_monitor',{}).get('enabled'):
            spec=importlib.util.spec_from_file_location('h8_compile_external_monitor',
                Path(__file__).with_name('h8_b0i_family_external_monitor_v1.py'))
            external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)
            monitor_factory=external.monitor_class(g)
        monitor=monitor_factory(grant);monitor.phase='C_COMPILE_'+task['name'];monitor.install()
        api.guard=monitor.tick
        signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
        compile_entered=True;row=api.compile(task['wrapper'],task['fingerprint'])
        monitor.tick(force=True)
        value=dict(status='PASS',name=task['name'],row=row)
        status=0
    except BaseException as error:
        value=dict(status='STOP',name=task['name'],reason=type(error).__name__+':'+str(error)[:1000])
    try:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        value.update(identity=g.identity(),cpu_seconds=usage.ru_utime+usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024)
        value['compile_call_entered']=compile_entered
        payload=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
        if len(payload)>MAX_MESSAGE:raise RuntimeError('COMPILE_IPC_CAP')
        view=memoryview(payload)
        while view:
            count=os.write(write_fd,view);view=view[count:]
    except BaseException:status=2
    finally:
        if monitor is not None:monitor.close()
        os.close(write_fd)
        os._exit(status)


class ForkBackend:
    """Parent-only ownership registry; every child has exactly one axis."""
    def __init__(self,grant,api,monitor,g):
        grant.require('H8_NATIVE_B0I_FULL_FAMILY12_V1')
        if grant.synthetic:raise g.Stop('ARTIFICIAL_PARALLEL_REQUIRES_MOCK_BACKEND')
        self.grant,self.api,self.monitor,self.g=grant,api,monitor,g
        self.handles=[];self.parent=os.getpid();self.clock_ticks=os.sysconf('SC_CLK_TCK')
        self.prior_cpu=monitor.child_cpu_seconds

    def start(self,task):
        if len(list(Path('/proc/self/task').iterdir()))!=1:raise self.g.Stop('FORK_PARENT_MUST_BE_SINGLE_THREAD')
        read_fd,write_fd=os.pipe();os.set_blocking(read_fd,False)
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,BLOCKED_SIGNALS)
        handle=None
        try:
            handle=dict(pid=None,identity=None,fd=read_fd,task=task,payload=bytearray(),
                exit_code=None,reaped=False,cpu_seconds=0.,peak_rss_bytes=0,
                started=time.monotonic(),affinity_confirmed=False)
            self.handles.append(handle)
            pid=os.fork()
            if pid==0:
                try:
                    os.close(read_fd)
                    worker_main(task,self.api,self.grant,self.g,write_fd,self.parent,mask,
                        [h['fd'] for h in self.handles if h is not handle and h['fd'] is not None])
                finally:os._exit(2)
            # All storage is allocated and registered before fork. Store the
            # kernel-returned PID before any identity IO that can raise.
            handle['pid']=pid
            handle['identity']=self.g.identity(pid)
            if handle['identity'] is not None and (handle['identity']['parent_pid']!=self.parent or
                    handle['identity']['uid']!=os.getuid()):raise self.g.Stop('COMPILE_CHILD_IDENTITY')
            return handle
        finally:
            os.close(write_fd)
            if handle is None or handle['pid'] is None:
                if handle is not None:self.handles.remove(handle)
                os.close(read_fd)
            signal.pthread_sigmask(signal.SIG_SETMASK,mask)

    def drain(self,h):
        if h['fd'] is None:return
        while True:
            try:chunk=os.read(h['fd'],4096)
            except BlockingIOError:return
            if not chunk:
                os.close(h['fd']);h['fd']=None;return
            h['payload'].extend(chunk)
            if len(h['payload'])>MAX_MESSAGE:raise self.g.Stop('COMPILE_IPC_CAP')

    def reap(self,h):
        if h['reaped']:return
        pid,status,usage=os.wait4(h['pid'],os.WNOHANG)
        if pid:
            h.update(reaped=True,exit_code=os.waitstatus_to_exitcode(status),
                cpu_seconds=usage.ru_utime+usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024)

    def observe(self,h):
        if h['reaped']:return
        current=self.g.identity(h['pid'])
        if current is None:return
        if h['identity'] is None or current!=h['identity']:raise self.g.Stop('COMPILE_OWNERSHIP_CHANGED')
        try:
            root=Path('/proc')/str(h['pid'])
            values=dict(line.split(':',1) for line in (root/'status').read_text().splitlines() if ':' in line)
            # A zombie can be observed between resource sampling and wait4.
            if values['State'].lstrip().startswith('Z'):return
            rss=int(values['VmRSS'].split()[0])*1024;virtual=int(values['VmSize'].split()[0])*1024
            fields=(root/'stat').read_text().rsplit(')',1)[1].split()
            h['cpu_seconds']=(int(fields[11])+int(fields[12]))/self.clock_ticks
            h['peak_rss_bytes']=max(h['peak_rss_bytes'],rss)
            if rss>self.grant.contract['budget']['rss_bytes']:raise self.g.Stop('COMPILE_WORKER_RSS_CAP')
            if virtual>self.grant.contract['budget']['address_space_bytes']:raise self.g.Stop('COMPILE_WORKER_AS_CAP')
            if int(values['Threads'])!=1:raise self.g.Stop('COMPILE_WORKER_THREADS')
            if os.sched_getaffinity(h['pid'])=={h['task']['cpu']}:
                h['affinity_confirmed']=True
            elif h['affinity_confirmed'] or time.monotonic()-h['started']>5:
                raise self.g.Stop('COMPILE_WORKER_CPU_BINDING')
            h['resource_last_seen']=time.monotonic()
        except KeyError as error:
            if error.args[0] not in ('VmRSS','VmSize'):raise
            # exit_mm can remove memory fields before State becomes Z and
            # before wait4(WNOHANG) reports exit. Never accept missing data as
            # a successful sample: retry/reap within the original 5s bound.
            self.reap(h)
            if h['reaped']:return
            current=self.g.identity(h['pid'])
            if current is None:return
            if current!=h['identity']:raise self.g.Stop('COMPILE_OWNERSHIP_CHANGED')
            last=h.setdefault('resource_last_seen',h['started'])
            if time.monotonic()-last>self.grant.contract['budget']['monitor_max_seconds']:
                raise self.g.Stop('COMPILE_MEMORY_OBSERVATION_MISSING')
            return
        except FileNotFoundError:return
        except ProcessLookupError as error:
            if error.errno!=errno.ESRCH:raise
            self.reap(h)
            return

    def poll(self,active):
        completed=[]
        for h in active:
            self.drain(h);self.reap(h)
            if h['reaped']:
                self.drain(h)
                if h['fd'] is not None:raise self.g.Stop('COMPILE_PIPE_NOT_CLOSED_AFTER_WAIT')
                completed.append(h)
            else:self.observe(h)
        own=resource.getrusage(resource.RUSAGE_SELF)
        self.monitor.child_cpu_seconds=self.prior_cpu+sum(h['cpu_seconds'] for h in self.handles)
        total=own.ru_utime+own.ru_stime+self.monitor.child_cpu_seconds
        if total>self.grant.contract['budget']['cpu_seconds']:raise self.g.Stop('AGGREGATE_STAGE_CPU_CAP')
        return completed

    def cleanup(self):
        errors=[]
        def wait_until(deadline):
            while time.monotonic()<deadline:
                for h in self.handles:
                    if h['reaped']:continue
                    try:self.reap(h)
                    except Exception as error:errors.append(str(error)[:200])
                if all(h['reaped'] for h in self.handles):return
                time.sleep(.05)
        for sig,seconds in CLEANUP_WAIT_SECONDS:
            for h in self.handles:
                if h['reaped']:continue
                try:
                    self.reap(h)
                    if not h['reaped']:
                        if h['identity'] is None:
                            current=self.g.identity(h['pid'])
                            if current is not None and current['parent_pid']==self.parent and current['uid']==os.getuid():
                                h['identity']=current
                        terminate_owned(h['pid'],h['identity'],sig,self.g)
                except Exception as error:errors.append(type(error).__name__+':'+str(error)[:200])
            try:
                if any(not h['reaped'] for h in self.handles):wait_until(time.monotonic()+seconds)
            except BaseException as error:errors.append('CLEANUP_WAIT:'+type(error).__name__+':'+str(error)[:200])
        for h in self.handles:
            if h['fd'] is not None:
                try:self.drain(h)
                except Exception as error:errors.append('CLEANUP_IPC:'+str(error)[:200])
            if h['fd'] is not None:
                try:os.close(h['fd']);h['fd']=None
                except OSError as error:errors.append(str(error)[:200])
        return dict(all_reaped=all(h['reaped'] for h in self.handles),errors=errors,
            workers=[dict(name=h['task']['name'],pid=h['pid'],identity=h['identity'],
                reaped=h['reaped'],native_exit_code=h['exit_code'],cpu_seconds=h['cpu_seconds'],
                peak_rss_bytes=h['peak_rss_bytes'],ipc_bytes=len(h['payload']),
                ipc_hex=bytes(h['payload']).hex()) for h in self.handles],automatic_retry=False)

    def emergency_cleanup(self):
        """Best effort after an exceptional cleanup; never reports success."""
        errors=[]
        for h in self.handles:
            try:
                self.reap(h)
                if not h['reaped']:
                    terminate_owned(h['pid'],h['identity'],signal.SIGKILL,self.g)
                    self.reap(h)
            except BaseException as error:errors.append(type(error).__name__+':'+str(error)[:200])
            finally:
                if h['fd'] is not None:
                    try:os.close(h['fd']);h['fd']=None
                    except BaseException as error:errors.append(str(error)[:200])
        return dict(all_reaped=False,errors=errors+['EXCEPTIONAL_CLEANUP_UNCONFIRMED'],automatic_retry=False)


def compile_batch(grant,cases,api,monitor,artifacts,calls,*,g,metrics_check,backend_factory=ForkBackend):
    grant.require('H8_NATIVE_B0I_FULL_FAMILY12_V1')
    config=grant.contract['parallel_compile'];count=grant.contract['workers']
    if (config.get('enabled') is not True or type(count) is not int or count!=4 or
            config.get('max_cases_in_memory')!=2 or
            config.get('max_workers')!=count or len(config.get('worker_cpus',[]))!=count or
            len(set(config['worker_cpus']))!=count):raise g.Stop('PARALLEL_HELPER_SCOPE')
    if grant.synthetic and backend_factory is ForkBackend:raise g.Stop('ARTIFICIAL_PARALLEL_REQUIRES_MOCK_BACKEND')
    external=None
    if grant.contract.get('external_resource_monitor',{}).get('enabled'):
        spec=importlib.util.spec_from_file_location('h8_parent_external_monitor',
            Path(__file__).with_name('h8_b0i_family_external_monitor_v1.py'))
        external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)
    tasks=[]
    for case in cases:
        if case['validation']['status']!='PASS':raise g.Stop('PARALLEL_REQUIRES_ALL_CORRECTNESS')
        for axis in ('cosine','sine'):
            wrapper=case['built']['wrappers'][axis];fp=case['validation']['fingerprints'][axis]
            # All native correctness checks already passed. A worker validates
            # its inherited COW input against this exact digest before compile
            # and rechecks it afterwards. Avoid the redundant parent pre-pass.
            tasks.append(dict(name=f"{case['prefix']}_{case['q']}_{axis}",prefix=case['prefix'],
                q=case['q'],axis=axis,wrapper=wrapper,fingerprint=fp))
    allowed={(p,q) for p in (7,14,21) for q in (1,2,4,8)}
    if (not 1<=len(cases)<=2 or len(tasks)!=2*len(cases) or
            len({t['name'] for t in tasks})!=len(tasks) or
            any((t['prefix'],t['q']) not in allowed for t in tasks)):
        raise g.Stop('H8_AT_MOST_TWO_CASES_FOUR_AXES_REQUIRED')
    expected_tasks=len(tasks)
    cleanup_name=f"wave_{cases[0]['prefix']}_{cases[0]['q']}_parallel_cleanup_v1.json"
    prior_cpu=monitor.child_cpu_seconds
    backend=backend_factory(grant,api,monitor,g);active=[];rows={};first=None;cleanup=None
    cleanup_signals=CleanupSignals(enabled=not grant.synthetic)
    try:
        while tasks or active:
            monitor.tick(force=True)
            while tasks and len(active)<count:
                used={h['task']['cpu'] for h in active}
                task=tasks.pop(0);task['cpu']=next(cpu for cpu in config['worker_cpus'] if cpu not in used)
                calls.take('compile');h=backend.start(task);active.append(h)
                started=dict(pid=h['pid'],identity=h['identity'],name=task['name'],
                    cpu=task['cpu'],reserved_before_fork=True)
                artifacts.write('compile_started_'+task['name']+'.json',started,monitor.tick)
                if external is not None:
                    external.atomic(Path(grant.contract['control_directory'])/
                        ('observed_compile_started_'+task['name']+'.json'),started)
            for h in backend.poll(active):
                task=h['task']
                try:payload=g.d0.strict_json(bytes(h['payload']))
                except Exception:
                    if h['exit_code']!=0:raise g.Stop('COMPILE_CHILD_STOP:'+task['name']+':native_exit='+str(h['exit_code'])+':IPC_MISSING_OR_INVALID')
                    raise g.Stop('COMPILE_IPC_MISSING_OR_INVALID:'+task['name'])
                if h['exit_code']!=0 or payload.get('status')!='PASS':
                    raise g.Stop('COMPILE_CHILD_STOP:'+task['name']+':native_exit='+str(h['exit_code'])+':'+str(payload.get('reason','NO_PASS')))
                if payload.get('name')!=task['name'] or payload.get('identity')!=h['identity']:
                    raise g.Stop('COMPILE_RESULT_OWNER_BINDING')
                row=payload['row'];metrics_check(row['metrics'])
                fp=task['fingerprint']
                if (row['fingerprint']!=fp or row.get('input_fingerprint_check')!=dict(before=fp,after=fp) or
                        row['compiler_hash']!=grant.contract['binding']['compiler']):
                    raise g.Stop('PARALLEL_COMPILED_IDENTITY')
                row=dict(row,worker=dict(identity=h['identity'],native_exit_code=h['exit_code'],
                    cpu_seconds=h['cpu_seconds'],peak_rss_bytes=h['peak_rss_bytes']))
                rows[(task['prefix'],task['q'],task['axis'])]=row
                artifacts.write('case_'+task['name']+'_cost.json',row,monitor.tick);active.remove(h)
            if active:time.sleep(.05)
        if len(rows)!=expected_tasks:raise g.Stop('PARALLEL_COVERAGE')
    except BaseException as error:first=error
    finally:
        try:
            cleanup_signals.defer()
            cleanup=backend.cleanup()
        except BaseException as error:
            if first is None:first=error
            cleanup=dict(all_reaped=False,errors=['CLEANUP_EXCEPTION:'+type(error).__name__+':'+str(error)[:200]])
            emergency=getattr(backend,'emergency_cleanup',None)
            if emergency is not None:
                try:cleanup['emergency']=emergency()
                except BaseException as fallback:cleanup['errors'].append('EMERGENCY:'+str(fallback)[:200])
        # Retain exact wait4 CPU after the backend and all children are gone.
        # Phase D and the final tick must not reset the stage CPU budget.
        monitor.child_cpu_seconds=prior_cpu+sum(h['cpu_seconds'] for h in backend.handles)
        completed=set(getattr(monitor,'completed_compile_names',()))
        completed.update(h['task']['name'] for h in backend.handles if h['reaped'])
        monitor.completed_compile_names=completed
        cleanup['child_cpu_seconds']=monitor.child_cpu_seconds
        if external is not None:
            try:
                external.atomic(Path(grant.contract['control_directory'])/'observed_compile_cpu_carry_v1.json',
                    dict(contract_sha256=grant.contract_sha,driver_identity=g.identity(),
                        child_cpu_seconds=monitor.child_cpu_seconds,completed_names=sorted(completed),all_reaped=cleanup.get('all_reaped') is True))
            except Exception as error:
                if first is None:first=error
        failed=first is not None or cleanup.get('all_reaped') is not True or bool(cleanup.get('errors'))
        # Failure never restores a pending signal before terminal recording,
        # so a later TERM/ALRM cannot replace the original STOP or skip wait4.
        monitor.cleanup_signal_deferral=cleanup_signals.receipt()
        cleanup['signal_deferral']=dict(monitor.cleanup_signal_deferral)
        cleanup['signal_deferral']['retained_until_process_exit']=bool(cleanup_signals.deferred and failed)
        try:artifacts.write(cleanup_name,cleanup,terminal=True)
        except Exception as error:
            if first is None:first=error
        cleanup_signals.finish(resume=not failed and first is None)
        monitor.cleanup_signal_deferral=cleanup_signals.receipt()
    if first is not None:raise first
    if not cleanup['all_reaped'] or cleanup['errors']:raise g.Stop('PARALLEL_CLEANUP_UNCONFIRMED')
    monitor.tick(force=True)
    return rows
