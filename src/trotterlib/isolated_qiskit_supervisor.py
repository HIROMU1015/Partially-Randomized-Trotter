"""Linux-only, private-process supervisor for fresh-interpreter case batches.

No Qiskit or science imports. Deadlines are contract inputs; computation and
readiness deadlines default to None. Cleanup grace periods limit shutdown only.
Launch this supervisor in its own Python process (see the preparation runner).
Frozen H2 launchers and their recorded source bytes are deliberately preserved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import threading
import time
import traceback

from .isolated_qiskit_execution import digest, read, save


def publish(path,value):
    """Atomic JSON publication with no overwrite (ready/release cannot tear)."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    try:
        with temporary.open("x") as stream:
            json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write("\n");stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class WatchPolicy:
    ready_timeout: float | None = None
    run_timeout: float | None = None
    heartbeat_timeout: float | None = None
    poll_seconds: float = .05
    terminate_grace: float = 5.
    kill_grace: float = 5.

    def __post_init__(self):
        for name,value in asdict(self).items():
            if value is None and name in {"ready_timeout","run_timeout","heartbeat_timeout"}:
                continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError("INVALID_WATCH_POLICY:"+name)


def proc_table():
    """Only return process identities, not argv or other users' environment."""
    table={}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try:
            text=(p/"stat").read_text()
            fields=text[text.rindex(")")+2:].split()
            table[int(p.name)]=dict(pid=int(p.name),state=fields[0],ppid=int(fields[1]),
                pgid=int(fields[2]),sid=int(fields[3]),start_ticks=int(fields[19]))
        except (OSError,ValueError,IndexError):
            continue  # Ordinary /proc exit race; Popen/pidfds decide exit status.
    return table


def _subreaper(value=None):
    libc=ctypes.CDLL(None,use_errno=True)
    if value is None:
        previous=ctypes.c_int()
        result=libc.prctl(37,ctypes.byref(previous),0,0,0)  # PR_GET_CHILD_SUBREAPER
        if result!=0: raise OSError(ctypes.get_errno(),"PR_GET_CHILD_SUBREAPER")
        return previous.value
    if libc.prctl(36,int(value),0,0,0)!=0:  # PR_SET_CHILD_SUBREAPER; this process only
        raise OSError(ctypes.get_errno(),"PR_SET_CHILD_SUBREAPER")


def worker_ready(output, release, *, poll_seconds=.05):
    """Worker handshake checks its own identity and the exact release record."""
    output,release=Path(output),Path(release)
    marker=read(output/"marker.json")
    parent=os.getppid()
    save(output/"worker_consumed.json",dict(pid=os.getpid(),no_retry=True))
    publish(output/"ready.json",dict(pid=os.getpid(),parent_pid=parent,case=marker["case"],
        launch_token=marker["launch_token"]))
    while True:
        if os.getppid()!=parent:
            raise RuntimeError("SUPERVISOR_DISAPPEARED_BEFORE_RELEASE")
        if release.exists():
            token=read(release)["tokens"].get(marker["case"])
            if token!=marker["launch_token"]: raise ValueError("WRONG_RELEASE_TOKEN")
            return
        time.sleep(poll_seconds)


def run_supervised(jobs, run, *, cwd, policy=None, stop_request=None, claim_directory=None):
    """Run a fixed batch once in a dedicated main-thread supervisor process.

    Output paths are owned by this run. Any abnormal event stops remaining work,
    sends TERM then KILL using identity-bound pidfds, and reaps direct/adopted
    descendants. Failures never cause re-submission. SIGINT/SIGTERM are caught;
    the supervisor's own parent death is detected by ppid/pidfd. SIGKILL of the
    supervisor, kernel uninterruptible sleep and system failure cannot be given
    a universal cleanup guarantee; unresolved identities are explicit in STOP.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("PRIVATE_MAIN_THREAD_SUPERVISOR_REQUIRED")
    if not hasattr(os,"pidfd_open") or not hasattr(signal,"pidfd_send_signal"):
        raise RuntimeError("LINUX_PIDFD_REQUIRED")
    run=Path(run).resolve()
    cwd=Path(cwd).resolve()
    policy=WatchPolicy() if policy is None else policy
    if not jobs: raise ValueError("EMPTY_BATCH")
    names=[j["case"] for j in jobs]
    identities=[j.get("case_id",j["case"]) for j in jobs]
    outputs=[str(Path(j["output"]).resolve()) for j in jobs]
    if len(set(names))!=len(names) or len(set(identities))!=len(identities):
        raise ValueError("DUPLICATE_CASE")
    if len(set(outputs))!=len(outputs): raise ValueError("OUTPUT_COLLISION")
    for name,out,job in zip(names,outputs,jobs,strict=True):
        if not isinstance(name,str) or not name or Path(out).parent!=run:
            raise ValueError("OUTPUT_MUST_BE_DIRECT_CHILD_OF_NEW_RUN")
        if not job.get("argv") or any(not isinstance(x,str) for x in job["argv"]):
            raise ValueError("INVALID_ARGV")
        if Path(out).exists(): raise FileExistsError(out)
    # No other children may be present: adopted orphans then have unambiguous
    # ownership. This excludes using the API inside an unrelated job manager.
    if any(v["ppid"]==os.getpid() for v in proc_table().values()):
        raise RuntimeError("SUPERVISOR_ALREADY_HAS_CHILDREN")
    # Validate kernel setup before consumption, without starting any children.
    previous_subreaper=_subreaper()
    run.mkdir()  # Atomic non-refundable one-shot; caller owns the parent dir.
    started=time.monotonic()
    parent=os.getppid()
    parent_fd=None
    parent_poll=select.poll()
    previous_handlers={}
    interrupted=[]
    children=[]
    owned={}
    errors=[]
    reports=[]
    released=None
    reason=None
    pending=[]
    persistence_errors=[]
    sent=set()

    def evidence(path,value):
        try: publish(path,value)
        except BaseException as exc:
            persistence_errors.append(dict(path=str(path),reason=repr(exc)))
            raise

    def journal(event,**value):
        try:
            with (run/"ledger.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event=event,monotonic=time.monotonic(),**value),sort_keys=True)+"\n")
        except BaseException as exc:
            persistence_errors.append(dict(path="ledger.jsonl",reason=repr(exc)))
            raise

    def receive(signum,frame):
        if not interrupted: interrupted.append(signal.Signals(signum).name)

    def discover():
        table=proc_table()
        leaders={c["p"].pid:c["job"]["case"] for c in children}
        # Repeat ancestry closure so grandchildren are tracked even when /proc
        # order differs. A private subreaper also owns newly adopted escapees.
        changed=True
        while changed:
            changed=False
            for pid,v in table.items():
                if pid==os.getpid() or pid in owned: continue
                ppid=v["ppid"]
                if pid in leaders or ppid in owned or ppid==os.getpid():
                    case=leaders.get(pid,owned.get(ppid,{}).get("case","adopted_descendant"))
                    try:
                        fd=os.pidfd_open(pid)
                        current=proc_table().get(pid)
                        if current is None or current["start_ticks"]!=v["start_ticks"]:
                            os.close(fd);continue
                    except ProcessLookupError: continue
                    owned[pid]=dict(**v,case=case,fd=fd,reaped=False)
                    changed=True
        # Reap only adopted descendants; direct leader statuses belong to Popen.
        for pid,item in owned.items():
            if pid in leaders: continue
            try:
                found,status=os.waitpid(pid,os.WNOHANG)
                if found: item["reaped"]=True;item["wait_status"]=status
            except ChildProcessError:
                pass  # Descendant is still owned by its original parent.
        return table

    def alive(table):
        return [v for pid,v in owned.items() if pid in table and
                table[pid]["start_ticks"]==v["start_ticks"] and not v["reaped"]]

    def send(items,signum):
        for v in items:
            key=(v["pid"],v["start_ticks"],signum)
            if key in sent: continue
            try:
                signal.pidfd_send_signal(v["fd"],signum)
                sent.add(key)
            except ProcessLookupError: pass
            except OSError as exc: errors.append(dict(pid=v["pid"],signal=signum,reason=repr(exc)))

    try:
        parent_fd=os.pidfd_open(parent)
        parent_poll.register(parent_fd,select.POLLIN)
        _subreaper(1)
        for signum in [signal.SIGINT,signal.SIGTERM]:
            previous_handlers[signum]=signal.signal(signum,receive)
        evidence(run/"marker.json",dict(state="CONSUMED_NONREFUNDABLE",no_retry=True,
            cases=names,policy=asdict(policy),supervisor_pid=os.getpid(),parent_pid=parent))
        journal("BATCH_CONSUMED",cases=names)
        claims=Path(claim_directory) if claim_directory is not None else run.parent/".case_claims"
        if claims.is_symlink(): raise ValueError("CLAIM_DIRECTORY_SYMLINK")
        claims.mkdir(parents=True,exist_ok=True)
        if claims.stat().st_uid!=os.getuid(): raise ValueError("CLAIM_DIRECTORY_NOT_OWNED")
        for job in jobs:
            identity=dict(case_id=job.get("case_id",job["case"]),binding=job.get("binding",{}))
            # Fixed per-grant claim_directory must be recorded in a future
            # scientific contract. Changing only run ID cannot refund a claim.
            evidence(claims/(digest(identity)+".json"),dict(**identity,run=str(run),no_retry=True))
        # All outputs/markers/logs are opened before any worker is launched.
        for job,out in zip(jobs,outputs,strict=True):
            out=Path(out);out.mkdir()
            token=digest(dict(run=str(run),case=job["case"],binding=job.get("binding",{})))
            evidence(out/"marker.json",dict(case=job["case"],case_id=job.get("case_id",job["case"]),
                state="CONSUMED_NONREFUNDABLE",launch_token=token,binding=job.get("binding",{})))
            stdout=(out/"stdout.log").open("x")
            try: stderr=(out/"stderr.log").open("x")
            except BaseException: stdout.close();raise
            children.append(dict(job=job,out=out,p=None,stdout=stdout,stderr=stderr,token=token,
                                 launched=None,finished=None,code=None))
        for c in children:
            if interrupted: raise InterruptedError(interrupted[0])
            p=subprocess.Popen(c["job"]["argv"],cwd=cwd,stdout=c["stdout"],stderr=c["stderr"],
                               start_new_session=True)
            c.update(p=p,launched=time.monotonic())
            evidence(c["out"]/"launch.json",dict(pid=p.pid,fresh_interpreter=True,
                start_new_session=True,argv=c["job"]["argv"],binding=c["job"].get("binding",{})))
            journal("LAUNCHED",case=c["job"]["case"],pid=p.pid)
        while True:
            table=discover()
            for c in children:
                code=c["p"].poll()
                if code is not None and c["finished"] is None:
                    c.update(code=code,finished=time.monotonic())
                    if released is None: raise RuntimeError("WORKER_EXITED_BEFORE_RELEASE:"+c["job"]["case"])
                    if code!=0: raise RuntimeError("NONZERO_WORKER_EXIT:"+c["job"]["case"])
            if interrupted: raise InterruptedError(interrupted[0])
            if os.getppid()!=parent or parent_poll.poll(0): raise InterruptedError("PARENT_DISAPPEARED")
            if stop_request is not None and Path(stop_request).exists(): raise InterruptedError("STOP_REQUEST")
            now=time.monotonic()
            if released is None:
                ready=[]
                for c in children:
                    path=c["out"]/"ready.json"
                    if path.exists():
                        record=read(path)
                        if (record.get("pid")!=c["p"].pid or record.get("parent_pid")!=os.getpid()
                                or record.get("launch_token")!=c["token"] or record.get("case")!=c["job"]["case"]):
                            raise ValueError("INVALID_READY_IDENTITY:"+c["job"]["case"])
                        ready.append(c)
                    elif policy.ready_timeout is not None and now-c["launched"]>policy.ready_timeout:
                        raise TimeoutError("READY_TIMEOUT:"+c["job"]["case"])
                if len(ready)==len(children):
                    evidence(run/"release.json",dict(tokens={c["job"]["case"]:c["token"] for c in children}))
                    released=now
                    journal("RELEASED",cases=names)
            else:
                if all(c["finished"] is not None for c in children):
                    if alive(discover()): raise RuntimeError("LEFTOVER_DESCENDANT_AFTER_WORKER_EXIT")
                    break
                if policy.run_timeout is not None and now-released>policy.run_timeout:
                    raise TimeoutError("RUN_TIMEOUT")
                if policy.heartbeat_timeout is not None:
                    for c in children:
                        if c["finished"] is not None: continue
                        heartbeat=c["out"]/"heartbeat.json"
                        mtime=heartbeat.stat().st_mtime if heartbeat.exists() else None
                        age=time.time()-mtime if mtime is not None else now-released
                        if age>policy.heartbeat_timeout: raise TimeoutError("HEARTBEAT_TIMEOUT:"+c["job"]["case"])
            time.sleep(policy.poll_seconds)
    except BaseException as exc:
        reason=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
    finally:
        # Drop unlaunched placeholders before ancestry/cleanup.
        launched=[c for c in children if c["p"] is not None]
        placeholders=[c for c in children if c["p"] is None]
        children=launched
        try:
            if reason is not None:
                # Discovery continues during TERM grace, including descendants
                # that change session or are adopted after leader termination.
                for signum,grace in [(signal.SIGTERM,policy.terminate_grace),(signal.SIGKILL,policy.kill_grace)]:
                    end=time.monotonic()+grace
                    while True:
                        for c in children: c["p"].poll()
                        table=discover()
                        live=alive(table)
                        if not live: break
                        send(live,signum)
                        if time.monotonic()>=end: break
                        time.sleep(policy.poll_seconds)
            for c in children:
                code=c["p"].poll()
                if code is not None:
                    c["p"].wait()  # Already terminated: never an unbounded wait.
                    if c["p"].pid in owned: owned[c["p"].pid]["reaped"]=True
                report=dict(case=c["job"]["case"],pid=c["p"].pid,exit_code=code,
                    launched_monotonic=c["launched"],finished_monotonic=c["finished"],
                    elapsed_until_reaped=time.monotonic()-c["launched"],aborted=reason is not None,
                    binding=c["job"].get("binding",{}))
                reports.append(report)
                try: evidence(c["out"]/"process.json",report)
                except BaseException: pass
            for c in placeholders:
                report=dict(case=c["job"]["case"],pid=None,exit_code=None,status="NOT_LAUNCHED",aborted=True)
                reports.append(report)
                try: evidence(c["out"]/"process.json",report)
                except BaseException: pass
            pending=[{k:v for k,v in x.items() if k!="fd"} for x in alive(discover())]
        except BaseException as exc:
            errors.append(dict(cleanup_error=repr(exc),traceback=traceback.format_exc()))
            pending=[{k:v for k,v in x.items() if k!="fd"} for x in owned.values() if not x["reaped"]]
        finally:
            # Never report successful reaping merely because /proc discovery
            # failed or a task is uninterruptible. Popen remains authoritative
            # for every directly launched child, even if it has no owned entry.
            pending_pids={p["pid"] for p in pending}
            for c in children:
                if c["p"].poll() is None and c["p"].pid not in pending_pids:
                    pending.append(dict(pid=c["p"].pid,case=c["job"]["case"],
                        state="UNREAPED_DIRECT_CHILD",identity_known_by_Popen=True))
            for c in launched+placeholders:
                for stream in [c["stdout"],c["stderr"]]:
                    try: stream.close()
                    except BaseException as exc: persistence_errors.append(dict(close_error=repr(exc)))
            for item in owned.values(): os.close(item["fd"])
            if parent_fd is not None: os.close(parent_fd)
            for signum,handler in previous_handlers.items(): signal.signal(signum,handler)
            _subreaper(previous_subreaper)
        success=reason is None and not errors and not persistence_errors and not pending
        status="COMPLETE_MANDATORY_STOP" if success else "FAILED_NO_RETRY_MANDATORY_STOP"
        final=dict(status=status,reason=reason,errors=errors,persistence_errors=persistence_errors,
            reports=reports,pending_owned_processes=pending,all_children_reaped=not pending,
            elapsed_seconds=time.monotonic()-started,automatic_retry=False,automatic_next_stage=False,
            identities=[{k:v for k,v in x.items() if k!="fd"} for x in owned.values()])
        try:
            journal("MANDATORY_STOP",status=status,all_children_reaped=not pending)
            evidence(run/"STOP.json",final)
        except BaseException as exc:
            # Storage failure cannot be made into a successful persistence claim.
            # stderr is a separate channel retained by the caller of this CLI.
            print(json.dumps(dict(status="STOP_PERSISTENCE_FAILED",run=str(run),reason=repr(exc),
                pending_owned_processes=pending)),file=__import__("sys").stderr,flush=True)
            final.update(status="STOP_PERSISTENCE_FAILED",persistence_errors=persistence_errors)
    return final
