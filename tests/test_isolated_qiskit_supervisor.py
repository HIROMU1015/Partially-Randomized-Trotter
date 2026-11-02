"""Artificial owned processes only; no SCF, Hamiltonian or molecular evolution."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from trotterlib.isolated_qiskit_execution import read, save
from trotterlib.isolated_qiskit_supervisor import WatchPolicy

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"scripts/run_non_df_h4_readiness.py"


@pytest.fixture
def area(tmp_path,request):
    configured=os.environ.get("NON_DF_READINESS_EVIDENCE_ROOT")
    if configured:
        out=Path(configured)/request.node.name.replace("/","_")
        out.mkdir(parents=True)
        return out
    return tmp_path


def policy(**changes):
    values=dict(ready_timeout=3.,run_timeout=2.,poll_seconds=.02,terminate_grace=.2,kill_grace=1.)
    values.update(changes)
    return values


def launch(area,modes,*,name="run",watch=None,patch=None):
    run=area/name
    jobs=[dict(case=f"case{i}",mode=mode) for i,mode in enumerate(modes)]
    spec=dict(run=str(run),jobs=jobs,policy=policy() if watch is None else watch)
    path=area/(name+".spec.json");save(path,spec)
    argv=[sys.executable,str(RUNNER),"--supervise",str(path)]
    if patch:
        code="import sys,runpy; "+patch+"; sys.argv="+repr([str(RUNNER),"--supervise",str(path)])+"; runpy.run_path("+repr(str(RUNNER))+",run_name='__main__')"
        argv=[sys.executable,"-c",code]
    stdout=(area/(name+".stdout.log")).open("x")
    stderr=(area/(name+".stderr.log")).open("x")
    p=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr)
    return p,run,stdout,stderr


def finish(handle):
    p,run,stdout,stderr=handle
    try:
        code=p.wait(timeout=15.)  # Artificial test watchdog; no science deadline.
    finally:
        stdout.close();stderr.close()
    result=read(run/"STOP.json")
    assert result["automatic_retry"] is False
    assert result["all_children_reaped"] and not result["pending_owned_processes"]
    assert all(not Path(f"/proc/{r['pid']}").exists() for r in result["reports"] if r["pid"] is not None)
    return code,result


def wait_path(path,process):
    end=time.monotonic()+8.
    while not path.exists():
        assert process.poll() is None
        if time.monotonic()>end: raise AssertionError("artificial worker did not reach checkpoint")
        time.sleep(.02)


def test_four_normal_workers_and_bound_handshake(area):
    h=launch(area,["normal"]*4)
    code,result=finish(h)
    assert code==0 and result["status"]=="COMPLETE_MANDATORY_STOP"
    assert len({r["pid"] for r in result["reports"]})==4
    for i in range(4):
        out=h[1]/f"case{i}"
        assert "artificial stdout" in (out/"stdout.log").read_text()
        assert "artificial stderr" in (out/"stderr.log").read_text()
        assert read(out/"ready.json")["launch_token"]==read(out/"marker.json")["launch_token"]


@pytest.mark.parametrize("mode,reason",[("noready","READY_TIMEOUT"),("earlyexit","WORKER_EXITED_BEFORE_RELEASE"),
    ("nonzero","NONZERO_WORKER_EXIT"),("hang","RUN_TIMEOUT"),("ignoreterm","RUN_TIMEOUT"),
    ("badready","INVALID_READY_IDENTITY")])
def test_failure_modes_stop_and_reap(area,mode,reason):
    watch=policy(ready_timeout=.5 if mode=="noready" else 3.,run_timeout=3. if mode=="nonzero" else .3)
    code,result=finish(launch(area,[mode],watch=watch))
    assert code==1 and result["status"]=="FAILED_NO_RETRY_MANDATORY_STOP"
    assert reason in result["reason"]["reason"]
    if mode=="earlyexit":
        assert "ARTIFICIAL_STARTUP_EXCEPTION" in (area/"run/case0/stderr.log").read_text()
    if mode=="nonzero": assert result["reports"][0]["exit_code"]==7
    if mode=="ignoreterm": assert result["reports"][0]["exit_code"]==-signal.SIGKILL


def test_partial_failure_aborts_only_owned_peer_and_preserves_unrelated_process(area):
    sentinel=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])
    try:
        code,result=finish(launch(area,["hang","nonzero"],watch=policy(run_timeout=None)))
        assert code==1 and "NONZERO_WORKER_EXIT" in result["reason"]["reason"]
        assert len(result["reports"])==2 and sentinel.poll() is None
    finally:
        sentinel.terminate();sentinel.wait(timeout=5)


@pytest.mark.parametrize("signum",[signal.SIGINT,signal.SIGTERM])
def test_supervisor_interruption_reaps_workers(area,signum):
    h=launch(area,["ignoreterm","hang"],watch=policy(run_timeout=None))
    wait_path(h[1]/"release.json",h[0])
    h[0].send_signal(signum)
    code,result=finish(h)
    assert code==1 and signal.Signals(signum).name in result["reason"]["reason"]


def test_external_stop_request(area):
    h=launch(area,["ignoreterm"],watch=policy(run_timeout=None))
    wait_path(h[1]/"release.json",h[0])
    (h[1]/"STOP.request").write_text("user cancellation")
    code,result=finish(h)
    assert code==1 and result["reason"]["reason"]=="STOP_REQUEST"


def test_heartbeat_timeout_is_opt_in(area):
    h=launch(area,["hang"],watch=policy(run_timeout=None,heartbeat_timeout=.3))
    code,result=finish(h)
    assert code==1 and "HEARTBEAT_TIMEOUT" in result["reason"]["reason"]


def test_contract_can_leave_normal_compute_without_wall_limit(area):
    h=launch(area,["normal"],watch=policy(run_timeout=None,ready_timeout=None))
    code,result=finish(h)
    assert code==0 and read(h[1]/"marker.json")["policy"]["run_timeout"] is None


def test_detached_descendants_and_grandchild_are_killed_and_reaped(area):
    h=launch(area,["descendants"],watch=policy(run_timeout=None))
    wait_path(h[1]/"case0/family.json",h[0])
    family=read(h[1]/"case0/family.json")
    (h[1]/"STOP.request").write_text("family created; exercise cleanup")
    code,result=finish(h)
    assert code==1 and "STOP_REQUEST" in result["reason"]["reason"]
    assert all(not Path(f"/proc/{pid}").exists() for pid in family.values())
    assert set(family.values()).issubset({r["pid"] for r in result["identities"]})


@pytest.mark.parametrize("target",["stdout.log","stderr.log"])
def test_log_open_failure_has_STOP_and_no_worker_launch(area,target):
    patch=("from pathlib import Path; old_open=Path.open; "
        "exec(\"def faulty_open(self,*a,**k):\\n if self.name=="+repr(target)+": raise OSError('ARTIFICIAL_LOG_SAVE_FAILURE')\\n return old_open(self,*a,**k)\"); Path.open=faulty_open")
    code,result=finish(launch(area,["normal"],patch=patch))
    assert code==1 and "ARTIFICIAL_LOG_SAVE_FAILURE" in result["reason"]["reason"]
    assert not result["identities"]


def test_second_popen_failure_reaps_first_child(area):
    patch=("import subprocess; original_popen=subprocess.Popen; calls=0; "
        "exec(\"def faulty_popen(*a,**k):\\n global calls\\n calls+=1\\n if calls==2: raise OSError('ARTIFICIAL_POPEN_FAILURE')\\n return original_popen(*a,**k)\"); subprocess.Popen=faulty_popen")
    code,result=finish(launch(area,["hang","hang"],patch=patch))
    assert code==1 and "ARTIFICIAL_POPEN_FAILURE" in result["reason"]["reason"]
    assert len(result["reports"])==2
    assert result["reports"][1]["status"]=="NOT_LAUNCHED"


def test_parent_runtime_exception_reaps_already_started_children(area):
    patch=("import trotterlib.isolated_qiskit_supervisor as s; original=s.publish; "
        "exec(\"def faulty_publish(path,value):\\n if str(path).endswith('release.json'): raise RuntimeError('ARTIFICIAL_PARENT_EXCEPTION')\\n return original(path,value)\"); s.publish=faulty_publish")
    code,result=finish(launch(area,["hang","hang"],patch=patch))
    assert code==1 and "ARTIFICIAL_PARENT_EXCEPTION" in result["reason"]["reason"]


def test_run_consumption_and_cross_run_case_claim_are_not_refunded(area):
    code,result=finish(launch(area,["normal"]))
    assert code==0
    old=(area/"run/STOP.json").read_bytes()
    second=launch(area,["normal"],name="another_run")
    code,result=finish(second)
    assert code==1 and "FileExistsError"==result["reason"]["type"]
    assert not result["identities"] and (area/"run/STOP.json").read_bytes()==old
    with (area/"duplicate.stdout.log").open("x") as so,(area/"duplicate.stderr.log").open("x") as se:
        p=subprocess.run([sys.executable,str(RUNNER),"--supervise",str(area/"run.spec.json")],cwd=ROOT,stdout=so,stderr=se,timeout=5)
    assert p.returncode!=0 and (area/"run/STOP.json").read_bytes()==old


def test_claim_directory_symlink_is_rejected_without_writing_target(area):
    external=area/"other_job";external.mkdir()
    (external/"protected.txt").write_text("protected")
    (area/".case_claims").symlink_to(external.resolve(),target_is_directory=True)
    code,result=finish(launch(area,["normal"]))
    assert code==1 and "CLAIM_DIRECTORY_SYMLINK" in result["reason"]["reason"]
    assert sorted(p.name for p in external.iterdir())==["protected.txt"]


@pytest.mark.parametrize("field",["case","case_id"])
def test_duplicate_case_identity_rejected_before_launch(area,field):
    spec=dict(run=str(area/"run"),jobs=[dict(case="a",mode="normal",case_id="one"),dict(case="b",mode="normal",case_id="two")])
    spec["jobs"][1][field]=spec["jobs"][0][field]
    path=area/"spec.json";save(path,spec)
    p=subprocess.run([sys.executable,str(RUNNER),"--supervise",str(path)],cwd=ROOT,capture_output=True,text=True,timeout=5)
    (area/"rejection.stdout.log").write_text(p.stdout)
    (area/"rejection.stderr.log").write_text(p.stderr)
    assert p.returncode!=0 and "DUPLICATE_CASE" in p.stderr
    assert not (area/"run").exists()


@pytest.mark.parametrize("value",[0.,-.1,float("inf"),float("nan"),True])
def test_invalid_watch_configuration(value):
    with pytest.raises(ValueError): WatchPolicy(run_timeout=value)


def test_h4_science_entry_is_denied_before_any_generation(area):
    p=subprocess.run([sys.executable,str(RUNNER),"--h4-execute"],cwd=ROOT,capture_output=True,text=True,timeout=5)
    (area/"rejection.stdout.log").write_text(p.stdout)
    (area/"rejection.stderr.log").write_text(p.stderr)
    assert p.returncode==2 and "H4_SCIENCE_NOT_AUTHORIZED" in p.stderr


def test_supervisor_parent_disappearance_is_detected_and_descendants_reaped(area):
    area=area.resolve()
    spec=dict(run=str(area/"run"),jobs=[dict(case="case0",mode="hang")],policy=policy(run_timeout=None))
    sp=area/"spec.json";save(sp,spec)
    # A private wrapper adopts and reaps the supervisor after its intermediate
    # parent exits. No pytest or shared-session subreaper setting is changed.
    middle=("import pathlib,subprocess,sys,time,os; p=subprocess.Popen("+
        repr([sys.executable,str(RUNNER),"--supervise",str(sp)])+
        "); release=pathlib.Path("+repr(str(area/"run/release.json"))+
        "); exec('while not release.exists():\\n time.sleep(.02)'); os._exit(0)")
    wrapper=("import ctypes,subprocess,os; ctypes.CDLL(None).prctl(36,1,0,0,0); "
        "p=subprocess.Popen("+repr([sys.executable,"-c",middle])+ "); p.wait(); "
        "exec('while True:\\n try: os.waitpid(-1,0)\\n except ChildProcessError: break')")
    with (area/"parent-death.stdout.log").open("x") as so,(area/"parent-death.stderr.log").open("x") as se:
        p=subprocess.run([sys.executable,"-c",wrapper],cwd=ROOT,stdout=so,stderr=se,timeout=15)
    assert p.returncode==0
    result=read(area/"run/STOP.json")
    assert result["status"]=="FAILED_NO_RETRY_MANDATORY_STOP"
    assert "PARENT_DISAPPEARED" in result["reason"]["reason"]
    assert result["all_children_reaped"]


def test_output_collision_is_rejected_without_touching_existing_file(area):
    run=area.resolve()/"run"
    existing=area.resolve()/"external";existing.mkdir()
    (existing/"marker.json").write_text("protected")
    code=("from pathlib import Path; from trotterlib.isolated_qiskit_supervisor import run_supervised; "
        "jobs=[dict(case='a',output=Path("+repr(str(existing))+"),argv=['unused']),"
        "dict(case='b',output=Path("+repr(str(existing))+"),argv=['unused'])]; "
        "run_supervised(jobs,Path("+repr(str(run))+"),cwd=Path.cwd())")
    p=subprocess.run([sys.executable,"-c",code],cwd=ROOT,capture_output=True,text=True,timeout=5)
    (area/"rejection.stdout.log").write_text(p.stdout)
    (area/"rejection.stderr.log").write_text(p.stderr)
    assert p.returncode!=0 and "OUTPUT_COLLISION" in p.stderr
    assert (existing/"marker.json").read_text()=="protected" and not run.exists()


def test_STOP_storage_failure_is_reported_without_claiming_success(area):
    patch=("import trotterlib.isolated_qiskit_supervisor as s; original=s.publish; "
        "exec(\"def faulty_publish(path,value):\\n if str(path).endswith('STOP.json'): raise OSError('ARTIFICIAL_STOP_STORAGE_FAILURE')\\n return original(path,value)\"); s.publish=faulty_publish")
    h=launch(area,["normal"],patch=patch)
    p,run,stdout,stderr=h
    try: code=p.wait(timeout=15)
    finally: stdout.close();stderr.close()
    assert code==1 and not (run/"STOP.json").exists()
    text=(area/"run.stderr.log").read_text()
    assert "STOP_PERSISTENCE_FAILED" in text and "ARTIFICIAL_STOP_STORAGE_FAILURE" in text
    row=read(run/"case0/process.json")
    assert row["exit_code"]==0 and not Path(f"/proc/{row['pid']}").exists()
