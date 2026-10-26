"""Synthetic recovery scope, STOP preservation and completed-data reuse."""
from pathlib import Path
import json
import shutil
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_b3_trace_recovery as r


@pytest.mark.parametrize('error',[dict(exitcode=-9,native_STOP=False),dict(exitcode=-7,native_STOP=True)])
def test_scientific_stops_and_non_sigbus_exits_are_not_reopened(error):
    with pytest.raises(ValueError,match='ONLY_TECHNICAL_SIGBUS'):
        r.technical_failures(dict(errors=[error]))


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def result(row,index):
    return dict(status='PASS',candidate=row,index=index,cohort='B3_EXPLORATION',conditional=False,orders={'0':row['s']})


def setup(tmp_path,monkeypatch):
    private=tmp_path/'private';public=tmp_path/'public'
    monkeypatch.setattr(r.b,'PRIVATE',private);monkeypatch.setattr(r.b,'PUBLIC',public)
    monkeypatch.setattr(r,'WORK',private/'diagnostic_run03')
    monkeypatch.setattr(r,'save',put)
    row=dict(id='synthetic',s=2)
    put(public/'native_selection.json',dict(candidates=[row]))
    (private/'interruptions').mkdir(parents=True)
    progress=dict(errors=[dict(exitcode=-7,native_STOP=False)],complete=1)
    put(private/'B3_EXPLORATION_progress.json',progress)
    return row,progress,private/'cost/B3_EXPLORATION/synthetic'


def test_adoption_preserves_successes_and_failed_originals(tmp_path,monkeypatch):
    row,progress,parent=setup(tmp_path,monkeypatch)
    put(parent/'0/result.json',result(row,0));old=(parent/'0/result.json').read_bytes()
    put(parent/'1/technical_exit.json',dict(exitcode=-7))
    diagnostic=r.WORK/'replays/cost/B3_EXPLORATION/synthetic/1'
    put(diagnostic/'result.json',result(row,1))
    digest=r.b.signal.sha(diagnostic/'result.json')
    put(diagnostic/'diagnostic_replay_check.json',dict(status='PASS',result_sha256=digest))
    r.preserve_and_adopt(progress)
    assert (parent/'0/result.json').read_bytes()==old
    assert (parent/'1_unfinished_run03/technical_exit.json').exists()
    assert r.b.signal.sha(parent/'1/result.json')==digest
    assert not diagnostic.exists()


def test_failed_diagnostic_leaves_original_directories_untouched(tmp_path,monkeypatch):
    row,progress,parent=setup(tmp_path,monkeypatch)
    put(parent/'1/technical_exit.json',dict(exitcode=-7))
    put(r.WORK/'replays/cost/B3_EXPLORATION/synthetic/1/STOP.json',dict(reason='synthetic'))
    with pytest.raises(ValueError,match='DIAGNOSTIC_NOT_ACCEPTED'):
        r.preserve_and_adopt(progress)
    assert (parent/'1/technical_exit.json').exists()
    assert not (parent/'1_unfinished_run03').exists()
    assert not (r.b.PRIVATE/'interruptions/run03_SIGBUS').exists()


def test_wrong_event_count_is_not_accepted():
    row=dict(id='synthetic',s=2);value=result(row,0);value['orders']={'0':1}
    with pytest.raises(ValueError,match='RECOVERED_EVENT_COUNT'):
        r.check_trial(value,row,0)


@pytest.mark.skipif(shutil.which('strace') is None,reason='strace unavailable')
@pytest.mark.parametrize('crash',[False,True])
def test_traced_child_reports_real_pid_and_exit_signal(tmp_path,monkeypatch,crash):
    script=tmp_path/'synthetic_worker.py'
    script.write_text('import os,resource,signal,time\nresource.setrlimit(resource.RLIMIT_CORE,(0,0))\ntime.sleep(.2)\n'+
        ('os.kill(os.getpid(),signal.SIGBUS)\n' if crash else ''))
    monkeypatch.setattr(r,'WORK',tmp_path/'trace')
    monkeypatch.setattr(r,'SCRIPT',script)
    monkeypatch.setattr(r.b,'ROOT',tmp_path)
    process=r.TracedProcess(None,((dict(id='synthetic'), 'B3_EXPLORATION',0),))
    process.start()
    assert process.pid!=process.process.pid
    process.join()
    assert process.exitcode==(-7 if crash else 0)
    if crash:
        logs=list((r.WORK/'run04_traces/B3_EXPLORATION/synthetic/0').glob('signals.*'))
        assert any('SIGBUS' in p.read_text() and 'SI_USER' in p.read_text() for p in logs)
