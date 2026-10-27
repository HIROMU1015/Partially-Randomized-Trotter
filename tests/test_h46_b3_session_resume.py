"""Artificial checkpoints and real tiny child processes; no science kernels."""
from pathlib import Path
import json
import subprocess
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_b3_session_resume as resume


def checkpoint(root,row,index,*,stop=False):
    directory=root/row['id']/str(index);directory.mkdir(parents=True)
    value=dict(status='PASS',candidate=row,cohort='B3_EXPLORATION',index=index,
        conditional=False,orders={'0':row['s']})
    (directory/'result.json').write_text(json.dumps(value))
    (directory/'events_PRIVATE.json').write_text('saved events')
    if stop:(directory/'STOP.json').write_text('{}')
    return directory


def test_completed_and_unfinished_inventory_preserves_bytes(tmp_path):
    row={'id':'synthetic','s':2}
    done=checkpoint(tmp_path,row,0);before=(done/'result.json').read_bytes()
    unfinished=tmp_path/row['id']/'1';unfinished.mkdir()
    (unfinished/'events_PRIVATE.json').write_text('interrupted events')
    observed=resume.inspect_trials([row],tmp_path)
    assert observed==dict(complete=1,unfinished=[unfinished],unstarted=7)
    assert (done/'result.json').read_bytes()==before
    assert (unfinished/'events_PRIVATE.json').read_text()=='interrupted events'


@pytest.mark.parametrize('stop',['STOP.json','technical_exit.json'])
def test_no_failure_is_reopened_or_any_directory_moved(tmp_path,stop):
    row={'id':'synthetic','s':2}
    checkpoint(tmp_path,row,0)
    interrupted=tmp_path/row['id']/'1';interrupted.mkdir()
    (interrupted/stop).write_text('{}')
    with pytest.raises(ValueError,match='PRESERVED_TRIAL_STOP'):
        resume.inspect_trials([row],tmp_path)
    assert interrupted.exists()
    assert not interrupted.with_name('1'+resume.SUFFIX).exists()


def test_changed_trial_binding_is_rejected(tmp_path):
    row={'id':'synthetic','s':2};directory=checkpoint(tmp_path,row,0)
    result=json.loads((directory/'result.json').read_text());result['cohort']='B3_CONFIRMATION'
    (directory/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='RECOVERED_TRIAL_BINDING'):
        resume.inspect_trials([row],tmp_path)


@pytest.mark.parametrize('exitcode',[0,7])
def test_process_completion_reports_real_exit_without_restart(tmp_path,exitcode):
    count=tmp_path/'runs'
    script="from pathlib import Path;import sys,time;Path(sys.argv[1]).write_text('one run');time.sleep(.1);sys.exit(int(sys.argv[2]))"
    child=subprocess.Popen([sys.executable,'-c',script,str(count),str(exitcode)])
    assert resume.process_state(child)['ActiveState']=='active'
    child.wait(timeout=5)
    state=resume.process_state(child)
    assert state['ActiveState']=='inactive'
    assert state['ExecMainStatus']==str(exitcode)
    assert state['Result']==('success' if exitcode==0 else 'exit-code')
    assert count.read_text()=='one run'
