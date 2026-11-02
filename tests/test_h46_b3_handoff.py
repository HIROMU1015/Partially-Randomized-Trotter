"""Real synthetic processes: all workers finish before controller interruption."""
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_b3_handoff as handoff


def test_workers_finish_before_controller_is_interrupted(tmp_path,monkeypatch):
    script=tmp_path/'synthetic_controller.py'
    script.write_text('''import multiprocessing as mp,time
from pathlib import Path
def work(i):
    time.sleep(.4+i*.15)
    Path('finished_'+str(i)).write_text('PASS')
if __name__=='__main__':
    ctx=mp.get_context('spawn')
    for i in range(3):ctx.Process(target=work,args=(i,)).start()
    Path('ready').write_text('ready')
    while True:time.sleep(.1)
''')
    monkeypatch.setattr(handoff.b,'ROOT',tmp_path)
    parent=subprocess.Popen([sys.executable,str(script)],cwd=tmp_path,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+10
        while not (tmp_path/'ready').exists():
            assert time.monotonic()<deadline
            time.sleep(.01)
        observed=handoff.pause_and_drain(parent.pid,str(script),minimum_startup_rss=0,poll_seconds=.03)
        parent.wait(timeout=10)
        assert len(observed['worker_exitcodes'])==3
        assert set(observed['worker_exitcodes'].values())=={0}
        assert observed['healthy_workers_signalled']==0
        assert all((tmp_path/('finished_'+str(i))).read_text()=='PASS' for i in range(3))
    finally:
        if parent.poll() is None:
            info=handoff.process(parent.pid)
            if info and info['state'] in ('T','t'):os.kill(parent.pid,signal.SIGCONT)
            parent.terminate();parent.wait()
