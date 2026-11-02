"""Private minimal synthetic driver/worker entry for exit/cleanup tests only."""
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).absolute().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h4_geometry.observer import IndependentObserver,identity,process_sample


def main():
    resource.setrlimit(resource.RLIMIT_AS,(64*2**20,64*2**20));signal.alarm(20)
    root=Path(sys.argv[2]);assert root.is_relative_to('/home/AbeHiromu')
    if sys.argv[1]=='worker':
        (root/'worker-ready.json').write_text(json.dumps(identity(process_sample(os.getpid()))))
        while True:time.sleep(0.1)
    assert sys.argv[1]=='driver'
    worker=subprocess.Popen([sys.executable,'-P','-B',__file__,'worker',str(root)],close_fds=True,
                             stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    obs=IndependentObserver(sys.executable,root/'orphan-observer.jsonl',scope='SYNTHETIC_ONLY',workers=1,output_cap=65536)
    obs.own_child(worker.pid)
    report={'driver':identity(process_sample(os.getpid())),'worker':identity(process_sample(worker.pid)),
            'observer':identity(process_sample(obs.process.pid))}
    pending=root/'family.json.partial';pending.write_text(json.dumps(report));pending.rename(root/'family.json')
    # Deliberate own driver exit: observer must stop only its registered worker.
    os._exit(0)


if __name__=='__main__':main()
