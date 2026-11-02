"""Small cooperative Linux processes for guard acceptance tests only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def burn(seconds):
    end=time.process_time()+seconds
    while time.process_time()<end:pass

mode=sys.argv[1];root=Path(sys.argv[2]);root.mkdir(exist_ok=True,parents=True)
(root/f'{os.getpid()}.pid.json').write_text(json.dumps({'pid':os.getpid(),'ppid':os.getppid(),'mode':mode}))
if mode=='normal':burn(.12);time.sleep(.15)
elif mode=='sleep':time.sleep(5)
elif mode=='busy':burn(2);time.sleep(2)
elif mode=='memory':
    data=bytearray(64*1024**2)
    for i in range(0,len(data),4096):data[i]=1
    time.sleep(3)
elif mode=='output':
    for i in range(8):
        with (root/f'generated-{i}.bin').open('xb') as f:f.write(b'x'*(128*1024))
        time.sleep(.025)
    time.sleep(3)
elif mode in ['grandchild','multiple','orphan','kill_tree']:
    child_mode='busy' if mode=='kill_tree' else 'normal'
    children=[subprocess.Popen([sys.executable,'-B',__file__,child_mode,str(root)])
              for _ in range(2 if mode=='multiple' else 1)]
    if mode=='orphan':os._exit(0)
    for child in children:child.wait()
    time.sleep(.1)
else:raise RuntimeError(mode)
