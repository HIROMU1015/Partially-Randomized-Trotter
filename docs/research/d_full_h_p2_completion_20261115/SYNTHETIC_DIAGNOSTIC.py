import faulthandler,time
from pathlib import Path
p=Path('/tmp/d7-synthetic-faulthandler-20261115.txt')
with p.open('w') as f:
 faulthandler.dump_traceback_later(.01,repeat=True,file=f)
 end=time.monotonic()+3
 def frames(n):
  return sum(i*i for i in range(n))
 while time.monotonic()<end:frames(100)
 faulthandler.cancel_dump_traceback_later()
