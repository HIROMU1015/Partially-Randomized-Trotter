import sys,time,json,faulthandler,gc
from pathlib import Path
mode=sys.argv[1];timed=sys.argv[2]=='timer';seconds=float(sys.argv[3]);log=Path(sys.argv[4])
if mode=='fast_writer':
 sys.path.insert(0,str(Path.cwd()/'scripts'))
 import h46_long_time_signal as S
 S.install()
 from h46_b3_fast_fingerprint import FingerprintSession
 from qiskit import QuantumCircuit
 circuit=QuantumCircuit(3)
 for i in range(100):
  circuit.crz(.2,0,1);circuit.cx(1,2);circuit.rz(.1,2)
 fp=FingerprintSession().fingerprint
 expected=fp(circuit)
count=0;start=time.monotonic()
with log.open('w') as stream:
 if timed:faulthandler.dump_traceback_later(.002,repeat=True,file=stream)
 while time.monotonic()-start<seconds:
  if mode=='fast_writer':
   assert fp(circuit)==expected
  else:
   code=compile('def f(): return sum(i*i for i in range(20))','ARTIFICIAL_DYNAMIC_FRAME_'+str(count),'exec')
   ns={};exec(code,ns);ns['f']();ns.clear();del code;gc.collect(0)
  count+=1
 if timed:faulthandler.cancel_dump_traceback_later()
print(json.dumps(dict(scope='ARTIFICIAL_NO_H4_NO_COMPILE',mode=mode,timed=timed,iterations=count,seconds=time.monotonic()-start,python=sys.version)))
