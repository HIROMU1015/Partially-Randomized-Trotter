from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from collections import Counter
import json,sys,threading,traceback
LAYERS=int(sys.argv[1]) if len(sys.argv)>1 else 32
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from trottertracks.c1_stage3_cost_compat import CompileSerialCalls
GUARDED=len(sys.argv)>2 and sys.argv[2]=='guarded'
guard=CompileSerialCalls(lambda kind,label,fn:fn())
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from trottertracks.pr_dictionary_validation.gates import compiled_wrapper
from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
# Only analytic/artificial gates; no input or saved QPY decode.
def circuit(i):
 q=QuantumCircuit(6)
 for j in range(LAYERS):
  a=(j+i)%6;b=(a+1)%6
  q.ry((i+1)*(j+1)/173,a);q.rz(-(j+2)/139,b)
  q.cp((i+j+1)/101,a,b);q.append(XXPlusYYGate((j+1)/97,(i+1)/53),[a,b])
 q.p(-(i+1)/19,5)
 return q
circuits=[circuit(i) for i in range(8)];barrier=threading.Barrier(len(circuits))
def task(i):
 barrier.wait()
 try:
  qc=guard('compile',str(i),lambda:compiled_wrapper(circuits[i],CompilerProfile())) if GUARDED else compiled_wrapper(circuits[i],CompilerProfile())
  return dict(index=i,status='PASS',gates=len(qc.data))
 except Exception as e:
  return dict(index=i,status='FAIL',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
with ThreadPoolExecutor(max_workers=len(circuits)) as pool:
 rows=[f.result() for f in as_completed([pool.submit(task,i) for i in range(len(circuits))])]
print(json.dumps(dict(artificial=True,layers=LAYERS,guarded=GUARDED,guarded_compile_completed=guard.completed,H4_or_saved_QPY_loaded=False,status_counts=dict(Counter(x['status'] for x in rows)),tasks=sorted(rows,key=lambda x:x['index'])),indent=2))
