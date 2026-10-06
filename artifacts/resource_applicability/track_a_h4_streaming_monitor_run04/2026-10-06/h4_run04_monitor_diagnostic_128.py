"""One bounded JSON metadata diagnostic; no circuits, arrays or science runner."""
import hashlib,json,os,sys,threading,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/AbeHiromu/projects/partially-randomized-trotter')
CONTROL=ROOT/'.server-preparation/executions/h4-signal-compile-run03-revision'
sys.path.insert(0,'/tmp/track-a-h4-streaming-monitor-run04-20261006/src')
from trottertracks.resource_applicability.h4_geometry.identity import fingerprint
from trottertracks.resource_applicability.h4_geometry.resources import Monitor,observe_memory,limit_owned_address_space
limit_owned_address_space()
allowed=set(os.sched_getaffinity(0))
assert allowed=={3,5,6,7,8,9,10,11,12,13,14,15}
before=observe_memory();monitor=Monitor(12,before)
# Repeated 256x256 parameter metadata as used by dense 8-spin gate identity.
# No numpy/science arrays, molecular files, circuit or compiler calls.
item={'complex128_hex':['0x1.123456789abcdep-1','-0x0.0p+0']}
row=[item for _ in range(256)];matrix=[row for _ in range(256)]
payload={'instructions':[{'parameters':[{'array':matrix,'shape':[256,256],'dtype':'<c16'}],'axis':'cosine','label':'ARTIFICIAL_METADATA_ONLY'} for _ in range(128)]}
rows=[];finished=threading.Event();failure=[]
def watch():
    while not finished.wait(1):
        started=time.monotonic();previous=monitor.last
        try:
            monitor.poll()
            rows.append({'started':started,'ended':time.monotonic(),'since_previous_check':started-previous,'PASS':True})
        except BaseException as exc:
            failure.append({'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc(),'started':started,'ended':time.monotonic(),'since_previous_check':started-previous})
            return
thread=threading.Thread(target=watch,daemon=True);thread.start()
start=time.monotonic();value=None;error=None
try:value=fingerprint('ARTIFICIAL_METADATA_DIAGNOSTIC',payload)
except BaseException as exc:error=type(exc).__name__+': '+str(exc)
elapsed=time.monotonic()-start
# Allow the observation begun during encoding to finish before stopping it.
finished.set();thread.join(timeout=10)
report={'schema_version':'h4-json-monitor-metadata-diagnostic-v1','observed_utc':datetime.now(timezone.utc).isoformat(),'scope':'ONE_ARTIFICIAL_JSON_PAYLOAD','metadata_matrix_shape':[256,256],'repetitions':128,'elapsed_seconds':elapsed,'fingerprint':value,'main_error':error,'monitor_failure':failure,'monitor_samples':rows,'monitor_thread_ended':not thread.is_alive(),'scientific_arrays_circuits_transpiles_created':0,'production_runner_workers_launched':0,'GPU_queries':0,'shared_changes':0}
with (CONTROL/'metadata_streaming_monitor_diagnostic_128_v1.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(report,sort_keys=True))
