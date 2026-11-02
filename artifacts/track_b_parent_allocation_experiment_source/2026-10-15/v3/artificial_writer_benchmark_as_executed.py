"""One old/new byte-I/O comparison on artificial records, no science data."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,os,resource,sys,tempfile,time
root=Path.cwd();sys.path.insert(0,str(root/'src'))
from trottertracks.algorithm_codesign.parent_experiment.evidence import BatchedEvidenceWriter
from trottertracks.algorithm_codesign.rte_development.evidence import EvidenceWriter
from trottertracks.algorithm_codesign.rte_development.launch import Guard
payload=[dict(index=i,coefficient=F(i+1,317),proposal=F(1,48),label='artificial off-domain byte payload',
              IR=[['CX',0,1],['R',0,'2/11',-1]]*8,notes='日本語😀') for i in range(48)]
caps=dict(parent_RSS_MiB=2048,worker_RSS_MiB=512,aggregate_RSS_MiB=3072,
          per_key_CPU_seconds=120,total_CPU_seconds=3600,virtual_address_MiB=4096,
          workers=2,output_bytes=67108864,per_key_log_bytes=65536)
results=[]
with tempfile.TemporaryDirectory() as tmp:
 for label,cls in [('legacy',EvidenceWriter),('buffered',BatchedEvidenceWriter)]:
  d=Path(tmp)/label;d.mkdir()
  for i in range(200):(d/f'receipt_{i:03d}.json').write_text('{"artificial": true}\n')
  guard=Guard(caps,d);calls=[]
  def check():calls.append(1);guard.check()
  start=time.monotonic();cpu=time.process_time()
  with guard:
   identity=cls(d,caps['output_bytes'],check).write('payload.json',payload)
  results.append(dict(label=label,wall_seconds=time.monotonic()-start,
      CPU_seconds=time.process_time()-cpu,writer_guard_calls=len(calls),
      identity=identity,raw=(d/'payload.json').read_bytes()))
 assert results[0]['raw']==results[1]['raw'] and results[0]['identity']==results[1]['identity']
for r in results:r.pop('raw')
record=dict(status='ARTIFICIAL_WRITER_BYTES_AND_GUARD_BENCHMARK_PASS',fixture_records=48,
 existing_receipt_files=200,results=results,same_bytes_sha256=True,
 writer_block_bytes=16384,legacy_evidence_code_unchanged=True,periodic_Guard_used=True,
 runtime_executable=sys.executable,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 numeric_threads=1,benchmark_repetitions_per_implementation=1,registered_inputs_read=0,
 science_calls=0,synthesis=0,matrix=0,new_forecasts=0,not_registered_run_speedup=True,
 mandatory_STOP=True)
p=root/'artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v3/artificial_writer_benchmark.json'
p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record))
