"""Two direct preparation workers; immutable QPY spool before serial compilation."""
import gc,hashlib,math,os,time
from pathlib import Path
import h8_phase_b2_recovery_contract as c
import h8_phase_b2_recovery_runtime as r
import track_a_phase_a_parallel as pool


def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 return h.hexdigest()


class LimitedFile:
 def __init__(self,stream,cap):self.stream,self.cap=stream,cap
 def write(self,data):
  if self.stream.tell()+len(data)>self.cap:raise ValueError('QPY_FILE_BYTE_CEILING')
  return self.stream.write(data)
 def __getattr__(self,name):return getattr(self.stream,name)


def dump_wrapper(wrapper,path,qpy,cap):
 path=Path(path)
 with path.open('xb') as f:
  qpy.dump(wrapper,LimitedFile(f,cap));f.flush();os.fsync(f.fileno())
 return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))


def load_wrapper(receipt,output,qpy,cap):
 path=Path(receipt['path'])
 if (not path.resolve().is_relative_to(Path(output).resolve()) or path.is_symlink() or
     not 0<receipt['bytes']<=cap or path.stat().st_size!=receipt['bytes'] or file_sha(path)!=receipt['sha256']):raise ValueError('QPY_SPOOL_IDENTITY')
 with path.open('rb') as f:circuits=qpy.load(f)
 if len(circuits)!=1:raise ValueError('ONE_SPOOLED_WRAPPER_REQUIRED')
 return circuits[0]


def release_parent_circuits():
 def rss():
  rows=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
  return int(rows['VmRSS'].split()[0])*1024
 before=rss();gc.collect();trimmed=False
 try:
  import ctypes
  libc=ctypes.CDLL(None)
  if hasattr(libc,'malloc_trim'):libc.malloc_trim(0);trimmed=True
 except (OSError,AttributeError):pass
 return dict(parent_RSS_before_bytes=before,parent_RSS_after_bytes=rss(),gc_collected=True,
             glibc_trim_called=trimmed,big_circuit_objects_returned_to_parent=False)


def batch(tasks,callback,permit,name,*,reserve,map_owned=pool.map_owned):
 limits=permit.grant['resources'];cpus=permit.grant['worker_cpus'][:2]
 if len(cpus)!=2:raise ValueError('TWO_PREPARATION_CPUS_REQUIRED')
 directory=permit.output/name
 receipt=map_owned(tasks,callback,cpus=cpus,limits=dict(max_workers=2,AS_bytes=limits['worker_AS_bytes'],
  RSS_bytes=limits['worker_RSS_bytes'],worker_CPU_seconds=limits['worker_CPU_seconds'],aggregate_CPU_seconds=limits['aggregate_CPU_seconds']),
  directory=directory,deadline=permit.started+limits['wall_seconds'],guard=permit.guard,reserve=reserve,
  font_seed=permit.config.get('font_seed'),prior_child_CPU_seconds=permit.child_CPU_seconds)
 if receipt['status']!='PASS' or not receipt['all_reaped'] or receipt['cleanup_errors']:raise ValueError('PARALLEL_PREPARATION_CLEANUP')
 permit.child_CPU_seconds=receipt['child_CPU_seconds'];values=[]
 if [v['task_id'] for v in receipt['workers']]!=[v['id'] for v in tasks]:raise ValueError('PREPARATION_TASK_ORDER')
 for worker in receipt['workers']:
  path=Path(worker['directory'])/'task_result.json'
  if worker['exit_code']!=0 or not worker['reaped'] or file_sha(path)!=worker['result_sha256']:raise ValueError('PREPARATION_NATIVE_RESULT')
  values.append(r.read(path))
 return values


def build_case(case,blocks,scalar,qstate,target,permit,native,m,*,n=16,cap=4_000_000,map_owned=pool.map_owned):
 permit.require(c.SCOPE)
 from qiskit import qpy
 permit.take('native_evolution_build')
 evolution=native.build_deterministic_native(blocks,num_system_qubits=n,T=.8,q=case['q'],formula='4th',scalar=scalar,control_policy='symmetric_directional',max_instructions=cap)
 np=m.np;low=np.concatenate((qstate,np.zeros_like(qstate)));high=np.concatenate((np.zeros_like(qstate),qstate))
 def directional(task,path):
  if task['id']=='left':
   left=m.simulate_statevector(evolution.circuit,low)
   return dict(errors=[float(np.linalg.norm(left-low))])
  right=m.simulate_statevector(evolution.circuit,high);z=complex(np.vdot(qstate,right[len(qstate):]))
  return dict(overlap=dict(real=z.real,imag=z.imag),errors=[float(np.linalg.norm(right[:len(qstate)]))]+[abs(z-complex(target[key]['real'],target[key]['imag'])) for key in ('corrected','independent_forward')])
 values=batch([dict(id=x) for x in ('left','right')],directional,permit,'directional_controls',
  reserve=lambda task:permit.take('control_probe'),map_owned=map_owned)
 errors=[error for value in values for error in value['errors']]
 if any(not math.isfinite(v) or v<0 or v>=1e-8 for v in errors):raise ValueError('S4_DIRECTIONAL_CONTROL_1E8')
 z=complex(values[1]['overlap']['real'],values[1]['overlap']['imag']);wrappers={}
 for axis in ('cosine','sine'):
  permit.take('measured_wrapper_build')
  wrappers[axis]=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=cap)
 del evolution,directional;gc.collect()
 def axis_validation(task,path):
  axis=task['id'];w=wrappers.pop(axis);wrappers.clear();gc.collect()
  probe=w.remove_final_measurements(inplace=False)
  measured=m.hadamard_expectation(m.simulate_statevector(probe,low),n);del probe
  error=abs(measured-(z.real if axis=='cosine' else z.imag))
  if not math.isfinite(error) or error>=1e-8:raise ValueError('S4_XY_CONTROL_1E8')
  fingerprint=m.canonical_qiskit_circuit_fingerprint(w)
  spool=dump_wrapper(w,path/'scratch/validated_wrapper.qpy',qpy,permit.grant['resources']['qpy_file_bytes'])
  return dict(axis=axis,max_control_error=error,fingerprint=fingerprint,qpy_spool=spool)
 def reserve_axis(task):
  permit.take('probe_wrapper_copy');permit.take('control_probe');permit.take('qpy_dump')
 values=batch([dict(id=x) for x in ('cosine','sine')],axis_validation,permit,'axis_controls_and_fingerprints',reserve=reserve_axis,map_owned=map_owned)
 error=max(errors+[v['max_control_error'] for v in values])
 # Return small descriptors only. Neither closure nor circuit escapes this call.
 return dict(spools={v['axis']:v['qpy_spool'] for v in values},fingerprints={v['axis']:v['fingerprint'] for v in values},max_control_error=error)
