"""Parallel control/fingerprint from B2; general formula and no process resource caps."""
import gc,hashlib,math,os,time
from pathlib import Path
import h8_key_compile_contract as c
import h8_key_compile_parallel as pool
r=c


def file_sha(path):return c.sha(path)

def dump_wrapper(wrapper,path,qpy,cap=None):
 path=Path(path)
 with path.open('xb') as f:qpy.dump(wrapper,f);f.flush();os.fsync(f.fileno())
 return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))

def load_wrapper(receipt,output,qpy,cap=None):
 path=Path(receipt['path'])
 c.require(path.resolve().is_relative_to(Path(output).resolve()) and not path.is_symlink() and receipt['bytes']>0 and path.stat().st_size==receipt['bytes'] and file_sha(path)==receipt['sha256'],'QPY_SPOOL_IDENTITY')
 with path.open('rb') as f:circuits=qpy.load(f)
 c.require(len(circuits)==1,'ONE_SPOOLED_WRAPPER_REQUIRED')
 return circuits[0]

def batch(tasks,callback,permit,name,*,reserve,map_owned=pool.map_owned):
 cpus=permit.grant['worker_cpus']
 if len(cpus)!=2:raise ValueError('TWO_PREPARATION_CPUS_REQUIRED')
 directory=permit.output/name
 receipt=map_owned(tasks,callback,cpus=cpus,limits=dict(max_workers=2),
  directory=directory,deadline=math.inf,guard=lambda:None,reserve=reserve,
  font_seed=permit.config.get('font_seed'),prior_child_CPU_seconds=0)
 if receipt['status']!='PASS' or not receipt['all_reaped'] or receipt['cleanup_errors']:raise ValueError('PARALLEL_PREPARATION_CLEANUP')
 values=[]
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
 evolution=native.build_deterministic_native(blocks,num_system_qubits=n,T=.8,q=case['q'],formula=case['formula'],scalar=scalar,control_policy='symmetric_directional',max_instructions=cap)
 np=m.np;low=np.concatenate((qstate,np.zeros_like(qstate)));high=np.concatenate((np.zeros_like(qstate),qstate))
 def directional(task,path):
  if task['id']=='left':
   left=m.simulate_statevector(evolution.circuit,low)
   return dict(errors=[float(np.linalg.norm(left-low))])
  right=m.simulate_statevector(evolution.circuit,high);z=complex(np.vdot(qstate,right[len(qstate):]))
  return dict(overlap=dict(real=z.real,imag=z.imag),errors=[float(np.linalg.norm(right[:len(qstate)]))]+[abs(z-complex(target[key]['real'],target[key]['imag'])) for key in ('corrected','independent_forward')])
 values=batch([dict(id=x) for x in ('left','right')],directional,permit,case['id']+'_directional_controls',
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
  spool=dump_wrapper(w,path/'scratch/validated_wrapper.qpy',qpy,None)
  return dict(axis=axis,max_control_error=error,fingerprint=fingerprint,qpy_spool=spool)
 def reserve_axis(task):
  permit.take('probe_wrapper_copy');permit.take('control_probe');permit.take('qpy_dump')
 values=batch([dict(id=x) for x in ('cosine','sine')],axis_validation,permit,case['id']+'_axis_controls_and_fingerprints',reserve=reserve_axis,map_owned=map_owned)
 error=max(errors+[v['max_control_error'] for v in values])
 # Return small descriptors only. Neither closure nor circuit escapes this call.
 return dict(spools={v['axis']:v['qpy_spool'] for v in values},fingerprints={v['axis']:v['fingerprint'] for v in values},max_control_error=error)
