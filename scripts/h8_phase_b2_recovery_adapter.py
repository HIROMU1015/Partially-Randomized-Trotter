"""Remaining two axes only. Parallel control/fingerprint, serial model-gated compile."""
from dataclasses import asdict
from types import SimpleNamespace
import importlib,json
from pathlib import Path
import h8_phase_b2_recovery_contract as c
import h8_phase_b2_recovery_runtime as r
import h8_phase_b2_recovery_parallel as parallel
import h8_phase_b2_adapter as legacy
import h8_b0_anchor_adapter as common
import h8_strengthened_export as exported
import track_a_phase_a_parallel as pool
observe_prefix=legacy.observe_prefix;scalar_for=legacy.scalar_for

def restore(export,permit,**kwargs):
 permit.require(c.SCOPE)
 def require(scope):
  if scope!=c.original.SCOPE:raise ValueError('UNEXPECTED_REPLAY_SCOPE')
  permit.require(c.SCOPE)
 return legacy.restore(export,SimpleNamespace(require=require,take=permit.take),**kwargs)

def serial_axes(plan,permit,writer,build,compile_axis,*,map_owned=pool.map_owned,font_seed=None):
 if plan['axes']!=c.AXES or plan['cases']!=c.CASES:raise ValueError('EXACT_TWO_REMAINING_AXES')
 built=build(plan['cases'][0]);writer.write('parent_circuit_release.json',parallel.release_parent_circuits());rows=[]
 for ordinal,axis in enumerate(plan['axes'],1):
  permit.require(c.SCOPE)
  task=dict(id=axis['case_id']+'_'+axis['axis'],case_id=axis['case_id'],axis=axis['axis'],counts={'compile':1,'qpy_load':1})
  limits=plan['resources'];directory=permit.output/('axis_'+str(ordinal).zfill(2))
  receipt=map_owned([task],lambda t,path:compile_axis(t,built,path),cpus=permit.grant['worker_cpus'][:1],
   limits=dict(max_workers=1,AS_bytes=limits['worker_AS_bytes'],RSS_bytes=limits['worker_RSS_bytes'],worker_CPU_seconds=limits['worker_CPU_seconds'],aggregate_CPU_seconds=limits['aggregate_CPU_seconds']),
   directory=directory,deadline=permit.started+limits['wall_seconds'],guard=permit.guard,reserve=permit.reserve,
   font_seed=font_seed,prior_child_CPU_seconds=permit.child_CPU_seconds)
  permit.child_CPU_seconds=receipt['child_CPU_seconds']
  if receipt['status']!='PASS' or not receipt['all_reaped'] or receipt['cleanup_errors'] or len(receipt['workers'])!=1:raise ValueError('COMPILE_CLEANUP')
  worker=receipt['workers'][0];path=Path(worker['directory'])/'task_result.json'
  if worker['task_id']!=task['id'] or worker['exit_code']!=0 or not worker['reaped'] or parallel.file_sha(path)!=worker['result_sha256']:raise ValueError('COMPILE_NATIVE_RESULT')
  row=r.read(path)
  if (row['case_id'],row['axis'])!=(axis['case_id'],axis['axis']):raise ValueError('AXIS_IDENTITY')
  row.update(native_exit_code=0,reaped=True,result_sha256=worker['result_sha256'],worker_identity=worker['identity'],model_sha256=c.MODEL_SHA)
  permit.settle(task['id']);permit.take('model_error_check');row['model_check']=c.axis_check(axis['exact_prediction'],row['measured_RZ'])
  writer.write('axis_'+str(ordinal).zfill(2)+'_accepted.json',row);rows.append(row)
  if not row['model_check']['PASS']:raise ValueError('AXIS_MODEL_ERROR_GE_2_PERCENT')
 return rows

def evaluate(config,permit,writer):
 """Read the validated sine QPY once; never regenerate completed controls or cosine."""
 permit.require(c.SCOPE)
 from h8_b2_saved_spool_acceptance import accept_saved
 accepted=accept_saved(config)
 m=importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
 qiskit=importlib.import_module('qiskit')
 if not Path(m.__file__).resolve().is_relative_to(Path(config['provider_root'])/'src'):raise ValueError('PINNED_PROVIDER_IMPORT')
 p=c.plan();cp=p['compiler_options'];compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
 spool=dict(accepted['sine_spool'],path=config['input_mapping']['validated_sine_QPY'])
 fingerprint=accepted['sine_fingerprint'];identity=accepted['preparation_binding']
 built=dict(case=c.CASES[0],identity=identity,spools={'sine':spool},fingerprints={'sine':fingerprint},max_control_error=accepted['max_control_error'])
 def compile_axis(task,built,directory):
  if task['axis']!='sine':raise ValueError('ONLY_REMAINING_SINE')
  def entered():r.Writer(directory).write('compile_entered.json',dict(task=task,SOURCE=config['SOURCE'],scope=c.SCOPE,model_sha256=c.MODEL_SHA,fingerprint=fingerprint,reserved_in_parent=True,API_call_entered_not_completion=True))
  from qiskit import qpy
  wrapper=parallel.load_wrapper(spool,Path(spool['path']).parent,qpy,p['resources']['qpy_file_bytes'])
  row=common.compile_wrapper(wrapper,fingerprint,m,compiler,cap=p['transpiled_instruction_cap'],before_compile=entered)
  row.update(case_id=task['case_id'],axis='sine',preparation_binding=identity,SOURCE=config['SOURCE'],
   profile_sha256=c.PROFILE,scalar_policy=accepted['scalar_policy'],max_control_error=accepted['max_control_error'])
  return row
 return serial_axes(p,permit,writer,lambda case:built,compile_axis,font_seed=config['font_seed'])
