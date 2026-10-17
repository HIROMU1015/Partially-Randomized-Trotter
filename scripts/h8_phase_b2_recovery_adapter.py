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
 permit.require(c.SCOPE)
 p=c.plan();paths=config['input_mapping'];models=r.read(paths['B1_frozen_models'])
 predictions=r.read(paths['B1_B2_predictions']);terminal=r.read(paths['B1_phase_terminal'])
 if (c.digest(models)!=c.MODEL_SHA or predictions['model_sha256']!=c.MODEL_SHA or terminal['model_sha256']!=c.MODEL_SHA or
  terminal['status']!='COMPLETE_MANDATORY_STOP' or [(x['candidate_id'],x['axis'],x['exact_predicted_RZ']) for x in predictions['axes']]!=
  [(x['case_id'],x['axis'],x['exact_prediction']) for x in c.original.AXES]):raise ValueError('B1_FROZEN_INPUT_BINDING')
 m=importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
 loader=importlib.import_module('trottertracks.resource_applicability.h8_signal_preflight_port_v1')
 native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
 specs=importlib.import_module('trotterlib.df_partial_s2');basis=importlib.import_module('trotterlib.df_rte_tail')
 gates=importlib.import_module('qiskit.circuit.library');qiskit=importlib.import_module('qiskit')
 for module in (m,loader,native,specs,basis):
  if not Path(module.__file__).resolve().is_relative_to(Path(config['provider_root'])/'src'):raise ValueError('PINNED_PROVIDER_IMPORT')
 permit.take('snapshot_load')
 ham,sector,full,state,md,_=loader.load_h8_snapshot(Path(paths['h8_input_snapshot.npz']),r.read(paths['snapshot_receipt.json']),r.read(paths['df_receipt.json']))
 if ((ham.n_qubits,len(ham.lambdas),sector.dimension)!=(16,28,4900) or md['hamiltonian_hash']!=config['scientific_binding']['hamiltonian'] or md['state_hash']!=config['scientific_binding']['state']):raise ValueError('H8_LAYOUT_OR_STATE_INPUT')
 _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
 export=r.read(paths['h8_preparation_export']);exported.validate_export(export,expected_hashes=config['expected_preparation_hashes'])
 def factory(name,params):return {'rz':gates.RZGate,'xx_plus_yy':gates.XXPlusYYGate}[name](*params)
 def inspect(ops):
  described=basis.describe_basis_change_operations(ops);return described,json.loads(json.dumps([asdict(x) for x in described]))
 blocks=restore(export,permit,factory=factory,inspect=inspect,one_type=specs.DFDeterministicOneBodySpec,fragment_type=specs.DFDeterministicFragmentSpec)
 if any(float(ham.lambdas[i]).hex()!=block.lam.hex() for i,block in enumerate(blocks[1:])):raise ValueError('RESTORED_LAMBDA')
 targets={r['case_id']:r for r in r.read(paths['saved_control_targets_v1'])['cases']}
 views={}
 for prefix in (21,):
  permit.take('prefix_view');expected=p['binding']['prefix_preparations'][str(prefix)]
  identity=observe_prefix(blocks,export,prefix,expected,inspect)
  if float(ham.constant).hex()!=identity['constant_hex']:raise ValueError('CONSTANT_IDENTITY')
  views[prefix]=(blocks[:prefix+1],identity)
 cp=p['compiler_options'];compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
 def build(case):
  bs,identity=views[case['prefix']];scalar=scalar_for(case['method'],identity['constant_hex'],targets[case['id']])
  built=parallel.build_case(case,bs,scalar,qstate,targets[case['id']],permit,native,m,cap=p['native_instruction_cap'])
  built.update(case=case,identity=identity)
  writer.write(case['id']+'_control_preparation.json',dict(case=case,preparation_binding=identity,
   basis_observed_from_live_operations=True,scalar_policy=p['scalar_policies'][case['method']],scalar_hex=scalar.hex(),
   max_control_error=built['max_control_error'],wrapper_fingerprints=built['fingerprints'],SOURCE=config['SOURCE'],profile_sha256=c.PROFILE))
  return built
 def compile_axis(task,built,directory):
  axis=task['axis']
  def entered():r.Writer(directory).write('compile_entered.json',dict(task=task,SOURCE=config['SOURCE'],scope=c.SCOPE,model_sha256=c.MODEL_SHA,fingerprint=built['fingerprints'][axis],reserved_in_parent=True,API_call_entered_not_completion=True))
  from qiskit import qpy
  wrapper=parallel.load_wrapper(built['spools'][axis],permit.output,qpy,p['resources']['qpy_file_bytes'])
  row=common.compile_wrapper(wrapper,built['fingerprints'][axis],m,compiler,cap=p['transpiled_instruction_cap'],before_compile=entered)
  row.update(case_id=task['case_id'],axis=axis,preparation_binding=built['identity'],SOURCE=config['SOURCE'],
   profile_sha256=c.PROFILE,scalar_policy=p['scalar_policies'][built['case']['method']],max_control_error=built['max_control_error'])
  return row
 return serial_axes(p,permit,writer,build,compile_axis,font_seed=config['font_seed'])
