"""Native S4 adapter; imports scientific APIs only after the new B2 permit."""
from dataclasses import asdict
import importlib,json,math
from pathlib import Path
import h8_phase_b2_contract as c
import h8_phase_b2_runtime as r
import h8_strengthened_export as exported
import h8_b0_anchor_adapter as common
import track_a_phase_a_parallel as pool


def restore(export,permit,**kwargs):
 permit.require(c.SCOPE)
 # The legacy restoration routine is pure callback replay. Translate only its
 # one internal capability check; never grant the legacy acquisition scope.
 class ReplayPermit:
  def require(self,scope):
   if scope!=common.c.SCOPE:raise ValueError('UNEXPECTED_REPLAY_CAPABILITY')
   permit.require(c.SCOPE)
  def take(self,name):
   if name!='saved_preparation_decode':raise ValueError('UNEXPECTED_REPLAY_CALL')
   permit.take(name)
 return common.restore(export,ReplayPermit(),**kwargs)


def observe_prefix(blocks,export,p,expected,inspect,*,n=16):
 observed=[]
 for i,b in enumerate(blocks[:p+1]):
  _,ops=inspect(b.runtime_basis_operations)
  h=c.digest(dict(num_system_qubits=n,operations=ops,hash_policy='ordered_local_basis_operations_v1'))
  observed.append(dict(block_id=b.block_id,basis_id='df-basis-'+h,basis_hash=h,operations=ops,
   diagonal_hex=[float(v).hex() for v in (b.diagonal_eigenvalues if i==0 else b.diagonal_eta)],lambda_hex=None if i==0 else float(b.lam).hex()))
 full=export['faithful:p'+str(len(blocks)-1)]
 if (observed!=full['blocks'][:p+1] or len(observed)!=expected['block_count'] or
     c.digest(observed)!=expected['ordered_block_description_sha256'] or full['constant_hex']!=expected['constant_hex'] or
     c.digest(expected['view_recipe'])!=expected['view_recipe_sha256'] or full['preparation_hash']!=expected['full_preparation_hash']):
  raise ValueError('LIVE_PREFIX_BASIS_IDENTITY')
 if expected['saved_root_preparation_hash'] is not None:
  saved=export['discard:p'+str(p)]
  if saved['blocks']!=observed or saved['preparation_hash']!=expected['saved_root_preparation_hash']:raise ValueError('SAVED_ROOT_IDENTITY')
 return expected


def scalar_for(method,constant_hex,target):
 original=float.fromhex(constant_hex)
 if method=='B0':
  if target['scalar_receipt'] is not None:raise ValueError('B0_MUST_NOT_ADD_TAIL_SCALAR')
  return original
 if method!='B0I':raise ValueError('FIXED_METHOD')
 saved=target['scalar_receipt'];receipt=saved['receipt']
 if (saved['policy']!='REUSED_SAVED_P21_FULL_FOCK_SCALAR' or receipt['full_Fock_identity'] is not True or
  receipt['new_scientific_certificate'] is not False or not math.isfinite(receipt['C'])):raise ValueError('SAVED_FULL_FOCK_SCALAR')
 return original+receipt['C']


def build_case(case,blocks,scalar,qstate,target,permit,native,m,*,n=16,cap=4_000_000):
 permit.require(c.SCOPE);permit.take('native_evolution_build')
 evolution=native.build_deterministic_native(blocks,num_system_qubits=n,T=.8,q=case['q'],formula='4th',
  scalar=scalar,control_policy='symmetric_directional',max_instructions=cap)
 np=m.np;low=np.concatenate((qstate,np.zeros_like(qstate)));high=np.concatenate((np.zeros_like(qstate),qstate))
 permit.take('control_probe');left=m.simulate_statevector(evolution.circuit,low)
 permit.take('control_probe');right=m.simulate_statevector(evolution.circuit,high)
 overlap=complex(np.vdot(qstate,right[len(qstate):]))
 errors=[float(np.linalg.norm(left-low)),float(np.linalg.norm(right[:len(qstate)]))]
 errors += [abs(overlap-complex(target[key]['real'],target[key]['imag'])) for key in ('corrected','independent_forward')]
 wrappers={};fingerprints={}
 for axis in ('cosine','sine'):
  permit.take('measured_wrapper_build');w=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=cap)
  permit.take('probe_wrapper_copy');probe=w.remove_final_measurements(inplace=False)
  permit.take('control_probe');measured=m.hadamard_expectation(m.simulate_statevector(probe,low),n)
  errors.append(abs(measured-(overlap.real if axis=='cosine' else overlap.imag)))
  wrappers[axis]=w;fingerprints[axis]=m.canonical_qiskit_circuit_fingerprint(w);del probe
 if any(not math.isfinite(v) or v<0 or v>=1e-8 for v in errors):raise ValueError('S4_CONTROL_ERROR_GE_1E8')
 return dict(wrappers=wrappers,fingerprints=fingerprints,max_control_error=max(errors))


def serial_axes(plan,permit,writer,build,compile_axis,*,map_owned=pool.map_owned,font_seed=None):
 rows=[];prior_cpu=0.;ordinal=0
 if plan['axes']!=c.AXES or plan['cases']!=c.CASES:raise ValueError('FIXED_SERIAL_AXES')
 for case in plan['cases']:
  permit.require(c.SCOPE);built=build(case)
  for axis in [x for x in plan['axes'] if x['case_id']==case['id']]:
   permit.require(c.SCOPE);ordinal+=1
   task=dict(id=case['id']+'_'+axis['axis'],case_id=case['id'],axis=axis['axis'],counts={'compile':1})
   directory=permit.output/('axis_'+str(ordinal).zfill(2));limits=plan['resources']
   receipt=map_owned([task],lambda t,path:compile_axis(t,built,path),cpus=permit.grant['worker_cpus'],
    limits=dict(max_workers=1,AS_bytes=limits['worker_AS_bytes'],RSS_bytes=limits['worker_RSS_bytes'],
     worker_CPU_seconds=limits['worker_CPU_seconds'],aggregate_CPU_seconds=limits['aggregate_CPU_seconds']),
    directory=directory,deadline=permit.started+limits['wall_seconds'],guard=permit.guard,reserve=permit.reserve,
    font_seed=font_seed,prior_child_CPU_seconds=prior_cpu)
   if (receipt['status']!='PASS' or not receipt['all_reaped'] or receipt['cleanup_errors'] or len(receipt['workers'])!=1):raise ValueError('AXIS_WORKER_CLEANUP')
   worker=receipt['workers'][0]
   if worker['task_id']!=task['id'] or worker['exit_code']!=0 or not worker['reaped']:raise ValueError('AXIS_WORKER_NATIVE_EXIT')
   path=Path(worker['directory'])/'task_result.json';h=r.sha(path);row=r.read(path)
   if worker.get('result_sha256',h)!=h or (row['case_id'],row['axis'])!=(task['case_id'],task['axis']):raise ValueError('AXIS_RESULT_IDENTITY')
   row.update(native_exit_code=0,reaped=True,result_sha256=h,worker_identity=worker['identity'],model_sha256=c.MODEL_SHA)
   permit.settle(task['id']);permit.take('model_error_check')
   row['model_check']=c.axis_check(axis['exact_prediction'],row['measured_RZ']);rows.append(row)
   writer.write('axis_'+str(ordinal).zfill(2)+'_accepted.json',row)
   if row['model_check']['PASS'] is not True:raise ValueError('AXIS_MODEL_ERROR_GE_2_PERCENT')
   prior_cpu=receipt['child_CPU_seconds']
  del built
 return rows


def evaluate(config,permit,writer):
 permit.require(c.SCOPE)
 p=c.plan();paths=config['input_mapping'];models=r.read(paths['B1_frozen_models'])
 predictions=r.read(paths['B1_B2_predictions']);terminal=r.read(paths['B1_phase_terminal'])
 if (c.digest(models)!=c.MODEL_SHA or predictions['model_sha256']!=c.MODEL_SHA or terminal['model_sha256']!=c.MODEL_SHA or
  terminal['status']!='COMPLETE_MANDATORY_STOP' or [(x['candidate_id'],x['axis'],x['exact_predicted_RZ']) for x in predictions['axes']]!=
  [(x['case_id'],x['axis'],x['exact_prediction']) for x in c.AXES]):raise ValueError('B1_FROZEN_INPUT_BINDING')
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
 for prefix in (15,21):
  permit.take('prefix_view');expected=p['binding']['prefix_preparations'][str(prefix)]
  identity=observe_prefix(blocks,export,prefix,expected,inspect)
  if float(ham.constant).hex()!=identity['constant_hex']:raise ValueError('CONSTANT_IDENTITY')
  views[prefix]=(blocks[:prefix+1],identity)
 cp=p['compiler_options'];compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
 def build(case):
  bs,identity=views[case['prefix']];scalar=scalar_for(case['method'],identity['constant_hex'],targets[case['id']])
  built=build_case(case,bs,scalar,qstate,targets[case['id']],permit,native,m,cap=p['native_instruction_cap'])
  built.update(case=case,identity=identity)
  writer.write(case['id']+'_control_preparation.json',dict(case=case,preparation_binding=identity,
   basis_observed_from_live_operations=True,scalar_policy=p['scalar_policies'][case['method']],scalar_hex=scalar.hex(),
   max_control_error=built['max_control_error'],wrapper_fingerprints=built['fingerprints'],SOURCE=config['SOURCE'],profile_sha256=c.PROFILE))
  return built
 def compile_axis(task,built,directory):
  axis=task['axis']
  def entered():r.Writer(directory).write('compile_entered.json',dict(task=task,SOURCE=config['SOURCE'],scope=c.SCOPE,model_sha256=c.MODEL_SHA,fingerprint=built['fingerprints'][axis],reserved_in_parent=True,API_call_entered_not_completion=True))
  row=common.compile_wrapper(built['wrappers'][axis],built['fingerprints'][axis],m,compiler,cap=p['transpiled_instruction_cap'],before_compile=entered)
  row.update(case_id=task['case_id'],axis=axis,preparation_binding=built['identity'],SOURCE=config['SOURCE'],
   profile_sha256=c.PROFILE,scalar_policy=p['scalar_policies'][built['case']['method']],max_control_error=built['max_control_error'])
  return row
 return serial_axes(p,permit,writer,build,compile_axis,font_seed=config['font_seed'])
