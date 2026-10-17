"""Byte-bound saved controls, cosine native PASS and exact sine QPY acceptance."""
import hashlib,json,math
from pathlib import Path
import h8_phase_b2_recovery_contract as c


def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while raw:=f.read(1024*1024):h.update(raw)
 return h.hexdigest()


def validate_saved(values,hashes,registry):
 """Pure metadata checks; never re-evaluate signal, fit or measured costs."""
 def require(value,reason):
  if not value:raise ValueError(reason)
 terminal=values['prior_execution_terminal'];observer=values['prior_observer_terminal'];native=values['prior_native_exit']
 require(terminal['status']==observer['status']=='STOP' and terminal['SOURCE']==registry['origin_SOURCE'],'ORIGIN_STOP_SOURCE')
 require(observer['all_reaped'] and not observer['cleanup_errors'] and native['wait_completed'] and native['native_exit_code']==1,'ORIGIN_NATIVE_WAIT_CLEANUP')
 task='B0I_p21_q8_4th_sine';require(task in terminal['irrevocable_reservations'] and task not in terminal['settlements'] and terminal['compile_entry_markers']==1,'OLD_SINE_RESERVATION_AND_ENTRY_RETAINED')
 require(terminal['completed_calls_lower_bound']['compile']==1 and values['prior_resource_stop']['reason']=='MEMORY_PRESSURE','TECHNICAL_STOP_ONLY')
 accepted=values['saved_cosine_accepted'];raw=values['saved_cosine_task']
 require(accepted['case_id']=='B0I_p21_q8_4th' and accepted['axis']=='cosine' and accepted['SOURCE']==registry['origin_SOURCE'],'SAVED_COSINE_IDENTITY')
 require(all(accepted.get(k)==v for k,v in raw.items()),'SAVED_COSINE_RAW_FIELDS')
 require(accepted['measured_RZ']==registry['cosine_measured_RZ']==1852640 and accepted['model_check']['PASS'] and accepted['model_check']['exact_relative_error']=='0','SAVED_COSINE_PASS')
 require(accepted['model_sha256']==c.MODEL_SHA and accepted['profile_sha256']==c.PROFILE and accepted['compiler_hash']==c.COMPILER,'SAVED_COSINE_COMPILER_MODEL')
 require(accepted['native_exit_code']==0 and accepted['reaped'] and accepted['fingerprint_before']==accepted['fingerprint_after'] and accepted['result_sha256']==hashes['saved_cosine_task'],'SAVED_COSINE_NATIVE')
 for receipt_name,names in [('saved_cosine_receipt',['saved_cosine_task']),('saved_directional_receipt',['saved_directional_left','saved_directional_right']),('saved_axes_receipt',['saved_axis_cosine','saved_axis_sine'])]:
  receipt=values[receipt_name]
  require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors'] and len(receipt['workers'])==len(names),'SAVED_WORKER_CLEANUP:'+receipt_name)
  for worker,name in zip(receipt['workers'],names):require(worker['exit_code']==0 and worker['reaped'] and worker['result_sha256']==hashes[name],'SAVED_WORKER_RESULT:'+name)
 errors=[]
 for name in ('saved_directional_left','saved_directional_right'):errors.extend(values[name]['errors'])
 for name in ('saved_axis_cosine','saved_axis_sine'):errors.append(values[name]['max_control_error'])
 require(all(type(x) in (int,float) and math.isfinite(x) and 0<=x<1e-8 for x in errors),'SAVED_CONTROL_ALLOWANCE')
 prep=values['saved_control_preparation'];sine=values['saved_axis_sine'];cosine=values['saved_axis_cosine']
 require(sine['axis']=='sine' and cosine['axis']=='cosine','SAVED_CONTROL_AXIS_IDENTITY')
 require(prep['SOURCE']==registry['origin_SOURCE'] and prep['profile_sha256']==c.PROFILE and prep['preparation_binding']==accepted['preparation_binding'],'SAVED_PREPARATION_PROFILE')
 require(prep['scalar_policy']==accepted['scalar_policy']=='ORIGINAL_PLUS_FULL_FOCK_TAIL_IDENTITY_ONCE','SAVED_SCALAR_POLICY')
 require(prep['wrapper_fingerprints']==dict(cosine=cosine['fingerprint'],sine=sine['fingerprint']) and cosine['fingerprint']==accepted['fingerprint_before'],'SAVED_FINGERPRINT_BINDING')
 require(sine['fingerprint']==registry['sine_fingerprint'] and sine['qpy_spool']['sha256']==registry['sine_spool_sha256']==hashes['validated_sine_QPY'] and sine['qpy_spool']['bytes']==registry['sine_spool_bytes'],'VALIDATED_SINE_QPY_BYTES')
 return dict(status='ACCEPT_SAVED_COSINE_AND_SINE_CONTROL_FOR_NEW_GRANT_ONLY',origin_status='STOP',origin_SOURCE=registry['origin_SOURCE'],old_unsettled_reservations_retained=True,saved_cosine=accepted,sine_fingerprint=sine['fingerprint'],sine_spool={k:sine['qpy_spool'][k] for k in ('bytes','sha256')},preparation_binding=prep['preparation_binding'],scalar_policy=prep['scalar_policy'],max_control_error=max(errors),new_control=0,new_signal=0,new_compile=0,C_authorized=False)


def accept_saved(config):
 registry=c.spool_registry();mapping=config['input_mapping'];values={};hashes={}
 for row in registry['required_input_receipts']:
  name=row['logical_name'];p=Path(mapping[name])
  if p.is_symlink() or p.stat().st_size!=row['bytes'] or file_sha(p)!=row['sha256']:raise ValueError('SAVED_RECEIPT_BYTES:'+name)
  hashes[name]=row['sha256']
  if name!='validated_sine_QPY':values[name]=json.loads(p.read_text())
 if Path(values['saved_axis_sine']['qpy_spool']['path']).resolve()!=Path(mapping['validated_sine_QPY']).resolve():raise ValueError('EXACT_ORIGIN_SPOOL_PATH')
 return validate_saved(values,hashes,registry)
