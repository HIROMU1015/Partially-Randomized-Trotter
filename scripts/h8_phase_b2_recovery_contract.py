"""New unapproved scope: accept saved B0, parallel preparation, two serial B0I axes."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import h8_phase_b2_contract as original
ROOT=Path(__file__).resolve().parents[1]
SCOPE='H8_B2_THREE_SAVED_AXES_VALIDATED_SINE_QPY_ONE_COMPILE_V1'
PROFILE=original.PROFILE;COMPILER=original.COMPILER;PROVIDER=original.PROVIDER
RUNTIME_MAP_SHA=original.RUNTIME_MAP_SHA;PROVIDER_MAP_SHA=original.PROVIDER_MAP_SHA;MODEL_SHA=original.MODEL_SHA
digest=original.digest;axis_check=original.axis_check
REGISTRY='artifacts/resource_applicability/track_a_h8_phase_b2_recovery_readiness/2026-10-17/saved_B0_acceptance_registry_v1.json'
REGISTRY_SHA='6523a32943df938b72c2cdd41cd1eb57a6fb99acacf22c23ebd4c22add88d6d7'
AXES=deepcopy(original.AXES[3:]);CASES=deepcopy(original.CASES[1:])
RESOURCES=deepcopy(original.RESOURCES)
RESOURCES.update(physical_cores=4,max_workers=1,max_compile_workers=1,max_preparation_workers=0,
 witness_processes=1,witness_AS_bytes=128*2**20,witness_RSS_bytes=64*2**20,witness_CPU_seconds=120,
 aggregate_RSS_bytes=int(64.3125*2**30),output_bytes=256*2**20,qpy_file_bytes=2**30,
 stage_storage_bytes=8*2**30,observer_CPU_seconds=1800)
CAPS={k:0 for k in original.CAPS}
CAPS.update(compile=1,model_error_check=1,qpy_load=1,qpy_dump=0)
SPOOL_REGISTRY='artifacts/resource_applicability/track_a_h8_b2_sine_spool_recovery/2026-10-17/saved_spool_registry_v1.json'
SPOOL_REGISTRY_SHA='bccb67b4963af4d3e06c59f59901398a0e0059d8b857be5d7405230121b9b734'

def spool_registry():
 p=ROOT/SPOOL_REGISTRY
 if hashlib.sha256(p.read_bytes()).hexdigest()!=SPOOL_REGISTRY_SHA:raise ValueError('SPOOL_REGISTRY_CHANGED')
 return json.loads(p.read_text())

def binding():
 value=deepcopy(original.binding());p=ROOT/REGISTRY
 if hashlib.sha256(p.read_bytes()).hexdigest()!=REGISTRY_SHA:raise ValueError('SAVED_B0_REGISTRY_BYTES_CHANGED')
 value['inputs'].append(dict(logical_name='saved_B0_acceptance_registry',bytes=p.stat().st_size,sha256=REGISTRY_SHA))
 value['inputs'].extend(spool_registry()['required_input_receipts'])
 return value

def saved_B0():
 binding();value=json.loads((ROOT/REGISTRY).read_text())
 if value['original_execution_status']!='STOP' or value['model_sha256']!=MODEL_SHA:raise ValueError('OLD_STOP_AND_FROZEN_MODEL_REQUIRED')
 for item in value['original_published_files']:
  p=ROOT/item['path']
  if p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('SAVED_B0_EVIDENCE_CHANGED')
 rows=value['axes']
 if [(r['case_id'],r['axis']) for r in rows]!=[(x['case_id'],x['axis']) for x in original.AXES[:2]]:raise ValueError('EXACT_SAVED_B0_AXES')
 for r in rows:
  if not (r['measured_RZ']==168353 and r['model_check']['PASS'] and r['native_exit_code']==0 and r['reaped'] and r['fingerprint_before']==r['fingerprint_after'] and r['model_sha256']==MODEL_SHA and r['profile_sha256']==PROFILE and r['compiler_hash']==COMPILER):raise ValueError('SAVED_B0_NATIVE_RECEIPT')
 return value

def plan():
 p=deepcopy(original.plan());p.update(schema='H8_PHASE_B2_RECOVERY_EXECUTION_V1',scope=SCOPE,cases=CASES,axes=AXES,
 binding=binding(),resources=RESOURCES,call_caps=CAPS,accept_saved_B0_requires_new_grant=True,
 original_B2_run01_status='STOP',original_unsettled_compile_reservations_retained=1,origin_sine_unsettled_reservation_retained=1,
 reuse_saved_cosine_and_validated_sine_spool=True,skip_all_control_and_circuit_regeneration=True,
 preparation_concurrency=0,compile_concurrency=1,parallel_phases=[],
 compiler_threads=1,QPY_roundtrip_requires_identical_canonical_fingerprint=True,
 free_parent_circuits_before_compile=True,dispatcher_returns_after_detached_witness=True,
 observer_pinned_before_exec=True,witness_native_wait_mode='BLOCKING_WAITPID',
 owned_process_CPU_window_accounting=True,technical_restart_requires_new_attempt_and_recorded_user_authority=True,
 authorization=dict(approved=False,runtime_authorization=False),C_authorized=False,next_stage_authorized=False)
 return p

def validate_complete(rows,calls):
 if [(r['case_id'],r['axis']) for r in rows]!=[(x['case_id'],x['axis']) for x in AXES] or calls!=CAPS:raise ValueError('EXACT_RECOVERY_AXES_AND_CAPS')
 for r in rows:
  if not (r['model_check']['PASS'] and r['fingerprint_before']==r['fingerprint_after'] and r['native_exit_code']==0 and r['reaped'] and r['model_sha256']==MODEL_SHA and r['profile_sha256']==PROFILE and r['compiler_hash']==COMPILER and 0<=r['max_control_error']<1e-8):raise ValueError('RECOVERY_AXIS_CHECK')
 return dict(status='TWO_SAVED_B0_PLUS_SAVED_COSINE_PLUS_ONE_NEW_SINE_ONLY',full_grid_guaranteed=False,C_authorized=False,next_stage_authorized=False)

SOURCES=original.SOURCES+tuple('scripts/'+name+'.py' for name in (
 'h8_phase_b2_recovery_contract','h8_phase_b2_recovery_runtime','h8_phase_b2_recovery_adapter',
 'h8_phase_b2_recovery_parallel','h8_phase_b2_recovery_observer','h8_phase_b2_recovery_witness',
 'run_h8_phase_b2_recovery','launch_h8_phase_b2_recovery','prepare_h8_phase_b2_recovery'))+(REGISTRY,SPOOL_REGISTRY,'scripts/h8_b2_saved_spool_acceptance.py')
