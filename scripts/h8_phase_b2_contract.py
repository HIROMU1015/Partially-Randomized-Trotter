"""Fixed H8 B2 scope: four serial S4 axes; no fitting or reaccounting."""
from fractions import Fraction
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
BUNDLE='artifacts/resource_applicability/track_a_h8_phase_b2_readiness/2026-10-17'
BINDING=BUNDLE+'/B2_input_binding_v1.json'
BINDING_SHA='a0f3a620f4f07dc1d74794a0f9681a558e5790807f4a51e34f12d09932b7b6a2'
SCOPE='H8_B2_FROZEN_S4_FOUR_AXIS_SERIAL_VALIDATION_V1'
PROFILE='622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PROVIDER='c31ef424175c02379eafebe8fc6c7548c1d9aabe'
RUNTIME_MAP_SHA='850568c8018943b91dc9dbb031198b684112f88f8870f5a217253c61ee5fb958'
PROVIDER_MAP_SHA='9382176df9a1353c9f02b842edec5943dfbeb06b223387f6a99542e53c878bc0'
MODEL_SHA='8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0'
AXES=[dict(case_id=cid,axis=a,exact_prediction=prediction) for cid,prediction in
 [('B0_p15_q1_4th','1175933/7'),('B0I_p21_q8_4th','1852640')] for a in ('cosine','sine')]
CASES=[dict(id=cid,method=method,prefix=p,q=q,formula='4th') for cid,method,p,q in
 [('B0_p15_q1_4th','B0',15,1),('B0I_p21_q8_4th','B0I',21,8)]]
RESOURCES=dict(driver_processes=1,max_compile_workers=1,observer_processes=1,physical_cores=3,
 threads_per_role=1,max_cases_in_parent=1,driver_AS_bytes=64*2**30,driver_RSS_bytes=32*2**30,
 worker_AS_bytes=64*2**30,worker_RSS_bytes=32*2**30,observer_AS_bytes=512*2**20,
 observer_RSS_bytes=256*2**20,aggregate_RSS_bytes=int(64.25*2**30),admission_reserve_bytes=16*2**30,
 CPU_seconds=86400,worker_CPU_seconds=86400,observer_CPU_seconds=1800,aggregate_CPU_seconds=86400,
 wall_seconds=43200,max_workers=1,AS_bytes=64*2**30,RSS_bytes=32*2**30,
 stage_storage_bytes=8*2**30,output_bytes=256*2**20,logs_bytes=128*2**20,log_bytes=128*2**20,
 fresh_free_disk_bytes=16*2**30,fresh_free_inodes=40000,minimum_free_inodes=40000,output_files_max=20000)
CAPS=dict(snapshot_load=1,saved_preparation_decode=1,prefix_view=2,native_evolution_build=2,
 measured_wrapper_build=4,probe_wrapper_copy=4,control_probe=8,compile=4,model_error_check=4,
 cost_fit=0,cost_reaccount=0,new_signal=0,new_full_H_reference=0,SCF=0,DF=0,state_solver=0,
 PR_sampling=0,law=0,tail_scalar=0,unregistered_compile=0)


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def binding():
 raw=(ROOT/BINDING).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=BINDING_SHA:raise ValueError('B2_BINDING_BYTES_CHANGED')
 return json.loads(raw)


def plan():
 return dict(schema='H8_PHASE_B2_EXECUTION_V1',scope=SCOPE,cases=CASES,axes=AXES,
 binding=binding(),call_caps=CAPS,resources=RESOURCES,model_sha256=MODEL_SHA,
 compiler_options=binding()['compiler_options'],control='symmetric_directional',
 scalar_policies=dict(B0='ORIGINAL_CONSTANT_ONCE_NO_TAIL_IDENTITY',B0I='ORIGINAL_PLUS_FULL_FOCK_TAIL_IDENTITY_ONCE'),
 native_instruction_cap=4_000_000,transpiled_instruction_cap=10_000_000,control_tolerance=1e-8,
 compile_concurrency=1,check_axis_before_next_admission=True,
 relative_error_denominator='MEASURED_RZ',PASS='abs(predicted-measured)/measured < 1/50',
 mismatch='STOP_NO_REFIT_NO_RETRY_NO_EXTRA_COMPILE',original_B0_run02_status='STOP',
 original_B1_result='83a3c4a6688e66576f85471bf41a45e29cbfb705',
 retry=False,resume=False,refit=False,one_shot=True,C_authorized=False,next_stage_authorized=False,
 mandatory_stop=True)


def axis_check(prediction,measured):
 if type(measured)is not int or measured<=0:raise ValueError('POSITIVE_INTEGER_MEASUREMENT_REQUIRED')
 prediction=Fraction(prediction)
 if prediction<=0:raise ValueError('POSITIVE_FROZEN_PREDICTION_REQUIRED')
 error=abs(prediction-measured)/measured
 return dict(exact_prediction=str(prediction),measured_RZ=measured,exact_relative_error=str(error),
  relative_error_percent=float(error*100),PASS=error<Fraction(1,50))


def validate_complete(rows,actual_calls):
 if len(rows)!=4 or [(r['case_id'],r['axis']) for r in rows]!=[(r['case_id'],r['axis']) for r in AXES]:
  raise ValueError('EXACT_SERIAL_FOUR_AXIS_ORDER')
 if actual_calls!=CAPS:raise ValueError('EXACT_B2_CALL_COUNTS')
 for row,axis in zip(rows,AXES):
  if not (row['model_sha256']==MODEL_SHA and row['profile_sha256']==PROFILE and row['compiler_hash']==COMPILER
    and row['native_exit_code']==0 and row['reaped'] is True and row['fingerprint_before']==row['fingerprint_after']
    and 0<=row['max_control_error']<1e-8 and row['model_check']==axis_check(axis['exact_prediction'],row['measured_RZ'])
    and row['model_check']['PASS']):raise ValueError('B2_COMPLETE_BINDING_OR_ERROR')
 return dict(status='VALIDATED_ONLY_ON_FOUR_REGISTERED_S4_AXES',full_grid_guaranteed=False,
  rigorous_bias_or_PR_population_guaranteed=False,C_authorized=False,next_stage_authorized=False)


SOURCES=(
 'scripts/h8_phase_b2_contract.py','scripts/h8_phase_b2_runtime.py','scripts/h8_phase_b2_adapter.py',
 'scripts/run_h8_phase_b2.py','scripts/launch_h8_phase_b2.py','scripts/prepare_h8_phase_b2.py',
 'scripts/h8_b0_anchor_adapter.py','scripts/h8_b0_anchor_contract.py','scripts/h8_b0_anchor_runtime.py',
 'scripts/h8_strengthened_export.py','scripts/h8_strengthened_contract.py','scripts/h8_strengthened_runtime.py',
 'scripts/launch_h8_strengthened_phase_a.py','scripts/track_a_phase_a_parallel.py',
 'src/trotterlib/h8_b0_anchor_plan.py','src/trotterlib/h8_b0_saved_anchor_audit.py',
 'scripts/probe_h8_b0_anchor_cpus.py',BINDING,
 BUNDLE+'/saved_control_targets_v1.json',BUNDLE+'/prefix_preparation_binding_v1.json',
 BUNDLE+'/instruction_allocation_bounds_v1.json')
