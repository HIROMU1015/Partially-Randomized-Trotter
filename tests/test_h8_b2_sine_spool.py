"""Sine-only new scope: artificial receipts, tamper and first-failure gates."""
from copy import deepcopy
import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import h8_phase_b2_recovery_contract as c
import h8_phase_b2_recovery_runtime as r
import h8_phase_b2_recovery_adapter as a
import h8_b2_saved_spool_acceptance as accepted
import h8_phase_b2_recovery_witness as witness


def fixture():
 reg=dict(origin_SOURCE='a'*40,cosine_measured_RZ=1852640,sine_fingerprint='sine',sine_spool_sha256='spool',sine_spool_bytes=3)
 raw=dict(case_id='B0I_p21_q8_4th',axis='cosine',SOURCE=reg['origin_SOURCE'],measured_RZ=1852640,profile_sha256=c.PROFILE,compiler_hash=c.COMPILER,fingerprint_before='cosine',fingerprint_after='cosine',preparation_binding={'fixed':True},scalar_policy='ORIGINAL_PLUS_FULL_FOCK_TAIL_IDENTITY_ONCE')
 cos=dict(raw,model_sha256=c.MODEL_SHA,model_check=dict(PASS=True,exact_relative_error='0'),native_exit_code=0,reaped=True,result_sha256='cosine-raw')
 names=['saved_cosine_task','saved_directional_left','saved_directional_right','saved_axis_cosine','saved_axis_sine'];hashes={name:name for name in names};hashes['saved_cosine_task']='cosine-raw';hashes['validated_sine_QPY']='spool'
 values=dict(prior_execution_terminal=dict(status='STOP',SOURCE=reg['origin_SOURCE'],irrevocable_reservations={'B0I_p21_q8_4th_sine':{}},settlements={'B0I_p21_q8_4th_cosine':{}},compile_entry_markers=1,completed_calls_lower_bound={'compile':1}),prior_observer_terminal=dict(status='STOP',all_reaped=True,cleanup_errors=[]),prior_native_exit=dict(wait_completed=True,native_exit_code=1),prior_resource_stop=dict(reason='MEMORY_PRESSURE'),saved_cosine_accepted=cos,saved_cosine_task=raw,saved_directional_left=dict(errors=[1e-14]),saved_directional_right=dict(errors=[0.,1e-13]),saved_axis_cosine=dict(axis='cosine',max_control_error=1e-14,fingerprint='cosine'),saved_axis_sine=dict(axis='sine',max_control_error=2e-14,fingerprint='sine',qpy_spool=dict(sha256='spool',bytes=3)),saved_control_preparation=dict(SOURCE=reg['origin_SOURCE'],profile_sha256=c.PROFILE,preparation_binding={'fixed':True},scalar_policy=raw['scalar_policy'],wrapper_fingerprints=dict(cosine='cosine',sine='sine')))
 for receipt,ns in [('saved_cosine_receipt',names[:1]),('saved_directional_receipt',names[1:3]),('saved_axes_receipt',names[3:])]:
  values[receipt]=dict(status='PASS',all_reaped=True,cleanup_errors=[],workers=[dict(exit_code=0,reaped=True,result_sha256=hashes[n]) for n in ns])
 return values,hashes,reg

class SavedAcceptance(unittest.TestCase):
 def test_good_saved_metadata_no_recalculation(self):
  v,h,g=fixture();out=accepted.validate_saved(v,h,g);self.assertEqual(out['origin_status'],'STOP');self.assertEqual(out['new_compile'],0);self.assertEqual(out['saved_cosine']['measured_RZ'],1852640)
 def test_native_model_profile_control_and_fingerprint_mismatch_reject(self):
  for key,field,new in [('saved_cosine_accepted','native_exit_code',2),('saved_cosine_accepted','model_sha256','wrong'),('saved_cosine_accepted','profile_sha256','wrong'),('saved_axis_sine','fingerprint','wrong'),('saved_axis_sine','max_control_error',1e-8),('prior_native_exit','wait_completed',False),('prior_observer_terminal','all_reaped',False),('prior_execution_terminal','status','PASS')]:
   v,h,g=fixture();v[key][field]=new
   with self.subTest(key=key,field=field),self.assertRaises(ValueError):accepted.validate_saved(v,h,g)
 def test_original_unsettled_reservation_never_refunded(self):
  v,h,g=fixture();v['prior_execution_terminal']['settlements']['B0I_p21_q8_4th_sine']={}
  with self.assertRaisesRegex(ValueError,'RESERVATION'):accepted.validate_saved(v,h,g)
 def test_receipt_or_spool_hash_mismatch_reject(self):
  for name in ('saved_cosine_task','saved_axis_sine','validated_sine_QPY'):
   v,h,g=fixture();h[name]='tamper'
   with self.subTest(name=name),self.assertRaises(ValueError):accepted.validate_saved(v,h,g)
 def test_byte_tamper_rejected_before_metadata_claim(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'receipt';p.write_bytes(b'bad');reg={'required_input_receipts':[dict(logical_name='one',bytes=3,sha256=hashlib.sha256(b'yes').hexdigest())]}
   with patch.object(c,'spool_registry',return_value=reg),self.assertRaisesRegex(ValueError,'SAVED_RECEIPT_BYTES'):accepted.accept_saved(dict(input_mapping={'one':str(p)}))

class ScopeAndGate(unittest.TestCase):
 def test_one_sine_and_no_other_science_caps(self):
  p=c.plan();self.assertEqual([x['axis'] for x in p['axes']],['sine']);self.assertEqual(p['preparation_concurrency'],0);self.assertEqual(c.CAPS['compile'],1);self.assertEqual(c.CAPS['qpy_load'],1)
  self.assertTrue(all(v==0 for k,v in c.CAPS.items() if k not in ('compile','qpy_load','model_error_check')));self.assertEqual(c.RESOURCES['physical_cores'],4)
 def test_scope_denied_before_metadata_or_provider_import(self):
  permit=Mock();permit.require.side_effect=ValueError('DENIED')
  with patch.object(accepted,'accept_saved') as evidence,patch.object(a.importlib,'import_module') as imp:
   with self.assertRaisesRegex(ValueError,'DENIED'):a.evaluate({},permit,None)
   evidence.assert_not_called();imp.assert_not_called()
 def test_new_scope_requires_explicit_saved_acceptance(self):
  cfg=dict(SOURCE='a'*40,execution_sealed=True,plan_sha256=c.digest(c.plan()),resources=c.RESOURCES,output='/home/AbeHiromu/projects/h6-handoff-evidence/artificial-sine/o',control='/home/AbeHiromu/projects/h6-handoff-evidence/artificial-sine/c',lock='/home/AbeHiromu/projects/h6-handoff-evidence/artificial-sine/l')
  g=dict(schema='H8_PHASE_B2_RECOVERY_NEW_SCOPE_GRANT_V1',scope=c.SCOPE,approved_by_user=True,runtime_authorization=True,approval_text='artificial',SOURCE=cfg['SOURCE'],config_sha256='cfg',plan_sha256=c.digest(c.plan()),resources=c.RESOURCES,call_caps=c.CAPS,one_shot=True,retry=False,resume=False,refit=False,C_authorized=False,next_stage_authorized=False,accept_saved_B0_2_axes=True,accept_saved_cosine_and_sine_spool=True,driver_cpu=0,monitor_cpu=1,worker_cpus=[2],witness_cpu=3,allowed_cpus=[0,1,2,3])
  r.verify_authorization(cfg,g,config_sha='cfg',affinity=set(range(4)));g['accept_saved_cosine_and_sine_spool']=False
  with self.assertRaises(ValueError):r.verify_authorization(cfg,g,config_sha='cfg',affinity=set(range(4)))
 def test_saved_QPY_loaded_exactly_once_and_mismatch_never_retries(self):
  for measured in (1852640,100):
   with tempfile.TemporaryDirectory() as td:
    permit=r.Permit({},dict(scope=c.SCOPE,runtime_authorization=True,resources=c.RESOURCES,call_caps=c.CAPS,worker_cpus=[min(os.sched_getaffinity(0))]),td);permit.child_CPU_seconds=0.;calls=[]
    def mapped(tasks,callback,*,directory,reserve,**kwargs):
     task=tasks[0];calls.append(task['axis']);self.assertEqual(len(tasks),1);reserve(task);directory.mkdir();work=directory/task['id'];work.mkdir();p=work/'task_result.json';p.write_text(json.dumps(dict(case_id=task['case_id'],axis=task['axis'],measured_RZ=measured)))
     return dict(status='PASS',all_reaped=True,cleanup_errors=[],child_CPU_seconds=1,workers=[dict(task_id=task['id'],directory=str(work),exit_code=0,reaped=True,result_sha256=accepted.file_sha(p),identity=[1,2,3,4])])
    if measured==100:
     with self.assertRaisesRegex(ValueError,'AXIS_MODEL_ERROR'):a.serial_axes(c.plan(),permit,r.Writer(td),lambda case:{},None,map_owned=mapped)
    else:self.assertEqual(len(a.serial_axes(c.plan(),permit,r.Writer(td),lambda case:{},None,map_owned=mapped)),1)
    self.assertEqual(calls,['sine']);self.assertEqual(permit.actual()['compile'],1);self.assertEqual(permit.actual()['qpy_load'],1)

if __name__=='__main__':unittest.main()
