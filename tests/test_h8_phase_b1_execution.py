"""Artificial exact-fit, holdout/STOP, and grant-boundary checks only."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/trotterlib'));sys.path.insert(0,str(ROOT/'scripts'))
import h8_phase_b1_fit as f
import h8_phase_b1_execution as run
from h8_phase_b1_contract import REQUIRED,PROFILE,COMPILER


def fixture():
 rows=[]
 for m,axis in f.STRATA:
  for p in (14,21):
   for q in ((1,8) if m=='B0' else (1,2,4,8)):
    a,b,d=(4,7,11) if m=='B0' else (9,13,17)
    rows.append(dict(id=f'{m}_p{p}_q{q}_2nd_{axis}',method=m,axis=axis,prefix=p,q=q,formula='2nd',saved_RZ=a+b*q+d*p*q,
     role='HOLDOUT' if m=='B0' and (p,q) in f.HOLDOUT else 'TRAIN',
     origin_SOURCE='db559a518214fd27fa96e2f1451911df05c4f4a8' if m=='B0' else '54ed98c5fa5b7190f77255271f4e3bec8db15630',
     runtime_profile_sha256=PROFILE,compiler_hash=COMPILER,control='symmetric_directional',
     scalar_policy='ORIGINAL_CONSTANT_ONCE_NO_TAIL_IDENTITY' if m=='B0' else 'ORIGINAL_PLUS_FULL_FOCK_TAIL_IDENTITY_ONCE',
     evidence={k:True for k in REQUIRED}))
 axes=[dict(candidate_id=c,axis=axis) for c in ('B0_p15_q1_4th','B0I_p21_q8_4th') for axis in ('cosine','sine')]
 return rows,axes


def execute(rows,axes,used,saved):
 def take(key):
  assert used.get(key,0)<f.CAPS[key];used[key]=used.get(key,0)+1
 return f.execute(rows,axes,take=take,write=lambda name,value:saved.__setitem__(name,deepcopy(value)),guard=lambda:None)


class Fit(unittest.TestCase):
 def test_exact_OLS_and_independent_methods_axes(self):
  rows,axes=fixture();used={};saved={};result=execute(rows,axes,used,saved)
  self.assertTrue(result['all_four_strata_PASS']);self.assertEqual(used['cost_fit'],4)
  self.assertEqual([m['coefficients'] for m in saved['frozen_models.json']['models']],[['4','7','11']]*2+[['9','13','17']]*2)
  self.assertEqual(used['B0_holdout_checks'],2);self.assertEqual(used['B0I_anchor_checks'],16)
  self.assertFalse(result['B2_authorized']);self.assertFalse(saved['frozen_models.json']['S4_measured_validation'])
 def test_holdout_not_in_fit_and_failure_prevents_next_axis(self):
  rows,axes=fixture();hold=next(r for r in rows if r['method']=='B0' and r['axis']=='cosine' and r['role']=='HOLDOUT');hold['saved_RZ']*=2
  used={};saved={}
  with self.assertRaisesRegex(ValueError,'GE_2_PERCENT'):execute(rows,axes,used,saved)
  self.assertEqual(used['cost_fit'],1);self.assertNotIn('B0_sine_partial.json',saved)
  self.assertEqual(saved['B0_cosine_partial.json']['coefficients'],['4','7','11']);self.assertNotIn('frozen_models.json',saved)
 def test_B0I_residual_failure_stops_fourth_fit(self):
  rows,axes=fixture();next(r for r in rows if r['method']=='B0I' and r['axis']=='cosine')['saved_RZ']*=10
  used={};saved={}
  with self.assertRaisesRegex(ValueError,'GE_2_PERCENT'):execute(rows,axes,used,saved)
  self.assertEqual(used['cost_fit'],3);self.assertNotIn('B0I_sine_partial.json',saved)
 def test_exact_threshold_and_measured_denominator(self):
  self.assertFalse(f.check(Fraction(98),100)['PASS']);self.assertFalse(f.check(Fraction(102),100)['PASS'])
  self.assertTrue(f.check(Fraction(10199,100),100)['PASS']);self.assertTrue(f.check(Fraction(100),102)['PASS'])
 def test_rank_deficiency_and_nonpositive_predictions_rejected(self):
  with self.assertRaisesRegex(ValueError,'SINGULAR'):f.solve([dict(prefix=14,q=1,saved_RZ=10)]*3)
  with self.assertRaisesRegex(ValueError,'NONPOSITIVE'):f.predict([Fraction(-1),Fraction(0),Fraction(0)],14,1)
 def test_all_metadata_checked_before_any_fit(self):
  for mutate in (lambda r:r.pop(),lambda r:r.append(deepcopy(r[0])),lambda r:r[0].update(origin_SOURCE='old'),
    lambda r:r[0].update(role='HOLDOUT'),lambda r:r[0]['evidence'].update(runtime_profile_bound=False),
    lambda r:r[0].update(scalar_policy='tail'),lambda r:r[0].update(saved_RZ=0)):
   rows,axes=fixture();mutate(rows);used={}
   with self.assertRaises(ValueError):execute(rows,axes,used,{})
   self.assertEqual(used,{})
 def test_B2_registry_changes_rejected_before_fit(self):
  rows,axes=fixture();axes[0]['candidate_id']='B0_p16_q1_4th';used={}
  with self.assertRaisesRegex(ValueError,'PREREGISTERED'):execute(rows,axes,used,{})
  self.assertEqual(used,{})
 def test_S4_estimates_preserve_single_wrapper_scalar_intercept(self):
  rows,axes=fixture();saved={};execute(rows,axes,{},saved)
  p=saved['B2_predictions_UNAPPROVED.json']['axes'][0]
  self.assertEqual(Fraction(p['exact_predicted_RZ']),4+3*(7+11*15));self.assertFalse(p['compile_authorized'])


class Execution(unittest.TestCase):
 def test_artificial_end_to_end_success_and_holdout_STOP(self):
  code='''
import json,os,sys
from pathlib import Path
sys.path[:0]=['scripts','src/trotterlib','tests']
import h8_phase_b1_execution as run
import h8_phase_b1_fit as f
from test_h8_phase_b1_execution import fixture
base=Path(sys.argv[1]);fail=sys.argv[2]=='1';rows,axes=fixture()
if fail:next(r for r in rows if r['method']=='B0' and r['axis']=='cosine' and r['role']=='HOLDOUT')['saved_RZ']*=2
run.write(base/'anchors.json',dict(rows=rows));run.write(base/'axes.json',dict(axes=axes))
inventory=run.cpu.inventory(os.sched_getaffinity(0));selected=[];seen=set()
for c,row in sorted(inventory.items()):
 key=(row['package_id'],row['core_id'])
 if key not in seen:selected.append(c);seen.add(key)
 if len(selected)==2:break
cfg=dict(SOURCE='a'*40,plan_sha256=f.digest(f.plan()),python=sys.executable,source_hashes={},
 files=[dict(path=str(base/n),bytes=(base/n).stat().st_size,sha256=run.a.sha(base/n)) for n in ['anchors.json','axes.json']],
 anchors=str(base/'anchors.json'),B2_selection=str(base/'axes.json'),output=str(base/'output01'),control=str(base/'control01'),lock=str(base/'one_shot.lock'))
g=dict(schema='H8_B1_NEW_SCOPE_GRANT_V1',approved_by_user=True,runtime_authorization=True,approval_text='ARTIFICIAL FIXTURE ONLY',
 scope=f.SCOPE,SOURCE=cfg['SOURCE'],config_sha256=f.digest(cfg),plan_sha256=cfg['plan_sha256'],resources=f.RESOURCES,call_caps=f.CAPS,
 accept_saved_B0_with_original_STOP=True,one_shot=True,retry=False,resume=False,B2_authorized=False,C_authorized=False,next_stage_authorized=False,allowed_cpus=selected)
run.write(base/'config.json',cfg);run.write(base/'grant_APPROVED_ONCE.json',g)
run.verify_all=lambda cfg:run.check_rows(cfg['files'])
run.gate.measure_idle=lambda cpus:dict(status='PASS',artificial_CPU_gate=True)
run.gate.filesystem=lambda base:dict(bytes=2**40,inodes=1000000,quota_verified=True)
try:run.launch(base)
except RuntimeError:
 if not fail:raise
else:assert not fail
terminal=run.a.read(base/'phase_terminal.json');owned=run.a.read(base/'control01/launcher_terminal.json')
assert terminal['status']==('STOP' if fail else 'COMPLETE_MANDATORY_STOP')
assert owned['all_reaped'] and not owned['cleanup_errors']
driver=run.a.read(base/'output01/terminal.json')
assert driver['actual_calls']['cost_fit']==(1 if fail else 4)
if fail:assert not (base/'output01/B0_sine_partial.json').exists()
try:run.launch(base)
except ValueError as error:assert str(error)=='B1_ATTEMPT_ALREADY_SPENT'
else:raise AssertionError('synthetic attempt reused')
'''
  for fail in (False,True):
   with tempfile.TemporaryDirectory() as td:
    r=subprocess.run([sys.executable,'-B','-c',code,td,'1' if fail else '0'],cwd=ROOT,capture_output=True,text=True,timeout=30)
    self.assertEqual(r.returncode,0,r.stderr)
 def test_default_CLI_has_no_scientific_imports_or_launch(self):
  code='import runpy,sys;sys.argv=["h8_phase_b1_execution.py"];runpy.run_path("scripts/h8_phase_b1_execution.py",run_name="__main__");assert not any(x in sys.modules for x in ("numpy","qiskit","scipy"))'
  r=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=10)
  self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['scope'],f.SCOPE)
 def test_old_or_unapproved_grant_denied_before_data_or_science(self):
  for g in ({},dict(schema='H8_B1_NEW_SCOPE_GRANT_V1',approved_by_user=False),dict(scope='H8_B0_CURRENT_PROFILE_S2_ANCHOR4_ACQUISITION_V1')):
   with self.assertRaisesRegex(ValueError,'NEW_EXPLICIT'):run.authorized(dict(SOURCE='a'*40),g)
 def test_valid_exact_scope_and_no_retry(self):
  cfg=dict(SOURCE='a'*40,plan_sha256=f.digest(f.plan()));g=dict(schema='H8_B1_NEW_SCOPE_GRANT_V1',approved_by_user=True,runtime_authorization=True,approval_text='artificial',scope=f.SCOPE,SOURCE=cfg['SOURCE'],config_sha256=f.digest(cfg),plan_sha256=cfg['plan_sha256'],resources=f.RESOURCES,call_caps=f.CAPS,accept_saved_B0_with_original_STOP=True,one_shot=True,retry=False,resume=False,B2_authorized=False,C_authorized=False,next_stage_authorized=False,allowed_cpus=[0,1])
  with patch.object(run.os,'sched_getaffinity',return_value={0,1}):
   self.assertEqual(run.authorized(cfg,g),[0,1])
   for key,value in [('retry',True),('B2_authorized',True),('accept_saved_B0_with_original_STOP',False),('SOURCE','b'*40)]:
    changed=dict(g,**{key:value})
    with self.assertRaises(ValueError):run.authorized(cfg,changed)
 def test_new_scope_keeps_fit_and_compile_caps_separate(self):
  self.assertEqual(f.CAPS['cost_fit'],4);self.assertEqual(f.CAPS['new_compile'],0)
  self.assertEqual(f.RESOURCES['driver_AS_bytes'],512*2**20);self.assertEqual(f.RESOURCES['driver_RSS_bytes'],256*2**20)
  self.assertEqual(f.RESOURCES['CPU_seconds'],240);self.assertEqual(f.RESOURCES['wall_seconds'],600)


if __name__=='__main__':unittest.main()
