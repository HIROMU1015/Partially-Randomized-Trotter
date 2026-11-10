"""Read-only Phase0 record checks. No normalizer, N, price or R optimizer call."""
import csv,hashlib,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_r_normalization_phase0 as phase
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'run01/result.json').read_bytes());h=json.loads((HERE/'run01/HANDOFF.json').read_bytes())
plan=json.loads((ROOT/phase.PLAN_PATH).read_bytes());packet=phase.preflight(ROOT)
assert phase.public_plan(packet)==plan
assert r['scientific_source']=='6a75ebf967aaae2698ce6d295bf6fbec9d5e9929'
assert r['status']==h['status']=='PHASE0_NORMALIZATION_RECORDED' and not r['errors']
assert r['source_preserved'] and r['input_preserved'] and h['normalization_evidence_valid']
assert r['records']==h['records'] and len(r['records'])==24
assert r['scope_observed']==dict(normalization_rows=24,piece_slots_represented=144,signal_phase_radius_N_C_G=0)
assert r['saved_tails']==plan['saved_tails'] and r['shot_inflation_budget'] is None
for path,digest in r['source_files'].items():
 data=(ROOT/path).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
 assert data==phase.blob(ROOT,r['scientific_source'],path)
for ref,digest in r['saved_sources'].items():
 commit,path=ref.split(':',1);assert hashlib.sha256(phase.blob(ROOT,commit,path)).hexdigest()==digest
for actual,expected in zip(r['records'],plan['requests']):
 assert all(actual[k]==expected[k] for k in ('system','m','T','q_RPE','q_PF','delta','candidate_id','R_outer','formula','prefix','K','structure','request'))
 assert actual['N'] is actual['C'] is actual['G'] is None
 assert actual['phase_radius_N_status']=='NOT_EVALUATED' and actual['recommended_R'] is None and actual['scientific_candidate_PASS_FAIL'] is None
 n=actual['normalization'];assert n['status']=='NORMALIZATION_SCORED_NOT_N_QUALIFIED'
 assert n['lambda_source']==plan['saved_tails'][actual['system']]['source']
 assert math.isfinite(n['log_B']) and n['log_B']>=0 and n['optimistic_shot_inflation_log']==2*n['log_B']
 assert math.isfinite(n['one_ulp_accumulation_log_sensitivity']) and n['one_ulp_accumulation_log_sensitivity']>=0
 assert n['sensitivity_is_rigorous_bound'] is False and n['leading_term_is_used_to_accept'] is False
 assert actual['attenuation']['status']==actual['shot_inflation']['status']=='FINITE_BINARY64'
 assert 0<actual['attenuation']['value']<=1 and actual['shot_inflation']['value']>=1
 # Check stored exponential displays against each other, without reevaluating B.
 assert math.isclose(actual['shot_inflation']['value']*actual['attenuation']['value']**2,1.,rel_tol=4*sys.float_info.epsilon)
 st=actual['structure'];assert sum(st['allocation'])==actual['R_outer'] and st['event_count']==actual['q_PF']*actual['R_outer']
 assert n['relation_to_explicit_budget']=='NOT_CLASSIFIED_NO_BUDGET'
with (HERE/'run01/NORMALIZATION.csv').open(newline='') as f:rows=list(csv.DictReader(f))
assert len(rows)==24
for c,j in zip(rows,r['records']):
 assert c['candidate_id']==j['candidate_id'] and json.loads(c['allocation'])==j['structure']['allocation']
 assert float(c['log_B'])==j['normalization']['log_B'] and float(c['attenuation'])==j['attenuation']['value']
assert (HERE/'EXIT_CODE.txt').read_text().strip()=='0'
assert (HERE/'run01.stderr.log').read_bytes()==b''
for path in ['docs/research/h4_df10_rpe_batch_results_20261110/N_HANDOFF.json','docs/research/h6_df19_rpe_m8_12_results_20261110/run01/result.json']:
 assert subprocess.check_output(['git','rev-parse','HEAD:'+path],cwd=ROOT)==subprocess.check_output(['git','rev-parse',r['scientific_source']+':'+path],cwd=ROOT)
checks=dict(status='PASS_FIXED24_RECORD_AND_SOURCE_CHECKS',scientific_source=r['scientific_source'],records=24,piece_slots=144,
 source_files_preserved=9,input_hashes_matched=7,law_ID_allocation_unchanged=True,JSON_CSV_handoff_equal=True,
 overflow_displays=0,underflow_displays=0,numerical_normalization_unresolved=0,
 one_ulp_sensitivity_max=max(x['normalization']['one_ulp_accumulation_log_sensitivity'] for x in r['records']),
 old_H4_H6_N_artifacts_preserved=True,new_signal_phase_radius_N_C_G=0,R_chosen=False,independent_scientific_review=False,mandatory_stop=True)
(HERE/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(checks))
