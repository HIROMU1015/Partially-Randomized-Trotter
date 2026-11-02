"""Read-only JSON/Fraction/SHA audit; no Arb, Qiskit or scientific runner."""
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
KEY='h4_central_arb_pr0_preparation_20261025'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify():
    checks=[]
    def check(ok,name):
        if not ok:raise AssertionError(name)
        checks.append(name)
    old=read(BASE/'preservation_before.json');inputs=read(BASE/'inputs.json')['files']
    check(all(sha(ROOT/p)==s for p,s in old.items()),'OLD_EVIDENCE_AND_PF_SCIENTIFIC_CODE_UNCHANGED')
    check(all(sha(ROOT/p)==s for p,s in inputs.items()),'SAVED_INPUTS_UNCHANGED')
    prior=read(BASE/'prior_manifest_rows.json');manifest=read(ROOT/'artifacts/validation_manifest.json')
    check(set(manifest)==set(prior)|{KEY} and all(digest(manifest[k])==h for k,h in prior.items()),'OLD_MANIFEST_ROWS_UNCHANGED')
    c=read(BASE/'pilot_contract_DRAFT_DISABLED.json')
    check(all(c[k] is False for k in ('execution_authorized','dispatch_enabled','grant_valid')),'CONTRACT_DISABLED')
    check(all(c[k] is None for k in ('grant','scientific_runner_SOURCE','launch_command')),'NO_SCIENTIFIC_GRANT_SOURCE_OR_LAUNCH')
    check(c['candidate']['key']=='PR0_cosine' and c['candidate']['index']==0 and c['candidate']['axis']=='cosine','ONE_PR0_COSINE_ONLY')
    check(c['precision_schedule_bits']==[64,96,128] and c['max_initial_state_actions']==6 and len(c['state_action_limits'])==6
          and all(v==1 for v in c['state_action_limits'].values()) and c['max_residual_scalar_reductions']==3,'FINITE_FUTURE_SIX_ACTION_SCOPE')
    check(not c['width_is_mandatory'] and not c['width_is_new_allowance'] and not c['successful_action_replay_allowed'],'NO_ADVISORY_WIDTH_ESCALATION_OR_REPLAY')
    p=read(BASE/'PR0_selected_C_plan_NO_EXECUTION.json');program=read(BASE/'PR0_compiled_program_NO_EXECUTION.json')
    target=read(BASE/'PR0_target_QUOTED.json');guard=read(BASE/'PR0_guard_QUOTED.json');inspection=read(BASE/'structural_inspection.json')
    model=read(ROOT/target['state_model']);case=model['cases']['PR'];join=read(ROOT/target['price_join'])
    check(sha(BASE/'PR0_selected_C_plan_NO_EXECUTION.json')==inspection['C_target_plan_sha256']==c['input_plans']['selected_C_plan_sha256'],'SELECTED_PLAN_SHA')
    check(sha(BASE/'PR0_compiled_program_NO_EXECUTION.json')==inspection['compiled_descriptor_sha256']==c['input_plans']['compiled_descriptor_sha256'],'COMPILED_DESCRIPTOR_SHA')
    check(p['domain']=='SAVED_PR0_NO_EXECUTION' and p['n_system']==8 and p['q']==8 and p['r']==4,'REAL_PLAN_STATIC_DOMAIN')
    check(p['trajectory_id']==target['trajectory_id']==join['trajectory_id'] and p['event_record_hash']==join['event_record_hash']
          and p['trajectory_record_hash']==join['trajectory_record_hash'],'SELECTED_EVENT_PROVENANCE')
    check(len(p['events'])==32 and all(e['taylor_order']==0 for e in p['events']),'ALL_32_PR0_EVENTS_ORDER0')
    check(len(p['retained_blocks'])==27 and all(b['label']==source['label'] and Q(b['angle_exact'])==Q(source['coefficient'])*Q(model['step_time'])/2
          for b,source in zip(p['retained_blocks'],case['blocks'])),'RETAINED_EXACT_OPERANDS_NO_BINARY64_COERCION')
    check(Q(p['native_scalar_exact_sum'])==sum(map(Q,case['native_scalar_phases']),Q(0)),'NATIVE_SCALAR_EXACT_SUM')
    check(not p['PR_law_mean_used'] and not p['probability_weights_applied'] and not p['normalization_B_applied_to_target'],'NO_LAW_MEAN_POINT_OR_PREAPPLIED_B')
    check(p['B_saved_binary64_exact']==c['B_saved_exact']==guard['scale']['B_saved_exact'] and c['rho_budget_exact']==guard['rho_guard_exact'],'UNCHANGED_SAVED_B_AND_BUDGET')
    check(program['axis']=='cosine' and program['ancilla']==8 and len(program['instructions'])==3139
          and program['origin']['qpy_sha256']==target['output_sha256'] and program['origin']['fingerprint']==target['output_fingerprint'],'FIXED_COMPILED_LAYOUT_FINGERPRINT')
    check(inspection['QPY_decodes']==4 and inspection['state_actions']==0 and all(inspection[k] is None for k in ('target_interval','compiled_signal_interval','rho_interval')),'STATIC_INSPECTION_NO_REAL_NUMERICS')
    tests=read(BASE/'arb_artificial_tests.json');qpy=read(BASE/'qpy_artificial_tests.json')
    check(tests['total']+qpy['total']==38 and all(r['exit_code']==0 for r in tests['records']+qpy['records']),'ARTIFICIAL_38_PASS')
    check(not (BASE/'run01').exists() and not list(BASE.glob('*grant.json')) and not list(BASE.glob('*ledger*')),'NO_SCIENCE_NAMESPACE_GRANT_OR_LEDGER')
    stop=read(BASE/'STOP.json')
    check(stop['mandatory_STOP'] and not stop['automatic_next_stage'] and stop['real_H4_state_actions']==0
          and stop['science_grants_issued']==stop['science_grants_consumed']==0,'TECHNICAL_MANDATORY_STOP')
    check(not c['formal_certificate_accepted'] and not c['LOCAL_BUDGET_MET'] and c['PR_law_rho'] is None and not c['old_N_G_changed'],'NO_CERTIFICATE_OR_GLOBAL_PROMOTION')
    return dict(status='PASS_PREPARATION_METADATA_ONLY',checks=checks,check_count=len(checks),preserved_files=len(old),saved_inputs=len(inputs),
        old_manifest_rows=len(prior),artificial_PASS=38,QPY_decodes=4,real_H4_state_actions=0,science_grants_issued=0,science_grants_consumed=0,
        local_PR0_rho_upper=None,C_target_interval=None,contract_execution_authorized=False,mandatory_STOP=True)


if __name__=='__main__':print(json.dumps(verify(),indent=2,sort_keys=True))
