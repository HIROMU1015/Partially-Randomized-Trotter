#!/usr/bin/env python3
"""Reproduce configuration syntax failure; audit an unapplied patch, no science."""
import ast,difflib,hashlib,json,subprocess,sys
from fractions import Fraction
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
ART='artifacts/track_b_m1s_finite_signal_v2_preflight_stop/2026-10-16/v1'
C='c6254be7c41d800a8ee417e8dc124f03ba6b382d'
S='fbc2edea3816bf8f68508f62c6da595631df03b6'
CONTRACT='artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1/finite_signal_contract_v2_adopted.json'

def main():
    checks=[]
    def check(name,value):
        checks.append({'name':name,'status':'PASS' if value else 'FAIL'})
        if not value:raise AssertionError(name)
    def sha(b):return hashlib.sha256(b).hexdigest()
    def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a])
    raw=(ROOT/CONTRACT).read_bytes();contract=json.loads(raw)
    check('adopted_contract_bytes_preserved',raw==git('show',C+':'+CONTRACT) and sha(raw)=='36fcbad56a6f9a241f4ce2b796b45dc39887a931b0b6b388360f631f8382bb10')
    for p,h in contract['source_sha256'].items():
        b=(ROOT/p).read_bytes();ast.parse(b);check('frozen_SOURCE_preserved:'+p,sha(b)==h and b==git('show',S+':'+p))
    results=[]
    for key in ['eps_signal_abs','eps_phase_rad','phase_visibility_floor']:
        value=contract['precision'][key]
        try:Fraction(value)
        except (ValueError,TypeError) as exc:result={'key':key,'literal':value,'parses':False,'exception':type(exc).__name__,'reason':str(exc)}
        else:result={'key':key,'literal':value,'parses':True}
        results.append(result)
        check('expected_configuration_parse_outcome:'+key,result['parses']==(key=='phase_visibility_floor'))
    p='src/trottertracks/algorithm_codesign/m1s_finite_signal_v2/pilot.py';original=(ROOT/p).read_text();tree=ast.parse(original)
    lines=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Fraction' and n.args:
            expr=ast.unparse(n.args[0])
            if "POLICY['precision']['eps_" in expr:lines.append(n.lineno)
    check('three_real_failing_call_sites',sorted(lines)==[44,104,127])
    helper='''def _precision_target(name):
    # Proposed metadata notation fix only: accept exactly the adopted literals.
    targets={'eps_signal_abs':('2^-24',24),'eps_phase_rad':('2^-18',18)}
    literal,exponent=targets[name]
    if POLICY['precision'][name]!=literal:
        raise RuntimeError('unsupported fixed precision literal; no relaxation')
    return Fraction(1,1<<exponent)


'''
    proposed=original.replace('class Pilot:',helper+'class Pilot:')
    for key in ['eps_signal_abs','eps_phase_rad']:proposed=proposed.replace("Fraction(POLICY['precision']['"+key+"'])","_precision_target('"+key+"')")
    proposed_tree=ast.parse(proposed)
    expected_patch=''.join(difflib.unified_diff(original.splitlines(True),proposed.splitlines(True),fromfile='a/'+p,tofile='b/'+p,n=0))
    check('patch_exactly_metadata_helper_and_three_replacements',(ROOT/ART/'PROPOSED_NOT_APPLIED_precision_literal_fix.patch').read_text()==expected_patch)
    check('candidate_helper_AST_only_not_executed',any(isinstance(n,ast.FunctionDef) and n.name=='_precision_target' for n in ast.walk(proposed_tree)) and '_precision_target' not in original)
    check('no_eval_exec_in_proposal',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['eval','exec','compile'] for n in ast.walk(proposed_tree)))
    manifest=json.loads((ROOT/ART/'unissued_draft_manifest.json').read_bytes())
    for x in manifest['files']:
        b=(ROOT/x['archived_path']).read_bytes();check('unissued_draft_bytes:'+x['archived_path'],sha(b)==x['sha256'] and len(b)==x['bytes'])
        check('planned_authority_or_supervisor_not_active:'+x['planned_path'],not(ROOT/x['planned_path']).exists())
    check('no_finite_v2_external_marker',not list(Path('/tmp').glob('m1s_finite_signal_v2_once_*.launch.json')))
    for prefix in ['artifacts/track_b_m1s_finite_signal_v2_execution_authorization','artifacts/track_b_m1s_finite_signal_v2_result']:
        check('no_execution_namespace:'+prefix,not(ROOT/prefix).exists())
    stop=json.loads((ROOT/ART/'technical_preflight_stop.json').read_bytes())
    check('zero_starts_and_no_consumption',stop['launch_attempts']==stop['scientific_worker_starts']==stop['first_numerical_correctness_gate_runs']==stop['finite_step_pilot_runs']==0 and stop['scientific_one_shot_consumed'] is False)
    check('no_numerical_modules_loaded',not any(x.split('.')[0] in ['numpy','scipy','mpmath','openfermion','qiskit','trottertracks'] for x in sys.modules))
    report={'schema':'m1s_finite_v2_preflight_configuration_static_audit','static_audit_PASS':len(checks),'static_audit_FAIL':0,'checks':checks,
      'preflight_configuration_gate':'FAIL','configuration_parse_results':results,'invalid_precision_fields':2,
      'SOURCE_unchanged':True,'contract_unchanged':True,'patch_applied':False,'patch_numerical_correctness_verified':False,
      'configuration_syntax_probes':3,'scientific_worker_starts':0,'new_scientific_calculations':0,'new_scientific_numerical_correctness_tests':0,
      'Pauli_generation':0,'finite_signal_evaluations':0,'issued_execution_authorizations':0,'new_external_markers':0,'one_shot_consumed':False,'mandatory_STOP':True}
    (ROOT/ART/'configuration_static_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','configuration_parse_results']}))
if __name__=='__main__':main()
