#!/usr/bin/env python3
"""Adoption metadata, hashes, AST and closed-guard checks; no science kernels."""
import ast,copy,hashlib,importlib.abc,io,json,subprocess,sys
from contextlib import redirect_stdout
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
ART='artifacts/track_b_m1s_finite_signal_v2_formal_adoption/2026-10-16/v1'
OLD='artifacts/track_b_m1s_finite_signal_preparation/2026-10-16/v2'
CANDIDATE='68fa558043bb7b18f40bdc02f45d2c0bba95f8fa'
SOURCE='fbc2edea3816bf8f68508f62c6da595631df03b6'
BASE='3e742894e2c79e7e46b14de9801e7e1519062879'

class BlockScience(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ['numpy','scipy','mpmath','openfermion','qiskit'] or fullname.endswith(('.exact_input','.numerics','.word_reference','.pilot','.reference','.execution')):
            raise RuntimeError('numerical import forbidden: '+fullname)

def main():
    sys.meta_path.insert(0,BlockScience());calls={'numerical_kernel_calls':0}
    def profile(frame,event,arg):
        if event=='call' and '/m1s_finite_signal_v2/' in frame.f_code.co_filename and not frame.f_code.co_filename.endswith(('/control.py','/__init__.py')):
            calls['numerical_kernel_calls']+=1;raise RuntimeError('numerical call forbidden')
    sys.setprofile(profile);sys.path.insert(0,str(ROOT/'src'))
    from trottertracks.algorithm_codesign.m1s_finite_signal_v2 import control as c
    checks=[]
    def check(name,value):
        checks.append({'name':name,'status':'PASS' if value else 'FAIL'})
        if not value:raise AssertionError(name)
    def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a])
    def sha(b):return hashlib.sha256(b).hexdigest()
    def load(p):return c.read_json((ROOT/p).read_bytes())
    original=load(OLD+'/finite_signal_contract_candidate.json');adopted=load(ART+'/finite_signal_contract_v2_adopted.json')
    raw=(ROOT/OLD/'finite_signal_contract_candidate.json').read_bytes()
    check('exact_user_selected_candidate',sha(raw)=='03944ae9aa462df10e2d9a0aab68e1db531413a731c20707b4048136f6a873f3' and raw==git('show',CANDIDATE+':'+OLD+'/finite_signal_contract_candidate.json'))
    check('candidate_original_unchanged',original['adopted'] is False and original['coverage_model_adopted'] is False)
    changed={k for k in original if not c._same(original[k],adopted[k])}
    check('exact_four_adoption_changes',changed=={'status','adopted','coverage_model_adopted','adoption_record'})
    for k in sorted(set(original)-changed):check('frozen_machine_field:'+k,c._same(original[k],adopted[k]))
    state=c.validate_contract(adopted,True)
    check('complete_adopted_contract',state['valid'] and state['contract_complete'] and state['pending']==[])
    check('complete_does_not_grant_execution',state['actual_execution_permitted'] is False and adopted['contract_grants_execution'] is False)
    record=load(adopted['adoption_record']['path'])
    check('adoption_record_hash',sha((ROOT/adopted['adoption_record']['path']).read_bytes())==adopted['adoption_record']['sha256'])
    check('record_kind_matches_frozen_guard',record['record_kind']=='FINITE_SIGNAL_V2_CONTRACT_ADOPTION_NOT_EXECUTION_AUTHORIZATION')
    check('user_contract_adoption_only',record['scientific_execution_authorized'] is False and record['contract_grants_execution'] is False)
    check('conditional_adoption_is_not_certification',record['conditional_coverage_adopted'] is True and record['platform_certified'] is False and record['numerical_correctness_verified'] is False and record['evidence_class']=='CONDITIONAL_ENGINEERING_NOT_CERTIFIED')
    for k,v in record['conditions_adopted'].items():check('adopted_conditions:'+k,c._same(v,adopted[k]))
    for key in ['user_statement','conversation_context']:
        b=record[key];check('conversation_binding:'+key,sha((ROOT/b['path']).read_bytes())==b['sha256'])
    context=load(record['conversation_context']['path']);statement=(ROOT/record['user_statement']['path']).read_text()
    check('current_user_not_template_is_authority',statement==context['current_user_message']['content']+'\n' and context['explicit_user_adoption'] is True and '科学計算は承認しません。' in statement)
    check('candidate_source_base_order',all(a!=b and subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',a,b]).returncode==0 for a,b in [(adopted['base_commit'],SOURCE),(SOURCE,CANDIDATE),(CANDIDATE,BASE)]))
    check('SOURCE_pin_unchanged',adopted['source_commit']==SOURCE)
    for p,h in adopted['source_sha256'].items():
        b=(ROOT/p).read_bytes();ast.parse(b,filename=p)
        check('SOURCE_AST_bytes_hash:'+p,b==git('show',SOURCE+':'+p) and sha(b)==h)
    for key in ['input','saved_law','independent_result','review','amendment_review','result_schema']:
        binding=adopted[key];b=(ROOT/binding['path']).read_bytes()
        check('frozen_evidence_hash:'+key,sha(b)==binding['sha256'] and b==git('show',CANDIDATE+':'+binding['path']))
    rows=c.planned_rows(adopted)
    check('planned_108_rows_unchanged',rows==load(OLD+'/planned_rows.json')['rows'] and len(rows)==108)
    check('zero36_nonzero72',sum(r['s']==0 for r in rows)==36 and sum(r['s']>0 for r in rows)==72)
    for argv,code in [([],0),(['validate-config','--contract',str(ROOT/ART/'finite_signal_contract_v2_adopted.json')],0),(['run','--contract',str(ROOT/ART/'finite_signal_contract_v2_adopted.json')],2)]:
        with redirect_stdout(io.StringIO()) as capture:actual=c.main(argv)
        value=json.loads(capture.getvalue());check('metadata_or_missing_authority_refusal:'+repr(argv),actual==code and value['actual_execution_permitted'] is False)
    for authority in ['artifacts/track_b_m1s_gate0_v5_execution_authorization/old.json','artifacts/track_b_m1s_finite_signal_execution_authorization/old.json']:
        try:c.authorize(ROOT,ART+'/finite_signal_contract_v2_adopted.json','0'*40,authority)
        except c.ExecutionClosed:refused=True
        else:refused=False
        check('old_authority_rejected_before_IO:'+authority,refused)
    try:c.require_permit(None)
    except c.ExecutionClosed:refused=True
    else:refused=False
    check('no_bare_execution_permit',refused)
    for prefix in [c.AUTH_PREFIX,c.OUTPUT_PREFIX,'artifacts/track_b_m1s_finite_signal_execution_authorization/','artifacts/track_b_m1s_finite_signal_result/']:
        check('no_new_authority_or_results:'+prefix,not(ROOT/prefix).exists())
    check('no_numerical_kernel_calls',calls['numerical_kernel_calls']==0)
    for p in sorted((ROOT/ART).glob('*.json')):
        if p.name not in ['evidence_index.json','publication_receipt.json','adoption_static_validation.json']:
            load(str(p.relative_to(ROOT)));check('strict_JSON:'+p.name,True)
    result={'schema':'m1s_finite_signal_v2_adoption_static_validation','PASS':len(checks),'FAIL':0,'checks':checks,'adopted_contract_status':state,
      'candidate_commit':CANDIDATE,'source_commit':SOURCE,'contract_adopted':True,'coverage_model_adopted':True,'contract_grants_execution':False,
      'raw_input_file_hash_reads':1,'NPZ_array_decodes':0,'numerical_kernel_calls':calls['numerical_kernel_calls'],
      'new_scientific_calculations':0,'new_numerical_correctness_tests':0,'Pauli_generation':0,'finite_signal_evaluations':0,
      'new_execution_authorizations_grants_markers':0,'numerical_correctness_verified':False,'platform_certified':False,'mandatory_STOP':True}
    (ROOT/ART/'adoption_static_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['checks','adopted_contract_status']}))
if __name__=='__main__':main()
