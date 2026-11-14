"""C consumer admission of saved m4 receipts/comparisons; no price arithmetic."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from trottertracks import price_implementation_guard as G
import admit_pr_df_ld1_q3_prices as A

ROOT=Path(__file__).resolve().parents[1]
BASE='docs/research/pr_df_ld1_m4_holdout_admission_20261115'
D='063f2b3b6ea271a179a469be91252fa4e5a7d096'
EXEC='7bdfc6f629aa74a5c1347abf84ad6486c669c9d8'
RAW='3042e6e66ad693ed514274334adecd3d1b42dc02'
D_DIR='docs/research/d_pr_df_ld1_m4_holdout_results_20261115/'
FROZEN=dict(commit='3cac53da1a4a8b2228245593c74c1ff9ec1f5212',path='docs/research/pr_df_ld1_residual_freeze_20261115/run01/FROZEN_MODELS.json')
PRIOR=dict(commit='c99d511e2422b8f250490b11054f754a81e536c5',path=A.BASE+'/ADMISSION.json')
FACTS=dict(commit=PRIOR['commit'],path=A.BASE+'/SOURCE_FACTS.json')
HANDOFF=dict(commit=D,path=D_DIR+'C_HANDOFF.json')
COMPARISON=dict(commit=D,path=D_DIR+'COMPARISON.json')
INPUTS=dict(commit=EXEC,path=D_DIR+'INPUTS.json')
PID='3df0cfd3356c358aeda8'
COHORT='D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1'
VERDICT='NOT_ASSESSED_NO_ADOPTED_TOLERANCE'
EXACT_KEYS=('P','O','signed_error','absolute_error','signed_relative_error','absolute_relative_error')


def at(ref,*selector):return dict(ref,selector=list(selector))


def receipt(root,ref):
    raw=G.C.blob(root,ref['commit'],ref['path'])
    if ref.get('sha256'):A.equal(hashlib.sha256(raw).hexdigest(),ref['sha256'],'Original receipt hash')
    return G.evidence(root,ref)


def check_row(r,original,path,axis_receipt,structural,gate,cell,prediction,request,profile):
    axis=r['axis'];a,name=A.AXES[axis]
    A.equal((r['physical_id'],r['m'],r['cohort'],r['role']),(PID,4,COHORT,'HOLDOUT'),'m4 physical/cohort/role')
    A.equal(r['seeds'],cell['seeds'][r['index']],'Pre-fixed m4 seeds')
    A.equal(r['frozen_prediction_identity'],prediction['identity'],'Frozen/current physical identity')
    A.equal((request['candidate']['id'],request['candidate']['T'],request['candidate']['q'],request['candidate']['R_outer']),
            (PID,1.6,16,32),'m4 T/q/R request')
    A.equal(r['original_price_configuration_id'],request['identity_configuration_id'],'Original configuration carrier')
    A.equal(r['original_price_canonical_id'],request['V2_candidate_id'],'Original canonical carrier')
    pair=A.path_binding({},path,original,axis_receipt,axis,cell,request['candidate'],EXEC,profile['compiler'])
    A.equal((r['C'],r['S'],r['event_digest'],r['wrapper_fingerprint']),
            (pair['C'],pair['S'],pair['event_digest'],pair['fingerprint']),'Handoff/receipt same-axis pair')
    A.equal(structural['S'],path['S'],'S receipt')
    A.equal(structural['event_digest'],r['event_digest'],'S/C event digest')
    A.equal(r['native_axis'],name,'Measurement axis')
    A.equal((r['execution_source'],r['result_publication_commit']),(EXEC,RAW),'Original execution/raw SOURCE')
    A.equal(path['lowering'],G.NEW_ZZ,'Original new ZZ')
    A.equal(r['lowering'],path['lowering'],'Handoff lowering')
    A.equal((r['scalar_hex'],r['scalar_application']),(path['scalar_hex'],'ONCE_AT_TOTAL_T'),'Scalar binding')
    A.equal(path['scalar_application'],'ONCE_AT_TOTAL_T','Scalar once')
    for key in ('physical_request','preparation_input_scalar_fragment_order','normal_seed','signed_event_allocation',
                'same_saved_event_digest','axis_compiler_fingerprint','wrapper_measurement_and_mutation_checks'):
        A.require(gate[key] is True,'D2 recorded identity gate '+key)
    A.equal(gate['status'],'SAME_PATH_SC_IDENTITY_CHECKED','D2 identity status')
    A.require(r['valid'] is True and r['reason'] is None,'D2 valid receipt')
    for obj in (r,original,path,gate):
        A.require(obj['full_wrapper_statevector_validation'] is False,'Keep unperformed statevector validation')
    A.equal(r['numerical_action_gate'],'NOT_RUN_AS_IN_ORIGINAL_P2_FRESH','Keep original action-gate scope')
    return pair


def check_comparison(unit,handoff_unit,prediction,source_rows,admitted):
    """Copy exact reported metrics. Deliberately never average or subtract."""
    A.equal(unit,handoff_unit,'D2 comparison/handoff consistency')
    A.equal((unit['m'],unit['axis'],unit['n_paths']),(4,prediction['axis'],4),'Axis comparison identity')
    A.equal(unit['P'],prediction['predicted_RZ'],'Original frozen exact prediction')
    A.require(unit['valid'] is True and unit['reasons']==[],'D2 comparison validity')
    A.equal(sorted(r['index'] for r in source_rows),[0,1,2,3],'Four original holdout paths')
    A.equal(unit['receipts'],source_rows,'Comparison uses these original receipts')
    A.require(len(admitted)==4 and all(r['status']=='ACCEPTED_SAVED_RECEIPT' for r in admitted),'All four axis receipts accepted')
    return {key:deepcopy(unit[key]) for key in EXACT_KEYS}


def build(root):
    frozen=G.evidence(root,FROZEN);prior=G.evidence(root,PRIOR);facts=G.evidence(root,FACTS)
    h=G.evidence(root,HANDOFF);cmp=G.evidence(root,COMPARISON);di=G.evidence(root,INPUTS)
    A.equal((h['status'],cmp['campaign']),('DESCRIPTIVE_COMPLETE','DESCRIPTIVE_COMPLETE'),'Saved descriptive status')
    for obj in (h,cmp):
        A.equal(obj['model_verdict'],VERDICT,'No adopted model tolerance')
        A.require(obj['tolerance'] is None,'No posthoc tolerance')
    A.equal(h['frozen_prediction_source'],FROZEN,'D2 frozen model SOURCE')
    A.equal(h['comparison_rule'],frozen['comparison_rule'],'Original comparison rule')
    A.equal(di['comparison_rule'],frozen['comparison_rule'],'Acquisition rule')
    A.equal(di['profile'],facts['profile'],'Compiler/control/measured wrapper/P0')
    A.equal(di['scalar_saved'],facts['scalar_saved'],'Scalar/preparation SOURCE')
    A.equal(di['expected_preparation_hash'],facts['preparation_hash'],'Preparation hash')
    for key in ('hamiltonian','state','snapshot','rank','rank_policy','geometry','basis'):
        A.equal(di['menus']['H4']['binding'][key],facts['input_binding'][key],'Input '+key)
    cell=di['cells'][0];future=frozen['holdout_metadata']['cell']
    for key in ('id','candidate_id','cohort','n','path_indices','seeds'):
        A.equal(cell[key],future[key],'Pre-acquisition holdout '+key)
    request=di['menus']['H4']['candidates'][0]
    A.equal(request,frozen['holdout_metadata']['row'],'Original physical/law/signed structure')
    A.equal(len(h['rows']),8,'Expected eight axis receipts')
    A.equal({(r['index'],r['axis']) for r in h['rows']},{(i,a) for i in range(4) for a in ('X','Y')},'Unique 4path x 2axis')
    admissions=[]
    for i,r in enumerate(h['rows']):
        entry=dict(source=at(HANDOFF,'rows',i),original_record=deepcopy(r),status='HELD',reason=None)
        try:
            pair=check_row(r,receipt(root,r['original_receipt']),receipt(root,r['path_result']),receipt(root,r['axis_receipt']),
                           receipt(root,r['structural_receipt']),G.evidence(root,r['gate']),cell,
                           frozen['models'][r['axis']]['m4_prediction'],request,facts['profile'])
            A.equal(r['scalar_hex'],facts['scalar_saved']['hex'],'Original total scalar')
            entry.update(status='ACCEPTED_SAVED_RECEIPT',pair=pair)
        except (KeyError,ValueError,OSError) as error:entry['reason']=str(error)
        admissions.append(entry)
    comparisons=[];candidate_rows=[]
    for axis in ('X','Y'):
        idx=next(i for i,u in enumerate(cmp['units']) if u['axis']==axis)
        unit=cmp['units'][idx];pred=frozen['models'][axis]['m4_prediction']
        rows=[r for r in h['rows'] if r['axis']==axis];accepted=[r for r in admissions if r['original_record']['axis']==axis]
        item=dict(axis=axis,m=4,physical_id=PID,status='HELD',reason=None,
            frozen_prediction=at(FROZEN,'models',axis,'m4_prediction','predicted_RZ'),
            comparison=at(COMPARISON,'units',idx),model_verdict=VERDICT,tolerance=None)
        try:
            item['saved_exact_comparison']=check_comparison(unit,next(u for u in h['units'] if u['axis']==axis),pred,rows,accepted)
            A.equal(unit['prediction_source'],at(FROZEN,'models',axis,'m4_prediction'),'Saved axis prediction locator')
            item['status']='ACCEPTED_SAVED_DESCRIPTIVE_COMPARISON'
        except (KeyError,ValueError) as error:item['reason']=str(error)
        comparisons.append(item)
        old=next(r for r in prior['rows'] if r['m']==4 and r['axis']==axis)
        A.equal(old['physical_id'],PID,'C alignment physical point')
        A.equal(pred['identity'],{k:old[k] for k in ('configuration_id','total_configuration_id','physical_id','original_canonical_id','total_candidate_id')},'C candidate/forecast identity')
        candidate_rows.append(dict(candidate='PR-DF LD1',route='PR_DF',configuration_id=old['configuration_id'],
            total_configuration_id=old['total_configuration_id'],m=4,T=1.6,q=16,R_outer=32,axis=axis,physical_id=PID,
            identity=pred['identity'],alignment_source=old['alignment'],
            frozen_prediction=item['frozen_prediction'],comparison_admission=['comparisons',len(comparisons)-1],
            receipt_admissions=[['receipt_admissions',i] for i,r in enumerate(admissions) if r['original_record']['axis']==axis]))
    n=sum(r['status']=='ACCEPTED_SAVED_RECEIPT' for r in admissions)
    c=sum(r['status']=='ACCEPTED_SAVED_DESCRIPTIVE_COMPARISON' for r in comparisons)
    return dict(schema='PR_DF_LD1_M4_C_CONSUMER_ADMISSION_V1',sources=dict(frozen=FROZEN,handoff=HANDOFF,comparison=COMPARISON,inputs=INPUTS,prior_alignment=PRIOR),
        status='DESCRIPTIVE_COMPLETE' if (n,c)==(8,2) else 'PARTIAL_ADMISSION',model_verdict=VERDICT,tolerance=None,
        receipt_counts=dict(requested=8,accepted=n,held=8-n),comparison_counts=dict(requested=2,accepted=c,held=2-c),
        candidate_rows=candidate_rows,receipt_admissions=admissions,comparisons=comparisons,
        full_wrapper_statevector_validation=False,numerical_action_gate='NOT_RUN_AS_IN_ORIGINAL_P2_FRESH',
        mean_or_error_or_prediction_recomputed=False,refit=False,qualified_E_C=None,CI=None,law_I_Q_distance=None,
        long_round_price=None,N=None,G=None,winner=None,mandatory_stop=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    A.require(not args.out.exists(),'Reuse the existing fixed admission; do not overwrite')
    result=build(ROOT);args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'ADMISSION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(receipts=result['receipt_counts'],comparisons=result['comparison_counts'],status=result['status'],new_scientific_calculations=0)))


if __name__=='__main__':main()
