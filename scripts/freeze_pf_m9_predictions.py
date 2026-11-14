"""Twelve exact m9 predictions from original fixed coefficients; never fit/read observations."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from trottertracks import baseline_signal_pipeline as C
from trottertracks import price_implementation_guard as G
from trottertracks import price_model_validation as V

ROOT=Path(__file__).resolve().parents[1]
BASE='docs/research/pf_m9_prediction_freeze_20261115'
MODELS=dict(commit='12372fa6b53c878dc5dd124e036572ba0cd82061',path='docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json')
ALIGN=dict(commit='0b4b2d941cd669aaa45231e1236bc2d3158a9d12',path='docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json')
D_PLAN=dict(commit='bcb67b2b2bb78776db1dd8d0625a3facb37cd298',path='docs/research/d_pf_allround_price_coverage_plan_20261115/NEXT_VALIDATION_PLAN.md')
POINTS=[
    (36,'FULL_PF','2nd','16/7',224,'a4ec53ff4e60d60a2930'),
    (38,'TRUNCATED_PF','2nd','16/7',224,'1bd0a5dd4bbf876da1d6'),
    (88,'FULL_PF','4th','16/3',96,'b2c479e7805c4d1f0b30'),
    (90,'TRUNCATED_PF','4th','16/3',96,'9924c226ee246283ec54'),
    (140,'FULL_PF','4th(m5_best)','16',32,'a53547a5fb707b9550d5'),
    (142,'TRUNCATED_PF','4th(m5_best)','16',32,'60111109e0024d1a1064'),
]
D_PRIVATE_ROOTS=[Path('/home/AbeHiromu/projects/d-full-pf-shortlist-20261114-private'),
                 Path('/home/AbeHiromu/projects/d-truncated-p7-t3-20261114-private')]
POLICY=dict(id='DETERMINISTIC_PF_M9_FROZEN_OBSERVED_1_OVER_50_V1',status='ADOPTED_FOR_THIS_FIXED_M9_SCOPE_ONLY',
    quantity='FULL_WRAPPER_MEAN_COMPILED_PRICE',metric='compiled_rz_count',
    kind='RELATIVE',scale='observed',value=dict(numerator=1,denominator=50),
    formula='abs(frozen_predicted_RZ-observed_RZ)/observed_RZ',rounds=[9],axes=['X','Y'],
    physical_ids=[p[-1] for p in POINTS],required_individual_results=12,
    comparison='EXACT_UNROUNDED_LESS_THAN_OR_EQUAL',
    PASS='VALID_OBSERVATION_AND_RELATIVE_ERROR_LE_1_OVER_50',
    FAIL='VALID_OBSERVATION_AND_RELATIVE_ERROR_GT_1_OVER_50',
    UNKNOWN='ACQUISITION_OR_SOURCE_OR_IMPLEMENTATION_GATE_NOT_ESTABLISHED',
    required_prerequisites=['SAME_PHYSICAL_ID_AND_AXIS_Q_ACTUAL_WIDTH_SCALAR_SPLIT',
        'SAME_FULL_MEASURED_CONTROLLED_WRAPPER_P0_COMPILER_AND_LOWERING',
        'ORIGINAL_D_IMPLEMENTATION_ACTION_GATES_AND_TOLERANCES_UNCHANGED'],
    campaign_success='ALL_12_INDIVIDUAL_PASS',zero_observed='UNKNOWN_RELATIVE_DENOMINATOR_UNDEFINED',
    automatic_refit=False,automatic_feature_or_threshold_change=False,
    other_round_qualification=False,confidence_statement='NOT_A_STATISTICAL_95_PERCENT_GUARANTEE')


def at(ref,*selector):return dict(ref,selector=list(selector))


def targets(root):
    """Metadata only. Neither m3 prices nor m6 observations are loaded."""
    models=G.evidence(root,MODELS)['models'];alignment=G.evidence(root,ALIGN)['rows']
    plan=C.blob(root,D_PLAN['commit'],D_PLAN['path']).decode();out=[]
    for x,route,formula,width,q,pid in POINTS:
        planrow=next(line for line in plan.splitlines() if '| '+pid+' |' in line)
        if '| '+str(q)+' |' not in planrow or str(x)+'/'+str(x+1) not in planrow:
            raise ValueError('D fixed plan physical point/q/row selector differs')
        for offset,axis in enumerate(('X','Y')):
            row=alignment[x+offset];physical=row['physical_candidate'];group=[route,formula,width,axis];key='|'.join(group)
            model=models[key]
            if (row['m']!=9 or row['axis']!=axis or row['physical_id']!=pid or physical['id']!=pid
                    or physical['q']!=q or physical['formula']!=formula or physical['T']!=51.2
                    or physical['method']!=('FULL' if route=='FULL_PF' else 'B0I')
                    or physical['prefix']!=(10 if route=='FULL_PF' else 7)):
                raise ValueError('Fixed m9 candidate identity differs')
            if (model['model_group_id']!=key or model['features']!=['one','q']
                    or model['domain']['axis']!=axis or model['domain']['route']!=route
                    or model['input_locators'][0]['configuration_id']!=row['configuration_id']
                    or model['status']!='CALIBRATED_UNVALIDATED'
                    or float(model['input_locators'][0]['actual_delta']).hex()!=float(physical['delta']).hex()):
                raise ValueError('Original frozen model group/domain/actual width differs')
            out.append(dict(model_group_id=key,model_group=group,configuration_id=row['configuration_id'],
                canonical_id=row['original_canonical_id'],m=9,T=51.2,physical_candidate=deepcopy(physical),
                physical_id=pid,axis=axis,q=q,actual_delta=physical['delta'],
                actual_delta_binary64_hex=float(physical['delta']).hex(),
                features=dict(one=1,q=q),alignment_source=at(ALIGN,'rows',x+offset),
                coefficient_source=at(MODELS,'models',key,'coefficients'),coefficients=deepcopy(model['coefficients']),
                model_source=at(MODELS,'models',key),domain=deepcopy(model['domain']),
                scalar=deepcopy(model['input_locators'][0]['scalar']),
                input_binding=deepcopy(model['input_locators'][0]['input_binding']),
                split=deepcopy(model['input_locators'][0]['split']),
                fragment_order=model['input_locators'][0]['fragment_order'],
                signed_structure=deepcopy(row['structure'])))
    return out


def preflight(root,*,after_prediction=False):
    """Read public ref/file/job NAMES and existence only; never observation contents."""
    root=Path(root);ids={p[-1] for p in POINTS};reasons=[]
    refs=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/m5-m4-receipt-admission-20261114',
        'refs/heads/codex/d-price-records-c-bridge-20261114','refs/heads/codex/*m9*'],cwd=root,text=True).splitlines()
    if any(re.search(r'refs/heads/codex/[^\s]*m9',r,re.I) for r in refs):
        reasons.append('M9_PUBLIC_REF_METADATA_NEEDS_DUPLICATE_OR_ACQUISITION_RESOLUTION')
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','docs/research'],cwd=root,text=True).splitlines()
    other_predictions=[p for p in paths if re.search(r'(m9|longq.*predict|predict.*longq)',p,re.I)
        and not p.startswith((BASE+'/','docs/research/h4_structural_'))]
    if other_predictions:reasons.append('EXISTING_M9_RELATED_RESULT_METADATA_NEEDS_IDENTITY_RESOLUTION')
    jobs=[]
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,comm=,args='],text=True).splitlines():
        c=line.strip().split(None,2)
        if len(c)<3 or c[0]==str(os.getpid()) or 'python' not in c[1]:continue
        args=shlex.split(c[2]);scripts=[s for s in args[1:] if s.endswith('.py')]
        if any(re.search(r'(m9|run_full_pf_shortlist_prices|run_truncated_p7_prices|pf.*(?:longq|prediction))',Path(s).name,re.I) for s in scripts):
            jobs.append(dict(pid=int(c[0]),scripts=scripts))
    if jobs:reasons.append('RELEVANT_RUNNING_JOB_REQUIRES_TARGET_METADATA_RESOLUTION')
    namespaces=[]
    for base in D_PRIVATE_ROOTS:
        names=sorted(p.name for p in base.iterdir()) if base.exists() else []
        target_names=[n for n in names if re.search(r'(m9|long.?q)',n,re.I) or any(pid in n for pid in ids)]
        namespaces.append(dict(root=str(base),exists=base.exists(),top_level_names=names,target_names=target_names))
        if target_names:reasons.append('TARGET_ACQUISITION_NAMESPACE_EXISTS')
    old=Path('/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/cost')
    target_results=[dict(physical_id=pid,path=str(old/co/pid/'0/result.json'),exists=(old/co/pid/'0/result.json').exists())
        for pid in sorted(ids) for co in ('EXPLORATION','CONFIRMATION')]
    held_ids={r['physical_id'] for r in target_results if r['exists']}
    if reasons:held_ids=ids
    if not after_prediction and (root/BASE/'run01/PREDICTIONS.json').exists():
        reasons.append('CURRENT_PREDICTION_RESULT_ALREADY_EXISTS_REUSE_INSTEAD_OF_RECALCULATING');held_ids=ids
    return dict(checked_at_UTC=datetime.now(timezone.utc).isoformat(),public_refs=refs,
        other_prediction_path_candidates=other_predictions,relevant_same_user_jobs=jobs,
        acquisition_namespaces=namespaces,target_result_file_existence=target_results,
        held_physical_ids=sorted(held_ids),reasons=reasons,observed_price_files_opened=0,
        status='CLEAR_IN_CHECKED_SOURCE_AND_SERVER_METADATA' if not held_ids else 'HELD_INDEPENDENCE_NOT_ESTABLISHED',
        scope='C/D published refs; C tracked result names excluding unrelated PR structural m9; same-user relevant job names; two known D PF namespaces and exact legacy target result paths',
        limitation='Unlisted names/archives/other servers are not claimed exhaustively checked; no observation content or private worktree changes')


def predict(rows,models,opening_check,source):
    expected={(p[-1],a,p[4]) for p in POINTS for a in ('X','Y')}
    if (len(rows)!=12 or {(r['physical_id'],r['axis'],r['q']) for r in rows}!=expected
            or any(r['m']!=9 or r['T']!=51.2 for r in rows)):
        raise ValueError('Only the fixed m9 twelve-point prediction scope is permitted')
    held=set(opening_check['held_physical_ids']);out=[];calls=0
    for row in rows:
        entry=deepcopy(row);entry.update(predicted_RZ=None,observed_RZ=None,decision=None,
            qualified_for_candidate=False,status='HELD_INDEPENDENCE_NOT_ESTABLISHED')
        if row['physical_id'] not in held:
            value=V.predict(models[row['model_group_id']],row['features']);calls+=1
            entry.update(predicted_RZ=V._enc(value),status='FROZEN_PREDICTION_READY_FOR_D_NOT_VALIDATED')
        out.append(entry)
    return dict(schema='PF_M9_FROZEN_PREDICTIONS_V1',source=source,model_source=MODELS,physical_plan=D_PLAN,
        policy=deepcopy(POLICY),rows=out,accepted_predictions=calls,held_predictions=12-calls,
        prediction_calls=calls,fit_calls=0,observation_reads=0,holdout_judgements=0,
        preopening=opening_check,campaign_pass=None,other_rounds_qualified=False,N=None,G=None,winner=None,
        unchanged_roles=dict(m3='SAVED_DIRECT_POINT_NOT_CALIBRATION_OR_HOLDOUT',m6='EXISTING_INDEPENDENT_HOLDOUT'),mandatory_stop=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    rows=targets(ROOT)
    if not args.execute:print(json.dumps(dict(targets=len(rows),prediction_calls=0,policy=POLICY['id'])));return
    check=preflight(ROOT);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if C.blob(ROOT,source,'scripts/freeze_pf_m9_predictions.py')!=Path(__file__).read_bytes():raise ValueError('Runner SOURCE not committed')
    if C.blob(ROOT,MODELS['commit'],V.MODULE)!=(ROOT/V.MODULE).read_bytes():raise ValueError('Original exact prediction implementation changed')
    out=ROOT/BASE/'run01';out.mkdir(parents=True,exist_ok=False)
    data=predict(rows,G.evidence(ROOT,MODELS)['models'],check,source)
    (out/'PREDICTIONS.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(source=source,accepted=data['accepted_predictions'],held=data['held_predictions'],prediction_calls=data['prediction_calls'],observation_reads=0,mandatory_stop=True)))


if __name__=='__main__':main()
