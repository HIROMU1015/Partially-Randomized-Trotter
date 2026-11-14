"""Fixed 12-group Stage 1 calibration and m6 prediction; no holdout reader.

Reuse C's exact price_model_validation.fit/predict and D's published accepted
calibration records. No circuit/sampling/risk/price-acquisition entry is loaded.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from trottertracks import price_model_validation as V
from trottertracks import price_implementation_guard as G
_reader=importlib.util.spec_from_file_location('_stage1_existing_receipt_reader',
    Path(__file__).resolve().with_name('check_m5_m4_receipt_admission.py'))
M=importlib.util.module_from_spec(_reader);_reader.loader.exec_module(M)

ROOT=Path(__file__).resolve().parents[1]
BASE='docs/research/pf_stage1_model_freeze_20261115'
INDEX=dict(commit='b190bd815c886446291fac8b281c3c7df7343372',
           path='docs/research/d_pf_stage1_calibration_20261115/C_HANDOFF_INDEX.json')
POLICY_SOURCE=dict(commit='91365dcf0610362a96aab101fbe0ec0e78f58af1',path='scripts/pf_price_stages.py')
CELL_SOURCE=dict(commit='43c15bfa34a22f313e55b824223b200332b7f23f',
                 path='docs/research/truncated_p7_admission_pr_df_evidence_20261114/CELL_PLAN_V2.json')
FEATURES=['one','q']
EXPECTED_Q={'2nd':[7,14,28],'4th':[3,6,12],'4th(m5_best)':[1,2,4]}


def read(root, ref):
    # Fixed calibration roots, saved receipts, policy and identity-only CELL_PLAN.
    if '/holdout/' in ref['path'].lower():
        raise ValueError('Holdout observations are outside this C execution')
    result=G.evidence(root,ref)
    if 'axis' in ref and isinstance(result,list):
        result=next(x for x in result if x.get('axis')==ref['axis'])
    return result


def at(ref,*keys):
    return dict(ref,selector=list(ref.get('selector',[]))+list(keys))


def policy(root):
    tree=ast.parse(M.C.blob(root,POLICY_SOURCE['commit'],POLICY_SOURCE['path']))
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='COMPARISON_POLICY' for t in n.targets))
    scope={};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'pinned comparison policy','exec'),scope)
    return scope['COMPARISON_POLICY']


def check_records(records,group):
    if (len(records)!=2 or any(r['role']!='CALIBRATION' or r['kind']!='DIRECT_COMPILED_DETERMINISTIC'
            or r['route']!=group[0] or r['axis']!=group[3] for r in records)
            or [r['cell_id'] for r in records]!=['%s:%s:%s:m%d'%(*group[:3],m) for m in (4,5)]
            or [V._exact(r['features']['q'],'q') for r in records]!=EXPECTED_Q[group[1]][:2]
            or any(V._exact(r['features']['one'],'one')!=1 for r in records)):
        raise ValueError('Exactly this group/axis m4,m5 one/q calibration required')
    V._domain(records)


def guard_rows(guard):
    if 'checks' in guard:
        return [x for c in guard['checks'] for x in c['guard']['checked']]
    return guard['guard']['checked']


def prepare(root):
    """Input/identity/accepted-guard checks only, no fit or prediction."""
    root=str(root);index=read(root,INDEX);p=policy(root)
    G._same(index['comparison_policy'],p,'D index/pinned comparison policy differs')
    if index['counts']['calibration_prices']!=24 or index['counts']['model_groups']!=12:
        raise ValueError('Wrong fixed Stage 1 input scope')
    cells=read(root,CELL_SOURCE)
    t,f=M.d_callers(root)
    templates={x['configuration_id']:x for x in f.plans(root)+t.plans(root)}
    groups={};blocked={}
    for key,entry in sorted(index['groups'].items()):
        try:
            group=entry['model_group'];g=read(root,entry['C_inputs_source'])
            if key!='|'.join(group) or g['model_group']!=group or g['stage']!='CALIBRATION' or g['features']!=FEATURES:
                raise ValueError('Group identity/stage/features differ')
            records=sorted([s['record'] for s in g['saved_records']]+g['new_records'],key=lambda r:r['cell_id'])
            check_records(records,group)
            metadata=[]
            for i,record in enumerate(records):
                m=(4,5)[i];e=next(e for e in entry['entries'] if e['m']==m)
                locator=dict(entry['C_inputs_source'],selector=e['C_inputs_selector'])
                G._same(read(root,locator),record,'C_INPUTS and index selector differ')
                G._same(e['record'],record,'D index and authoritative C_INPUTS differ')
                saved=e['origin']=='C_ADMITTED_SAVED_RECEIPT_NOT_RECOMPILED'
                if saved:
                    refs=e['references'];price_ref=refs['PRICES.json'];side_ref=refs['SIDECARS.json'];guard_ref=refs['GUARD_CHECK.json']
                    G._same(read(root,refs['SAVED_CALIBRATION_INPUTS.json']),record,'Saved original calibration record differs')
                else:
                    price_ref=e['price_reference'];side_ref=entry['bound_sidecars'];guard_ref=entry['guard']
                price=read(root,price_ref);sidecars=read(root,side_ref)
                # Axis-only saved locator returns one sidecar; stage locator returns its list.
                sides=[sidecars] if isinstance(sidecars,dict) else sidecars
                side=next(s for s in sides if s['axis']==group[3]
                    and s['identity']['candidate']['id']==price['identity']['candidate']['id']
                    and s['cohort']==price['cohort'] and M.C.p.same(s['price_source'],price['source']))
                G._same(side['identity'],price['identity'],'Sidecar canonical identity differs')
                G._same(side['law_binding'],price['law_binding'],'Sidecar law differs')
                G._same(side['physical_context'],price['physical_context'],'Sidecar physical context differs')
                G._same(record['source'],price['source'],'Calibration price SOURCE differs')
                G._same(record['profile'],price['price_profile'],'Calibration compiler profile differs')
                if V._exact(record['rz_count'],'record RZ')!=V._exact(price['mean']['rz_count'],'price RZ'):
                    raise ValueError('Calibration observed RZ differs')
                if (price['axis']!=group[3] or price['m']!=m or price['expectation']!='DETERMINISTIC'
                        or price['classification']!=record['kind']):
                    raise ValueError('Price axis/round/expectation differs')
                if saved and (m!=4 or group[1]!='4th(m5_best)' or price['cohort']!='EXPLORATION'):
                    raise ValueError('Only original EXPLORATION M5/m4 saved receipts allowed')
                ci=next(j for j,c in enumerate(cells) if c['model_group']==group and c['m']==m)
                cell=cells[ci];template=templates[cell['configuration_id']]
                physical=price['physical_context']['physical_candidate']
                G._same(cell['physical_candidate'],physical,'CELL_PLAN physical fields differ')
                G._same(M.PN.I.price_context(template['plan'],m),price['physical_context'],'Full physical SOURCE/split/scalar/profile differs')
                facts=side['implementation'];profile=price['price_profile']
                if (profile['qiskit_version']!='1.3.0' or profile['basis_gates']!=['rz','sx','x','cx']
                        or profile['optimization_level']!=1 or profile['seed_transpiler']!=17
                        or profile['P']!=0 or profile['control']!=p['control'] or profile['wrapper']!=p['measurement']
                        or price['quantity']!=p['quantity'] or facts['scalar_application']!=p['scalar_application']
                        or facts['scalar_quantity']!=p['scalar_by_route'][group[0]]
                        or facts['rte_rotation_lowering']!=p['rte_rotation_lowering']):
                    raise ValueError('Fixed measured compiled-RZ/profile/scalar contract differs')
                G._same(facts['resolved_scalar'],facts['expected_scalar'],'Resolved total scalar differs')
                guard=read(root,guard_ref)
                if not saved:
                    G._same(guard['comparison_policy'],p,'Accepted D guard comparison policy differs')
                    if guard['stage']!='CALIBRATION' or guard['route']!=group[0] or not guard['comparison_policy_adopted']:
                        raise ValueError('D accepted-guard stage/route/policy differs')
                if not any(x['m']==m and x['axis']==group[3] and x['evidence']==side['implementation_evidence']
                           for x in guard_rows(guard)):
                    raise ValueError('Accepted guard does not cover this axis/receipt')
                metadata.append(dict(m=m,origin=e['origin'],record_source=locator,price_source=price_ref,
                    sidecar_source=side_ref,guard_source=guard_ref,configuration_id=cell['configuration_id'],
                    physical_candidate=physical,q=physical['q'],actual_delta=physical['delta'],
                    observed_RZ=record['rz_count'],cohort=price['cohort'],original_price_source=price['source'],
                    implementation_evidence=side['implementation_evidence'],
                    scalar=dict(value=facts['resolved_scalar'],quantity=facts['scalar_quantity'],
                                application=facts['scalar_application'],source=facts['scalar_source']),
                    cell_plan_source=at(CELL_SOURCE,ci),
                    input_binding={k:price['physical_context']['binding'][k] for k in
                                   ('hamiltonian','state','snapshot','basis','rank','geometry','rank_policy')},
                    fragment_order=price['physical_context']['binding']['decomposition']['contract']['fragment_order'],
                    split=price['physical_context']['split'],profile=profile))
            hi=next(j for j,c in enumerate(cells) if c['model_group']==group and c['m']==6)
            hold=cells[hi]
            if hold['role']!='HOLDOUT' or hold['physical_candidate']['q']!=EXPECTED_Q[group[1]][2]:
                raise ValueError('Wrong m6 holdout identity/q')
            groups[key]=dict(model_group=group,records=records,inputs=metadata,
                C_inputs_source=entry['C_inputs_source'],holdout_cell=deepcopy(hold),holdout_cell_source=at(CELL_SOURCE,hi))
        except (ValueError,KeyError,StopIteration) as exc:
            blocked[key]=dict(status='UNRESOLVED_INPUT_GROUP_NOT_FITTED',reason=type(exc).__name__+': '+str(exc))
    return dict(index=INDEX,comparison_policy=p,groups=groups,blocked=blocked)


def calibrate(prepared,source):
    """Each allowed group exactly once; predict only its identity-only m6 cell."""
    out={};fit_calls=0;prediction_calls=0
    for key,g in sorted(prepared['groups'].items()):
        check_records(g['records'],g['model_group'])
        model=V.fit(g['records'],features=FEATURES,design_source=dict(
            source=source,plan=POLICY_SOURCE,comparison_policy=prepared['comparison_policy'],
            C_inputs_source=g['C_inputs_source'],model_group=g['model_group']))
        fit_calls+=1;h=g['holdout_cell']
        predicted=V.predict(model,h['features']);prediction_calls+=1
        model.update(model_group_id=key,model_group=g['model_group'],formula='C(q)=beta_one+beta_q*q',
            input_locators=g['inputs'],m6_prediction=dict(cell_source=g['holdout_cell_source'],
                configuration_id=h['configuration_id'],physical_candidate=h['physical_candidate'],
                m=6,axis=h['axis'],q=h['physical_candidate']['q'],features=h['features'],
                predicted_RZ=V._enc(predicted),display_RZ=str(predicted),observed_RZ=None,
                independent_holdout_status='NOT_EVALUATED',holdout_pass=None),
            coefficient_uncertainty=None,compiler_systematic=None,shared_covariance=None)
        out[key]=model
    return dict(schema='DETERMINISTIC_PF_STAGE1_FROZEN_MODELS_V1',source=source,
        status='CALIBRATED_COEFFICIENTS_AND_M6_PREDICTIONS_FROZEN_HOLDOUT_UNEVALUATED' if len(out)==12
               else 'PARTIAL_FREEZE_UNRESOLVED_INPUT_GROUPS',index_source=INDEX,
        comparison_policy=prepared['comparison_policy'],models=out,blocked=prepared['blocked'],
        fit_calls=fit_calls,m6_prediction_calls=prediction_calls,holdout_observation_reads=0,
        calibration_points=sum(len(m['calibration']) for m in out.values()),
        independent_holdout_status='NOT_EVALUATED',campaign_pass=None,long_q_extrapolation_qualified=False,
        N=None,G=None,winner=None,mandatory_stop=True)


def d_consumer(root,frozen,groups):
    """Run only D's fixed frozen-model metadata reader, not its stage runner."""
    scope=dict(json=json,deepcopy=deepcopy,hashlib=hashlib,FEATURES=FEATURES,
        MODEL='PRICE_LINEAR_FEATURE_MODEL_V1',_T3=lambda:SimpleNamespace(c_model=lambda:V),
        read_pinned=lambda root,ref:M.C.blob(root,ref['commit'],ref['path']))
    M.functions(root,POLICY_SOURCE['commit'],POLICY_SOURCE['path'],('_canonical','frozen_models'),scope)
    return scope['frozen_models'](root,frozen,{k:dict(model_group=g['model_group'],stage='CALIBRATION',
        saved_records=[dict(record=r) for r in g['records'] if ':m4' in r['cell_id'] and g['model_group'][1]=='4th(m5_best)'],
        new_records=[r for r in g['records'] if not (':m4' in r['cell_id'] and g['model_group'][1]=='4th(m5_best)')])
        for k,g in groups.items()})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    args=ap.parse_args();prepared=prepare(ROOT)
    if not args.execute:
        print(json.dumps(dict(metadata_checked_groups=len(prepared['groups']),blocked=prepared['blocked'],fit_calls=0)));return
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in (V.MODULE,'scripts/freeze_pf_stage1_models.py'):
        if M.C.blob(ROOT,source,path)!=(ROOT/path).read_bytes():raise ValueError('Execution code differs from committed SOURCE')
    out=ROOT/BASE/'run01';out.mkdir(parents=True,exist_ok=False)
    result=calibrate(prepared,source)
    (out/'FROZEN_MODELS.json').write_text(json.dumps(result,indent=2)+'\n')
    checks=dict(checked_groups=len(prepared['groups']),blocked=prepared['blocked'],
        saved_receipts=sum(i['origin']=='C_ADMITTED_SAVED_RECEIPT_NOT_RECOMPILED' for g in prepared['groups'].values() for i in g['inputs']),
        new_accepted_receipts=sum(i['origin']=='NEW_STAGE1_DIRECT_COMPILE' for g in prepared['groups'].values() for i in g['inputs']),
        execution_SOURCE=source,finished_UTC=datetime.now(timezone.utc).isoformat(),
        exact_method='EXACT_SQUARE_SOLVE_FRACTION',new_compile_calls=0,holdout_observation_reads=0,
        fit_calls=result['fit_calls'],m6_prediction_calls=result['m6_prediction_calls'],mandatory_stop=True)
    (out/'INPUT_AND_EXECUTION_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks))


if __name__=='__main__':main()
