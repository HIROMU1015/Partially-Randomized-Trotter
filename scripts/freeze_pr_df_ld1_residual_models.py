"""Fixed LD1 m0/m3/m6 paired residual fit, then ONLY two unopened m4 forecasts."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from trottertracks import price_implementation_guard as G
from trottertracks import price_model_validation as V
import admit_pr_df_ld1_q3_prices as A

ROOT=Path(__file__).resolve().parents[1]
BASE='docs/research/pr_df_ld1_residual_freeze_20261115'
ADMISSION=dict(commit='c99d511e2422b8f250490b11054f754a81e536c5',path=A.BASE+'/ADMISSION.json')
FACTS=dict(commit=ADMISSION['commit'],path=A.BASE+'/SOURCE_FACTS.json')
D_DIR='docs/research/d_pr_df_ld1_m3_results_20261115/'
D_HANDOFF=dict(commit='2592d442ae8f36f71e42820a26e847b1fa38b59d',path=D_DIR+'C_HANDOFF.json')
D_INPUTS=dict(commit='2f423befbba5075bad4ede3057f89edd788d2661',path=D_DIR+'INPUTS.json')
FEATURES=['one','q','qR']
FOCUS={0:('53e00d8b07d1547d19a7',1,2),3:('045843b2429bece6f29b',8,16),6:('2a8c6718220eec59d419',64,128)}
M4='3df0cfd3356c358aeda8'


def enc(x):return V._enc(Fraction(x))
def at(ref,*selector):return dict(ref,selector=list(selector))


def raw_receipt(root,ref):
    """Only saved calibration receipts, never a holdout price file."""
    if 'm4' in ref['path'].lower():raise ValueError('M4_OBSERVATION_READ_FORBIDDEN')
    raw=G.C.blob(root,ref['commit'],ref['path'])
    if ref.get('sha256'):A.equal(hashlib.sha256(raw).hexdigest(),ref['sha256'],'Receipt hash')
    return G.evidence(root,ref)


def archived(ref):
    return dict(commit=ADMISSION['commit'],path=ref['archived_path'],sha256=ref['sha256'],original_path=ref['original_path'])


def inputs(root):
    admission=G.evidence(root,ADMISSION);facts=G.evidence(root,FACTS)
    dh=G.evidence(root,D_HANDOFF);di=G.evidence(root,D_INPUTS)
    A.equal(dh['execution_source'],D_INPUTS['commit'],'D2 execution SOURCE')
    A.equal(di['profile'],facts['profile'],'D2 compiler/wrapper/P0')
    A.equal(di['scalar_saved'],facts['scalar_saved'],'D2 scalar metadata')
    A.equal(di['expected_preparation_hash'],facts['preparation_hash'],'D2 preparation')
    A.equal(di['menus']['H4']['binding']['snapshot'],facts['input_binding']['snapshot'],'D2 snapshot')
    for k in ('hamiltonian','state','geometry','basis','rank','rank_policy'):
        A.equal(di['menus']['H4']['binding'][k],facts['input_binding'][k],'D2 '+k)
    records=[]
    for i,r in enumerate(admission['rows']):
        m=r['m'];axis=r['axis']
        if m not in FOCUS:continue
        pid,q,R=FOCUS[m];A.equal((r['physical_id'],r['physical_candidate']['q'],r['physical_candidate']['R_outer']),(pid,q,R),'Calibration physical point')
        if m in (0,6):
            paths=[(k,p) for k,p in enumerate(r['finite_paths']) if p['phase']=='P2' and p['acquisition_phase']=='fresh']
            A.equal([p['index'] for k,p in paths],list(range(4)),'P2 four original fresh paths')
            for k,p in paths:
                A.equal(p['status'],'ACCEPTED_FINITE_PATH_ONLY','Original C admission')
                original=raw_receipt(root,archived(p['original_receipt']));receipt=raw_receipt(root,archived(p['axis_receipt']))
                a=A.AXES[axis][0];A.equal((original['C'][a],original['S'][a]),(p['pair']['C'],p['pair']['S']),'Original same-path S/C')
                A.equal(receipt['fingerprint'],p['pair']['fingerprint'],'P2 axis fingerprint')
                A.equal(receipt['metrics']['rz_count'],p['pair']['C'],'P2 axis RZ')
                A.equal(original['event_digest'],p['pair']['event_digest'],'P2 event digest')
                A.equal(original['cohort'],'D_P2_DIRECT_PILOT_V1','No P1/replay pooling')
                A.equal(original['candidate_id'],pid,'P2 physical ID')
                oldcell=next(c for c in di['calibration_saved_cells'] if c['m']==m)
                A.equal(G.evidence(root,p['seeds']),oldcell['seeds'][p['index']],'P2 frozen original seed reference')
                A.require(original['conditional'] is False and original['full_wrapper_statevector_validation'] is False,'Original normal/path validation status')
                records.append(dict(m=m,axis=axis,q=q,R=R,physical_id=pid,index=p['index'],cohort=p['cohort'],
                    C=p['pair']['C'],S=p['pair']['S'],original_role=p['role'],model_role='CALIBRATION',
                    admission=at(ADMISSION,'rows',i,'finite_paths',k),original_receipt=archived(p['original_receipt']),
                    axis_receipt=archived(p['axis_receipt']),event_digest=p['pair']['event_digest'],
                    seeds=p['seeds'],laws=p['laws'],scientific_source=p['scientific_source'],full_wrapper_statevector_validation=False))
        else:
            cell=di['cells'][0]
            A.equal(cell['cohort'],'D_PR_DF_LD1_M3_CAL_NORMAL_20261115_V1','m3 cohort')
            for k,p in enumerate(dh['rows']):
                if p['axis']!=axis:continue
                A.equal((p['m'],p['physical_id']),(m,pid),'m3 physical identity')
                req=G.evidence(root,p['request']);A.equal(req['candidate'],r['physical_candidate'],'m3 request')
                A.equal(req['structure'],G.evidence(root,r['structural']['source'])['identity']['structure'],'m3 signed structure')
                A.equal(A.law_projection(req['law_binding']),A.law_projection(G.evidence(root,r['structural']['law_source'])),'m3 law')
                path=raw_receipt(root,p['path_result']);original=raw_receipt(root,p['original_receipt'])
                receipt=raw_receipt(root,p['axis_receipt']);sr=raw_receipt(root,p['structural_receipt'])
                checked=A.path_binding(r,path,original,receipt,axis,cell,req['candidate'],D_INPUTS['commit'],facts['profile']['compiler'])
                A.equal((p['C'],p['S']),(checked['C'],checked['S']),'m3 handoff same-path S/C')
                A.equal(p['seeds'],path['seeds'],'m3 seed handoff')
                A.equal((p['event_digest'],p['wrapper_fingerprint']),(checked['event_digest'],checked['fingerprint']),'m3 axis/digest handoff')
                A.equal(sr['S'],path['S'],'m3 structural receipt')
                A.equal(sr['event_digest'],p['event_digest'],'m3 S/C event digest')
                A.equal((p['scalar_hex'],path['scalar_hex']),(facts['scalar_saved']['hex'],facts['scalar_saved']['hex']),'m3 scalar binary64')
                A.equal(path['lowering'],G.NEW_ZZ,'m3 new ZZ')
                A.equal(path['scalar_application'],'ONCE_AT_TOTAL_T','m3 scalar once')
                gate=G.evidence(root,p['gate'])
                A.equal(gate['status'],'SAME_PATH_SC_IDENTITY_CHECKED','D2 saved gate')
                for key in ('physical_request','preparation_input_scalar_fragment_order','normal_seed','signed_event_allocation',
                            'same_saved_event_digest','axis_compiler_fingerprint','wrapper_measurement_and_mutation_checks'):
                    A.require(gate[key] is True,'D2 saved gate '+key)
                A.require(p['full_wrapper_statevector_validation'] is False and original['full_wrapper_statevector_validation'] is False,'m3 validation remains false')
                records.append(dict(m=m,axis=axis,q=q,R=R,physical_id=pid,index=p['index'],cohort=p['cohort'],C=p['C'],S=p['S'],
                    original_role=p['role'],model_role='CALIBRATION',handoff=at(D_HANDOFF,'rows',k),original_receipt=p['original_receipt'],
                    axis_receipt=p['axis_receipt'],structural_receipt=p['structural_receipt'],event_digest=p['event_digest'],
                    seeds=at(D_INPUTS,'cells',0,'seeds',p['index']),laws=p['laws'],scientific_source=p['execution_source'],
                    full_wrapper_statevector_validation=False))
    # Only declared metadata and the existing ideal structural mean are opened for m4.
    A.equal(dh['future_holdout'],di['future_holdout'],'Pre-acquisition m4 plan')
    future=deepcopy(di['future_holdout']);c=future['row']['candidate'];cell=future['cell']
    A.equal((c['id'],c['q'],c['R_outer'],c['T']),(M4,16,32,1.6),'Fixed m4 physical point')
    A.equal((cell['cohort'],cell['n'],cell['path_indices']),('D_PR_DF_LD1_M4_HOLDOUT_NORMAL_20261115_V1',4,[0,1,2,3]),'Fixed m4 cohort')
    A.require(future['prices_opened'] is False and future['execution_authorized'] is False,'Original m4 pre-acquisition state')
    structure={}
    for r in admission['rows']:
        if r['m']!=4:continue
        A.equal(r['physical_candidate'],c,'M4 admitted identity')
        value=G.evidence(root,r['structural']['source'])
        A.equal(A.law_projection(value['law_binding']),A.law_projection(future['row']['law_binding']),'M4 ideal law')
        structure[r['axis']]=dict(source=r['structural']['source'],saved=value,admission_identity=dict(
            configuration_id=r['configuration_id'],total_configuration_id=r['total_configuration_id'],
            physical_id=r['physical_id'],original_canonical_id=r['original_canonical_id'],total_candidate_id=r['total_candidate_id']))
    A.equal(len(records),24,'24 paired calibration axis observations')
    return dict(schema='LD1_RESIDUAL_FIXED_INPUTS_V1',paths=records,features=FEATURES,source_facts=FACTS,
        profile=facts['profile'],scalar=facts['scalar_saved'],preparation_hash=facts['preparation_hash'],
        holdout_metadata=future,holdout_source=at(D_INPUTS,'future_holdout'),structural_m4=structure,
        holdout_rule=deepcopy(di['model_plan']['holdout_rule']),original_plan=dict(commit='8eef0ad7a9b04e04d766f5e05b3b32eee50a794e',path=A.BASE+'/NEXT_VALIDATION_PLAN.md'),
        original_roles_preserved=True,inputs_held=[],science_executed=False)


def preopening(root,*,allow_existing=False):
    """Only ref names, tracked path names, job names and known namespace names."""
    refs=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/*ld1*','refs/heads/codex/m5-m4-receipt-admission-20261114'],cwd=root,text=True).splitlines()
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','docs/research'],cwd=root,text=True).splitlines()
    suspects=[p for p in paths if 'ld1' in p.lower() and any(t in p.lower() for t in ('residual_freeze','residual_fit','m4_predict')) and not p.startswith(BASE+'/')]
    jobs=[]
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,comm=,args='],text=True).splitlines():
        a=line.split(None,2)
        if len(a)!=3 or a[0]==str(os.getpid()) or 'python' not in a[1]:continue
        args=shlex.split(a[2]);names=[Path(v).name for v in args if v.endswith('.py')]
        if any(('ld1' in n.lower() and any(t in n.lower() for t in ('residual','m4','holdout'))) for n in names):jobs.append(dict(pid=int(a[0]),scripts=names))
    root_private=Path('/home/AbeHiromu/projects/d2-pr-df-ld1-m3-20261115-private')
    namespaces=[]
    for directory in (root_private,root_private/'run01',root_private/'run01'/'fresh'):
        names=sorted(p.name for p in directory.iterdir()) if directory.exists() else []
        namespaces.append(dict(directory=str(directory),names=names))
    m4_names=[n for d in namespaces for n in d['names'] if 'm4' in n.lower() or M4 in n]
    m4_refs=[r for r in refs if 'pr-df-ld1' in r.lower() and ('m4' in r.lower() or 'holdout' in r.lower())]
    output=(Path(root)/BASE/'run01/FROZEN_MODELS.json').exists()
    clear=not(suspects or jobs or m4_names or m4_refs or (output and not allow_existing))
    return dict(checked_at_UTC=datetime.now(timezone.utc).isoformat(),status='CLEAR_IN_CHECKED_METADATA' if clear else 'HELD_METADATA_REQUIRES_RESOLUTION',
        public_refs=refs,duplicate_candidates=suspects,relevant_jobs=jobs,known_namespaces=namespaces,m4_namespace_candidates=m4_names,
        m4_ref_candidates=m4_refs,own_output_preexists=output,observation_price_reads=0,
        limitation='Focused published sources/names and known local namespaces only; unknown archives/other hosts not exhaustively checked')


def calculate(data,source):
    paths=data['paths'];models={};aggregates=[];residuals=[]
    A.equal(len(paths),24,'Exactly 24 calibration axis pairs')
    for axis in ('X','Y'):
        rows=[];means=[];X=[]
        for m,(pid,q,R) in FOCUS.items():
            chosen=[(i,p) for i,p in enumerate(paths) if p['axis']==axis and p['m']==m]
            A.equal(sorted(p['index'] for i,p in chosen),[0,1,2,3],'No missing/duplicate paths')
            values=[]
            for i,p in chosen:
                A.equal((p['physical_id'],p['q'],p['R']),(pid,q,R),'Calibration scope')
                A.require(type(p['C']) is int and type(p['S']) is int,'Integer same-path C/S')
                residual=Fraction(p['C']-p['S']);values.append(residual)
                residuals.append(dict(input_selector=['paths',i],m=m,axis=axis,index=p['index'],C=p['C'],S=p['S'],residual=enc(residual)))
            mean=sum(values,Fraction())/4
            aggregates.append(dict(m=m,axis=axis,n_paths=4,q=q,R=R,mean_residual=enc(mean),path_input_selectors=[['paths',i] for i,p in chosen]))
            X.append([Fraction(1),Fraction(q),Fraction(q*R)]);means.append(mean)
        # Existing exact square solver rejects any missing pivot: successful solve proves rank 3.
        coefficients=V._solve(X,means)
        model=dict(model='PAIRED_RESIDUAL_LINEAR_FEATURE_MODEL_V1',axis=axis,features=FEATURES,
            coefficients={f:enc(v) for f,v in zip(FEATURES,coefficients)},rank=3,
            rank_check='EXISTING_EXACT_GAUSS_JORDAN_THREE_NONZERO_PIVOTS',design_matrix=[[enc(v) for v in r] for r in X],
            mean_residuals=[enc(v) for v in means],calibration_rounds=list(FOCUS),calibration_points=3,coefficient_count=3,
            lack_of_fit_DOF=0,status='CALIBRATED_UNVALIDATED_HOLDOUT_UNOPENED',qualified_for_candidate=False)
        model['calibration_arithmetic_residuals']=[enc(V.predict(model,dict(zip(FEATURES,r)))-y) for r,y in zip(X,means)]
        A.require(all(v['numerator']==0 for v in model['calibration_arithmetic_residuals']),'Exact calibration arithmetic check')
        structural=data['structural_m4'][axis];saved=structural['saved'];s=Fraction.from_float(saved['mean_S'])
        residual=V.predict(model,dict(one=1,q=16,qR=512));predicted=s+residual
        model['m4_prediction']=dict(m=4,T=1.6,q=16,R_outer=32,axis=axis,physical_id=M4,
            identity=structural['admission_identity'],ideal_structural_source=structural['source'],
            ideal_structural=dict(quantity=saved['quantity'],expectation=saved['expectation'],saved_value=saved['mean_S'],
                binary64_hex=saved['mean_S'].hex(),exact_binary64_value=enc(s),saved_numerical_enclosure=saved['numerical_enclosure'],
                law_binding=saved['law_binding'],kernel_source=saved['kernel_source'],source_probability_factorization_relative_bound=saved['source_probability_factorization_relative_bound']),
            residual_prediction=enc(residual),predicted_RZ=enc(predicted),display_RZ=float(predicted),
            semantics='MIXED_IDEAL_STRUCTURAL_PLUS_SAMPLER_FINITE_COHORT_RESIDUAL_MODEL_NOT_QUALIFIED_E_C',
            observed_RZ=None,errors=None,law_distance=None,finite_cohort_uncertainty=None,CI=None,model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE')
        models[axis]=model
    return dict(schema='LD1_TWO_RESIDUAL_MODELS_M4_FORECASTS_V1',execution_source=source,
        input_source=dict(commit=source,path=BASE+'/INPUTS.json'),status='TWO_MODELS_AND_TWO_M4_PREDICTIONS_FROZEN_HOLDOUT_UNOPENED',
        paths=residuals,cell_means=aggregates,models=models,holdout_metadata=data['holdout_metadata'],holdout_source=data['holdout_source'],
        comparison_rule=data['holdout_rule'],fit_calls=2,m4_prediction_calls=2,calibration_arithmetic_checks=6,m4_observation_reads=0,
        inputs_held=[],qualified_E_C=None,coefficient_uncertainty=None,law_I_Q_distance=None,rare_contribution=None,shared_covariance=None,
        CI=None,G=None,winner=None,mandatory_stop=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();directory=ROOT/BASE
    if args.prepare:
        A.require(not directory.exists(),'Existing prepared scope must be reused')
        checked=preopening(ROOT);A.equal(checked['status'],'CLEAR_IN_CHECKED_METADATA','Resolve duplication/m4 metadata before proceeding')
        data=inputs(ROOT)
        directory.mkdir(parents=True,exist_ok=False)
        (directory/'INPUTS.json').write_text(json.dumps(data,indent=2)+'\n')
        (directory/'PREOPEN_CHECK.json').write_text(json.dumps(checked,indent=2)+'\n')
        print('24 same-path axis inputs accepted; only m4 metadata/old ideal structural values read; no fit or prediction',flush=True);return
    if not args.execute:raise SystemExit('Choose --prepare or --execute; no automatic science')
    checked=preopening(ROOT);A.equal(checked['status'],'CLEAR_IN_CHECKED_METADATA','Unresolved duplicate/acquisition metadata')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for path in ('scripts/freeze_pr_df_ld1_residual_models.py',BASE+'/INPUTS.json',V.MODULE):
        A.equal((ROOT/path).read_bytes(),G.C.blob(ROOT,source,path),'Fixed execution SOURCE/input '+path)
    out=directory/'run01';out.mkdir(exist_ok=False)
    start=time.perf_counter();print('BEGIN exact paired residual aggregation / two-axis fit / two m4 forecasts',flush=True)
    try:
        data=calculate(json.loads((directory/'INPUTS.json').read_text()),source)
    except Exception as error:
        (out/'STOP.json').write_text(json.dumps(dict(status='STOP_NO_RETRY',source=source,error=repr(error),
            input_path=BASE+'/INPUTS.json',m4_observation_reads=0,mandatory_stop=True),indent=2)+'\n')
        raise
    data['preopening']=checked
    data['wall_seconds']=time.perf_counter()-start
    (out/'FROZEN_MODELS.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(fit_calls=2,m4_predictions=2,rank_X=data['models']['X']['rank'],rank_Y=data['models']['Y']['rank'],held=0,m4_observation_reads=0,wall_seconds=data['wall_seconds'],mandatory_stop=True)),flush=True)


if __name__=='__main__':main()
