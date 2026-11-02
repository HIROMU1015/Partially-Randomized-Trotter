"""Read-only/static preparation audit; no registered Q/T/U or native work."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_et_resumption.launch import ART,RESULT,verify_local
from trottertracks.algorithm_codesign.df_correction_et_resumption.saved_data import load_saved_inputs,saved_cd_receipt

BASE='ef0b2419df2d25187ad14e865ae0a38d17a04628'
APPEND_ONLY={'PROJECT_MAP.md','docs/research/研究概要・現状.md','scripts/README.md','src/trotterlib/README.md','docs/README.md','docs/tracks/algorithm_codesign/README.md'}


def identity(path):
    path=Path(path);raw=path.read_bytes()
    return {'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def audit():
    contract=json.loads((ROOT/ART/'contract_v4.json').read_text())
    old=json.loads((ROOT/'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json').read_text())
    checks=[]
    def check(name,ok):
        checks.append({'name':name,'PASS':bool(ok)})
        if not ok:raise AssertionError(name)
    for k in ('arms','CD','T','caps','runtimes','numerical_certification'):
        check('unchanged '+k,contract[k]==old[k])
    for k,v in old['E'].items():
        if k!='A_native':check('unchanged E '+k,contract['E'][k]==v)
    check('E/T stages only',contract['resumption']['run_stages']==['E_NATIVE','T_KEYS'])
    check('review source validation delegation',contract['source_review_policy']['routine_final_source_validation'].startswith('DELEGATED_TO_CODEX'))
    check('new result namespace absent',not (ROOT/RESULT).exists())
    auth=json.loads((ROOT/ART/'authorization.json').read_text())
    check('authorization pending',auth['science_execution_authorized'] is False and auth['runs']==0 and auth['source_commit'] is None and auth['retries']==0 and auth['mandatory_STOP'] is True)
    try:verify_local(ROOT)
    except PermissionError:check('pending launch rejects',True)
    else:check('pending launch rejects',False)
    original=json.loads((ROOT/'artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/v2/source_manifest_v1.json').read_text())
    inherited=[];append=[]
    for row in original['files']:
        p=ROOT/row['path']
        if row['path'] in APPEND_ONLY:
            raw=subprocess.check_output(['git','show',BASE+':'+row['path']],cwd=ROOT)
            check('append-only inherited index '+row['path'],p.read_bytes().startswith(raw))
            append.append(row['path'])
        else:
            check('inherited protected '+row['path'],identity(p)==row)
            inherited.append(row)
    old_evidence=[]
    for namespace in ('artifacts/track_b_df_correction_comparison_result/2026-10-11/v2','artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2','artifacts/track_b_e_r3_identity_diagnostic_execution_audit/2026-10-12/v2'):
        for p in sorted((ROOT/namespace).rglob('*')):
            if p.is_file():
                raw=subprocess.check_output(['git','show',BASE+':'+str(p.relative_to(ROOT))],cwd=ROOT)
                check('old evidence '+str(p.relative_to(ROOT)),p.read_bytes()==raw)
                old_evidence.append(identity(p))
    frames=load_saved_inputs(ROOT,contract)
    check('six saved frame identities',set(frames)=={'3','6'} and all(len(f)==3 for f in frames.values()))
    receipt=saved_cd_receipt(ROOT,contract)
    check('CD six-row saved substage without recomputation',receipt['saved_rows']==6 and receipt['CD_recomputed'] is False and receipt['old_run_reclassified'] is False)
    paths=list((ROOT/'src/trottertracks/algorithm_codesign/df_correction_et_resumption').glob('*.py'))+[ROOT/'scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v4.py']
    for p in paths:
        tree=ast.parse(p.read_text())
        calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        check('no recipe/CD/profile entry '+p.name,not calls & {'cd_recipe','e_recipe','cd_task','random_unitary','default_rng','setprofile'})
    new_eval=ast.parse((ROOT/'src/trottertracks/algorithm_codesign/df_correction_et_resumption/native_evaluation.py').read_text())
    old_eval=ast.parse((ROOT/'src/trottertracks/algorithm_codesign/df_correction_comparison/native_evaluation.py').read_text())
    for name in ('serialize_frames','deserialize_frames','operator_numeric_guard'):
        a=next(n for n in new_eval.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in old_eval.body if isinstance(n,ast.FunctionDef) and n.name==name)
        check('numeric helper AST unchanged '+name,ast.dump(a)==ast.dump(b))
    return {'classification':'READY_FOR_SEPARATE_E_T_ONE_SHOT_AUTHORIZATION','checks':checks,'PASS':len(checks),'FAIL':0,
            'inherited_scientific_and_source_files_unmodified':len(inherited),'append_only_indexes':append,
            'old_evidence_files_unmodified':len(old_evidence),'inherited_files':inherited,'old_evidence':old_evidence,
            'CD_saved_receipt':receipt,'registered_E_T_execution':0,'registered_CD_recalculation':0,
            'new_authorization_only_commit_created':False,'mandatory_STOP':True,'scientific_winner':None,
            'limitations':['Registered r3/r6 target margins, cap sufficiency and T coverage await separate authorized one-shot.',
                           'Native guards retain conditional IEEE/BLAS/faithful-libm model; no change of scientific guarantee.',
                           'Historical direct failure cause remains nonunique; old run not reclassified.']}


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
