"""One-shot Phase 1 stages; backend import/work only after explicit execution.

Calibration -> committed frozen model -> independent validation -> rare pairs.
Existing STOPs are read-only; failures write a new STOP and are not retried.
"""
from copy import deepcopy
import csv
import json
import re
from pathlib import Path
import subprocess

from . import r_price_phase1 as m


def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def check_source(root, plan):
    for name in m.SOURCE_FILES:
        if (Path(root)/name).read_bytes()!=m.blob(root,plan['source'],name): raise ValueError('SCIENTIFIC_SOURCE_CHANGED:'+name)


def committed_frozen(root, out, commit):
    path=(Path(out)/'FROZEN_MODELS.json').resolve().relative_to(Path(root).resolve())
    data=m.blob(root,commit,str(path))
    if data!=(Path(out)/'FROZEN_MODELS.json').read_bytes(): raise ValueError('FROZEN_MODEL_NOT_COMMITTED_EXACTLY')
    return json.loads(data)


def execute_stage(root, plan, out, private, stage, *, execute=False, backend=None, frozen_commit=None,
                  saved_reader=None, frozen_reader=None):
    """No approval receipt. --execute avoids accidental science; chat scope governs."""
    if not execute: raise ValueError('EXPLICIT_EXECUTION_REQUIRED_NOT_PERMISSION_PROOF')
    m.validate_plan(plan);check_source(root,plan)
    out=Path(out);private=Path(private)
    if stage not in ('calibrate','validate','rare'): raise ValueError('PHASE1_STAGE')
    if stage=='calibrate':
        out.resolve().relative_to(Path(root).resolve())
        if private.resolve().is_relative_to(Path(root).resolve()):raise ValueError('PRIVATE_RAW_MUST_BE_OUTSIDE_REPOSITORY')
        # Refuse existing results or old STOP; never overwrite or resume a stage.
        if out.exists() or private.exists():raise FileExistsError('FRESH_PHASE1_OUTPUTS_REQUIRED')
        out.mkdir(parents=True,exist_ok=False);private.mkdir(parents=True,exist_ok=False)
        write(out/'PLAN.json',plan)
        write(out/'EXECUTION_PATHS.json',dict(private=str(private.resolve())))
    elif not out.is_dir() or read(out/'PLAN.json')!=plan:
        raise ValueError('EXISTING_PHASE1_PLAN_REQUIRED')
    if read(out/'EXECUTION_PATHS.json')['private']!=str(private.resolve()) or not private.is_dir():raise ValueError('PRIVATE_RESULT_LOCATION_CHANGED')
    if (out/'STOP.json').exists(): raise ValueError('STOP_NO_IMPLICIT_RETRY')
    if stage!='calibrate':
        if not frozen_commit or not re.fullmatch('[0-9a-f]{40}',frozen_commit): raise ValueError('COMMITTED_FREEZE_REQUIRED_BEFORE_VALIDATION')
        frozen=(frozen_reader or committed_frozen)(root,out,frozen_commit)
        if frozen['plan_digest']!=m.digest(plan) or frozen['validation_prices_observed']: raise ValueError('FROZEN_PLAN_OR_ROLE')
        if stage=='rare' and not (out/'VALIDATION.json').exists(): raise ValueError('NORMAL_VALIDATION_REQUIRED_FIRST')
    write(out/(stage+'.started.json'),dict(source=plan['source'],plan_digest=m.digest(plan),stage=stage,no_retry=True))
    try:
        if backend is None:
            from .r_price_phase1_science import Backend
            backend=Backend(root,plan)
        if stage=='calibrate':
            cells=[c for c in plan['cells'] if c['role']=='CALIBRATION'];data={}
            for c in cells:
                if c['saved']:data[c['id']]=m.saved_cell(root,plan,c,read=saved_reader)
            new=[c for c in cells if not c['saved']]
            data.update(backend.acquire_many(new,private/'calibration'))
            for c in cells:m.validate_records(plan,c,data[c['id']])
            write(out/'CALIBRATION.json',data)
            frozen=m.calibrate(root,plan,data);write(out/'FROZEN_MODELS.json',frozen)
            evaluations=[];prices=[]
            for c in cells:
                ev,_=m.evaluate_cell(root,plan,c,data[c['id']],frozen);evaluations.append(ev)
                prices+=m.price_records(plan,c,ev,frozen,dict(execution_source=plan['source'],result_path=str((out/'CALIBRATION.json').resolve().relative_to(Path(root).resolve()))))
            write(out/'CALIBRATION_EVALUATION.json',dict(evaluations=evaluations,prices=prices,full_rpe_resources=None))
            return dict(status='CALIBRATION_COMPLETE_COMMIT_FREEZE_BEFORE_VALIDATION',full_RPE_qualified=False)
        if stage=='validate':
            cells=[c for c in plan['cells'] if c['role']=='VALIDATION'];data=backend.acquire_many(cells,private/'validation')
            if set(data)!={c['id'] for c in cells}:raise ValueError('COMPLETE_VALIDATION_BATCH_REQUIRED')
            evaluations=[];residuals=[];prices=[]
            for c in cells:
                ev,rows=m.evaluate_cell(root,plan,c,data[c['id']],frozen);evaluations.append(ev);residuals+=rows
                prices+=m.price_records(plan,c,ev,frozen,dict(execution_source=plan['source'],result_path=str((out/'VALIDATION.json').resolve().relative_to(Path(root).resolve())),frozen_model_commit=frozen_commit))
            write(out/'VALIDATION.json',dict(frozen_model_commit=frozen_commit,data=data,evaluations=evaluations,
                prices=prices,validation_refit=False,all_law_expected_price=None,total_CI=None,full_rpe_resources=None,mandatory_stop=True))
            with (out/'PATH_RESIDUALS.csv').open('x',newline='') as f:
                writer=csv.DictWriter(f,list(residuals[0]),lineterminator='\n');writer.writeheader();writer.writerows(residuals)
            return dict(status='INDEPENDENT_VALIDATION_EVALUATED_NOT_ADOPTED',full_RPE_qualified=False)
        # Forced pairs are a separate unit and never enter fit or ordinary means.
        diagnostic=backend.rare_pairs(private/'rare')
        if len(diagnostic)!=4 or any(v['unit']!='FORCED_LEGAL_PAIR_NOT_UNCONDITIONAL_COHORT' for v in diagnostic):
            raise ValueError('FOUR_FORCED_PAIRS_REQUIRED')
        write(out/'RARE_DIAGNOSTICS.json',dict(pairs=diagnostic,cohort_pooling=False,all_law_contribution=None,total_CI=None,mandatory_stop=True))
        report='''# R依存compiled価格 Phase 1実行結果

校正3セル/systemを固定して係数をGitへ保存後、新R×qの独立検証価格を取得した。検証再fitなし。指標/paired共分散はVALIDATION.json、path誤差はPATH_RESIDUALS.csv、forced legal pairはRARE_DIAGNOSTICS.jsonを参照する。

有限cohortとforced診断は別単位。係数covarianceは有限標本のnominal量で、長q/R systematic・全law期待価格・共有joint uncertainty・total CI・実Gは未資格/null。性能の採用閾値は設定されておらず、modelのqualified_for_candidateはfalseを維持する。SOURCE/input/raw/過去STOPは保全。詳細科学レビュー・全RPE評価は実施していない。

GitHub公開とremote照合後mandatory STOP。追加R/q・再fit・再sampling・N/Gへ自動進行しない。
'''
        with (out/'REPORT.md').open('x') as f:f.write(report)
        write(out/'STOP.json',dict(status='PHASE1_COMPLETE_MANDATORY_STOP',mandatory_stop=True,new_N_G=0,full_RPE_qualified=False))
        return dict(status='PHASE1_COMPLETE_MANDATORY_STOP')
    except BaseException as error:
        if not (out/'STOP.json').exists():write(out/'STOP.json',dict(status='PHASE1_FAILED_NO_RETRY',stage=stage,reason=str(error),mandatory_stop=True))
        raise
