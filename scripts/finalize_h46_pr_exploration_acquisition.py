"""Technical acquisition projection only; explicit-file local commit, no push."""
from pathlib import Path
import subprocess
from types import FunctionType
import h46_pr_exploration_contract as c
import finalize_h46_pr_pilot_acquisition as previous

BUNDLE='artifacts/resource_applicability/track_a_h46_pr_exploration_run01/2026-10-21'
REPORT='docs/research/track_a_h46_pr_exploration_run01_results_20261021.md'
_project=FunctionType(previous.project.__code__,dict(previous.__dict__,c=c),'project',previous.project.__defaults__)
_preserve=FunctionType(previous.preserve_old_det.__code__,dict(previous.__dict__,c=c),'preserve_old_det',previous.preserve_old_det.__defaults__)

def project(cfg):
    row=_project(cfg);row.pop('n_E');row.update(cohort='EXPLORATION',
        n_E_adopted=cfg['decisions']['n_E_by_candidate'],independent_confirmation=False,
        pilot_samples_pooled=False,population_mean_or_variance_guarantee=False)
    return row

def finalize(configs,controller_root):
    out=c.ROOT/BUNDLE;out.mkdir(parents=True,exist_ok=False)
    rows=[project(cfg) for cfg in configs]
    try:preservation=_preserve(configs)
    except Exception:preservation=dict(status='STOP_PRESERVATION_NOT_CONFIRMED',details_private=True)
    summary=dict(SOURCE=configs[0]['SOURCE'],readiness_commit='2fcdec15341c0b3371b37ab5d5eb5bec1009de54',
        adopted_option=configs[0]['decisions']['option_id'],stages=rows,preservation=preservation,
        controller_finished_sha256=c.sha(Path(controller_root)/'CONTROLLER_FINISHED.json'),
        cohort='EXPLORATION',storage_policy=c.storage.storage_policy(),no_sampling_retry=True,
        no_top_up=True,no_statistical_analysis=True,AD03='DEFERRED',mandatory_STOP=True)
    c.save(out/'run_summary.json',summary)
    lines=['# H4/H6高次PF PR本探索 run01：取得・技術終了監査','',
        '科学SOURCE `'+configs[0]['SOURCE']+'`。承認済みの新規EXPLORATION単価取得だけを実行した。','',
        '| 系 | 状態 | 完了trajectory | 受理compile軸 | native exit/reap |',
        '| --- | --- | ---: | ---: | --- |']
    for r in rows:
        lines.append('| '+r['stage'][:2]+' | '+r['status']+' | '+str(r['completed_trajectories'])+'/'+str(r['planned_trajectories'])+
            ' | '+str(r['accepted_compile_axes'])+'/'+str(r['planned_compile_axes'])+' | '+str(r['native_exit'])+'/'+str(r['native_reaped'])+' |')
    lines+=['', '両軸native受入とbyte完全復元を別々に監査した。途中STOP・欠測をPASSへ読み替えない。', '',
        '[公開技術summary](../../'+BUNDLE+'/run_summary.json)に、各契約・結果SHA・archive容量・wall/RAM/scratch・終了状態を保存した。peakは定期観測値で厳密な上界ではない。', '',
        'MCSE目標2%・分散倍率4・群別下限33は保存pilotに基づく設計で、未観測S4・母平均費用・分散の保証ではない。pilot・独立確認との混合はしない。', '',
        '新しい分散/MCSE評価、shot計算、費用再会計、順位、独立確認、AD03採用は実施していない。既存結果・grant・ledger・marker・STOPを保全した。', '',
        '**mandatory STOP。次の担当：ユーザーが取得結果を確認し、新しい探索標本の限定統計評価を別途承認するか判断する。Codexは承認前に統計・会計へ進まない。','']
    report=c.ROOT/REPORT;c.require(not report.exists(),'NEW_COMPLETION_REPORT_ONLY');report.write_text('\n'.join(lines))
    (out/'README.md').write_text('# EXPLORATION取得の技術metadata\n\n科学統計・費用会計・seed・private原本は含まない。取得と終了監査を表す。\n')
    paths=[BUNDLE+'/run_summary.json',BUNDLE+'/README.md',REPORT]
    brief='## 2026-10-21 H4/H6 PR本探索：取得終了・mandatory STOP\n\n'
    brief+='[取得結果](docs/research/'+Path(REPORT).name+')。'
    brief+=' / '.join(r['stage'][:2]+': '+r['status']+' '+str(r['completed_trajectories'])+'/'+str(r['planned_trajectories'])+'本' for r in rows)
    brief+='。新統計・shot・会計・順位・P・AD03は未実施。次はユーザーの限定統計評価の判断。以下は履歴。\n\n'
    for name in ('PROJECT_MAP.md','VALIDATION_STATUS.md','docs/research/研究概要・現状.md'):
        p=c.ROOT/name;p.write_text((brief.replace('(docs/research/','(') if name.startswith('docs/research/') else brief)+p.read_text())
        paths.append(name)
    manifest=c.ROOT/'artifacts/validation_manifest.json';v=c.read(manifest)
    v['h46_pr_exploration_run01_results_20261021']=dict(report=REPORT,summary=BUNDLE+'/run_summary.json',
        stages={r['stage']:r['status'] for r in rows},statistics=False,shot=False,ranking=False,mandatory_STOP=True)
    # Existing index, intentionally updated rather than overwrite an evidence file.
    import json
    manifest.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');paths.append('artifacts/validation_manifest.json')
    c.require(not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=c.ROOT),'INDEX_MUST_BE_EMPTY_FOR_COMPLETION_COMMIT')
    for rel in paths:
        text=(c.ROOT/rel).read_text();c.require('/home/' not in text and '/tmp/' not in text and 'registry_seed_PRIVATE' not in text,'PUBLIC_TECHNICAL_PROJECTION_ONLY')
    subprocess.run(['git','add','--',*paths],cwd=c.ROOT,check=True)
    actual=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=c.ROOT).decode().strip('\0').split('\0')
    c.require(sorted(actual)==sorted(paths),'EXPLICIT_PUBLIC_COMPLETION_PATHS_ONLY')
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m',
        'Record H4 H6 PR exploration acquisition and mandatory STOP'],cwd=c.ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
    c.save(Path(controller_root)/'PUBLIC_LOCAL_COMMIT.json',dict(commit=commit,paths=paths,push=False,mandatory_STOP=True))
    return commit
