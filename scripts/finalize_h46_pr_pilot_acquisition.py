"""Public technical completion projection and explicit-file local commit only.

Never exports raw paths/seeds/arrays, evaluates scientific statistics, or pushes.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import h46_pr_pilot_contract as c

BUNDLE='artifacts/resource_applicability/track_a_h46_pr_design_pilot_run01/2026-10-20'
REPORT='docs/research/track_a_h46_pr_design_pilot_run01_results_20261020.md'

def project(cfg):
    root=Path(cfg['run_root']);audit=root/'end_audit.json'
    r=c.read(audit) if audit.exists() else {}
    completion=root/'completion_witness.json'
    done=c.read(completion) if completion.exists() else {}
    external=Path(cfg['supervisor']['exit']);exit=c.read(external) if external.exists() else {}
    return dict(stage=cfg['stage'],SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
      status=r.get('status','STOP_END_AUDIT_MISSING'),
      planned_trajectories=len(cfg['tasks']),planned_compile_axes=cfg['call_scope']['compile_axes'],
      completed_trajectories=r.get('completed_trajectories',len(done.get('results',[]))),
      accepted_compile_axes=r.get('accepted_compile_axes',2*len(done.get('results',[]))),
      reserved_compile_calls=r.get('reserved_compile_calls'),
      failed_or_incomplete_trajectories=len(cfg['tasks'])-r.get('completed_trajectories',len(done.get('results',[]))),
      native_scientific_full_stage_PASS=done.get('status')=='UNIT_COST_STAGE_COMPLETE_MANDATORY_STOP',
      byte_SHA_roundtrip_full_stage_PASS=r.get('byte_SHA_roundtrip_PASS',False),
      archive_count=r.get('archives'),compressed_byte_total=r.get('compressed_byte_total'),
      original_byte_total=r.get('original_byte_total'),logical_run_bytes=r.get('logical_run_bytes'),
      allocated_run_bytes=r.get('allocated_run_bytes'),worker_peak_RSS_KiB=r.get('worker_peak_RSS_KiB'),
      own_aggregate_RAM_peak_sampled_bytes=r.get('own_aggregate_RAM_peak_sampled_bytes'),
      owned_scratch_peak_sampled_bytes=r.get('owned_scratch_peak_sampled_bytes'),
      peaks_are_sampled_not_exact_bounds=True,wall_seconds=exit.get('wall_seconds'),
      native_exit=exit.get('native_exit'),native_reaped=exit.get('reaped',False),
      all_workers_exit0_reaped=bool(done.get('workers')) and all(v['exit']==0 and v['reaped'] for v in done.get('workers',{}).values()),
      result_set_sha256=c.digest(done.get('results',[])),
      end_audit_sha256=c.sha(audit) if audit.exists() else None,
      completion_sha256=c.sha(completion) if completion.exists() else None,
      external_exit_sha256=c.sha(external) if external.exists() else None,
      error_details_private=True,statistics_executed=False,n_E=None,
      shot_policy_adopted=False,AD03='DEFERRED',retry=False,next_phase_started=False,mandatory_STOP=True)

def preserve_old_det(configs):
    inventory=c.ROOT/'artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/2026-10-20/private_raw_inventory.json'
    # The public inventory has relative paths only. Locate the old fixed config
    # root from the independent signal-origin lineage's known acquisition root
    # explicitly supplied in the new private config; never search directories.
    base=Path(configs[0]['preservation_old_DET_root'])
    rows=c.read(inventory)
    for row in rows:
        p=base/row['path']
        c.require(p.stat().st_size==row['bytes'] and c.sha(p)==row['sha256'],'OLD_DET_BYTES_CHANGED')
    return dict(status='PASS',files=len(rows),bytes=sum(r['bytes'] for r in rows),
      inventory_sha256=c.sha(inventory),old_files_modified=0)

def finalize(configs,controller_root):
    out=c.ROOT/BUNDLE;out.mkdir(parents=True,exist_ok=False)
    rows=[project(cfg) for cfg in configs]
    try:preservation=preserve_old_det(configs)
    except Exception:preservation=dict(status='STOP_PRESERVATION_NOT_CONFIRMED',details_private=True)
    summary=dict(SOURCE=configs[0]['SOURCE'],base_commit='3d0b5477ce04604ce873c079d9c66d02f2782239',
      stages=rows,preservation=preservation,controller_finished_sha256=c.sha(Path(controller_root)/'CONTROLLER_FINISHED.json'),
      n_P_each=33,cohort='DESIGN_PILOT',storage_policy=c.storage.storage_policy(),
      no_sampling_retry=True,no_top_up=True,no_statistical_analysis=True,n_E=None,AD03='DEFERRED',mandatory_STOP=True)
    c.save(out/'run_summary.json',summary)
    lines=['# H4/H6 PR DESIGN_PILOT run01：取得・技術終了監査', '',
      '科学SOURCE `'+configs[0]['SOURCE']+'`。各群33本の承認済み取得だけを実行した。', '',
      '| 系 | 状態 | 完了trajectory | 受理compile軸 | native exit/reap |',
      '| --- | --- | ---: | ---: | --- |']
    for row in rows:
        lines.append('| '+row['stage'][:2]+' | '+row['status']+' | '+str(row['completed_trajectories'])+
          '/396 | '+str(row['accepted_compile_axes'])+'/792 | '+str(row['native_exit'])+'/'+str(row['native_reaped'])+' |')
    lines+=['', 'byte完全復元と科学native受入を分離した。非完了・STOPをPASSに読み替えず、未取得値を0補完しない。', '',
      '保存容量・native gate・SOURCE/結果SHA・wall/RAM/scratch・終了状態は[公開技術summary](../../'+BUNDLE+'/run_summary.json)を参照する。RAM/scratch peakは定期観測値であり厳密な上界ではない。private原本・seed一覧・private絶対pathは公開しない。', '',
      '分散・SE/MCSE・n_E・shot・総費用・順位を計算していない。母平均費用・rare tail保証、AD03採用も行っていない。既存pilot/探索/独立確認との標本混合、retry、top-up、条件変更は0。', '',
      '**mandatory STOP。** 次の担当はユーザー：取得・終了監査結果を確認し、pilot限定数値評価を別途承認するか判断する。Codexは新たな承認まで科学統計・本探索へ進まない。','']
    report=c.ROOT/REPORT
    c.require(not report.exists(),'NEW_COMPLETION_REPORT_ONLY');report.write_text('\n'.join(lines))
    (out/'README.md').write_text('# DESIGN_PILOT取得の技術metadata\n\nrun_summary.jsonは各系の独立取得・終了監査と保存容量を転記する。標本統計・費用会計・seed/private原本は含まない。\n')
    paths=[BUNDLE+'/run_summary.json',BUNDLE+'/README.md',REPORT]
    # Never pick up another process's staged work or unrelated dirty files.
    c.require(not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=c.ROOT),'INDEX_MUST_BE_EMPTY_FOR_COMPLETION_COMMIT')
    for rel in paths:
        text=(c.ROOT/rel).read_text()
        c.require('/home/' not in text and '/tmp/' not in text and 'registry_seed_PRIVATE' not in text,'PUBLIC_TECHNICAL_PROJECTION_ONLY')
    subprocess.run(['git','add','--',*paths],cwd=c.ROOT,check=True)
    actual=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=c.ROOT).decode().strip('\0').split('\0')
    c.require(sorted(actual)==sorted(paths),'EXPLICIT_PUBLIC_COMPLETION_PATHS_ONLY')
    subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m',
      'Record H4/H6 PR design-pilot acquisition and mandatory STOP'],cwd=c.ROOT,check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
    c.save(Path(controller_root)/'PUBLIC_LOCAL_COMMIT.json',dict(commit=sha,paths=paths,push=False,mandatory_STOP=True))
    return sha
