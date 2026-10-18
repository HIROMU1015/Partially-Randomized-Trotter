"""Metadata-only completion collector. Never launches or retries science."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import h4_followup_contract as c


def accepted(end,terminal,gate,result):
    c.require(end['exit_code']==terminal['exit_code']==0 and end['same_parent_reaped'] and end['wait4_complete'],'NATIVE_COMPLETION')
    c.require(end['SOURCE']==terminal['SOURCE']==gate['SOURCE'] and gate['status']=='PASS','COMPLETION_SOURCE_GATE')
    c.require(terminal['status']==result['status']=='COMPLETE_MANDATORY_STOP','MANDATORY_STOP_RESULT')


def collect(private):
    public=c.ROOT/'artifacts/resource_applicability/track_a_h4_followup/2026-10-18';results={};receipts={}
    for stage in ('R','B','T','F','FP'):
        run=private/(stage+'_run01');end=c.read(run/'native_exit.json');term=c.read(run/'output/terminal.json')
        gate=c.read(run/'output/final_gate.json');data=(run/'output/result.json').read_bytes();result=json.loads(data)
        accepted(end,term,gate,result)
        config=c.read(private/('config_FP.json' if stage=='FP' else 'config_RBTF.json'))
        for name,h in config['source_hashes'].items():c.require(c.sha(c.ROOT/name)==h,'SOURCE_STILL_UNCHANGED:'+name)
        workers=[]
        for p in (run/'output').glob('**/parallel_receipt.json'):
            v=c.read(p);c.require(v['status']=='PASS' and v['all_reaped'] and not v['cleanup_errors'],'WORKER_RECEIPT')
            c.require(all(w['exit_code']==0 and w['reaped'] for w in v['workers']),'WORKER_NATIVE_EXIT')
            for w in v['workers']:
                result_path=Path(w['directory'])/'task_result.json'
                c.require(result_path.resolve().is_relative_to((run/'output').resolve()) and c.sha(result_path)==w['result_sha256'],'WORKER_RESULT_CONTENT_BINDING')
            workers.extend(v['workers'])
        c.require(len(workers)==dict(R=12,B=32,T=82,F=0,FP=42)[stage],'EXACT_NATIVE_WORKER_COVERAGE')
        gz=public/(stage+'_result.json.gz')
        if not gz.exists():
            with gz.open('xb') as f:
                with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(data)
        c.require(gzip.decompress(gz.read_bytes())==data,'PUBLIC_RESULT_IDENTICAL')
        receipt=dict(stage=stage,SOURCE=end['SOURCE'],native_exit=end,terminal=term,final_gate=gate,
            result_uncompressed_sha256=hashlib.sha256(data).hexdigest(),result_gzip_sha256=c.sha(gz))
        path=public/(stage+'_completion_receipt.json')
        if not path.exists():c.save(path,receipt)
        else:c.require(c.read(path)==receipt,'EXISTING_RECEIPT_IDENTICAL')
        measured=len(list((run/'output').glob('**/measured_*.json')))
        c.require(measured==dict(R=0,B=64,T=144,F=0,FP=72)[stage],'EXACT_MEASURED_AXIS_COVERAGE')
        receipts[stage]=dict(SOURCE=end['SOURCE'],native_exit_code=0,final_gate='PASS',
            compile_axes=measured,workers=len(workers),all_workers_reaped=True,wall_seconds=end['wall_seconds'],
            peak_single_process_RSS_bytes=end['peak_RSS_bytes'],result_sha256=receipt['result_uncompressed_sha256'])
        results[stage]=result
    c.require(len(results['R']['signals'])==12 and len(results['T']['panels'])==2 and results['T']['new_T08_computation']==0,'REGISTERED_SCOPE_COMPLETE')
    c.require(len(results['B']['measurements'])==32 and results['F']['actual_calls']['df_decomposition']==1,'B_F_SCOPE_COMPLETE')
    summary=dict(status='COMPLETE_MANDATORY_STOP',stages=receipts,strict_bias_certificate=None,
        population_mean_certified=False,unmeasured_cost_zero_filled=False,new_geometry=0,next_stage_authorized=False)
    c.save(public/'final_completion.json',summary)
    lines=['\n\n## 最終完了：全stage正常終了・mandatory STOP\n',
        '各SOURCEのnative exit0、最終gate PASS、全owned worker回収と予定軸数を照合した。R12 signal、B64軸、T144新規軸、Fの入力生成1回、FP72軸が完了した。T=.8とlegacy代表panelの保存値は再利用し、追加sampling・compile・fitは起動しない。\n',
        '| Stage | 新compile軸 | native wall秒 | SOURCE |\n|---|---:|---:|---|\n']
    for stage,r in receipts.items():lines.append(f'|{stage}|{r["compile_axes"]}|{r["wall_seconds"]:.2f}|`{r["SOURCE"]}`|\n')
    lines+=['\n### 固定5構成内の点費用比較\n',
        '下表は通常bias・allowance倍率1・P=0・RZのPR削減率である。新しい全格子winnerや母平均優位を示さない。nullは適格な比較pairがなく、0費用を意味しない。全20条件・paired SEは各圧縮resultに保存した。\n',
        '| 系列 | ε=.05 | .01 | .005 | .001 | .0005 |\n|---|---:|---:|---:|---:|---:|\n']
    panels=[('legacy T=.8（保存）',results['T']['saved_T08'])]
    panels += [(f'legacy T={p["T"]}',p) for p in results['T']['panels']]
    panels += [('DF政策比較・legacy rank12',results['FP']['legacy']),('tol-only rank'+str(results['F']['rank']),results['FP']['new'])]
    for name,panel in panels:
        vals=[]
        for eps in c.plan()['epsilons']:
            row=next(r for r in panel['comparisons'] if r['bias_mode']=='USUAL_SIGNED' and r['allowance_factor']==1. and r['epsilon']==eps)
            value=row.get('PR_reduction');vals.append('null' if value is None else f'{100*value:.2f}%')
        lines.append('|'+name+'|'+'|'.join(vals)+'|\n')
    lines += ['\nTとDF政策のpanelはPRのKを含めて別登録集合であり、両panelを混ぜた最良選択をしていない。新DFはSCF軌道・状態も新しく、DF rankだけの因果効果を断定しない。Rの新PR構成について費用取得・ランキングは未実施。希少層は条件付きn8であり、無条件費用の母平均保証にはしない。\n',
        '\n公開証拠は `artifacts/resource_applicability/track_a_h4_followup/2026-10-18/`。SOURCE/plan/input/compiler/runtime/各grant digest・native回収・最終gateを分離して保存した。raw NPZ・環境・grant原本はprivateに保全した。旧H4/H6/H8の結果・SOURCE・grant・ledger・STOPを書き換えていない。\n']
    report=c.ROOT/'docs/research/track_a_h4_followup_20261018.md'
    with report.open('a') as f:f.writelines(lines)
    status='H4追加4検証完了：R12 signal、B64軸、T144新規軸、tol-only rank'+str(results['F']['rank'])+'入力とFP72軸、全native exit0/final gate PASS・mandatory STOP。詳細 `track_a_h4_followup_20261018.md`。以下は履歴。\n\n'
    for name in ('PROJECT_MAP.md','VALIDATION_STATUS.md','docs/research/研究概要・現状.md'):
        path=c.ROOT/name;path.write_text(status+path.read_text())
    note=c.ROOT/'docs/research/研究ノート/2026-10-18_h4_followup.md'
    with note.open('a') as f:f.write('\n全5実行単位のnative exit0・最終gate PASS・全回収・予定軸数を確認。各stage STOPを保持し、全結果を公開可能なmetadataと圧縮JSONへ保存。追加科学実行なし。\n')
    path=c.ROOT/'artifacts/validation_manifest.json';manifest=c.read(path)
    manifest['h4_followup_20261018'].update(status='COMPLETE_MANDATORY_STOP',stages=receipts,artifact=str(public.relative_to(c.ROOT)))
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    inventory={str(p.relative_to(public)):dict(bytes=p.stat().st_size,sha256=c.sha(p)) for p in public.iterdir() if p.is_file()}
    c.save(public/'inventory.json',inventory)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--commit',action='store_true')
    a=p.parse_args();c.save(a.private/'completion_collector_lock.json',dict(start=time.time(),metadata_only=True))
    try:
        expected_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
        while not all((a.private/(s+'_run01')/'native_exit.json').exists() for s in ('R','B','T','F','FP')):time.sleep(3)
        summary=collect(a.private)
        current_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=c.ROOT,text=True).strip()
        if a.commit and current_head==expected_head and not staged:
            paths=['artifacts/resource_applicability/track_a_h4_followup/2026-10-18','docs/research/track_a_h4_followup_20261018.md',
                'PROJECT_MAP.md','VALIDATION_STATUS.md','docs/research/研究概要・現状.md',
                'docs/research/研究ノート/2026-10-18_h4_followup.md','artifacts/validation_manifest.json']
            subprocess.run(['git','add','--',*paths],cwd=c.ROOT,check=True)
            subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Record completed H4 boundary rare time and tol-only results'],cwd=c.ROOT,check=True)
            summary['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.ROOT,text=True).strip()
        elif a.commit:summary['commit_skipped']='HEAD_CHANGED_OR_PREEXISTING_STAGED_CHANGES; results preserved'
        c.save(a.private/'completion_collector_result.json',summary);print(json.dumps(summary,indent=2));return 0
    except BaseException as exc:
        c.save(a.private/'completion_collector_STOP.json',dict(reason=str(exc),traceback=traceback.format_exc(),new_science_started=False));raise


if __name__=='__main__':raise SystemExit(main())
