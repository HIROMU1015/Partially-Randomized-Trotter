"""Add final audited comparison to repository indexes; preserve old result entries."""
import json
from pathlib import Path
from finalize_h6_pf_common_results import ROOT,OUTPUT


def read(p):return json.loads(Path(p).read_text())


def main():
    out=ROOT/OUTPUT;summary=read(out/'summary.json');audit=read(out/'independent_saved_result_checks.json')
    assert summary['complete']==588 and audit['status']=='PASS'
    comps=read(out/'comparisons_60.json');overall=[r for r in comps if r['formula']=='ALL_FORMULAS']
    summaryline='588候補の評価が全て完了。H2/H4は再計算せず参照結合。H6不足63metadata・新102signal・1380 endpoint・必要209 compile候補418軸を取得し終了監査PASS。'
    strict=[]
    for system in ('H2','H4','H6'):
        r=next(r for r in overall if r['system']==system and r['epsilon']==.0005)
        strict.append(f"{system} {r['PR_reduction_percent']:+.2f}%")
    limits='linear1Å/STO-3G/tol-only rank3/10/19、固定L_D/q≤8・δ=.8/.4/.2/.1/T=.8/P=0。ε=.0005の全公式PF対保存PR削減率：'+ '、'.join(strict)+'。経験的allowance・有限探索条件付きで母平均保証なし。旧SOURCE/raw/grant/STOP不変、local commit後mandatory STOP・push0。以下は履歴。'
    for name,target in [('PROJECT_MAP.md','docs/research/track_a_h246_pf_common_final_20261022.md'),('docs/research/研究概要・現状.md','track_a_h246_pf_common_final_20261022.md'),('docs/README.md','research/track_a_h246_pf_common_final_20261022.md'),('scripts/README.md','../docs/research/track_a_h246_pf_common_final_20261022.md'),('src/trotterlib/README.md','../../docs/research/track_a_h246_pf_common_final_20261022.md'),('VALIDATION_STATUS.md','docs/research/track_a_h246_pf_common_final_20261022.md')]:
        p=ROOT/name;p.write_text('## 2026-10-22 H246共通決定論PF：588候補の固定比較完了\n\n[全結果・契約・監査]('+target+')。'+summaryline+limits+'\n\n'+p.read_text())
    p=ROOT/'scripts/README.md';p.write_text(p.read_text().replace('以下は履歴。','新経路 `h6_pf_metadata_completion.py`、監査 `audit_h6_pf_metadata.py` / `audit_h6_pf_common_run.py`、固定参照統合 `finalize_h6_pf_common_results.py`、数値別照合 `check_h246_pf_common_final.py`。stage driverと消費済みgrantの再使用は禁止。以下は履歴。',1))
    p=ROOT/'artifacts/validation_manifest.json';manifest=read(p)
    assert 'h246_pf_common_final_20261022' not in manifest
    manifest['h246_pf_common_final_20261022']=dict(summary,report='docs/research/track_a_h246_pf_common_final_20261022.md',bundle=OUTPUT,independent_saved_checks=audit['checks_passed'],
        comparisons_60=True,all_system_formula_and_global_minima_complete=True,global_PR_reduction_percent_by_system={s:{str(r['epsilon']):r['PR_reduction_percent'] for r in overall if r['system']==s} for s in ('H2','H4','H6')})
    p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    note=ROOT/'docs/research/研究ノート/2026-10-22_h6_pf_metadata_STOP.md'
    note.write_text(note.read_text()+'\n## 追記：明示継続承認による完了\n\n旧STOP・snapshot_restore1消費を保持し、新SOURCE/新grantで63観測取得と旧52signal照合PASS。H6固定102signal・1380 endpoint・209compile候補418軸を取得。保存同native30価格を再利用し、H2/H4をbyte参照結合した588候補の全比較を終了監査した。科学grantのretry/resume0、旧PR統計不変。[最終結果と限界](../track_a_h246_pf_common_final_20261022.md)。ローカルcommit後mandatory STOP、push0。\n')
    print('Final indexes, current synthesis, dated note and manifest updated')


if __name__=='__main__':main()
