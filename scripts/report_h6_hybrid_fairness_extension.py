#!/usr/bin/env python3
"""Render completed, already-accounted H6 results. Never launch science."""
import csv,json,time
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/h6_hybrid_fairness_extension/2026-10-27'
def read(p):return json.loads(p.read_text())
def display(x):return 'null' if x is None or x=='' else format(float(Q(str(x))),'.7g')
def main():
    while not (OUT/'archive_restore_verification.json').exists():time.sleep(10)
    assert read(OUT/'archive_restore_verification.json')['status']=='PASS'
    summary=read(OUT/'EXECUTION_SUMMARY.json');v=read(OUT/'verification.json');plan=read(OUT/'sampling_design.json');cf=read(OUT/'confirmation_plan.json')
    comp=list(csv.DictReader((OUT/'COMPARISON_RESULTS.csv').open()));rows=read(OUT/'candidate_results.json');events=read(OUT/'event_orders.json')
    text=['# H6 Pauli-JW Hybrid PR：大event数への固定4候補検証','',
          '2026-10-27。科学SOURCE `'+summary['science_SOURCE']+'`、解析SOURCE `'+summary['analysis_SOURCE']+'`。',
          '固定linear H6/隣接1.00 Å/STO-3G/中性singlet、保存Pauli918項＋constant、同じraw ray/full-H reference/exact norm。S2/T=.8、central-rotation、full cosine/+Im sine、P=0、Qiskit1.3.0/rz,sx,x,cx/opt1/seed17。',
          '実行対象はp78/q8/r16,32,64/K2とp684/q1/r16/K4の4物理候補だけ。H2/H4、純RTE追加探索、旧成功cohort再取得、Hamiltonian/state/reference生成は0。','',
          '## 確認標本による条件付き比較','',
          '仮想compiled残差ρ=0、有限候補・有限価格標本による比較。GはRZ×shot、nは古典価格標本数で量子shot Nとは異なる。全5精度は同じ価格標本を共有する。新集合の確認後minと探索nomineeを区別する。旧14/15は元集合の結果として保存し、今回の拡張結果で上書きしない。','',
          '| ε_sig | 新Hybrid確認min / n | G新Hybrid | 拡張Hybrid最小（旧含む） | G純RTE | H/R | G SE Hybrid / RTE | 記述的判定 |',
          '|---|---|---:|---|---:|---:|---:|---|']
    for x in comp:
        text.append('| '+' | '.join([x['epsilon_sig'],x['new_Hybrid_id']+' / '+x['new_Hybrid_n'],display(x['new_Hybrid_G_exact']),x['expanded_Hybrid_id'],display(x['RTE_G_exact']),display(x['Hybrid_over_RTE_exact']),display(x['expanded_Hybrid_G_SE'])+' / '+display(x['RTE_G_SE']),x['classification']])+' |')
    text+=['','純RTEは保存H6_p000_q8_r64_K2・確認n48。旧Hybrid anchorはp78/q8/r8/K2とp684/q1/r8/K4・各n48を再利用した。INCONCLUSIVEは2 combined-SE以内という記述的規則であり、厳密な母平均winnerの信頼保証ではない。','',
           '## Signal・normalization・回路価格の内訳','',
           '| 候補 | 確認n | B_saved | C_cos / C_sin | ε=.05 N_total | ε=.0005 N_total | 適格性 |',
           '|---|---:|---:|---:|---:|---:|---|']
    for cid in cf['candidates']:
        rr=[r for r in rows if r['candidate_id']==cid and r['cohort']=='confirmation'];lo=next(x for x in rr if x['epsilon_sig']=='0.05');hi=next(x for x in rr if x['epsilon_sig']=='0.0005')
        text.append('| '+' | '.join([cid,str(cf['candidates'][cid]['n_C']),display(lo['B_exact']),display(lo['C_cos_exact'])+' / '+display(lo['C_sin_exact']),str(lo['N_total']),str(hi['N_total']),'; '.join(x['epsilon_sig']+':'+x['eligibility'] for x in rr)])+' |')
    text+=['','全4候補のlaw/meanは既存tight L2（scalar128bit・binary64 vector中心53bit）を各1作用。Bを一回適用し、raw normを共有、再正規化なし。数値allowance・guard・shot十分条件・任意精度ceilは既存のまま。新science retryや追加precision作用なし。',
           'Bias不適格のN/G=nullと、価格標本が取得済みであることを分ける。全候補が少なくともε=.05で適格なため価格を取得した。全精度・両cohortのbias/u/N/C/G/SEはCANDIDATE_RESULTS.csvへ保存。','',
           '## 標本設計と選抜','',
           '旧paired価格から分散・共分散を再計算し、新lawの非zero event確率と併せてsampling前にn_Eを固定した。旧SE由来のn要求は'+str(plan['cv_based_n_E_requirement'])+'、希少event観測の設計要求を含む共通n_Eは'+str(plan['n_E'])+'となった。探索上限128、確認下限64/上限1024はsampling前固定の有限科学budgetで、CPU制限ではない。',
           'p684の非zero eventを90%の名目確率で少なくとも一度見る設計要求は'+str(plan['rarity']['PR_p0684_q1_r16_K4']['n_for_nominal90pct_at_least_one'])+'本で探索上限を超える。観測目標は達成保証とせず、未観測寄与を0としない。',
           '確認対象は適格4候補すべて。探索paired分散、名目t係数、1%半幅目標、旧比較側SE floorからn_Cを決め、確認開始前に候補ごとのn_C・seed/indexを一度だけ固定した。確認結果を見たtop-upは0。探索・確認をpoolしない。',
           '探索nomineeは精度ごとに凍結し、nomineeの確認Gと確認後minをCOMPARISON_RESULTS.csvで別列にした。5条件を独立な5実験と数えない。名目SEは希少eventの母平均tailや選抜後同時coverageの上界ではない。','',
           '| ε | 探索前規則からのnominee | nominee確認G | 確認後新集合min | r64境界にminが残るか |',
           '|---|---|---:|---|---|']
    for x in comp:text.append('| '+' | '.join([x['epsilon_sig'],x['nominee_id'],display(x['nominee_confirmation_G_exact']),x['new_Hybrid_id'],x['minimum_at_r64_boundary']])+' |')
    text+=['','## 希少eventと検証の範囲','', '| 候補/cohort | n | order0 / order2 / order4 event数 |','|---|---:|---|']
    for key,e in events.items():text.append('| '+key+' | '+str(e['n'])+' | '+' / '.join(str(e['event_orders'][k]) for k in ['0','2','4'])+' |')
    text+=['','K2のorder4はsupport外、K4の未観測order4/componentは費用寄与0の証明ではない。実compiledρ、母平均費用tail、全パラメータ大域最適性、FT/QPE/RPE全体は未保証。q/r/Kを揃えてもprefix変更はtail/law/B/biasを変え、純粋な保持部の因果効果ではない。',
           '新trajectory '+str(summary['new_paired_trajectories'])+'、両軸compile '+str(summary['new_axis_compiles'])+'、mean状態作用4。native child '+str(summary['native_children'])+'件のexit0/reapを回収。'+str(v['qpy_gzip_raw_SHA_checked'])+' QPYのcompressed/raw SHAと代表'+str(v['representative_compiled_QPY_decoded'])+' compiled QPYのwire/basis/費用を照合。これは全compiler数値証明ではない。',
           'Phase Aの人工/mock27件と、再利用したbody/control/shot/paired処理を使用。科学SOURCE/kernelは保全し、新namespaceだけで実行した。実測free CPUに応じた低優先度workerを使用し、共有設定・他者process/ファイルは変更しない。',
           '科学raw/QPYは既存QPY-aware形式で可逆圧縮し、元bytesのSHA復元を全件検査。原rawは無変更。内部絶対pathを含むoperational logはlocalに保全し、公開版PUBLIC_NATIVE_AUDIT.jsonには終了・回収証拠を保存した。','',
           '## データ・終了・次の担当','',
           '- CANDIDATE_RESULTS.csv / candidate_results.json：全endpointと会計。',
           '- COMPARISON_RESULTS.csv / comparison_details.json：一次対照、nominee、確認後min、G差・両側SE。',
           '- SOURCE.json / ANALYSIS_SOURCE.json / sampling_design.json / confirmation_plan.json：固定SOURCE・設計。',
           '- verification.json / PUBLIC_NATIVE_AUDIT.json / data_archives.json / archive_restore_verification.json：保存証拠。',
           '科学計算はSTUDY_STOP.jsonで終了。公開・remote照合後mandatory STOP。旧14/15結果は保全。主agentの技術照合であり、詳細科学レビュー・中心claimの正式採択はしていない。',
           '次の担当はユーザー/GPT：この拡張有限集合の条件付き比較を研究上どう扱うか判断する。境界・近接・希少eventの追加検証が必要なら、対象を限定して別承認する。Codexは新候補や詳細レビューへ自動移行しない。','']
    (OUT/'REPORT.md').write_text('\n'.join(text))
    (OUT/'README.md').write_text('# H6 Hybrid公平性補完の結果\n\n[REPORT.md](REPORT.md)から読む。新4候補の条件付き比較で、旧14/15と別集合。科学SOURCEと実行・会計・可逆archiveの索引を同directoryに保存。\n\n新sampling/compile/再解析はSTUDY_STOP.jsonで終了。実ρ/母平均tail/大域最適性は未保証。\n')
    print('REPORT_FROM_SAVED_RESULTS_READY',flush=True)
if __name__=='__main__':main()
