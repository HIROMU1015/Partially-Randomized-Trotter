#!/usr/bin/env python3
"""Render the completed pilot's saved statistics; no simulator, fit or N API."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
O=ROOT/'artifacts/track_d_r16_price_calibration/2026-11-10/run01'
D=ROOT/'docs/research/track_d_r16_price_calibration_20261110'


def read(path):return json.loads(path.read_text())
def fmt(value):return f'{value:,.3f}'.rstrip('0').rstrip('.')


def main():
    result=read(O/'result.json')
    if result['status']!='FINITE_CAMPAIGN_COMPLETE_MANDATORY_STOP':
        raise ValueError('INCOMPLETE_CAMPAIGN_NO_COMPLETE_CASE_REPORT')
    stats=read(O/'paired_statistics.json');prices=read(O/'prices.json')
    refs=read(D/'r32_reference.json');source=read(O/'SOURCE.json')
    metrics=('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size','qubit_count')
    # Formatting-only TSV: values come from the already saved price records.
    with (O/'prices.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=('q','sample','sample_id','method','axis','reused',*metrics),
            delimiter='\t',lineterminator='\n');writer.writeheader()
        for r in prices:writer.writerow({**{k:r[k] for k in ('q','sample','sample_id','method','axis','reused')},**r['metrics']})
    pairs=[]
    for q,data in stats['by_q'].items():
        for r in data['rows']:
            pairs.append(dict(q=int(q),axis=r['axis'],metric=r['metric'],
                strong_mean=r['strong']['mean'],strong_SD=r['strong']['sample_SD'],
                reference_mean=r['reference']['mean'],reference_SD=r['reference']['sample_SD'],
                delta_mean=r['paired_delta']['mean'],delta_SD=r['paired_delta']['sample_SD']))
    with (O/'paired_summary.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(pairs)
    equal=True
    table={(r['q'],r['sample'],r['method'],r['axis']):r['metrics'] for r in prices}
    for q in (2,4,8):
        for i in range(4):
            for m in ('symmetric_directional','reference'):
                equal=equal and table[q,i,m,'X']==table[q,i,m,'Y']
    lines=['# r16有限paired価格結果','',
        '固定条件：Linear H4／1.00Å／STO-3G／Nα=Nβ2・元H ground仮定／準備費除外、number-CDF8／元順LD2／ordered partial S2/s1/r16/K0／h=.1。q2/4/8各4 path、中央complete-step cut、C1 OFF、P0はQiskit1.3/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free/MPFR160宣言phase保持。','',
        '12 path・48価格recordが適格に揃った。q4には事前指定の保存1 pathを含む。新規11 raw、44 compile/postcheckと全pathのpaired X/Y・χがPASS。既存4 compile/postcheckは原本照合のみで再実行していない。','',
        '## 全7資源の標本平均・SD・paired差','',
        '以下は各qの4 pathにおける探索的有限標本統計。表の±は**標本SD（ddof1）**で、標準誤差・母平均CIではない。paired差はReference−Strong。n4を所定精度の保証としない。','',
        '|q|軸|資源|Strong 平均 ± SD|Reference 平均 ± SD|paired差 平均 ± SD|',
        '|---:|---|---|---:|---:|---:|']
    for p in pairs:
        lines.append(f"|{p['q']}|{p['axis']}|{p['metric']}|{fmt(p['strong_mean'])} ± {fmt(p['strong_SD'])}|{fmt(p['reference_mean'])} ± {fmt(p['reference_SD'])}|{fmt(p['delta_mean'])} ± {fmt(p['delta_SD'])}|")
    lines+=['','## 共分散・相関・標本単位','',
        f'全7資源について同一pathのX/Y価格が一致したか：**{equal}**。X/Yを独立8 pathとは扱わず、各qは4 pathである。価格の軸依存性と、実RPEで別々に行う軸の量子測定trialの独立性を混同しない。','',
        '機械可読`paired_statistics.json`に、方式×軸×資源の28変量共分散／相関と、paired差の14変量共分散／相関を保存した。分散は共分散対角、SDは各rowにある。各共分散のrankは最大3で、信頼性の高い高次元共分散推定や逆行列を主張できない。定数qubit幅や変動0のpaired資源では相関をnull（未定義）とし、独立性や母分散0の証明としない。','',
        'q間は別seed/pathの群で、今回の標本から13round共同分布・モデル係数共分散を作らない。source/input/compilerは共有し、将来のモデル誤差にも相関が生じ得る。q4の既存1本は既知データの再利用でprospective holdoutではない。全4本を事前固定通りに含め、完全な原recordでreused flagを保持した。','',
        '## r32との参考比較（X、7資源）','',
        'r32 C6a q2/q4各n16、C6b q8 n32の保存値を引用し、N・A/F/S係数・分散は更新していない。旧`peak_qubits`は保存full-wrapperの幅であり、この表のlogical qubitsとの対応だけを明記する。別law・event数・path集合であるため、r変更だけの因果効果や母平均の差ではない。','',
        '|q|資源|r16 Strong平均|r32保存Strong平均|r16 Reference平均|r32保存Reference平均|r16 paired平均|r32保存paired平均|',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    old={(r['q'],'qubit_count' if r['metric']=='peak_qubits' else r['metric']):r for r in refs['rows']}
    for p in pairs:
        if p['axis']!='X':continue
        r=old[p['q'],p['metric']]
        lines.append(f"|{p['q']}|{p['metric']}|{fmt(p['strong_mean'])}|{fmt(r['strong']['mean'])}|{fmt(p['reference_mean'])}|{fmt(r['reference']['mean'])}|{fmt(p['delta_mean'])}|{fmt(r['paired_delta']['mean'])}|")
    lines+=['','A=affine、F=frozen C6a、S=sqrt_laneというr32既存scenarioを保持する。r32の保存88本でCX差=208−224qが観測されたが、r16の予測式・補正定数・all-support恒等式として使っていない。r16では今回の3qの有限価格を直接記録しただけで、新しいA/F/Sや長q価格表を作成していない。','',
        '## 十分Nとの接続の限界','',
        '前回S1の保存Nを読取専用で引用：q2/q4はr16/r32とも512/軸、q8は513/512、q4096は1359/910。PF/RTE誤差・Nを再評価していない。同一r16・同一qの理想測定同値性の下ではStrong/Referenceに共通Nを使えるが、短qの価格比を長q全roundへ移送しない。rを変えたときは価格低下とN増加を両方扱う必要があり、今回の有限結果だけでは全RPE順位は決まらない。','',
        'RZはT countではなく、topology-free depthは実機時間ではない。raw ordered event、QPY、NPZはlocalのみ。公開は価格・統計・hash索引・SOURCE・STOP。']
    (D/'PAIRED_PRICE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    report=['# Track D × PR＋RPE：r16期待回路価格の限定校正','',
        '**q=2・4・8で各4 pathの有限paired価格校正を完了。** 保存q4の1本・4 wrapperを再利用し、新規11 path・52 occurrence・832 event、44 P0 compile/postcheckを取得した。全raw/post/paired χはPASS。これは短qの探索的有限標本で、母平均・長q価格・全13round総資源の保証ではない。','',
        '## 条件・SOURCE・事前固定','',
        'Linear H4／1.00Å／STO-3G／Nα=Nβ2・元H ground仮定／準備費除外、number-CDF8／元frame順LD2／S2/s1/r16/K0／h=.1。中央complete-step cutはq2/4/8で1/2/4、whole-path一gadget、Strongの既存Gaussian/directional/head relative/tail共有/step境界融合とReference A/B内部最適化を維持。Referenceの16 CSWAP・χ・追加register・cut損失を含む。C1追加融合OFF。','',
        f"実行SOURCE `{source['execution_source']}`、基点は前回r16結果`e6c6edac084943843b66c87c7899afee327c51c1`。同じr16研究branchを再利用。新sampling／価格観測の前に [CALIBRATION_PLAN](CALIBRATION_PLAN.md) と全seedの [sampling_plan.json](sampling_plan.json) をcommit固定した。旧r16/r32 SOURCE・原本・結果・STOPは無変更。",'',
        '2水準だけではaffineの残差が識別できないためq2/4にq8を加えた。n4は分散自由度3の初期pilotとして選び、母平均精度の最適nとは扱わない。保存r32のpaired変動は計画の参考とし、r16へ分布を転用しなかった。q16/32は新規実行していない。','',
        '## 結果（X、Yは同一価格と確認した場合に限り共通）','',
        '|q|資源|Strong平均|Reference平均|paired差平均|paired SD|','|---:|---|---:|---:|---:|---:|']
    for p in pairs:
        if p['axis']=='X':report.append(f"|{p['q']}|{p['metric']}|{fmt(p['strong_mean'])}|{fmt(p['reference_mean'])}|{fmt(p['delta_mean'])}|{fmt(p['delta_SD'])}|")
    report+=['','全軸・7資源の平均/SDとr32参考比較は [PAIRED_PRICE_RESULTS](PAIRED_PRICE_RESULTS.md)。28変量と14変量の共分散・相関をJSONに保持し、X/Yを標本数に二重計上していない。変動0資源も全supportの母分散0とはしない。','',
        '## 再利用・正確性・実行状態','',
        '既存canonical sampler、r16 law、C3 prepare_steps/StrongBuilder、C6a/B2 CAR、r16 P0 compile/postcheckとpaired χ関数、既存batched Aerを再利用した。C6a/C6bの固定r32 cohort guardは変更せず、新集合のbinding/統計adapterを追加した。describe/covarianceの算術は既存と同じ。既存SOURCEへの変更はない。','',
        'P0=Qiskit1.3.0/opt1/seed17/rz,sx,x,cx/approx1/initially_zero=False/topology-free、既存SymEngine MPFR160宣言global phase保持。raw全256入力、AB=V、actual adjoint、vacuum ray/χ、signed event/scalar/identity、complex coherence/X/Y observable、compiled対CAR/rawとpairedを原precision（component1e−10、observable Frobenius2e−10、rtol0、phase fitなし）で検査。科学retry・resampling・threshold緩和0。','',
        f"実測wall timeは{result['wall_seconds']:.2f}秒。実空きCPU/RAMに従いraw11並列、qに応じたstate-action threadsとCPU重み付きcompile/post dispatchを使用した。共有環境設定や他ユーザーprocessには変更を加えていない。",'',
        '人工回帰15件PASS。sampling原本・raw/compiled QPY・参照NPZ・内部logはlocal限定で、公開は科学結果・SOURCE・hash索引・STOP。前後SOURCE/inputと旧r16原本を照合済み。','',
        '## N・長q・総費用の限界','',
        '旧r16 S1は全13roundで条件付き十分条件PASSのまま保存・再利用した。Nはq8で513/軸（r32保存512）、q4096で1359/軸（910）。新しいPF/RTE誤差やNは計算していない。r変更による単価差を同一r内のReference効果と分け、短qでの単価低下だけで総費用優位としない。','',
        '今回のq8までの有限平均からq1024〜4096の価格や13round共同誤差を保証できない。r32 A/F/S係数・固定総値を変更せず、新しい長qモデルfit/登録・全round価格表・総資源集計も行っていない。新A Q、Closed P3、α最適化、ground solve、DF fit、詳細科学レビューは0。','',
        '技術的な傾向と次段階の判断材料は [NEXT_EXECUTION_PROPOSAL](NEXT_EXECUTION_PROPOSAL.md)。正式な研究GOやモデル採択はユーザー／GPTへ戻す。','',
        '成果を既存r16専用branchへcommit/pushし、remote SHA/tree/公開bytesを照合する。最終SHAと照合記録はチャットとlocal publication記録に残す。公開照合後mandatory STOP。']
    (D/'REPORT.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':main()
