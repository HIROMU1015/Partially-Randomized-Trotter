# SCV224：既存証拠と限界

2026-11-07 JST。**固定REPORT/summaryの引用と既存設計の整理のみ。数値再解析・追加実行・方針採択なし。**

## 対象と数値の出典

数値は全て[完了REPORT（ba483ab3）][result]と同版の[RESULT_SUMMARY.json][summary]の既報値。
Raw、個々のbit・価格、archiveから分散・相関・費用を再計算していない。新しい比率・検定・break-even値も算出していない。

対象は保存linear H4 / 1.00 Å / STO-3G / legacy DF12 / 8 system qubits / 全4電子sector / L_D=3 /
m0,q1,δ=.02,r1,K2 / coefficient_atol=0 / X,Y / N1 actual finite-grid Q・selected full / 理想gate。
理論基準は元の保存DF Hamiltonianの厳密ground、実行にはN2の保存保証stateと認証済みexact Fraction engineを使用した。
保存DF外の未近似分子Hamiltonian、実機noise、長roundへの保証は含まない。
価格はQiskit1.3.0、basis=(rz,sx,x,cx)、opt2、seed17、状態準備だけを除くfull interrogation。
科学SOURCEは `996e434e63ce9b1d14c6ae191649c9abe213cf47`。[実行契約][execution]の一campaignは完了・mandatory STOP済み。

## 確認された結果の引用

| 項目 | 固定REPORTの記録 |
|---|---|
| control | rotation fragment群indicator、S={df-fragment-3}、h1だけ有効。他4層λ=0は取得前固定 |
| 条件付き平均 | ν₁≈0.842715247177。Actual Qからexact表を取得・train前に凍結 |
| X train | u=0が4件、u=1が28件。rank2・range PASS、λ₁≈1.0164×10⁻¹⁶ |
| Y train | u=0が2件、u=1が30件。rank2・range PASS、λ₁≈0.0666667 |
| 評価 | 学習64＋共通fresh評価160＝224 physical shots、全5正層、各軸quota(4,64,4,4,4) |
| 保存・停止 | 全224 bit/価格完備、正常終了。元SOURCE・結果・STOPとC1凍結を保全 |

| 軸 | B1-OLの推定量分散点指標 | SCVの同点指標 | 既報の解釈 |
|---|---:|---:|---|
| X | 1.39572692233×10⁻⁹ | 表示精度で同じ | 追加削減を観測せず |
| Y | 0.0151001791303 | 0.0150964477577 | 約0.024711%低下。統計的有意性は未確立 |

ここでの点指標は既存の \(\widehat V=\sum_h p_h^2s_h^2/n_h\)。層内raw sample varianceそのものとは区別する。
Xで0だったのは**h1のraw sample variance**。X全targetの分散点指標は上表の非零値で、真の母分散0という意味でもない。
Xの微小な非零OLS係数は事後に0へ丸めておらず、保存された残差の微小差を実用的な改善と呼ばない。
Yの低下も、真の分散差・相関の確立を意味しない。paired covarianceと推定平均差のsample-based varianceは保存されているが、**分散削減量自体のconfidence intervalではない**。[有限confidence整理][confidence]

| 費用区分 | 既報の実現compiled RZ |
|---|---:|
| 学習 | 306,990 |
| 共通評価・同quota B1-OL参照 | 772,251 |
| 物理合計・学習込みSCV参照 | 1,079,241 |

元REPORTでは、Vhat×各手法の実現RZという探索的point scoreはX/YともSCVの方が大きかった。
共通評価を実支出で二重課金せず、SCVから学習費を落とさない。cache hitでも量子RZは全額課金されている。
これは今回の実現支出・point scoreの記述。期待RZ、精度保証付き必要shot数、最適配分B1-OLへの優劣ではない。

## 現行controlについて確認できたこと

- 同じh1内で特徴が実際に変化し、既存3-controlの層内定数という制約を越えた。
- Actual Qから条件付き平均・二次momentを厳密に得て、同じfinite signalを保持する中心化を実装・接続できた。
- path metadataのfragment lookupでfeatureを取得できる。feature取得のための追加量子測定は不要だが、**係数学習のfresh bitは有料**。
- rank・range・独立学習/fresh評価・paired合成・全5層と全費用を、一つの固定条件で完遂できた。
- この具体的な群と学習結果では、追加削減と学習費回収の積極的な証拠は得られなかった。

最後の項目は「SCV全般が無効」「fragment群族全体が無効」「真の相関が0」の証明ではない。

## Balance、相関、学習精度は別

採用規則は正質量fragmentの非空proper subsetから、ν₁∈[1/8,7/8]かつ1/2最近接を選び、同点をcanonical整数maskで決めるもの。
**Qの質量だけを使い、測定bitとの相関を直接最適化していない。** 群の非定数性と平均の扱いやすさを確保しても、応答の説明力は未知だった。
この点から「別の群なら改善する」とも推論できない。別群の探索・旧bitへの再fitは今回していない。

rank2はfit可能性の検査で、係数精度の保証ではない。特にYのu=0が2件という記録は不確かさを残す。
X/Yは別応答であり、一方の係数・小さな点差を他方や別roundへ移植できない。
元Hの状態・回路・phaseで決まるpath平均変動のうち、metadata controlで説明できる部分だけが削減対象で、固定pathのBornノイズは残る。[既存設計][design]

## 同じcontrolに追加検証する場合の意味

| 解消したい疑問 | 別途許可された独立取得で確認し得ること | それだけでは得られないこと |
|---|---|---|
| 凍結済み係数の効果が今回の少数評価に依存するか | 同じS・λに対するfresh評価で点差の再現性・不確かさを調べる | 不正確な学習係数の修復、最適係数や別群の性能 |
| 学習の低出現countが結果を不安定にしたか | 別の有限学習契約で係数品質・rank/rangeの再現性を問う | 単一fitから再学習policy全体の期待性能を確定すること |
| 分散差が精度保証付きshot削減になるか | 独立calとfresh評価の別設計でactual残差の上界・十分整数Nを問う | 現224評価を遡及的にcalと呼ぶこと、期待単価の保証 |

追加量を増やすだけでは、群に相関が乏しい場合の説明力は増えない。追加train/calにもfull interrogation費がかかる。
保存λを固定利用する単発deploymentと、trainから再学習するpolicyを分け、過去学習費・償却の扱いを明示する必要がある。[費用接続][cost]
いずれも旧STOP済みcampaignへの追加入力ではなく、新しい有限scope・独立性・費用境界・STOPと明示許可が必要。shot数や実行条件は本書で設定しない。

[result]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/REPORT.md
[summary]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/RESULT_SUMMARY.json
[execution]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/ba483ab3038693048e58830e39a676493a892053/docs/research/o2_scv224_science_20261107/EXECUTION.md
[design]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/84e021b6a8decde7fa3ed3fa44d7148cc5428061/docs/research/intrastratum_cv_design_20261107/STRATIFIED_CV_DESIGN.md
[confidence]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107/SCV_FINITE_CONFIDENCE_MAP.md
[cost]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107/SCV_TOTAL_COST_COMPARISON.md
