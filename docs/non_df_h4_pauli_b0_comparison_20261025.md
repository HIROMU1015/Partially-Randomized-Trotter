# H4 Pauli B0/B0I：有限候補探索と条件付き総資源比較

2026-10-25。**固定H4・登録106構成では、ε_sig=.05の最良B0/B0Iはprefix132/q1で、保存Pauli central PRより条件付きGが約12.9%低かった。ε=.01/.005/.001/.0005ではPRが低かった。**
保存平均価格を固定し、各実装残差ρが指定guardに入ると仮定した比較である。
実compiled精度、PR母平均費用tail、全prefixの大域最適性、正式certificateは未保証のまま。
科学的な主要claimの正式採択・重要レビューは実施していない。

基点 `codex/non-df-h4-pauli-b0-execution-20261025` / `6ba7d7e2ee0ebc4ff246596e5ecae358c74b6278`。
専用branchは `codex/non-df-h4-pauli-b0-comparison-20261025`。

| 役割 | 固定SOURCE |
|---|---|
| 科学worker・初期候補・数値/compile/accounting | `47384f5698094e466e83027a5a54751e0cf6000e` |
| 保存値の有限集合選択・303点grid | `8ab472db8f90d6f64f6cc626584655c5a7487a56` |
| 再利用Cのfull-H参照・full PF target | `e80d7510880f8db993b777c627bb8e95b05a070d` |
| 再利用prefix27/q8 | `268e7e9b242196fac46aba1101a513c5e3cfb645` |
| 保存central価格 | D2 `74515d8eb4f073c40fc783681a3775dbca89e9b1`、A_AXES `a37232f85812688c3b9ab957c5073c5ce143b17e` |
| 保存PR条件付き会計 | `d96dfe5ca0a6c950a5f1791ff61659e6dc8f1c34` |

## 1. 固定した比較条件

Linear H4、隣接1.00 Å/STO-3G、保存Pauli-JW184非identity項＋constant、保存abs係数降順/label順を維持。
同じraw state/ray、同じ保存full-H参照**全区間**、共通exact norm、Qiskit little endian/system0–7/ancilla8。
S2、T=.8、q=1/2/4/8。B0は保持prefix以外の非identity項を切り捨て、constantを常に一度保持する。
このSOURCEではB0/B0Iは同じ候補であり、二重実行していない。prefix184はfull PFとして同じregistryの1候補とする。
PRは保存prefix27/q8/r4/K4・256 trajectoryを固定し、新しいPR sampling・compile・パラメータ探索を行っていない。

D2 central-rotation control、cosine/+Im sineのfull Hadamard wrapper、state preparation費用を除くcompiled RZ×shot。
Qiskit1.3.0、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。
他geometry、H6/H8、DF数値データ、新しいPRアルゴリズムを対象に含めていない。

## 2. 候補集合と適応的追加の理由

初期は **prefix {27,36,37,128,184} × q {1,2,4,8} =20構成**。
prefix36まではZ/ZZのみで、37項目が最初の非対角項 `XXYYIIII`。
27は元の失敗baseline、36/37は非対角項の境界、128は中間保持、184はfull PFの端点である。
DFのfragment番号をPauli項数へ機械的に対応させていない。

| batch | 構成数 | 事前に固定した追加理由 | 新target作用 | 新compile |
|---|---:|---|---:|---:|
| 01 | 20 | 初期代表候補。保存値・対角同値性の再利用を含む | 12 | 14 |
| 02 | 24 | prefix64/96/112/144/160/176 × 全q。初期の.05費用僅差と中間/near-fullの空白を確認 | 24 | 40 |
| 03 | 28 | 精度ごとの最小候補近傍。q1/q2の132–152、q2/q4/q8の168–183を重点化 | 28 | 56 |
| 04 | 34 | 最小候補の整数prefix近傍を補完。q1の129–147、q2の141–147/173–183、q4の179/181、q8の173/174/175/177 | 34 | 68 |
| 合計 | **106** | 初期20＋事後追加86 | **98** | **178** |

各batchの対象・SOURCE・入力hash・call上限・終了条件を、当該batchの実行前に固定した。
追加理由と直前batch結果のSHAを各planに保存し、過去runは再開していない。
局所追加後も5精度の最小候補は変わらず、同じqでの隣接prefixも確認できたため候補追加を終えた。
これは探索域全体の数学的な最適性証明ではない。

評価したprefixのunionは `{27,36,37,64,96,112,128–148,152,160,168,172–184}` の43種類。
全prefix×全qの直積ではない。[membership表](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/candidate_membership.csv)の106セルだけが登録済みである。
`NOT_EVALUATED` と精度不適格/nullを混同しない。
[batch01](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/batch01/plan.json)、
[batch02](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/batch02/plan.json)、
[batch03](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/batch03/plan.json)、
[batch04](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/batch04/plan.json)が正確な構成一覧である。

## 3. 既存資産と計算の再利用

`PauliSignalAdapter`/`h4_pauli_b0` のexplicit L_D、保持S2とconstant更新、D2制御、既存compiler/cost関数を使用した。
旧pilotのL_D=None経路を呼ばず、Hamiltonianは保存係数descriptorから復元しただけで、SCF/FCI/JW変換・state生成は行っていない。
新規接続は [comparison library](../src/trotterlib/h4_pauli_b0_comparison.py)、
[batch runner](../scripts/run_h4_pauli_b0_comparison.py)、[saved-value summary](../scripts/summarize_h4_pauli_b0_comparison.py)。

数値評価は前段のB0 target、Cのexact binary64 operand/phase/区間、PR0のexact Pauli作用、Arbのunitary L2 recenterを再利用した。
初期64 bitで全候補の判定が定まり、96/128 bitは未使用。数値allowanceは各bias包含区間のexact半幅で、任意のroundoff予算を置いていない。
対角prefixの理想保持項は可換で、T/qが2の冪でのexact除算となるため、q間の同じtargetを再利用した。
この恒等式をcompiled費用のq独立性とは扱っていない。

- prefix27の全qのtargetは保存q8の包含を再利用し、q8の868 RZも再compileしていない。他qの価格は不適格のため未取得。
- prefix36はq1の新targetをq2/4/8へ転送した。全qが指定guardで不適格だったのでcompileを省略した。
- full PF/q8のtargetは保存C **nominal PF raw signed-bias全区間**を共通exact νで割って再利用した。rounded native scalar phaseへの置換はしていない。
- full PF/q8両軸15028 RZは保存D2/A_AXESを再利用した。PR価格・5精度会計は保存値の引用である。

DF研究からは有限登録集合ごとの最小選択、null適格性、保存値gridという比較の構成を参照した。
DFのpure `grid_segments` を再利用し、gridの隣接bracketを示す。DFのgeometry/rank/価格・identity位相補正・旧承認runtimeは転用していない。

## 4. 適格性と会計

元と同じ `ε_axis=float(ε_sig)/sqrt(2)`、α_axis=.025、`u=u_enc+ρ`、
`h=ε_axis−|b0|−u`、`h>0`、`u/h≤.01`、外向きのHoeffding shot十分条件を使用する。
適格性はこの**固定軸配分/guard**に関するもの。他の配分・estimatorの不可能性を主張しない。
数値不確定・biasによる不適格・未compileを別の状態として記録した。

| ε_sig | 条件付きguard適格 | bias不適格 | 数値判定未確定 |
|---|---:|---:|---:|
| .05 | 90 | 16 | 0 |
| .01 | 69 | 37 | 0 |
| .005 | 52 | 54 | 0 |
| .001 | 26 | 80 | 0 |
| .0005 | 21 | 85 | 0 |

unitary決定論的候補の数学的B=1/log_B=0を用い、会計は既存 `scale_bounds(1,0)` の外向きB_upを記録した。
PRの保存normalization/数値allowance/guardは変更していない。
full PF/q8についても新しい共通unitary会計を別結果として適用し、旧log marginを含む歴史的PF会計は`historical_PF_q8_accounting`へそのまま残した。
旧契約・旧N/Gは書き換えていない。いずれの会計も実compiledρの保証を与えない。

## 5. 要求5精度での登録集合最良候補

以下のGは **仮想ρ=0、保存PR標本平均価格、状態準備除外**でのRZ×shot。
両軸価格が今回同値だった候補では `G=C_RZ×N_total`。最小選択にはexact有理数を使った。

| ε_sig | 最良B0/B0I prefix/q | RZ/軸 | N_total | B0 G | 最良full PF prefix/q：G | 保存PR G |
|---|---|---:|---:|---:|---|---:|
| .05 | 132/1 | 1248 | 18620 | 23,237,760 | 184/1：32,680,930 | 26,673,154.3984375 |
| .01 | 144/2 | 2820 | 510938 | 1,440,845,160 | 184/2：1,723,689,280 | 674,975,346.234375 |
| .005 | 176/2 | 3576 | 2722223 | 9,734,669,448 | 184/2：10,167,897,280 | 2,742,116,862.3984375 |
| .001 | 180/8 | 14676 | 39471499 | 579,283,719,324 | 184/8：615,615,107,120 | 78,560,515,459.78125 |
| .0005 | 180/8 | 14676 | 181520409 | 2,663,993,522,484 | 184/8：2,982,596,955,988 | 387,166,920,004.71875 |

最良決定論的PFをtruncated/fullのunionから選んでも、5条件では全て上のtruncated B0が選ばれた。
[best_by_precision.csv](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/best_by_precision.csv)、
[530行の全候補×精度表](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/candidate_precision_table.csv)、
[registry/信号bias/数値allowance/価格](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/candidate_registry.json)。
G=nullの候補を0・∞で補わず、最小選択から除外した。

| ε_sig | B0 G / PR G | 低Gの方式 | 独立ρ guard内のfamily最小比較 |
|---|---:|---|---|
| .05 | 約0.8712 | B0/B0I（約12.9%低い） | B0側が低い |
| .01 | 約2.1347 | PR | PR側が低い |
| .005 | 約3.5501 | PR | PR側が低い |
| .001 | 約7.3737 | PR | PR側が低い |
| .0005 | 約6.8807 | PR | PR側が低い |

全5条件で、保存平均価格を固定した独立guard全域でも**方式間**の順位が保たれた。
family最小のguard区間は、登録済み適格候補のG下端の最小とG上端の最小で構成する。
guard内で選ばれる個別prefix/qまで常に同一とは主張しない。実ρのguard所属も未保証である。

ε=.05ではB0の1248 RZ/shotは保存PR平均1140.8046875より大きいが、B=1のshot負担によってGが小さくなった。
元prefix27/q8の不適格結果は保持し、今回の他候補の結果と区別する。

## 6. 保存値だけを使った303点精度grid

301対数点 `.0005–.05` に元の5精度を合わせ、数値重複を除いた303点。
新しい分子・状態計算ではなく、保存bias/allowance/価格を使う会計である。
元5精度のPR N/Gは再計算せず引用し、追加精度だけ固定PR候補の保存値から会計した。
[解析plan](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/analysis_plan.json)、
[precision_grid.csv](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/precision_grid.csv)。

- 仮想ρ=0：PR低Gは283点（.0005〜.036782112723）、B0低Gは20点（.0373510949464〜.05）。
- 独立guard全域でPR低Gの点は280点、B0低Gは18点。
- .0356700187536〜.0379288787515の5点はguard区間が重なり、その範囲の順位を確定していない。
- 連続精度上の厳密な交点、303個の独立実験、全条件同時95%保証ではない。

## 7. 成立範囲と残る不確かさ

この有限登録集合と固定compiler/rayでは、要求精度によってB0/PRの条件付きG順位が異なることを確認した。
適格性・費用の候補依存を調べるための、DFと同種の有限候補比較は今回の範囲で完了した。
追加計算で候補集合が広がれば最小候補・grid境界は変わり得る。

1. compiled測定残差ρは未評価。Arbが包含したのはideal C-targetとfull-Hのbiasで、実compiled精度ではない。
2. PR価格は固定256本のpaired標本平均。元のSE/nominal CIを母平均tail保証へ格上げしない。
3. B0は106候補から選び、PRは固定候補を再利用した。PRまで最適化した対称的family最適化や一般的アルゴリズム優越性ではない。
4. 未登録prefix/qセル、他geometry・分子・state・compiler・stateprep/noise・全RPE費用へ外挿しない。
5. 各精度のα_axis=.025は当該条件での配分で、grid全体の同時保証ではない。

## 8. 技術検証・実行終了・保全

新人工5種類を両runtimeで10実行PASS。順序・exact T/q、保存PF区間とnorm換算、null除外、finite-family最小/同費用、guard重複を検査した。
既存B0/C/Arb/D2の固定実装・人工証拠を再利用し、独立科学レビューと呼んでいない。
全4batchの親engineと計365 childはnative exit0・wait/reap済み。解析親もexit0・回収済み。
新target作用98は全て64 bit、body/control89、両軸wrapper178、compile178。
失敗・retry・旧run再開・新科学grant・PR sampling/compile・新full-H参照/状態生成は0。

独立task数と実CPU affinity64に応じたprocess並列を使い、CPU/RAMの人工上限、共有設定・環境・他ユーザー操作は行っていない。
実行時間はbatch約17.9/24.1/26.7/25.9秒、保存値解析約10.9秒。準備・保存・公開時間は含まない。
旧入力/証拠22fileと旧SOURCE・raw・grant・ledger・marker・STOPを保全した。

raw JSON、native記録、lossless圧縮QPYはbatchごとのarchiveへ保存する。
archiveの各memberは元ファイルとSHAで照合でき、QPY gzipは元bytesのhashも保持する。
[データ入口とarchiveの使い方](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/README.md)、
[結果要約](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/result_summary.json)、
[保存検証](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/verification.json)、
[commit/remote状態](../artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/publication_status.json)。

**mandatory STOP。** 今回の範囲で追加作業なし。
次の担当は**ユーザー**：この条件付き有限集合比較を研究上どう扱うか判断する。
**Codex**は保存・公開で終了し、重要claimの採択、GPT詳細レビュー、追加物理条件・PR最適化へ自動移行しない。
