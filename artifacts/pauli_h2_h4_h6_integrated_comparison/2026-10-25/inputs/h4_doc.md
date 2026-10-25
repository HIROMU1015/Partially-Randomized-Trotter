# H4 Pauli-PRパラメータ探索と最良決定論的PFの比較

2026-10-25。**固定H4の141登録PR構成を探索し、事前に凍結した4候補を別seedで確認した。5要求精度すべてで、探索nomineeの独立確認による条件付きG点推定が、保存106構成内の最良B0/B0Iおよびfull PFを下回った。**

ε=.05における、以前の固定PRに対するB0約12.9%優位は、このパラメータ探索後の有限標本比較では維持されなかった。新PR側の点費用はB0より約48.42%低い。
これは仮想ρ=0・有限標本価格の条件付き結果であり、母平均winner、全パラメータ最適性、実compiled精度や無条件end-to-end優位の保証ではない。主要claimの正式採択は行っていない。

## 1. 固定条件・来歴

Linear H4、隣接1.00 Å、STO-3G、Pauli-JW184非identity項＋constant、abs(c)降順/label tie-break、保存raw state/ray、同じfull-H reference、S2/T=.8。
system8/ancilla8、D2 central-rotation control、cosine/+Im sine、状態準備を除くfull Hadamard wrapperのcompiled RZ×shot。
Qiskit1.3.0、NumPy1.26.4、rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing指定なし。
Arbは既存python-flint0.9.0環境を読取再利用した。

- 基点：`codex/non-df-h4-pauli-pr-search-preparation-20261025`、`351dd5fa15c65feb3c14738a933145fc62d2be2c`。
- 専用branch：`codex/non-df-h4-pauli-pr-search-execution-20261025`。
- 初期科学worker SOURCE：`52865921930b3afd11b1447dff0a6cbcb6085af8`。
- 選択・paired統計SOURCE：`4a7f91fa7ba8aa3ae1ce4ccb30bde9b973869410`。
- 最終比較SOURCE：`a17ad381b6ff2ac615e3cd930c641947c551bcbc`。後続workerも同じ数値/compile codeを保持してこのHEADへ接続した。

[準備報告](non_df_h4_pauli_pr_search_preparation_20261025.md)、[active科学範囲](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/active_scope.json)、[入力参照](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/input_references.json)。ユーザーが一括実行を明示承認した後に開始し、grantや承認証明機構は作成していない。

決定論的側は[保存106構成](non_df_h4_pauli_b0_comparison_20261025.md)の最良G・bias・単価を引用し、再計算/compileしていない。Pauli B0/B0I同等性もそのまま。旧PR prefix27/q8/r4/K4の256 paired cohortは歴史的baselineとして分離した。

## 2. 登録候補と段階的な選択

初期は100構成：prefix {0,8,16,27,36,37,64,96,128,176}×q{1,2,4,8}/r4/K4の40、同prefix×q{1,4,8}/r4/K2の30、prefix{8,27,36,96,176}×q{1,4,8}/r{2,8}/K4の30。
低保持、Z/ZZ境界36/37、中間、full近傍を含め、巨大な全直積は採っていない。

初期上位2/εから、事前規則のprefix±1/4/8・隣接q/r・K切替を一座標ずつ追加。round1は重複除去後30、1%以上の改善によりround2へ進み11追加、**100＋30＋11＝141構成**で候補追加を終了した。
2roundともproposal queueを使い切った範囲であり、許可最大32件を埋めるために別の規則を発明していない。未登録セルの大域探索は行わない。

各batchは対象/理由/直前結果SHA・CODE SOURCE/input/runtime/call上限を実行前に固定した。[各planとnativeを含むデータ入口](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/README.md)。初期と事後選択した追加を[candidate catalog](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/candidate_catalog.json)で区別する。

全候補のうち保存PR1候補を再利用し、新しいlaw-mean状態作用は140回、**全て64bit**。96/128bitや新reference作用は不要だった。

| ε_sig | 仮想ρ=0でguard適格 | bias不適格 | 数値未確定 |
|---|---:|---:|---:|
| 0.05 | 141 | 0 | 0 |
| 0.01 | 140 | 1 | 0 |
| 0.005 | 112 | 29 | 0 |
| 0.001 | 90 | 51 | 0 |
| 0.0005 | 83 | 58 | 0 |

これは固定軸配分・u/h guardでの適格性であり、他estimatorでの不可能性ではない。不適格/nullを0や∞へ読み替えていない。

探索価格は新候補ごと8 paired trajectoryから開始し、事前規則で選んだ6候補だけ64本へ追加、**新探索計1,456本**。
その後4 finalistのIDと各εのprimary/challengerを[確認前に凍結](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/confirmation_nominations.json)した。
p36/q2/r8/K2・K4は別seed各48本、p0/q8/r8/K2・K4は別seed各384本。後者のみ、48→96→192時点の僅差/相対nominal半幅条件により上限まで進んだ。確認計864本。探索とのpooling・確認後の候補差替えは0。

## 3. 探索で選んだ候補の独立確認：主結果

主表は**探索nomineeを固定したまま、独立確認標本だけでGを推定**したもの。単位RZ×shot。Nは十分条件の会計値で、実量子shotを実行した値ではない。

| ε_sig | PR prefix/q/r/K | 確認n | PR Ghat | 保存最良決定論的G | 対決定論的削減 | 対旧固定PR削減 |
|---|---|---:|---:|---:|---:|---:|
| 0.05 | 36/2/8/2 | 48 | 11,986,566.08 | 23,237,760 | 48.42% | 55.06% |
| 0.01 | 0/8/8/4 | 384 | 315,105,056.69 | 1,440,845,160 | 78.13% | 53.32% |
| 0.005 | 0/8/8/4 | 384 | 1,260,419,576.24 | 9,734,669,448 | 87.05% | 54.03% |
| 0.001 | 0/8/8/4 | 384 | 31,510,478,997.70 | 579,283,719,324 | 94.56% | 59.89% |
| 0.0005 | 0/8/8/4 | 384 | 126,041,917,942.39 | 2,663,993,522,484 | 95.27% | 67.45% |

出典：[best_by_precision.csv](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/best_by_precision.csv)、[結果JSONのepsilon_results](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/result_summary.json)。exact fraction・全cohortの行は[725行CSV](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/candidate_precision_table.csv)。

保存最良決定論的は全5点でtruncated B0/B0I。prefix/qは精度順に132/1、144/2、176/2、180/8、180/8。保存full PFの最良Gはそれぞれ32,680,930／1,723,689,280／10,167,897,280／615,615,107,120／2,982,596,955,988で、PR主候補はこれらも下回った。

### 3.1 回路単価とshotの内訳

| ε_sig | PR平均RZ/軸 | B_saved | N_cos | N_sin | paired G SE | nominal t95半幅 |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 429.041666667 | 1.383368202398 | 14,376 | 13,562 | 38,028.53 | 76,503.54 |
| 0.01 | 325.263020833 | 1.662369839186 | 484,385 | 484,385 | 1,216,699.23 | 2,392,246.28 |
| 0.005 | 325.263020833 | 1.662369839186 | 1,937,539 | 1,937,539 | 4,866,794.41 | 9,568,980.17 |
| 0.001 | 325.263020833 | 1.662369839186 | 48,438,458 | 48,438,460 | 121,669,820.12 | 239,224,425.17 |
| 0.0005 | 325.263020833 | 1.662369839186 | 193,753,833 | 193,753,845 | 486,679,288.01 | 956,897,715.49 |

両軸RZの等値は今回取得した当該cohortでの観測であり、一般に仮定していない。GのSEは同じtrajectoryの両軸を組み合わせ、軸間共分散を保持して計算した。
PRはnormalizationによるshot増を負担する一方、回路単価の低下がこの比較でそれを上回る。

p36は全Z/ZZを保持する部分ランダム化候補。p0はconstantだけを決定論的に保持し、全184非identity項をRTEへ置く**全ランダム化の境界候補**である。保持部が空なので、その場合に非自明なS2 retained誤差があるとは述べない。prefix0は承認済み初期集合に含まれ、アルゴリズムを新しく変更した結果ではない。

### 3.2 確認標本内で事後に最小を取った場合

| ε_sig | 確認集合内の点最小 | Ghat | 主表との差 |
|---|---|---:|---:|
| 0.05 | p036_q2_r8_K4 | 11,986,442.25 | 123.83 |
| 0.01 | p000_q8_r8_K2 | 315,076,782.82 | 28,273.87 |
| 0.005 | p000_q8_r8_K4 | 1,260,419,576.24 | 0.00 |
| 0.001 | p000_q8_r8_K4 | 31,510,478,997.70 | 0.00 |
| 0.0005 | p000_q8_r8_K4 | 126,041,917,942.39 | 0.00 |

.05ではK4、.01ではK2が確認標本の点最小になり、探索nomineeと異なる。ただし差はpaired SEよりはるかに小さい。**K2/K4の一意な母平均順位は確定しない。** 最小の見かけだけを主表へ差し替えず、主表の独立確認と確認後選抜を分離した。これらは凍結4候補内の比較で、全141候補の母平均winner保証ではない。

## 4. 会計・normalization・ρ guard

元の `G=N_cos*C_cos+N_sin*C_sin`、ε_axis=float(ε)/√2、α_axis=.025、`h=ε_axis−|b0|−(u_enc+ρ)>0`、`(u_enc+ρ)/h≤.01` を維持した。
各候補のeffective law mean/bias、実binary64 B_saved、外向きln(B_saved)、Arb区間半幅u_encからNとguardを計算する。別候補のbias/B/N/guardは転用しない。
初期表示は仮想ρ=0。実compiledρが0だと確認したものではない。

保存reference全区間、同じraw state/ray・exact norm、retained exact operand、native scalar、signed product/rotation/order phaseを維持した。B_savedは一度だけ適用。NumPy1.26.4のCDF/53bit uniform-wordモデルのeffective確率に基づく平均で、IID仮定は明記し、PCG64独立性を証明していない。

保存p27 descriptorの角度・CDF・B・scalarと新接続の対応はPASS。旧科学計算を再実行せず、保存Cのwhole physical-bias intervalを新会計へ接続した。
旧log margin・旧bias中心/uを含むN/Gはそのまま引用する一方、**common-policy固定PRの保存値再会計**を別列にした。純粋なパラメータ改善は`PR_over_fixed_common_policy_exact`、旧会計込みの比較は`PR_over_fixed_legacy_exact`で区別できる。旧Gを上書きしたものではない。

全5点で、保存/取得cohort平均価格を固定した独立ρ guard box全域でもPR主候補が最良決定論的より低Gだった。これは実ρのguard所属、PR母平均価格、実機精度の保証を含まない。旧6件の局所Arb保証は新パラメータの全lawρへ転用していない。

## 5. 再利用・数値検証・高次event

`PauliSignalAdapter`/audited adapterとsampler、D2 control、Hadamard wrapper、compiled cost cache、CのCDF/grid mass/Arb primitives、Pauli map、B0比較のN/G/finite selection、既存paired統計を再利用した。
新しい[batch runner](../scripts/run_h4_pauli_pr_search.py)、[保存値比較](../scripts/summarize_h4_pauli_pr_search.py)、[最終主表](../scripts/finalize_h4_pauli_pr_search.py)はこの研究に必要な接続だけを担い、旧固定runner/契約へ条件を上書きしていない。

Arb meanは同じ有限lawの凸unitary平均をFLINTでvector作用し、contraction norm≤1とL2 recenterで数値誤差を包含する。保持項/phaseを省略した別のsignalではない。
高速化後の人工Arb4件・既存接続6件、production price人工1件、finite選択/paired covariance人工3件がPASS。192bit dense C oracleとの包含比較は小系だけ。実H4に192bitを機械的に適用していない。作者の技術確認であり、別担当の独立レビューやimmutable CIとは呼ばない。

探索で観測したeventはorder0=37,451／order2=444／order4=9。独立確認はorder0=50,460／order2=228／order4=0。K2はlawの定義上order4を持たず、K4の確認cohortではorder4未観測である。
**未観測order4/componentの費用寄与を0とはしていない。** Ideal-effective meanはallowed全orderを含むが、価格は有限MC平均。SE・nominal Student-t95・384本までの確認によって母平均tail保証が得られたとはしない。逐次観察/候補選抜込みの同時coverageも主張しない。

固定p27の[母平均回路価格に関する限定scalar結果](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/07d4f6ff7f1df085f292d36c36e460354df8a740/docs/non_df_h4_pr_mean_cost_scalar_execution_20261025.md)の上界・母平均順位は、この新しいパラメータ集合へ転用していない。

## 6. 終了・保全・残る範囲

新mean状態作用140、paired trajectory2,320（探索1,456＋確認864）、body/central lift各2,320、wrapper/compile各4,640。
11 batch親、2,464 direct childは全てnative exit0・wait/reap。失敗科学call/retry/旧run再開0。新SCF/FCI/DF、Hamiltonian/state/full-H reference生成、科学grant0。

[主agent検証](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/verification.json)は共通raw norm、保存N/G恒等式、seed分離、2,320 event SHA、代表compiled QPY288件、旧入力12 SHAとnative終了を確認した。旧SOURCE/raw/結果/grant/ledger/marker/STOPを保全し、今回の正常終了で元H4の失われたnative証拠を代替していない。
独立候補/trajectoryを実affinity64に応じprocess並列化し、共有環境・他job・venvを変更していない。

raw・native・代表QPY・段階別解析をlossless archiveへ保存し、[data_archives.json](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/data_archives.json)でmember SHAを辿れる。価格全2,320行は[trajectory_prices.csv](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/trajectory_prices.csv)。QPYは各候補/cohort index0の両軸input/outputを保存し、他indexはcompiled byte SHAと費用・event・seed・SOURCEを保存した。

今回答えられるのは、**登録Pauli-PR familyを探索すると、固定旧PRだけの比較とは異なり、5精度で最良保存決定論的PFより低い条件付きG点推定を独立標本でも得られる**こと。ε=.05の旧B0優位はこの有限比較では維持されない。
一意なK最適性、全prefix/q/r/K最適性、母平均Gの厳密順位、実ρ/全law実精度、正式certificate、別geometryや全RPE費用は未確定。303点の新gridは中心5点の結論に不要なため実行していない。

[STUDY_STOP](../artifacts/non_df_h4_pauli_pr_search/2026-10-25/STUDY_STOP.json)：**mandatory STOP**。結果保存・commit/公開・GitHub remote SHA照合で終了する。最終commitとremote SHAは最終チャットに示す。
次の担当は**ユーザー**：この条件付きempirical比較の科学的解釈・採否を判断する。**Codex**は追加科学計算、PR最適化の拡張、正式certificate/主要claim採択、GPT詳細レビューへ自動移行しない。
