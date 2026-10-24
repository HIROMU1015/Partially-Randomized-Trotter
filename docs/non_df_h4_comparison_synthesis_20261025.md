# H4 Pauli-JW／DF-native PF・PR比較研究：既存証拠の再整理

2026-10-25 JST。**固定H4・固定実装・保存cohortの回路価格比較と、残差を条件とするC/N/G比較は揃っている。この範囲のcase studyを報告するために、追加科学計算や全compiler数値保証は必須ではない。** 一方、実際の精度を保証したPR母平均Gの優位性は、実装残差の転送と費用tailが未確定であるため完成していない。

今回の重要な照合結果は、初回D1の停止に加え、**後続のSOURCE基準BによるD1が9構成の測定まで完了している**ことである。保持/event価格を一律「未測定」とする旧段階の要約は、現時点の全証拠を表さない。ただしbasis単独やcompiler passの一意な寄与率は未測定のままで、推定で補わない。

本資料は掲載候補となる主張と判断材料の整理であり、新しい科学的再解析・再ランキング、研究方針の採択、重要な最終科学レビュー、正式certificate受理ではない。

## 1. 固定条件と証拠の来歴

基点branch `codex/non-df-h4-r2c-compiler-phase-proof-20261025`、commit `c45247deb9992d08594ad74b1114e6fac505292f`。local HEADとGitHub上の同commitを照合した。今回の専用branchは `codex/non-df-h4-comparison-synthesis-20261025`。

Linear H4、隣接1.00 Å、STO-3G、system8＋ancilla8、S2、T=.8、q8、r4、K4、ε_sig=.05/.01/.005/.001/.0005、cosine/sine。費用は**状態準備を除くfull Hadamard wrapperのcompiled RZ数**で、CX/depthは補助指標。Qiskit1.3.0、basis `rz/sx/x/cx`、optimization_level=1、seed17、backend/coupling/layout/routing指定なし。

[入力receipt](../artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json)は、同じMO積分からのPauli/DF生成、DF tol-only1e-8/rank10、`df_truncation_value=0.0`、full-H一致診断error `8.462237051574415e-15`（閾値1e-9）を保存する。これは数値的一致診断で、厳密同一の証明ではない。Cの両表現modelは同じraw state hash `8339b1a8b0bec35ae07688e667fde7ba602d56c1182523d58e0889ea114e8b80`を持つ。DF独立state診断のhashを、この共通rayと混同しない。

| 役割 | 固定科学SOURCE | 今回の扱い |
|---|---|---|
| 元H4、generic Pauli／DF-native | `0c329b30a81accb35c212be8defa66fee0dd0f12` | 価格・bias・descriptorの引用 |
| Cの包含評価 | `e80d7510880f8db993b777c627bb8e95b05a070d` | 保存encoded bias/normalization証拠 |
| 初回D1 | `48d4d27a05a529e05b11599f5f3c671302826617` | fingerprint FAILと停止を保持 |
| D1 SOURCE基準B | `9843b803878db6c3aee7ab5e7c6bb4818572c944` | 保存9構成の測定結果 |
| D2 central対照 | `74515d8eb4f073c40fc783681a3775dbca89e9b1` | 同じ保存Pauli bodyの2対照 |
| A_COHORT_COS／A_AXES | `34ef0a7ca2412a61f5215a4ce1192e99f3226b2a`／`a37232f85812688c3b9ab957c5073c5ce143b17e` | 保存256本の両軸価格 |
| generic条件付き会計／central会計 | `b1984dcde06a874d0fd6e9fe6395b7f3fb7504d8`／`d96dfe5ca0a6c950a5f1791ff61659e6dc8f1c34` | 既存N/G・guard領域の引用のみ |

各主張の保存field・ファイルSHAは[引用証拠JSON](../artifacts/non_df_h4_comparison_synthesis/2026-10-25/evidence_quotes.json)。本資料の表に使った[回路価格6系列](../artifacts/non_df_h4_comparison_synthesis/2026-10-25/circuit_costs_QUOTED.csv)と[仮想ρ=0の30行](../artifacts/non_df_h4_comparison_synthesis/2026-10-25/conditional_resources_QUOTED.csv)には、元JSONのpointerを付けた。新しい平均・SE・N・G・確率・順位は計算していない。

## 2. 4方式の比較：科学モデルと回路経路

Pauliのgenericとcentralは**別の実装系列**であり、旧genericの価格を上書きしない。

| 項目 | Pauli-JW PF | Pauli-JW PR | DF-native PF | DF-native PR |
|---|---|---|---|---|
| Hamiltonian単位 | 184非identity Pauli項＋constant | 同じ184項 | one-body＋10二乗fragment＋scalar | 同じDF表現 |
| 保持部 | 全184項 | prefix27、すべてZ/ZZ＋constant | one-body＋全10fragment | one-body＋fragment0、tail identityもphaseへ抽出 |
| prefix基準 | 全保持 | abs(c)降順の50%累積 | 全保持 | abs(lambda)×Frobenius norm²の50%累積、one-bodyは別枠 |
| random law | なし | tail Pauli、k=0/2/4のfinite paired Taylor | なし | tail rotated Z/ZZ involution、k=0/2/4 |
| 各step | 全項forward/reverse half | 保持half→4event→保持reverse half | one-body/全fragmentのhalf列 | 保持half→4event→保持reverse half |
| 制御化 | generic whole-body、またはD2 central rotation | 同左。productも発生順に制御 | Gaussian basisは非制御、central diagonalとphaseを制御 | basisは非制御、central Z/ZZ product/rotationとphaseを制御 |
| basis再利用 | Pauli local basis/parity、最終compiler依存 | Z/ZZ保持はlocal basis変更不要 | fragment内のdiagonalを同じbasisで集約 | occurrence内の隣接equal-basis reuseあり。step間はraw concatenation |
| 保存B | 1 | 1.2873709223899619 | 1 | 約1.17758514 |
| 費用標本 | singleton | 保存256 trajectory | singleton | 別lawの保存256 trajectory |

Pauli27対DF1というprefix数や「50%」を、同じ物理的分割・同じ誤差・同じ保持費用と解釈しない。同じfull Hに近くてもS2分解とfinite-RTE lawが異なる。Pauli/DFの同じtrajectory indexもpaired標本ではない。Pauli generic/centralの同じ保存trajectoryはpaired対照として対応する。

SOURCE対応は `h4_molecular_inputs.resolve_prefixes`、`non_df_h4_readiness.structural_prefix`、`pauli_signal_adapter.append_pauli_event`、`df_rte_qiskit.DFRTECircuitBuilder._build`、`df_partial_s2_repeated`、`h4_d2_pauli_control`。数値モデルや回路をこれらの関数から新生成してはいない。

## 3. 回路単価の完成範囲

### 3.1 Full-wrapper RZ：保存cohortの結果

| 系列 | PF RZ/shot | PR平均RZ/shot（n=256） | PR SE | PR nominal t95半幅 |
|---|---:|---:|---:|---:|
| Pauli generic | 255,500 | 9,321.1015625 | 13.847523668697294 | 27.270074727002452 |
| Pauli central | 15,028 | 1,140.8046875 | 1.059925039945424 | 2.0873215844123516 |
| DF native | 135,355 | 23,678.44140625 | 46.718033123081106 | 92.00231643184831 |

出典：[generic/DF cost summary](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/cost_summary.json)、[A_AXES statistics](../artifacts/non_df_h4_a_axes/2026-10-24/run01/statistics.json)。両軸のRZは保存cohortで同値。PFにはMC標本誤差がなく、PRのnominal t95は母平均の厳密tail上界ではない。

旧genericではPFはDFが低単価、PRはPauliが低単価という逆転があった。central後は対応するPF/PRの双方でPauliがDFより低単価となる。これは既存の固定実装/cohort比較であり、表現一般の順位を新しく決定したものではない。

### 3.2 CX/depth：singletonとPR index0の保存測定

| full cosine回路 | RZ | CX | depth | rz_depth | cx_depth |
|---|---:|---:|---:|---:|---:|
| Pauli generic PF | 255500 | 164096 | 358563 | 160288 | 150752 |
| Pauli generic PR index0 | 9652 | 6872 | 12973 | 5918 | 6494 |
| DF PF | 135355 | 78628 | 134134 | 61312 | 67164 |
| DF PR index0 | 23021 | 11134 | 18306 | 8411 | 8328 |
| Pauli central PF | 15028 | 26768 | 35694 | 9227 | 25200 |
| Pauli central PR index0 | 1145 | 1856 | 2035 | 664 | 1357 |

前4行は[D1 SOURCE基準Bのmetrics CSV](../artifacts/non_df_h4_d1_source_execution/2026-10-24/metrics_table.csv)、後2行は[D2 CSV](../artifacts/non_df_h4_d2_execution/2026-10-24/comparison_table.csv)を引用した。PR index0を256平均と混同しない。CX/depthの新しいcohort平均は集計していない。A_AXESには保存256本すべてでcosine/sineの5指標が一致した記録があるが、cohort外の等値定理ではない。

## 4. Pauliが低価格になる機構：何をどこまで測れたか

### 4.1 D1の初回停止と後続測定は別の証拠

[初回D1](non_df_h4_d1_execution_20261024.md)はPauli-PR index0の旧recursive fingerprint不一致によりStage Aで停止。top-level費用compile0/9、隔離回路未実行だった。このFAIL・旧payload欠測・排他的原因UNKNOWNは今も維持する。名前だけが原因と証明したわけではない。

その後、別SOURCE/別契約の[D1 SOURCE基準B](non_df_h4_d1_source_execution_20261024.md)で、保存QPY・descriptor・phase/controlのSOURCE対応を検証し、full4の旧RZを再現した後に隔離5を測定した。初回をPASSへ書き換えたのではなく、B基準の後続9/9測定である。

| PR index0 cosine | full RZ | retained-only | event-only | empty | 保存Δ=full−retained−event＋empty |
|---|---:|---:|---:|---:|---:|
| Pauli generic | 9652 | 6524 | 3110 | 0 | +18 |
| DF native | 23021 | 13483 | 5773 | 0 | +3765 |

出典：[interaction CSV](../artifacts/non_df_h4_d1_source_execution/2026-10-24/interaction_table.csv)。empty=0は実測で、欠測の0埋めではない。保持/eventは有効なsimulation候補ではなく診断ablation。Δは操作的な非加法性であり、basisだけ・compiler passだけの費用ではない。event-only CXはPauli1976/DF1932で、全指標でPauliが小さいとの主張も成立しない。

### 4.2 機構と因果解釈の限界

| 要因 | SOURCE・保存descriptorで確認した機構 | 対照測定／観測 | 未分離の寄与 |
|---|---|---|---|
| Pauli prefix27 | 全labelがI/Zのみでsupport1〜2。PFにある非対角stringの必須反復がPRではtailへ移る | D1 generic index0の保持部RZ6524、DF13483 | Z/ZZの効果だけを他条件から分離した値ではない |
| Pauli event粒度 | k個のlocal signed Pauli product→1 rotation。productごとに指数回路を実行しない | D1 event-only RZ3110対DF5773 | 異なるlaw/成分/角度/basisとの交絡 |
| DF保持部とevent | one-body/fragmentのGaussian basis・central RZ/RZZ。random成分はfragment全体ではなくrotated Z/ZZ | 保存descriptorとD1隔離測定が対応 | basis単独のcompiled価格や最大寄与率は未確定 |
| DF basis reuse | 同じbasis ID/hashが隣接するとreuse。identityでflush、occurrence末尾で閉じる。step間はraw concatenation | DF index0/step0は保存counterでnaive8→emitted2、cancelled pairs3 | literal counterでありcompiled RZ削減量ではない。reuse off対照は未実施 |
| control方式 | Pauli genericはbody全体、D2はbasis/parityを非制御にしてcentral rotationを制御 | 同じ保存bodyでPF255500→15028、PR0 9652→1145 RZ | control構成＋compiler応答の合成効果。各passの寄与ではない |
| compiler相互作用 | full wrapperを一括compile、1Q再合成・inverse cancellation等 | D1のΔ、D2のpaired差、A_COHORT/A_AXES | literal加算はfull価格と一致しない。pass trace・一意の費用配賦は未確定 |

DFのRZZ制御には内部generic decompositionも入り得るため「DF nativeなら全て最小制御費用」とはしない。DF basis reuseは存在するので「各eventで常にbasisを2回払う」とも数えない。SOURCE上の安価化機構、測定した対照効果、未識別の因果寄与をこの表で区別する。

D1の数値は**generic Pauli対DF**。central Pauliの保持/event別価格をこの表から推定しない。D2と同cohortのA結果によりcontrol方式の重要性は実測で支持されるが、prefix・law・basis・control・compilerの各寄与率のランキングはできない。比較研究のcase studyを報告するだけなら、その全配賦を完成条件にする必要もない。

## 5. Bias・normalization・必要shot・条件付きG

### 5.1 使用した誤差会計

保存会計は `u=u_enc+ρ`、`h=ε_axis−|b0|−u`、`ε_axis=ε_sig/√2`、`α_axis=.025`、`u/h≤.01`を使用する。B_savedを変更せず、十分条件には `B_up=max(1,B_saved,exp(saved_log_B_upper))` を使い、`N_axis=ceil(2 B_up² ln(2/α_axis)/h²)`。**今回この式を実行していない。**

| 方式 | 保存signed bias cosine / sine（既存表示） | u_enc cosine / sine | B_saved / B_up（既存表示） |
|---|---|---|---|
| Pauli PF | +8.60040e-5 / +1.40526e-5 | 7.58568e-15 / 7.51950e-15 | 1 / 1.00750723 |
| Pauli PR | +9.15986e-5 / +1.49687e-5 | 5.24159e-15 / 8.45520e-15 | 1.28737092 / 1.28929068 |
| DF PF | +1.05630e-4 / +1.72517e-5 | 4.84190e-12 / 4.84787e-12 | 1 / 1.00088015 |
| DF PR | +1.05820e-4 / +1.72829e-5 | 1.38613e-12 / 1.38719e-12 | 1.17758514 / 1.17821702 |

出典：[既存会計報告§2](non_df_h4_conditional_cng_results_20261024.md)、[exact各軸field](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/conditional_functions.json)。biasは旧signed値、u_encは採用会計でCの保存`bias_distance.upper`を用いた項。C totalの一部を引いたり、経験値をcertificateに格上げしたりしない。PFのB_up>1も旧log_B marginによる保存方針で、無断に1へ置き換えない。

centralも同じencoded target/bias/Bを使うため、**仮想ρ=0のNはgenericと同値**。これは実compiled残差が0との認定ではない。

| ε_sig | Pauli PF N_total | Pauli PR N_total | DF PF N_total | DF PR N_total |
|---|---:|---:|---:|---:|
| .05 | 14275 | 23381 | 14097 | 19535 |
| .01 | 360964 | 591666 | 357406 | 495293 |
| .005 | 1465005 | 2403669 | 1455550 | 2017168 |
| .001 | 41581910 | 68864124 | 42716009 | 59217525 |
| .0005 | 201460416 | 339380548 | 220461185 | 305829365 |

量子shot Nと価格のMC標本数256は別。各条件の両軸failure budget=.05であり、全条件同時95%保証でもenergy/RPEの精度でもない。量子shotを実行した結果ではない。

### 5.2 仮想ρ=0・保存平均価格での総費用

Gは `N_cos C_cos＋N_sin C_sin` のpaired標本平均。次表は[central会計の既存表](non_df_h4_central_conditional_cng_20261024.md)からの表示値引用で、exact値は出典pointer付きCSVに保持する。

| ε_sig | Pauli central PF | Pauli central PR | Pauli generic PF | Pauli generic PR | DF PF | DF PR |
|---|---:|---:|---:|---:|---:|---:|
| .05 | 2.145247e8 | 2.667315e7 | 3.647262e9 | 2.179367e8 | 1.908099e9 | 4.625584e8 |
| .01 | 5.424567e9 | 6.749753e8 | 9.22263e10 | 5.514979e9 | 4.837669e10 | 1.172777e10 |
| .005 | 2.20161e10 | 2.742117e9 | 3.743088e11 | 2.240484e10 | 1.97016e11 | 4.776339e10 |
| .001 | 6.248929e11 | 7.856052e10 | 1.062418e13 | 6.418895e11 | 5.781825e12 | 1.402179e12 |
| .0005 | 3.027547e12 | 3.871669e11 | 5.147314e13 | 3.163401e12 | 2.984052e13 | 7.241563e12 |

旧genericのPauli PR/PF G比は約.05975〜.06146、centralでは約.12434〜.12788。PFの改善が相対的に大きいため、centralはPR自体も安くするが**PRのPFに対する相対優位幅は小さくなる**。central PF/DF PFは約.10146〜.11243、central PR/DF PRは約.05346〜.05766。いずれも既存会計の引用で、新しい再ランキングではない。

### 5.3 ρ guard全域の頑健性と残る条件

generic/DFは保存30比較、centralは指定15比較において、方式・軸別のρを独立にguard内で動かしても順位反転なし。単一の共通ρ=0断面だけの結果ではない。[generic比較領域](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/comparison_regions.json)、[central比較領域](../artifacts/non_df_h4_central_conditional_cng/2026-10-24/recovery02/run01/comparison_regions.json)の保存端点と判定を照合した。centralの比較対象はPauli PR/PF、Pauli PF/DF PF、Pauli PR/DF PRの各5精度で、未指定pairを今回追加評価していない。

最厳ε=.0005の保存予算（cosine/sine）はPauli PF約2.64900e-6/3.36139e-6、Pauli PR約2.59361e-6/3.35232e-6、DF PF約2.45469e-6/3.32972e-6、DF PR約2.45280e-6/3.32941e-6。**予算は実残差の測定値ではない。** guard内の頑健性は、実ρのguard所属やPR母平均価格を保証しない。

PRの価格nominal SE/CIはsmall-cohortの統計的不確かさで、guard boxの幅とは別。Pauli両軸は同じtrajectoryに対応して共分散を保存している。Pauli/DF間をpairedとしてSEを小さくしてはいけない。

## 6. 高次RTE eventと母平均費用

### 6.1 論文の費用モデルとの関係

[PR原論文v2 Appendix A.2](https://arxiv.org/pdf/2503.05647v2#page=21)は全偶数Taylor次数を含め、各eventをPauli積と1回転で表す。高次eventが稀であっても除外はしない。[Appendix E.3/E21](https://arxiv.org/pdf/2503.05647v2#page=42)の解析的Toffoli/平均回転費用と、本研究のfull-wrapper compiled RZ標本平均は異なる評価モデルである。paper側のcontrolによるhalvingやRPE/RMSE設定も、この固定signal実験へ無断に移さない。

本研究はK4のfinite lawで、k=0/2/4の確率を正規化し、finite approximationのbiasも扱う。Pauli SOURCEはproductを発生順にappendし、DFでは異なるGaussian basisのinvolutionを実行する。論文の簡潔なPauli rotation数を、そのままDF Gaussian/compiled費用や高次event費用0へ読み替えない。ここでのk=2/4はRTE event次数で、PF S2の次数や別H2/Yoshida4研究とは別である。

### 6.2 保存lawと観測

| 保存項目 | Pauli PR | DF PR |
|---|---|---|
| nominal p0 | .9960625813453219 | .9974498244090985 |
| nominal p2 | .003934819613987418 | .002549087577779853 |
| nominal p4 | 2.5990406907583435e-6 | 1.0880131215414273e-6 |
| 保存8192 eventのk=0/2/4 | 8157 / 35 / 0 | 8167 / 25 / 0 |
| 何れかにorder4を含むtrajectoryの保存implemented質量 | 8.316595170215553e-5 | 3.481583274276441e-5 |
| order4 trajectory観測 | 0/256 | 0/256 |
| 未観測component集合U | 空 | 保存164 component、質量約.00327472917 |
| order4なしでUを含むtrajectoryの保存質量 | 0 | .1001190406163259 |

出典：[rare_cost_sensitivity.json](../artifacts/non_df_h4_conditional_cng_results/2026-10-24/run01/rare_cost_sensitivity.json)。nominal値とfinite-grid implemented exact massは別fieldで、後者を使った過去の解析結果を引用している。新しい確率計算・抽選はない。IID uniform-word product lawが前提で、PCG64の独立性を新証明したものではない。Uは同じ標本から選ばれた集合なので、固定Uに対する独立coverage検定として解釈しない。

Pauli centralの保存群別結果はorder0のみ221本の平均1139.4932126696833 RZ、order2含有35本の平均1149.0857142857142 RZ、order4群の費用はnull。これは記述統計で、order2だけの因果増分ではない。genericで計算したrare stratum費用感度をcentralの新価格へ移す再解析も実施しない。

### 6.3 言えること／費用upperの不足

- 有限256本の平均価格とそのcohortに対する条件付きGは確定した保存値。未観測eventが存在しても、この有限cohortの観測結果が消えるわけではない。
- 真の期待費用には、未観測order4やDFの未観測componentも正のlaw質量で入る。その条件付き平均費用はnullで、0ではない。観測最大やnominal t95も全law upperではない。
- 既存generic感度解析のDF約76万RZ/shot、Pauli約17億RZ/shotという反転水準は、ρ=0に加え、remainder母平均を標本平均と同一とする仮定付き。実tailの予測・上界でも、Pauli対DF PR順位の全域証明でもない。
- 母平均優位性を主張するなら、full-wrapper compiled費用の正当なenvelope、またはlaw/stratumを用いる妥当な期待費用boundsと統計的不確かさが必要。rare側だけを抑えても、remainder母平均の扱いを無条件に正当化したことにはならない。PR同士なら一方のupperと他方のlower等、主張に対応する比較が必要。

「費用の個数upper」と「compilerのoperator数値誤差保証」は別の仕事である。費用tailのために全compiler数値保証を要求しない。現在の限定比較を報告するために、全allowed trajectoryのcompileや標本の無制限拡大を必須工程にもしない。

## 7. 6局所Arb保証の適用範囲

以下は[保存6件一覧](../artifacts/non_df_h4_central_arb_pf_pr0_sine/2026-10-25/six_local_results.json)の外向き上端の引用。各参照resultのSHAも照合した。

| Pauli central対象 | cosineのρ上端（保存表示） | sineのρ上端（保存表示） |
|---|---:|---:|
| PF singleton | 9.56489249909976322304440e-15 | 1.23241819194079088800945e-14 |
| PR index0 | 6.79574416339626407633468e-16 | 7.73535930526261160074741e-16 |
| PR index1、order2含有 | 6.27209038472283370091352e-16 | 6.33356970130103477878240e-16 |

すべて保存局所予算内、64bitで終了。同じraw ray/exact norm、phase/control/+Im、B_saved一回という尺度で、selected C-targetと実保存compiled QPYを比較した保証である。PFは固定singletonの両軸を覆う。一方、PR0/PR1は選択2trajectoryだけで、256本全部・未観測order4・PR全law・DFへは拡張しない。

[R2-C](non_df_h4_r2c_compiler_phase_proof_20261025.md)の保存判定はPROOF_INCOMPLETE。今回その証明は継続せず、全lawρ=null・正式受理falseを維持した。局所ρと条件付き会計との接続・受理を自動実行していない。必要なのは主張に合う測定残差の証拠であり、常に一般compiler全体の形式的数値保証が必要という意味ではない。

Pauli/DFのC-referenceは各表現のfull Hに対応する別の保存値である。共通rayを使っていても、それだけで共通厳密Hamiltonianへの新しい精度転送を認定しない。この対応は[reference_scopes](../artifacts/non_df_h4_central_conditional_cng/2026-10-24/recovery02/run01/reference_scopes.json)に保存されている。

## 8. 現在支持される主張と、研究としての完成条件

| 掲載候補となる主張 | 現在の証拠での判断 | 不足／適切な限定 |
|---|---|---|
| 固定H4/cohortにおける4方式と2種類のPauli制御の回路価格 | **報告可能な観測結果** | 同じ分割/同じlawや全分子一般の比較とはしない |
| 同じPauli bodyでcontrol構成を変えると費用が大きく変わる | **D2で実測支持** | control＋compiler応答。各passや表現だけの寄与率ではない |
| genericの選択PR0で保持/event価格と非加法性が異なる | **後続D1で実測支持** | 初回hash FAILを保持し、採用B基準・単一代表・ablationと明記 |
| PRの単価低下はshot負担込みの条件付きGでも有効 | **5精度の保存会計で支持** | ρ=0表示だけでなく、指定guard全域・保存価格平均という条件付き |
| Pauli centralは対応するDF PF/PRより低G | **保存15比較の範囲で条件付き支持** | 実ρ所属、母平均tail、分割/law/controlの交絡を明記 |
| 実精度保証付きの無条件PR母平均G優位 | **未完成** | measurement残差の転送、期待費用/統計tail、必要なtarget整合 |
| Pauli表現自体がDFより本質的・普遍的に優れる | **支持しない** | 交絡を除く別研究設計が必要。今回の完成条件へ加えない |

**この固定実装のempirical/conditional case studyとしてなら、主要比較は完成しており、今回の整理を基に主張・caption・Methodsを限定すればよい。** 追加のD1再実行、basis費用の完全分解、R2-C完証、order4の大量samplingは、そのscopeで結論を述べるための必須作業ではない。掲載可否の最終レビューや研究方針の正式採択は、今回実施していない。

**精度保証付き母平均Gまで結論したい場合**は、費用tailと実装残差を独立に閉じる必要がある。片方の解消を、もう片方の解消や正式certificate受理とみなさない。PF同士の価格にはPR tailはなく、既存PF局所残差も再利用できるため、PR全law保証をPF比較の前提にする必要はない。

## 9. 最小の次検証候補：主張ごとの判断材料

以下は優先順位の判断材料で、採択・実行はしていない。追加実験を自動的に要求しない。

| 求める結論／優先条件 | 最小候補・担当 | 判断できること／限界 | 必要な承認 |
|---|---|---|---|
| 現在のempirical/conditional case studyを報告 | ユーザー/GPTが主張scopeを選び、Codexが既存表・captionを整える | 現在の証拠で十分。新科学計算0 | 重要な方針採択・最終レビューは担当を決めて別実施 |
| PR母平均費用まで強めたい | Codex/費用担当が既存structural counterと限定compiler個数変換からfull-wrapper envelopeを検討 | rareだけでなくremainder、PR対PRなら両側の妥当なboundsが必要。追加samplingより先に、何を抑えれば判断が変わるか定める | ここでは案のみ。新H4上界数値評価・sampling/compileは範囲を決めて別承認 |
| 実compiled精度込みGまで強めたい | Codex/精度担当が既存PF両軸Arb、PR selected証拠、C/Lを再利用し、未接続のactual measurement残差だけを設計 | PF fixed stateとPR law平均を区別。全compiler operator保証より弱い適切な転送でもよい | 新数値診断/接続再解析/正式受理はそれぞれ許可範囲を確認。今回R2-C継続0 |
| order4が非常に高価格という具体的懸念を検査 | 必要時だけ、事前固定したorder4代表のcosine最大1compileを別設計 | stress診断・仮説の反証。1代表から全law upperや母平均順位は証明できない | 新trajectoryの構築/compileを明示承認。samplingは別範囲 |
| DF basis reuseの因果効果を数値化したい | 必要時だけ既存DF PR0を対照に、reuse offのcosine最大1compile（D3案） | 同じevent列でのreuse＋compiler応答。basis全体の寄与率ではない | 別の回路build/compile承認。比較Gだけを報告するなら不要 |
| central内部の保持/event寄与を知りたい | 必要時だけ同じPR0の保持-only/event-only最大2compile。既存full/emptyを適合確認して再利用 | central系列の操作的非加法性。generic D1の費用を転用しない | build/compileと保存値の新解析を別承認。比較研究の既定工程にはしない |

結論を変える可能性が高い不足は、**強い母平均主張を選ぶ場合の費用tail**と、**実精度主張を選ぶ場合のρ転送/target対応**である。basis/control/compilerの詳細寄与率は機構論文をさらに強める資料になり得るが、現観測を報告するための最優先条件ではない。algorithm改善は別チャットのscopeとして今回扱わない。

## 10. 保全・公開・mandatory STOP

今回は標準libraryで保存JSON/CSVを読んでfieldを転記し、SHA・pointer・資料間の同値を照合しただけである。shot/G/CI/平均/確率の新計算、H4科学計算、Arb作用、回路build/compile、sampling/RNG replay、Hamiltonian/state生成、新PR実装、R2-C証明継続、正式受理はすべて0。

旧SOURCE/結果/grant/ledger/marker/STOPを変更せず、旧manifest行を保持する。[構造・保全確認](../artifacts/non_df_h4_comparison_synthesis/2026-10-25/verification.json)。元H4 native終了UNKNOWN、Pauli-PR historical hash FAIL/原因UNKNOWN、C/Lの未接続、旧20行N/G nullはそのまま。後続の正常終了証拠はそれぞれの新runだけを証明し、元終了証拠を補完しない。

最新版`research_project_instructions_unified_2026-10-25.md`の§8.4/10/10.3を適用。既存コード・数値パイプラインを調べ、今回それらの科学entrypointは実行せず保存表を再利用した。共有環境/他ユーザー/processへの操作、恣意的CPU/RAM制限、新しい承認証明の仕組みはない。

新branchのcommit/push・remote照合は[publication_status.json](../artifacts/non_df_h4_comparison_synthesis/2026-10-25/publication_status.json)へ記録する。**完了後mandatory STOP**。次の担当は**ユーザー/GPT**：empirical/conditional case studyで結論するか、母平均/実精度保証まで必要かを判断する。**Codex**は選ばれた主張に必要な最小範囲を、明示指示後に扱う。今回は新科学計算や重要な方針採択へ移行しない。

## 公開確認結果

local snapshot `c37ab940b858f12f46b2d5b203c61aaca1a7ac2a` の通常origin pushはHTTPS Username取得不能でexit128。当初のbranch/commit検索未検出も失敗記録へ保持した。

その後、許可済みGitHub接続のGit data APIを用い、全blobと完全tree `0c86ada77eb3fbf398835183ccb433d227a53ea2` がlocal snapshotと一致することを確認して公開した。公開整理commitは **`745bc0044c621112a10e7a1008e39dd4f577c787`**。APIが付すcommit metadataはlocal snapshotと異なるが、ファイル内容は完全一致する。元snapshotは別のlocal branchへ保全し、公開commitをgit fetchして専用local branchを同じSHAへ揃えた。公開状態の追記commit後もremote/local HEADを確認する。

既にAPI経由で公開済みのため再pushは不要。通常git用の手動コマンドは次のとおり：

```bash
git -C /home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-comparison-synthesis-20261025 push -u origin codex/non-df-h4-comparison-synthesis-20261025
```

**成果をlocal保全してmandatory STOP**。新科学計算・重要方針採択へは進まない。
