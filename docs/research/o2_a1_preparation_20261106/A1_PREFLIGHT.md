# A1：入力・round・Control Variatesの実行前照合

2026-11-06 JST。実装・人工確認まで。新規科学計算、保存科学値の再解析、回路build/compileは未実施。
設計根拠は固定[調査1bcf70b][SURVEY]の4文書。実行判断は[計画](A1_EXECUTION_PLAN.md)、作業実績は[REPORT](REPORT.md)へ。

**保存canonical PRに接続可能な最小候補は特定できた。推奨は最終opt2のPartial L_D=3、m=0/q=1/δ=.02/r=1/K=2、X/Y両軸。科学実行条件は未固定・未許可。**
モデル内の保存Hamiltonian/stateの来歴は対応するが、未近似の物理Hとの同一性、厳密な信号半径下界、実law/bias、fresh測定と全価格の取得条件は未閉鎖である。
従って「元の物理Hに対する実行可能性を保証した」「CVが改善する」とは判定しない。

## 1. 固定資料と読取範囲

| 資料 | 固定版・用途 |
|---|---|
| A1設計 | [1bcf70bのREPORT／文献／互換条件／候補計画][SURVEY]を確認。既知法の応用候補で新原理ではない |
| 実測CV | [CV4Quantum v1][CV] §§2.1–2.5、Eqs.28–35,38–39,47。factorized既知平均、pilot fit、残る測定noise、同標本学習のbias |
| bounded confidence | [Maurer–Pontil v1][EB] Theorem4/10。独立・有界標本、固定n。計画への適用条件を確認し数値評価はしていない |
| 原科学cohort | [carrier4591b8d][CARRIER]内のopt2・WP04・snapshot。旧dirty-worktree証拠の保存版であり、CI再現認定ではない |
| 入力・半径・branch | [V1 e520e6e][V1]、[V2 78c49d0][V2]。今回保存値を再解析せず、既存判定とmetadata欄を引用 |
| 旧O2 | [S2 dc42c3a][S2]のT不改善、[R1 b6305ee][R1]のNO_SUPPORTED_R1_CANDIDATE、強いClosed/CTSとC1凍結を維持 |
| 正式規約 | 実ファイルの共通指示2026-10-25を確認。特に§8.1/8.2/8.4/10.1/10.3。[公開固定コピー][COMMON]。AGENTS、PROJECT_MAP、研究概要全文、関連validation status/manifestも確認 |

以下の数字は**保存欄の引用**またはファイル来歴であり、新しい科学数値ではない。

## 2. H・model・stateの対応

| 項目 | 保存された対応・限界 |
|---|---|
| 系 | linear H4、隣接1.00 Å、STO-3G、legacy DF rank12、8qubit、4電子sector。Partial L_D3。C1の新basis/324-label packetとは別 |
| 元model H | WP04は同じDF Hamiltonian全体のsector groundをphysical_df_ground_stateとして使用。H_D groundへの置換はしていない |
| H hash | `56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5` |
| 保存snapshot | carrier内 `artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz` |
| snapshot byte確認 | Git blob `77a69d5021b33a7fa1fde723fd3d1806f3c77950`、3589 bytes、SHA256 `13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b`。既存記録と一致。NPZをdecodeしていない |
| 保存physical state | V1のfingerprint `a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90`。PartialのWP04/PF3/PF12間metadata一致という既存結果を継承 |
| 異なるcontrol state | δ=.02のPF有効固有状態controlは別fingerprint。高overlapを理由にphysical stateと同一視しない |
| 未近似物理H | 保存DF modelと未近似の分子Hを同一とするbyte/operator bindingは今回未取得。df_truncation_value=0等の保存欄だけで同一性を保証しない。厳密元H ground仮定を満たしたとの認定はできない |
| 情報アクセス | E₀/gap/state vector/path別信号をfit・feature・配分へ渡さない。snapshotの存在は無料古典stateアクセスを許可しない。状態準備費は除外 |

元Hへの対応を閉じられない場合、後日の評価は明示された**保存DF model限定の診断案**となる。その採否はユーザー／GPTが判断する。研究の物理H仮定を黙って緩和しない。

## 3. round候補と推奨

最終保存scheduleは[opt2 JSON][OPT2]の `/coherent_reoptimization/best_by_ld/3/best/rounds/*`、
対応metadataは[V2 CSV][V2CSV]。δ=.02のPartial m0..17を新規評価しない。

| 候補round | q | 保存δ／時間t | 保存r／K | 適合性・選定理由 |
|---|---:|---|---|---|
| **m0（推奨）** | 1 | .02／.02 | 1／2 | 最短、一つのRTE occurrence。標準short wrapperの受理領域、opt2にq1の直接価格資料あり。既存入力を最小に再利用 |
| m1（代替） | 2 | .02／.04 | 1／2 | short wrapperと直接較正q2に対応。ただし一測定のoccurrenceが増える。利益が出そうという理由で選ばない |
| m2（代替） | 4 | .02／.08 | 1／2 | short wrapperで構築可能。最終opt2のq1/q2較正と同じ「直接価格既取得」とは言わない |

長qを受けるbenchmark builderも既存だが、初手で使わない。旧δ=.1/r4/K2の4round、初期δ別schedule、最終δ=.02 opt2を混在させない。
推奨理由は実装・証拠の再利用性であり、保存meanからcovarianceや改善率を推測した選定ではない。

推奨m0で引用する旧条件：beta_PF=`.014050012886294481`、beta_RTE=`9.899584668737594e-08`、beta_stat=`.38594988811785885`。
両axisの旧coordinate epsilon=`.266146579872968`、旧alpha/axis=`2.8897922699779505e-08`、旧N/axis=`510`。
**510を新しいpilot/calibration/production数へ流用せず、CVのNを算出したとも扱わない。**
保存α_total=.05、beta_RPE=.4、energy目標CA/10の `.00015936001019904` Haは当時の18roundモデル条件で、新しい正式採択ではない。
単独roundの比較にも全RPE成功やenergy改善の認定を付けない。

## 4. law・回路・phase・価格の接続

| 接続 | 静的に確認したSOURCEと条件 |
|---|---|
| finite law | [rte.py][RTE]：finite_rte_distribution、make_rte_config、_make_event、iter_sample_rte_events。paired even order n=0,2（K2）。orderを選び、n+1個のlabelを復元抽出 |
| metadata順 | selected_component_ids は products → rotation。rotationは最後。内部抽出配列ではrotationが先なので混同しない。phase=(-1)^(n/2)とcoefficient_signを回路に保持 |
| 実生成 | PCG64、Generator.choice(p=...)。order pとcomponent p/sumのbinary64を使用。event_probability欄はalgebraic lawであり、そのまま実finite-grid qではない |
| interrogation | [short wrapper][WRAP]受理q=(1,2,4)。ordinary controlled diag(I,U)、D half-steps、中央RTE、scalar/identity、追加controlなし、measurement付き、状態準備なし |
| readout | X/cosineはH、Y/sineはSdg→H。0→+1、1→−1。Yはimaginary signalで、exp(-itH)、energy=-phase/δの規約。後処理でphaseやsine符号を再度掛けない |
| 保存selected policy | opt2は既存support_run_le_1とboundary処理を含む。対応は [research_direction_full_scope.py][FULL] の apply_run_threshold_policy と既存sequence-policy/builder。A1とbaselineは同じpolicyを共用。full-basis標準provider価格と無断で交換しない |
| full価格 | [df_rpe_hadamard_compiled_cost.py][PRICE]の短round provider、既存WP05/opt2のcomplete wrapper価格経路がある。q1/δ.02/r1/K2の保存opt2較正を参照できるが、任意fresh pathの価格・誤差保証を取得済みとはしない |
| 指標・compiler | 元opt2主指標はcompiled RZ count。Qiskit1.3.0、rz/sx/x/cx、opt2、seed17、backend/coupling/layout/routingなし。元6指標はRZ count/depth、CX count/depth、total depth、size。**native T価格は未取得**。T=RZまたはT=0にしない。1Qは実gate counts等が必要 |
| 旧統計・decoder | [rpe_resource_accounting.py][ACCOUNT]の±1 Hoeffding／reference_radius=1をCVへ流用しない。V2の決定論的phase/branch対応だけは後日再利用可。二項tailはCVに不適合 |

これら6つのcoreファイル（rte、wrapper、compiled-cost、accounting、partial_s2、repeated）はcarrier4591b8dと今回基点1bcf70bで差分なしを確認した。
selected policyのSOURCE・compilerとfresh測定/価格の結合は将来runnerで固定する必要がある。旧raw8本を今回のtrain/cal/prodに転用しない。

## 5. control bankと既知平均

finite bankは**outcomeを見る前**に固定する。次は候補設計であり採択済みcontrol数ではない。
保存順の正support labelからa,bを事前指定し、occurrence0に対して
`1[n=0,L0=a]`、`1[n=2,L0=a]`、`1[n=2,L0=a,L1=b]` のlocal/pair候補を使う。
label実IDを今回抽出・順位付けしておらず、まだbankは固定されていない。a,bの選定は既存metadataのみから行い、測定応答によるfeature探索はしない。
全path indicator表、未知path平均、state由来特徴は使わない。order-only bankは比較用で、独立の新候補に数えない。

実lawでorder質量をq_n、label質量をq_lとした既知平均は、それぞれq_0 q_a、q_2 q_a、q_2 q_a q_b。
同じpositionを二重条件付けした積は、そのpositionを一回だけ数える。異なるlabelやorderが同時に必要ならjoint momentは0。
異なるoccurrenceを使うときは、宣言された独立lawに基づく積であり、相関生成へ自動対応する接口ではない。

### 実lawの取得条件

[NumPy1.26.4 choice SOURCE][NPCHOICE]はbinary64の `p.cumsum()` を最終要素で正規化し、uniformへのright searchを使う。
[PCG64 next_double][NPCG]と[uint64_to_double][NPDOUBLE]は53-bit格子を用いる。
実CDFの境界c_jを**実装と同じ演算で取得した後**、格子モデルでの質量は
`q_j = (ceil(2^53 c_j)-ceil(2^53 c_(j-1)))/2^53`（c_-1=0）で取得できる。
これは実装された写像への適用上の計算式であり、新しいsampling algorithmではない。

追加関数law_from_choice_cdfはこの入力をexact Fractionへ変換する。科学CDFへの呼出しは未実施で、人工CDFのみテストした。
factorizationとconfidenceには既存のfresh-IID random-wordモデルが必要。固定seedの決定論列自体をIIDの証明とはしない。
実行時のNumPy/PCG64/生成順/CDFをSOURCEに結び、同seedや同測定の再利用を避けることが残る。
全dependencyのbitwise一致や承認証明を新要件にはしない。

Parentのdigital_proposal／dyadic_distributionは、別のlargest-remainder法でproposalを作る処理である。
既存momentのrational/interval表現と人工bit mappingは再利用できるが、canonical samplerのqの代用品にはできない。
本作業はsamplerを置換していない。

νの誤差e_jが残る場合はcentering bias `sum_j |lambda_j| e_j` をcoordinate予算から控除する。
実CDF未対応・support消失・未評価の相関等でe_jを正当に上界化できなければ科学実行しない。
さらに実qとalgebraic finite targetの差、native/PF/RTEのbiasは共通baselineにも残す。νだけを実qで直して元targetのbiasが消えたとはしない。
今回そのbias値を新計算せず、補正weightを勝手に変更していない。

## 6. 比較成立の範囲と未解決点

raw、利用可能なorder/label層別・配分、CVは同じScope（H/state/finite target/law/round/axis/weight/circuit）で接続できる。
追加実装のstratified_meanは、同じfeatureを細分した既知確率の正strataを漏れなく要求する。
order-only層別も認め、label bankのパターンだけよりorderでさらに細分する対照を排除しない。
小さな分割なら飽和した層平均・実行可能なcost-aware配分を強い対照とする。最適variance/配分のoracleは与えない。

層内標本は事前固定した条件付き取得と独立割当が必要。既存canonical lawの古典生成後、測定前の条件選別を用いるなら、棄却生成費・全support・有限budgetを含めて後日明示する。
本実装は条件付きsamplerや配分optimizerを持たず、IID標本から空層を削って再正規化する処理でもない。
科学実行前にはこの小さい取得adapter、実CDF接続、runnerの独立stream管理、confidence計算、full価格接続が必要。
旧sampler／量子builderの改造、全round化、RA/新law探索は必要条件にしない。

物理Hとの結合、正当な半径/残余bias、実lawのmoment、原policyの価格/指標、有限測定数/失敗配分が未固定。
**技術接口は準備できたが、これだけで科学計算を開始できる状態ではない。**

[SURVEY]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/1bcf70ba69f91c62763a8739ec620b040b097ed9/docs/research/o2_estimator_mechanisms_survey_20261106
[CV]: https://arxiv.org/html/2502.08735v1
[EB]: https://arxiv.org/pdf/0907.3740v1
[CARRIER]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/tree/4591b8d75a2a6a55c74a51225f0c61648023c8ee
[OPT2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json
[V1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/e520e6e396f7e0def9e522ea1cff58eb10ddfa8f/docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md
[V2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/REPORT.md
[V2CSV]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/78c49d0b5f1d71e63b5959e78813237923bbc648/docs/research/pr_rpe_v2_branch_success_20261103/V2_ROUND_RESULTS.csv
[S2]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/dc42c3a9f77e960f415d0475da2ba03bb56252c7/docs/research/o2_s2_multi_event_validation_v2_20261106/REPORT.md
[R1]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b6305ee2e33436ed96d8aefdee15588676242000/docs/research/o2_r1_server_candidate_screening_20261106/REPORT.md
[COMMON]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3571477093441ac4852166fe829f8ccc5db5ec4f/docs/archive/server1_main_collection_20261103/b05c68a693d31e3a15652faccfc4e6f0a05f8c34/research_project_instructions_unified_2026-10-25.md.original
[RTE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rte.py
[WRAP]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_hadamard_interrogation.py
[FULL]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/research_direction_full_scope.py
[PRICE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/df_rpe_hadamard_compiled_cost.py
[ACCOUNT]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/4591b8d75a2a6a55c74a51225f0c61648023c8ee/src/trotterlib/rpe_resource_accounting.py
[NPCHOICE]: https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_generator.pyx
[NPCG]: https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_pcg64.pyx
[NPDOUBLE]: https://github.com/numpy/numpy/blob/v1.26.4/numpy/random/_common.pxd
