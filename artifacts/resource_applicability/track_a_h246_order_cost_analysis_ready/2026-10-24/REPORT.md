# H2・H4・H6の高次PF費用変化：保存値定量解析の実行準備

最新のユーザー指示により、添付のH2/H4解析A〜DへH6も追加した。完了済みH6のSOURCE/worktree/実行領域を変更せず、基準 `77ea542152de1c9ff796aefaf9be91c68c7ff1cd` から専用branch/worktreeで準備した。**実保存値の新しいC/N/G比・寄与・統計・ε別方向判定は未実行。** 人工10テストmethodと封印runtimeの人工86 eventがPASSし、SOURCE/input/runtimeの1,333 fileを固定、metadata-only preflightがPASSした。添付指示§1・§10に従い、別途明示承認を待つ。新規grant0・実解析root未作成・mandatory STOP。

## 固定対象

linear H2/H4/H6、隣接1.0 Å、STO-3G、tol-only DF1e-8、rank3/10/19。T=.8、主ε_sig=.0005、α_axis=.025、P=0、USUAL_SIGNED、既存empirical allowance/reference/normalization、symmetric-directional、状態準備を除くfull Hadamard wrapperのcompiled RZ費用を維持する。

| 系 | 方法 | 固定候補 | prefix | q | trajectory数 |
|---|---|---|---:|---:|---:|
| H2 | Full S2 | B1_p3_q4_2nd | 3 | 4 | 1 |
| H2 | Full Yoshida4 | B1_p3_q2_4th | 3 | 2 | 1 |
| H2 | PR S2 | H2O32_c7ec590a32a859bc | 1 | 4 | 33 |
| H2 | PR Yoshida4 | H2O32_8a2ecdd0294ffb74 | 1 | 2 | 34 |
| H4 | Full S2 | H4_FULL_10_8_2nd | 10 | 8 | 1 |
| H4 | Full Yoshida4 | H4_FULL_10_4_4th | 10 | 4 | 1 |
| H4 | PR S2 | M5_MIN_72b305020499dd68 | 3 | 8 | 48 |
| H4 | PR Yoshida4 | M5_MIN_ab57ad5bd40dc583 | 3 | 4 | 33 |
| H6 | Full S2 | H6_FULL_19_8_2nd | 19 | 8 | 1 |
| H6 | Full Yoshida4 | H6_FULL_19_4_4th | 19 | 4 | 1 |
| H6 | PR S2 | M5_MIN_a08e8bfb6642eb93 | 5 | 8 | 33 |
| H6 | PR Yoshida4 | M5_MIN_37ff3428bb58f5cd | 5 | 4 | 33 |

PRは既存K4・ABS_TIME_PER_OUTER32。cosine/sineの同一trajectory pairingを保ち、公式間・分子間・探索/独立確認のcohortを混合しない。m5_best、B0/B0I、各εのbest候補を指定Yoshida4/Fullの代用品にしない。H6で未使用の旧rank-only契約も使用しない。

## 保存データの範囲と不足

[60セルの利用可否表](input_availability_60.csv)と[SHA付き入力索引](input_availability.json)を保存した。44 endpointは解析用fieldが保存済みであり、解析式の実データ検算は承認後に行う。

| 系 | 条件が一致する固定候補の保存endpoint | 今回の精度解析範囲 |
|---|---:|---|
| H2 | 4 | 主ε=.0005のみ |
| H4 | 20 | .05/.01/.005/.001/.0005の5精度 |
| H6 | 20 | .05/.01/.005/.001/.0005の5精度 |

H2の別共通PF表・旧保存費用表には他精度の値もあるが、完了比較のDIAGNOSTICS allowanceと一致しない。4候補それぞれの**元保存値を引用した**[政策差の照合](H2_saved_endpoint_policy_differences.json)により区別した。同じ候補名・Cだけを根拠に、別u/referenceのN/Gを同一系列へ接続しない。H2の16セルは欠測として残し、補間・別候補・別政策の代入を行わない。今回の契約は新しいshot/allowance計算でこの不足を埋めない。

Cのcosine/sine平均、Nの軸別整数、G、headroom、log_B・margin、u、paired G SEと保存軸統計を再利用する。Full endpointにN_totalや軸別Gの表示fieldがなくても、保存軸N/C/総Gは存在する。承認後の分解内でN_totalとexact有理数のC×Nを検算し、元のGを上書きせず併記する。FullのSEはnullのまま、PRのSEを流用しない。

PRのouter_PF_or_discard_bias・finite_RTE_bias・total_signed_biasは元signalに保存されていたため引用可能。H2では元signalの成分と現在のDIAGNOSTICS総biasがbit一致するかを別fieldで残した。shot関係には実際に完了比較で使用したbiasを使い、異なる丸め/経路の成分を無断で合成しない。Fullで保存のない成分はnullとする。新しい誤差分離や反実仮想計算は行わない。

## 再利用したSOURCEと追加処理

[再利用一覧・採用しない処理の理由](reuse_inventory.json)、[SOURCE/SHA](SOURCE.json)。

- `mechanism_analysis.explanation_side` を関数本体の変更なしに再利用する。保存headroom・u・bias・normalizationから連続shot式と整数丸めの整合を確認する。新しい整数shotを採用しない。
- 元 `h46_m5_plan.weights/schedule/allocation/frozen_allocation` と `h46_new2_core.allocation` を再利用する。元H2 DIAGNOSTICS acquisitionの対応3 SOURCE SHAとの一致を技術確認した。historical `SYSTEMS` のH4 DF12定数や分子生成関数は呼ばず、今回のH4 DF10入力を維持する。
- H2の既存ledger call本体とSOURCE/input/runtime seal・ImportSeal・終了closure検査を再利用し、独立した認可namespaceへ結合した。
- `Backend.allowance/account/rank`、`u_aware_shots`、`resource_row`、`metric_summary`、paired momentsは調査したが、元の保存値があるため再実行しない。
- 旧mechanismのcontrast validatorは歴史的なPR/PF比較・固定cohort向けであり、今回のS2/Yoshida4比較へそのまま使えない。旧saved-cost `comparison` のunpaired difference SEも、公式間の独立性仮定と欠測SEの扱いを含むため採用しない。

追加数式は軸別C/N/Gの恒等式と加法分解だけである。新しい科学kernel、Hamiltonian、回路、sampling算法を実装していない。

## A：費用分解の固定定義

S2を0、Yoshida4を1、軸をaとする。

`G_j = Σ_a N_ja C_ja`

`ΔG = Σ_a N_0a ΔC_a + Σ_a C_0a ΔN_a + Σ_a ΔN_a ΔC_a`

最終項をcross項として一度だけ数える。並行して `C_eff,j = G_j / N_total,j` とし、`R_G = R_Ntotal × R_Ceff`、およびS2の軸費用重みを用いる `R_G = Σ_a w_0a R_Na R_Ca` を有理数で検査する。各軸Cが等しい場合のスカラーC分解はこの一般式の特殊例である。元Gの演算順による丸め差を保持し、exact再構成と元float Gの整合を所定の8 ulp以内で確認する。

22比較の内訳はH2 Full/PRの主精度2比較、H4/H6 Full/PRの5精度各10比較。元paired SE/分散は保存値を引用する。公式間のpaired SE、差のSE、比のCI、新しい母平均保証は計算しない。

## B：q・RTE構成とcompiled費用

[既存SOURCEと証拠の対応](q_RTE_SOURCE_map.json)。承認後に12候補の元schedule関数を一度ずつ呼び、signed piece、負時間、q、各pieceのRTE配分を整理する。FullのRTE数は構造上0であり、元scheduleの未使用rte_steps placeholderとは区別する。

全214 PR trajectoryの既存raw結果metadata・価格cohortを結合した。**full eventは各PR候補の事前固定された最初の登録trajectory、計6件だけ**を照合対象とし、そのevent digestを元receiptと一致確認した。承認後にこの6件のpiece配分・Taylor order・phase・basis adjacencyを記述する。全QPY再解析や全trajectoryのoperation寄与推定は行わない。保存stratum統計と6例の観測を区別する。

basis adjacencyはreuseを説明する局所的な証拠であり、全軸のcompiled RZをoperation別に分解した測定ではない。step数からcompiled費用を推定したり、未観測event費用を0にしない。qと次数が同時に変わる比較であることを維持する。

## C・D：shot関係と精度依存性

44 endpointごとに元 `explanation_side` を呼び、保存のsigned bias/u/headroomと実使用log_B_upperによる連続shot式・保存整数を照合する。log_B_for_shotsの保存値がある場合はそれを用い、ない場合は契約内で保存log_B+marginを評価する。整合不成立はUNRESOLVEDとして残し、説明できたことにしない。保存bias成分の相関・会計関係を越える因果主張をしない。

系×Full/PRの6精度summaryは、固定候補の登録済み比較方向だけを整理する。εごとの候補再選択・再ランキング、未知精度への補間はしない。Full/PRそれぞれのS2/Y4方向フラグは費用分解22 callの出力に含め、別の4方式ranking関数は実行しない。

## 未承認の固定実行契約

| 項目 | 値 |
|---|---|
| SOURCE | 2e59a55beec8159c7832ff4ff02513b8976c0eb9 |
| canonical contract SHA256 | 2478152414dce3c468e48c17e04d0addf3143794190be686593bc94eae1708e8 |
| input manifest SHA256 | 30979e34bda28d0ed70fc53b8a8004e22cd94a77d45a45c4101c08a7cb42355a |
| series | H246_FIXED_ORDER_COST_DECOMPOSITION_V1 |
| 取得済みSOURCE/input/runtime seal | 1,333 file、isolated Python3.11 |
| bootstrap / bundle_load | 各1 |
| cost_decomposition | 22 |
| explanation_side | 44 |
| candidate_structure | 12 |
| precision_summary | 6 |
| ledger合計 | 86 event |
| sampling / build / compile / simulation / Hamiltonian・reference再計算 | すべて0 |
| allowance / u_aware_shots / 新しい標本分散・SE / 独立ranking関数 | すべて0 |
| authority | 別途明示承認待ち、有効grant0、専用root未作成 |
| STOP | 一回、retry=false、終了後mandatory STOP |

call単位は登録した解析関数である。分解1 call内に2 endpointのexact C/N/G検算、構成1 call内に元schedule呼出を含む。これらの内部処理も固定SOURCEの範囲内であり、別の科学kernelを呼ばない。[実行契約案](execution_contract_DRAFT.json)、[無効authority template](authority_templates.json)、[出力schema](output_schema_DRAFT.json)、[preflight](metadata_preflight.json)。結果は `results/analysis.json` とSOURCE/input/result SHA・ledger・completion・STOPへ保存する。欠測/不適格はnull比率と理由、SOURCE/入力/予算/G不一致は部分失敗STOP、shot関係の未解決は明示状態として記録する。

## 技術検証・保全・次の担当

[人工テスト10 method](technical_tests.txt)は、系/rank/公式/q/prefix/cohort/εの混同、pairing、単位、欠測0の拒否、交差項の一回計上と一般軸恒等式、SEの意味、未承認実データ/旧scope拒否を確認した。[封印isolated runtime人工smoke](isolated_artificial_smoke.json)でも人工86 event・元scheduleの未登録toy条件・loaded closureがPASS。実データを数値解析したテストではなく、作者自身による技術確認である。

[保全監査](preservation_audit.json)：既知1,622 fileのSHAが不変。実際に使う封印SOURCE・入力・runtime、完了H2/H4結果/ledger/marker/STOP、H6封印入力と既知保存結果・論文/Figure2資料を保全した。他worktreeの通常更新や全体HEAD凍結を要求しない。共有環境や他processを変更せず、SHA処理は利用可能CPUに合わせて並列化した。[資源観測](resource_observation.json)。private raw/event/seed/内部絶対path/有効grantは公開対象へ含めない。[file inventory](file_inventory.json)。

H6の旧Full費用は中断stageから受入済みのcomplete pairであり、元parentのexit/reap不明を保持する。H2のstrict identity残条件、empirical/reference/有限標本の限定も解除しない。詳細GPT/Claude科学レビュー、最終evidence freezeは開始していない。

次はユーザーが本契約による数値再解析を一度だけ明示承認する。承認後、Codexがfresh preflight・新規未使用grantで固定解析を一回実行し、結果・終了監査・commitをまとめて保存する。現在は実解析0・grant0、旧strict COMPARE BLOCKED、[mandatory STOP](STOP.json)。
