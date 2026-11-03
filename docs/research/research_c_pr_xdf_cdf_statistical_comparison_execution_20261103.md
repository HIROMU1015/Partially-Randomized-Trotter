# 研究C：新PR-aware対保存XDF8/CDF8・直接統計比較の実行結果

2026-11-03。**固定7/7 logical calls COMPLETE、correctness FAIL=0、retry=resume=0、mandatory STOP。**
保存native32の事後的・探索的一標本t近似モデル内で、新PRの平均条件付き総RZは保存XDF8より低いことを支持した。
登録した5%精度・分布診断基準も通過した。Confirmatory validation、保証付き優位、actual CAの認定ではない。

## 実行条件と固定SOURCE

対象は直鎖H4・隣接1.00 Å、STO-3G、Nα=Nβ=2、sector36、DF rank8、S2/s1、h=δ=.1、m0..12、CA=.00159 Ha。
新PRは元順LD2/r32/K0のpartial ordered S2、保存XDF8/CDF8は全frame決定論ordered S2でr/K非適用。
それぞれ自身のDF ground-state oracle、状態準備費用除外、保存P0（Qiskit1.3.0、opt1、seed17、rz/sx/x/cx、
approximation_degree1、qubits_initially_zero=false、MPFR160宣言phase）、head/X/Y shell込みfenced serial会計を保持した。
異なるDF表現の比較であり、同一J/U比較・再最適化後の比較・PR目的関数単独の因果比較ではない。

| 用途 | 固定commit |
|---|---|
| 今回の実行SOURCE（runner/契約を含むcheckout） | `e2884a071d60ff8d2ffcacc98705e2e5cfafb206` |
| 実装固定commit | `c761acaa5430100aaa0d605f523a13295d225133` |
| 再利用した新PR native32科学SOURCE | `4df8eca29c59ffd55e2dd898f05da16bb246ddbe` |
| 新PR native32結果 | `beb8de5cd8b71f80bb69556f92299435ff431c3e` |
| XDF/CDF共通ideal科学SOURCE | `2d20ad196a406752de8e4c0ce9516b101e49d927` |
| XDF/CDF共通ideal結果 | `39ee9705b7a459c40c205c542d660b5d2ab90d33` |

PR endpoint SHA256は `a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a`。
主比較 `B3_XDF_spin_relative`、副比較 `B2_CDF_relative`。
[技術準備・再利用調査・元証拠への入口](research_c_pr_xdf_cdf_statistical_comparison_preparation_20261103.md)と
[固定契約](research_c_pr_xdf_cdf_statistical_comparison_contract_unapproved_20261103.json)の条件を変更していない。
契約ファイル名/statusは作成時の未承認proposalを表す。今回の会話で与えられた固定7-callの明示実行許可に従って一度実行し、
契約/SOURCEの過去のbytesやstatusを後から書き換えていない。

- 契約SHA256：`b6ec9f13247c989187c09d066c70b1ec0f99aa40a37cc18b7ca5376100eec460`
- 技術SOURCE manifest SHA256：`0f35c863224a197641bef433d3034d73f26756035c787069b3aecd44e031be2a`
- 実行開始：2026-11-03 07:28:15 UTC（journal）。計測wall time約0.171秒。
- 新出力：[run01](../../artifacts/research_c_pr_xdf_statistics/2026-11-03/run01/)（9ファイル）。

## 主比較：固定XDF8に対する新PR平均総RZ

推定対象は固定candidate/r32/RPE政策と保存finite-bit sampling実装に対する
`μ_PR = W_PR E[complete controlled r32 kernel RZ] + S_PR`。
固定W=12,539,808、shell RZ=51,583。r32・価格校正数32・13roundを追加の乗数や独立標本にしない。
保存PR価格32件だけを再利用し、旧update064の256件やupdate005の640件、ENTRYをpoolしない。

`H0: μ_PR >= C_XDF8`、`H1: μ_PR < C_XDF8`。
固定XDF8総RZは100,267,421,030で、XDF側のsampling分散は置かない。
既存 `c_r32_price_statistics.price_statistics` とSciPy `ttest_1samp(alternative='less')` を使用した。
一側α=.025と両側95%区間の上端（97.5%一側上側信頼限界）は対応し、結果も整合した。

| 指標 | 今回の保存出力 |
|---|---:|
| PR標本数 | 32 |
| complete kernel平均RZ | 7,519.09375 |
| kernel平均SE | 107.32188319090913 |
| PR平均総RZ点推定 | 94,288,043,542 |
| PR平均総RZのSE | 1,345,795,809.412428 |
| PR平均総RZの近似95%区間 | [91,543,274,892.59941, 97,032,812,191.40059] |
| PR−XDFの差 | −5,979,377,488 |
| 差の近似95%区間 | [−8,724,146,137.400589, −3,234,608,838.599411] |
| PR/XDF点比 | 0.9403656997798819 |
| 比の近似95%区間 | [0.9129912184059235, 0.9677401811538404] |
| t統計量 / 自由度 | −4.443004983505327 / 31 |
| 一側p値 | 0.000052615399948145244 |

判定：**`POSTHOC_EXPLORATORY_NOMINAL_MODEL_PR_LOWER`**。
区間上端は固定XDF費用を下回り、一側p<.025、`test_interval_consistent=true`。
比の分母は固定定数で、PR側だけに価格sampling不確かさがある。
このp値は近似モデル内の名目値であり、事後的な比較選択を補正したconfirmatory p値ではない。

精度halfwidthは2,744,768,649.400584で、登録したhalfwidth/XDF≤5%を通過。
Moment skewness=.5823647225066234（閾値2）、最大delete-one shift/halfwidth=.22930148846748832（閾値.25）、
診断警告なし、観測除去0。これらの診断でIID・未観測tail・interval coverageを検証できたとは扱わない。
[Primary JSON](../../artifacts/research_c_pr_xdf_statistics/2026-11-03/run01/primary.json)。

## 副比較：固定CDF8と同じPR区間

固定CDF8総RZは102,056,444,569。主比較と同じPR平均区間から差・固定分母比を得た。
追加t検定0、追加cohort0、CDFのp値はnull。

| 指標 | 保存出力 |
|---|---:|
| PR−CDFの差 | −7,768,401,027 |
| 差の近似95%区間 | [−10,513,169,676.400589, −5,023,632,377.599411] |
| PR/CDF点比 | 0.9238813280258082 |
| 比の近似95%区間 | [0.8969867143540096, 0.9507759416976068] |

判定：**`POSTHOC_EXPLORATORY_SHARED_BOUND_PR_LOWER`**。
XDF<CDFという固定順序と同じ上側boundから含意する副比較で、独立な統計的確認ではない。
[Secondary JSON](../../artifacts/research_c_pr_xdf_statistics/2026-11-03/run01/secondary.json)。

## 標本資格と科学的な限界

Scope/native監査PASS。保存32 sample idと各32 K0 microevent、candidate/SOURCE/cohort、hash-derived seed、
raw/compiled/actual adjoint/XY correctness、head/shell会計を照合した。
実現event列の完全重複0、drop/reweight/resample0、新PRNG draws0。
Sampling lawは保存PCG64/binary64 categorical実装、cohortは `near_practical_PR_r32_fixed32_v1`。
種の衝突がないことだけで数学的独立性・IID・ideal-lawへの移送を保証しない。

今回の仮説は既にPR点価格とXDF価格を見た後に設計したため、保存32件を新規prospective holdoutと呼ばない。
固定N近似Student-tのcoverageは未検証。未観測support/tail・有限bit lawのideal-law母平均への移送は未資格。
共通ideal laneのnative synthesis error=0は反実仮想の仮定であり、実測0ではない。

Actual CA、全law synthesis、implemented-law signal誤差、保証付き資源優位、一般PR law・他距離・H6等への一般化、
XDF/CDF再最適化後の順位、独立新規性、重要科学レビュー・論文化GOは認定しない。
Equivalence/noninferiorityはnull。最新Evidence Map C29/Q02等は元固定版を保持し、この結果報告を別の証拠として接続する。

## 実行・技術確認・保全

既定passive preflightで143 SOURCEと213入力、候補・科学SOURCE、scope、保存STOP、fresh不在を確認後、
固定runnerへ `--execute` を付けて一度実行した。計算コード・統計政策・入力・契約は修正していない。

| call | attempt / COMPLETE |
|---|---:|
| saved_metadata_load | 1 / 1 |
| fixed_scope_gate | 1 / 1 |
| saved_native32_load | 1 / 1 |
| native_row_gate | 1 / 1 |
| primary_one_sample_comparison | 1 / 1 |
| secondary_fixed_threshold | 1 / 1 |
| result_summary | 1 / 1 |
| 合計 | **7 / 7** |

[STOP](../../artifacts/research_c_pr_xdf_statistics/2026-11-03/run01/STOP.json)はfailed=false、retry=resume=0、mandatory_stop=true。
Journalは7 attempt＋7 COMPLETE、FAIL0。統計engineは追加実行せず、出力schema・閾値整合・SOURCE/input hashを静的確認した。
契約とSOURCEのrun保存bytesは技術準備版と一致。既存人工・回帰66 PASSの固定実装をそのまま使った。

新native sampling/build/compile、PF/RTE、shots solver、ground eigensolve、DF再評価、fit/U更新、NPZ/QPY decodeは0。
実行資源は小さなJSON読取りと依存する32-row統計で、重いkernelはない。CPU32・available RAM約53GiBを観測し、
人工CPU/RAM/worker/wall capなし。Process-local BLAS1のみで、共有venv/OS/cache/jobを変更していない。
元dirty checkoutのHEAD/index、旧SOURCE/raw/契約/grant/ledger/marker/STOPを保持した。
既知のmanifest旧schema FAILは今回と区別し、全体checker PASSとは主張しない。

研究概要・validation status・当日の研究ノート・machine-readable索引にこの限定結果を追加し、
旧Claim/旧結果・過去の準備時点の未承認記録は書き換えない。公開後はremote SHA/tree・本文・主要blobを照合する。

**Mandatory STOP。** 次の担当はユーザー／研究C：この探索的結果を踏まえて独立prospective確認や重要科学レビューが必要か判断する。
それらは今回の7-call許可に含めない。Codexは追加sampling・再解析・移送・レビューを自動開始しない。
