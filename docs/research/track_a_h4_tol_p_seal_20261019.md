# H4 tol-only DF10 独立確認P 最終seal（2026-10-19）

**Pの最終技術sealを完了し、mandatory STOP。Pのsampling・compile・科学実行は未承認であり、実行していない。** S_STATS完了commit `d728d088c010089c6b672eef0521261791b2d209` と指定nを正式受入したユーザー指示に従い、保存結果の照合、seed識別子の固定、科学SOURCE・入力・runtime・one-shot契約・INVALID grant・終了監査を準備した。既存標本の新規平均・分散・power・n計算は0。

## 固定した実行契約

P科学SOURCEは `3c664ca4a9f003880b9f6d56f2a0e55ed5b56603`、contract digestは `d930b6adbffa78ab2f9facb8ff2c06e768d8071c6b8e028e7d702d6d08e10b52`。SOURCE closure25ファイル、独自コピーしたprovider/kernel251ファイル、入力29ファイル、runtime34,273ファイルのSHAを照合済み。実行時も同じSOURCE・入力・compiler/runtimeを照合する。封印後の資料commitや無関係なworktreeのHEAD/status更新は拒否条件にしない。

| ε | 固定PR候補 | 独立した新cohortのn | D190の保存PF対照 | 保存PF G_RZ | 保存基準方向 |
|---|---|---:|---|---:|---:|
|0.05|`B2_p3_q1_2nd_r4_K2`|1,108|`B0I_p5_q1_2nd`|70,108,214|+1|
|0.0005|`B2_p2_q8_2nd_r2_K4`|78|`B0_p7_q8_2nd`|7,856,558,002,614|+1|

linear H4、隣接1 Å、STO-3G、tol-only threshold1e-8のDF rank10、8 system qubits＋ancilla、Nα=Nβ=2／sector36、T=.8／P0。B2のL_Dは順に3と2、δ=.8と.1、通常signed bias・allowance倍率1を固定。H4の保存dense eigh・phase canonicalization・追加の明示的な事後再正規化なしという来歴、およびH6/H8のsolver・正規化との差を保持する。状態・DF・積分・full-H referenceを生成し直さない。

Qiskit1.3.0、basis `[rz,sx,x,cx]`、opt1、seed_transpiler17、backend/coupling/layout/routing指定なし。`symmetric_directional`、original constant＋extracted tail identityを一度だけ加え、referenceを回転しない。compiler hashは `1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b`。正本はmachine-readable bindingであり、本文の転記も終了時に照合する。

D190 result commit `406f70570323366da185355550a3900388717fbb`、A50 result commit `85cb5b4d1808434165c421115ed09dab3b63dd00`、S_STATS result commit `d728d088c010089c6b672eef0521261791b2d209` をgit bytesと完成audit/native exitで照合した。D/A科学SOURCE `84db3909639b34ec521004c3d8f7ef9b5e6ad851`、S_STATS科学SOURCE `8752489bde3903d80d306c4500535c79ac65ef31`、既存の全結果・grant・ledger・marker・STOPを保持している。

## 科学call予算と判定

| 項目 | 固定値／上限 |
|---|---:|
| 新しい無条件trajectory／native evolution | 各1,186 |
| cosine/sine compiled wrapper軸 | 2,372 |
| outer occurrence／RTE event枠 | 1,732／5,680 |
| 先頭trajectoryのcontrol probe／独立event oracle | 8／2 |
| 保存rank10からのpreparation再構築 | 2 prefixを各1回 |
| 新P cohortのendpoint会計／比較 | 1 pass／2比較 |
| 新master／master・step整数slot | 1,186／1,810 |
| 新PF compile・signal・shot再算出・bootstrap・SCF/DF/state/reference | 全て0 |
| 既存raw標本の新統計解析・cohort混合・候補再選択 | 全て0 |

各trajectoryでcosine/sineは同一event列を共有し、実測RZ×保存shotをpairedに合計する。固定nの全件完了後、2比較を一回だけ会計する。`sign(saved PF_G − new PR paired mean)`と固定方向+1が一致すれば`REPRODUCED`、逆なら`NOT_REPRODUCED`、厳密な同値は`INCONCLUSIVE`。逆方向も科学的結果として保存する。技術PASSと方向再現を混同せず、追加標本・再選択・optional stopping・自動retryは行わない。

## seed固定と並行研究の保護

namespaceは`H4_TOL10_MAIN_V1_20261019_CONFIRMATION`。固定SOURCE・plan digest・rank10 input binding・cell・replicaのSHA256からmasterを一度だけ導出し、元の`direct_master_to_step_seeds_v1`を維持する。q1はmasterをそのまま使用し、q8は元のPython Randomから8 step seedを導出する。候補間CRNは使わない。cosine/sineの共有は同一trajectory内だけ。

新1,186 master／1,810整数を全て固定し、既知70,270整数との衝突0。旧H4、F/FP、条件付きpilot、固定δ、旧rank12 Pの予約・使用seed、rank10 A50の392 trajectory／2,760整数、H6/H8のlegacy・探索・条件付き・高次PF・確認予約/使用seedを除外した。さらに登録geometry6群・1,308候補中1,164 stochastic caseの保守的予約枠7,737,216 slotを6自然単位に並列化して検査し、衝突0。seed検査は整数識別子の生成・照合だけであり、RTE eventをsamplingしていない。

検査cutoffは2026-10-19 05:40:56 UTC。衝突0はPRNG独立性や未知の将来campaignを保証しない。別途P実行承認後、既知の登録metadataを再読して新規予約との衝突をrefreshし、封印seedを引き直さず、衝突があればSTOPする。他worktreeの更新を凍結せず、他job・共有環境へ書き込まない。新seedの整数値・raw event・実科学入力はprivateに保持し、公開候補projectionはdigest・件数・範囲のみ。

## 並列化と実行拒否・終了監査

既存のsampler・native lowering・control/event oracle・paired軸compile・fingerprint reuseを再利用した。元A kernel SOURCEは `31ef62a12407010e9c1a07f9b37facedb8dff5fe`、native provider SOURCEは `217c971dd537b1cc7fbc13742bc0f6c263ff7f45`。P専用adapterは保存rank10ローダーを使い、旧rank12ローダーへ接続しない。

実行時に利用可能なphysical CPU、既存affinity/quota、effective RAMと既知job割当を観測してworkerを割り当てる。CPU/RAM/worker/wallの恣意的上限は設けず、過去の64 workerを固定上限として引き継がない。各workerは1 thread、parentの2 prefix preparationをfork/COWで共有する。先頭2 trajectoryを並列実行し、control8／oracle2のPASSと全回収後、残1,184 trajectoryを並列実行する。各trajectory内のcosine→sine順は元kernelのまま。PSIは診断のみで、競合や監視遅延だけでworkerを停止しない。共有設定や他processを変更しない。

人工36件＋既存paired kernel29件＋rank10契約32件＋followup13件＋S_STATS21件、計131件PASS。人工値だけで固定n・欠落/重複・seed/axis/event/compiler不整合・探索cohort混入拒否、rank12拒否、preparation hash不一致、control先行、失敗時lock/STOP保持、同値/逆方向、native exit・終了監査schemaを検証した。今回のprimary agent自身の技術チェックであり、別AIによる独立レビューではない。

実物のpending契約とINVALID grantでdirect/witness両経路の拒否を検証し、科学provider import・入力ロード・driver launchの前に拒否された。P output・ONE_SHOT lock・有効grantはいずれも未作成。既存9,216記録のSHAを前後で確認した。終了監査コードはnative exit0／同一親wait/reap／全worker task・compiler/event/seed・control/oracle・nonrefundable ledger・固定n・sole accounting・旧seal保全を要求し、統計値を再計算しない。

## 科学的限界と次の担当

採用nはfamily α=.05／planning power=.8／SD倍率4／検出差.02というS_STATS近似normal設計の計画値であり、有限標本の点順位再現を目的とする。実power、母平均保証、選択後coverageを認定しない。探索n8の両候補は保存stratum count上S0_ONLY=8で、未観測S2/S4の費用・分散はunknownのまま。新Pで未観測でも0とみなさない。探索n8、FP n32、legacyの標本はP統計へ混ぜない。

未解決の科学条件変更はない。保留は別途P実行承認、実行時fresh資源割当、最新の既知seed登録refresh。今回は追加の公開対象が具体的に指定されていないため、前段と同様に新規pushは行わず、ローカル専用branchに成果物をcommitする。公開範囲未確定のprivate実入力・seed整数・元台帳はpushしない。

次の担当は**ユーザー**：固定SOURCE・n1108/78・1,186 trajectory／2,372軸・希少event限界・失敗時自動retryなしを確認し、P科学実行one-shotを別途明示承認するか判断する。**Codex**：その承認後に限り、科学契約を変更せずfresh資源・seed refresh・承認receiptを結合して有効grantを作成する。今回の承認では開始しない。詳細科学レビュー・H6/H8最終統合へ進まない。

証跡：[contract projection](../../artifacts/resource_applicability/track_a_h4_tol_p_seal/2026-10-19/contract_projection.json)、[seed検査](../../artifacts/resource_applicability/track_a_h4_tol_p_seal/2026-10-19/seed_inventory_summary.json)、[テスト](../../artifacts/resource_applicability/track_a_h4_tol_p_seal/2026-10-19/test_report.json)、[拒否検証](../../artifacts/resource_applicability/track_a_h4_tol_p_seal/2026-10-19/refusal_report.json)、[完成監査](../../artifacts/resource_applicability/track_a_h4_tol_p_seal/2026-10-19/completion_audit.json)。
