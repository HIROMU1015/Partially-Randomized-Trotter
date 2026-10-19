# H4 tol-only DF10 A50 完了（2026-10-19）

**承認済みA50を一回完了し、native exit0・終了監査PASS、mandatory STOP。** 科学SOURCE `84db3909639b34ec521004c3d8f7ef9b5e6ad851`、最終seal commit `50479f2f6da76523cdfb3e8bb06bde33213fea67` を継承した。利用者の「次の作業に進んで大丈夫」と選択された「A50科学実行を別途承認」をAのみの承認receiptに固定した。S_STATS/Pや新しい科学レビューは開始していない。

## 固定条件・実行範囲

linear H4・隣接1 Å・STO-3G、保存済みtol-only threshold1e-8 DF rank10、system8 qubits＋ancilla、Nα=Nβ=2のsector36、T=.8、P0。B2二次、prefix1/2/3、q1/2/4/8/16（δ=.8/.4/.2/.1/.05）、r4/K2,4と中央prefix2のr2,8感度を合わせたPhase 0の50候補を維持。通常signed bias、allowance倍率1、ε=.05/.01/.005/.001/.0005。

Qiskit1.3.0、rz/sx/x/cx、opt1、seed_transpiler17、backend/coupling/layout/routing=None、symmetric_directional control、元constant＋tail identityは一度、reference無回転、元の無条件finite-RTE lawを維持した。H4/H6/H8のsolver・正規化・compiler来歴・探索領域の差は[Phase 0](track_a_h4_tol_main_phase0_20261019.md)の記録のまま。SCF・integral・DF・state solver・full-H referenceの新生成は0。

保存FP `B2_p3_q4_2nd_r4_K4` の1候補n32/64軸と保存signal/mean/SEはそのまま利用した。新規49候補はEXPLORATION_N8の別cohortで、保存FPとpoolしていない。受入済みD190 commit `406f70570323366da185355550a3900388717fbb` の5精度の最良PFと費用も再計算せずjoinした。

## 保存結果の有限標本点比較

以下はrunnerの一回の保存会計からの転記。各候補の必要shot×実測full wrapper compiled RZをcosine/sineで合計し、新cohort内で平均・paired SE・精度間covarianceを一回集計した。実際に必要shot回数の量子測定を行った結果ではない。

| ε | 保存最良PR | n | 保存G_RZ平均（exact） | 保存paired SE | 受入済み最良PF | 保存PF G_RZ | 保存点方向 |
|---|---|---:|---:|---:|---|---:|---|
|0.05|B2_p3_q1_2nd_r4_K2|8|542760385/8|1472260.835308543|B0I_p5_q1_2nd|70108214|PR低費用|
|0.01|B2_p3_q2_2nd_r4_K2|8|13442946399/4|65882183.32927127|B0_p7_q2_2nd|5203661250|PR低費用|
|0.005|B2_p2_q4_2nd_r4_K2|8|42636088469/2|345396315.91301435|B0_p7_q4_2nd|33735246876|PR低費用|
|0.001|B2_p2_q8_2nd_r2_K4|8|3808269974525/4|8468572863.672162|B0I_p7_q2_4th|1446021078216|PR低費用|
|0.0005|B2_p2_q8_2nd_r2_K4|8|4912729573525|43698381292.52138|B0_p7_q8_2nd|7856558002614|PR低費用|

最良候補は全適格候補の費用が揃った後、exact費用・同値ならcell_id辞書順の登録規則で選んだ。欠測を不適格として除外せず、未知費用を0にしていない。n8は探索の有限標本であり、母平均・希少tail・selection-adjusted coverageの保証ではない。保存FPがwinnerなら上表のn32と新規n8を区別する。global optimum、厳密bias certificate、分子サイズだけの効果や独立確認Pの成功を主張しない。新たなratio・bootstrap・power/n設計・有意性解析を追加していない。

## 実行・終了監査

| 項目 | 実績 |
|---|---:|
|新規base signal|49|
|新規適格候補|49|
|新trajectory / compiled軸|392 / 784|
|occurrence / RTE event|2448 / 10784|
|control probe / 独立event oracle|196 / 49|
|preparation / primitive probe|3 / 120|
|endpoint / 会計|250 / 1 pass|
|worker枠 / native worker process|64 / 441|
|witness壁時計秒|438.58948223665357|
|中断 / retry|0 / 0|

fresh資源で64物理CPU枠・worker内thread1を割り当てた。既知H6確認scopeは正常終了済みと読み取り専用で観測し、実効空きRAM約469 GiB・既存affinity/quotaを確認した。49 signalの並列取得、全replica0 control/oracle完了後の残343 trajectory並列、fork/COW・3 preparation cache・paired evolution共有・fingerprint再利用を使った。CPU/RAM/worker/wallの独自上限、共有設定変更、他job/他worktreeへの書込はない。PSIは診断のみで、正常計算を競合理由で停止・再起動しなかった。

固定392 master/2,760 seed整数は不変。起動前に既知H4/H6/H8 registryを追補照合し、追加除外値0・衝突0。元の67,510整数＋geometry7,737,216予約slotの照合receiptを保持した。未知・未来のjobまで網羅する主張や数学的なPRNG独立性の証明ではない。

終了auditorは全workerのnative exit0/同一親回収・result SHA、固定seed/law/event_digest、両軸/compiler/fingerprint、非払い戻しcall予約、control、適格subset各n8、250 endpoint、FP統計の逐語不変、保存winner/tieとD join、SOURCE/input/provider/runtimeを検査した。科学accounting・平均/SE/covariance・signal・sampling・compileを再dispatchしていない。保存raw費用から平均・SEを再計算して監査値を作っていない。監査は著者自身の技術確認であり、独立科学レビューではない。seal時の人工/回帰96件と無効grant2経路拒否の証拠は[最終seal](track_a_h4_tol_a_seal_20261019.md)に保持した。

SOURCE17・入力877行・provider251・runtime34,273とPython binaryを開始/終了時に照合。旧774・完成D2,033・A最終seal原本も不変。新実行の5652 fileを私有preservation manifestで固定した。結果SHA256は `3c217dc6c6cafe3e13d9ee767088d671dee18273dff300dd9bf34cb03d8dbb69`。

[保存result](../../artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/result.json)、[終了監査](../../artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/completion_audit.json)、[実行binding](../../artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/execution_binding.json)、[承認receipt](../../artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/approval_receipt.json)、[native exit](../../artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/native_exit.json)、[control](../../artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/control_validation.json)、[worker回収](../../artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/worker_receipts.json)、[seed追補照合](../../artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/seed_refresh_projection.json)、[artifact SHA](../../artifacts/resource_applicability/track_a_h4_tol_a_completed/2026-10-19/artifact_manifest.json)。同directoryのgzipにはsignalとseedを除いたpaired費用projectionを保存する。raw seed/event・実行契約/grant/ONE_SHOT原本は `.h4_tol_main_private/A_seal01_20261019/` 内に保全し、公開しない。

CLIのHTTPS認証が利用できない場合は認証済みGitHub接続で公開し、全tree一致とremote SHAを確認する。元のlocal履歴はbackup branch/bundleに保持し、科学SOURCE・seed・結果bytesを変更しない。実際の公開SHAは最終チャットと私有publication receiptに記録する。

未解決の技術異常はない。不利な点方向も科学結果として保存し、追加samplingやretryを行わない。**現在mandatory STOP。S_STATS/Pや新しい科学レビューには進んでいない。** 次の作業は、**ユーザー**がA50結果の受入と次段階の指示を判断すること。**Codex**は次の指示を受けるまで停止する。
