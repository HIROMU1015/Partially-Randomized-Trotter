# H4 tol-only DF10 S_STATS 完了（2026-10-19）

**承認された数値解析1 pass・24 scenario・2基準方向・12感度設定を完了。native exit0／同一親回収／終了監査PASS、mandatory STOP。** ユーザーの「次の作業に進んで大丈夫」と選択文脈「S_STATS数値解析1回を別途承認」をS_STATSだけの承認receiptに固定した。sampling・compile・P科学実行・新seedは0。raw標本再集計・bootstrap・law再構成も実行していない。

計算SOURCEは`8752489bde3903d80d306c4500535c79ac65ef31`、準備commitは`92780e253a8483679fd03e8fa7a1641aad3db6bb`。固定計算コード・対象・感度条件を変更せず、[最終準備契約](track_a_h4_tol_s_stats_preparation_20261019.md)を継承した。今回新規の公開pushは行わず、専用branchへ独立commitで保存する。

## 基準シナリオの提案n

family α=.05、固定primary数m2の片側Bonferroni、planning power=.8、SD倍率f4・相対検出差η=.02を事前規則どおり採用した。次表は一回の保存resultからの転記であり、終了監査・文書化でn式を再実行していない。

| ε | 固定PR候補 | 提案n | D190の保存PF対照 | 保存PF G_RZ | 保存基準方向 |
|---|---|---:|---|---:|---:|
|0.05|B2_p3_q1_2nd_r4_K2|1108|B0I_p5_q1_2nd|70108214|1|
|0.0005|B2_p2_q8_2nd_r2_K4|78|B0_p7_q8_2nd|7856558002614|1|

提案trajectory合計1,186、両軸compile計2,372。基準方向+1は探索n8の保存平均が保存PF費用より小さいことを表し、独立Pによる再現・母平均優位の認定ではない。Pの最終n採用・SOURCE・seed・実行grantは未固定。今回Pは開始していない。

## 固定条件と入力

linear H4・隣接1 Å・STO-3G、tol-only threshold1e-8 DF rank10、8 system qubits＋ancilla、Nα=Nβ=2のsector36、T=.8/P0、通常signed bias・allowance倍率1。PRは既存B2二次、p3/q1/r4/K2（δ=.8）とp2/q8/r2/K4（δ=.1）。保存状態のH4/H6/H8 solver・正規化・compiler来歴の差を保持した。

A50 commit `85cb5b4d1808434165c421115ed09dab3b63dd00` のEXPLORATION_N8保存mean・paired SE・n・stratum集計と、D190 commit `406f70570323366da185355550a3900388717fbb` の固定PF費用を使用した。保存FP n32、legacy DF12、他分子、条件付きpilotの標本や分散を混ぜていない。D/A科学SOURCE `84db3909639b34ec521004c3d8f7ef9b5e6ad851` と保存結果bytesは不変。科学compilerはQiskit1.3.0/rz-sx-x-cx/opt1/seed17・symmetric_directionalの来歴を保持し、S_STATSでは呼ばない。

`SD_plan=SE_saved×sqrt(n_saved)`、検出差`η×PF_G_saved`、`n=max(2,ceil(((z_(1−.05/2)+z_.8)×f×SD_plan/(η×PF_G_saved))²))`の近似normalモデルを一回適用した。f1/2/4/8とη=.01/.02/.05の24 scenario。n capはなく、観測された探索費用差を検出差に置き換えていない。実power・CI/p値・bootstrap・選択後補正を計算していない。

## 保存された感度表

| SD倍率 | 検出差η | p3/q1/r4/K2 n | p2/q8/r2/K4 n | 提案trajectory計 | compile軸 |
|---:|---:|---:|---:|---:|---:|
|1|0.01|277|20|297|594|
|1|0.02|70|5|75|150|
|1|0.05|12|2|14|28|
|2|0.01|1108|78|1186|2372|
|2|0.02|277|20|297|594|
|2|0.05|45|4|49|98|
|4|0.01|4431|311|4742|9484|
|4|0.02|1108|78|1186|2372|
|4|0.05|178|13|191|382|
|8|0.01|17722|1244|18966|37932|
|8|0.02|4431|311|4742|9484|
|8|0.05|709|50|759|1518|

小さなnのシナリオへの差し替え、候補追加・再選択、同一候補の標本二重計数は行っていない。基準f4/η.02を維持した。

## Pに必要な提案科学call予算

| 項目 | 提案値 |
|---|---:|
| 新無条件trajectory / native evolution | 各1,186 |
| cosine/sine wrapper compile | 2,372軸 |
| occurrence | 1,732 |
| RTE event | 5,680 |
| control probe / 独立event oracle | 最大8 / 2 |
| master＋step seed予約slot | 1,810（未生成） |
| endpoint会計 / 固定比較 | 1 pass / 2比較 |
| 新PF compile・signal・SCF/DF/state/reference・shot再計算・bootstrap | すべて0 |

これらはPの提案値であり、今回のS_STATS科学call実績ではない。[Pへ引き渡すmetadata草案](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_completed/2026-10-19/P_after_S_STATS_DRAFT.json)は保存resultの転記のみで、最終P契約・seed registry・有効grantは作成していない。

## 希少eventの限界

両対象の保存n8はS0_ONLY8、S2_SINGLE0、S2_MULTI0、S4_PRESENT0。K2候補の未観測S2、K4候補の未観測S2/S4の費用・分散は今回の保存SEで保証できない。候補固有のlaw確率・visibilityはnull、未知層費用はnullのまま。観測0を確率0／未知費用0へ置き換えていない。

f4/f8のSD拡大とplanning power=.8は、未観測層を含む母平均精度・実power・coverageの保証ではない。提案nはS0のみの探索n8の保存SEを入力とした近似計画である。P最終契約ではこの限界を含むユーザー判断が必要。law確率の新評価・raw再集計・条件付きpilotの追加利用を、このS_STATS承認に自動包含しない。

## 実行監査と保全

旧pending spec・INVALID grant・準備STOPを保持し、新しい承認receipt／execution spec／grant／fresh資源receiptを`.h4_tol_s_stats_private/activation01/`に固定した。24条件の小さなスカラー計算は1 processで1 pass。CPU/RAM/worker/wallの恣意的制限、affinity変更、共有設定・他job・他worktreeへの書込はない。

witness wall秒は`0.4519287757575512`、numeric部分と終了後closure照合を含む秒は`0.20961726270616055`。native exit0、同一親回収、終了後SHA PASS、中断/retry0。SOURCE9 file・入力16 file（元13＋activation3）・runtime128 file・Python binaryを照合した。旧9178レコードと準備seal20レコードを保全し、新14レコードをmanifestへ固定した。

[終了auditor](../../scripts/audit_h4_tol_s_stats_completed.py)はone-shot/ledger、全24 scenarioと12設定、基準設定、保存nとP call予算の整数恒等式、2方向の入力join、希少層null、SOURCE/input/runtimeと保全SHAを検査した。数値kernel・normal n式・quantile・SD・powerを再呼出ししていない。作者自身の終了確認であり、独立科学レビューではない。準備時の人工35件・無効grant2経路拒否の証拠は前段に保存した。

結果SHA256：`85d9d5fa41020ebf3ef724672d1f755b45ad64b6f50537f0f5f345f9406a0c06`。

[保存result](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_completed/2026-10-19/result.json)、[native exit](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_completed/2026-10-19/native_exit.json)、[終了監査](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_completed/2026-10-19/completion_audit.json)、[実行binding](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_completed/2026-10-19/execution_binding.json)、[承認receipt](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_completed/2026-10-19/approval_receipt.json)、[call ledger](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_completed/2026-10-19/calls_ledger.json)、[artifact SHA](../../artifacts/resource_applicability/track_a_h4_tol_s_stats_completed/2026-10-19/artifact_manifest.json)。原本・有効grant・入力snapshot・旧STOPは私有directoryに保全する。

未解決の技術異常はない。提案nをPの最終nに採用する判断、希少層の限界、P SOURCE・seed・最終call契約と科学実行承認は未確定。**mandatory STOP。** 次は**ユーザー**がS_STATS結果・n1108/78・希少層の限界を確認し、採用とP最終契約準備を指示するか判断する。**Codex**は次の指示まで停止し、P科学実行は別の明示承認を必要とする。今回新規pushはしていない。
