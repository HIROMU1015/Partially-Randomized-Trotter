# H2独立確認結果の正式採用と研究資料統合

2026-10-19（Asia/Tokyo）。**ADOPTED_FINITE_SAMPLE_POINTS_ONLY**。ユーザーの明示指示「H2の独立確認Pの2条件を、有限標本の結果として正式採用します。」を記録する。結果の正式受入と、論文本文への掲載、プロトコル改訂案の採択、母平均保証は別の判断である。

固定結果は [P完了 f2d1a86c](https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/f2d1a86c431f835db700f348ce618d8604e7ee23)、科学SOURCEは `ce1f511c4f7429190adcee58cde86be8a87a2fb1`。本作業は専用branch `codex/h2-results-integration-20261019` で行う文書整理と構造検証に限定する。

入口：[採用記録](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/adoption_record.json)、[全6段階の証拠](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/stage_evidence.json)、[GitHub読取記録](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/github_read_receipt.json)、[8条件比較](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/H2468_comparison_separate.md)、[v2へのH2追記案](track_a_pr_resource_protocol_v2_h2_update_DRAFT_20261019.md)。

## 受け入れた条件と結果

linear H2、結合距離1.0 Å、STO-3G。DFはtol-only `1e-8`・final_rank指定なしの実rank3。4 qubits、Nα=Nβ=1のspin sector次元4。T=.8、P=0、通常signed bias、allowance倍率1。full cosine/sine Hadamard wrapperのcompiled RZに保存analytic shotを掛けた両軸総費用を用いる。

|ε|固定PR|L_D / δ / r / K|n|保存G_PR（exact）|保存paired SE|固定PF / 保存G_PF|点費用・方向判定|
|---|---|---|---|---|---|---|---|
|0.05|B2_p1_q1_2nd_r2_K4|1 / .8 / 2 / 4|80|384135011/80|15992.018778605378|B1_p3_q1_2nd / 5533984|PR低費用、REPRODUCED|
|0.0005|B2_p1_q4_2nd_r2_K4|1 / .2 / 2 / 4|12|998482376201/4|276760628.92856276|B1_p3_q4_2nd / 305695076994|PR低費用、REPRODUCED|

費用・SE・n・候補・判定は[P公開projection](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/f2d1a86c431f835db700f348ce618d8604e7ee23/artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/P_completed_run01/confirmation_result.json)の `/accounting/comparisons/0` と `/accounting/comparisons/1` の保存値を転記した。新しい比・削減率・差・CI・検定・順位・標本数を計算していない。Pに保存されていない比や削減率は空欄を維持する。

独立Pは固定92 trajectory、184 compiled軸、256 RTE event。元の有限RTE law、固定n、cos/sin paired trajectoryの別cohortとして完了し、探索n8やFP診断n1を混合していない。q1はS0_ONLY79/S2_SINGLE1、q4はS0_ONLY12と保存監査に記録されている。観測した層以外の費用・分散やlaw確率はunknownのまま扱う。S2_SINGLEを1件観測したことや小さい保存SEから母平均の優位を保証しない。

## 完了証拠と来歴

|stage|科学SOURCE|完了commit|受入範囲|
|---|---|---|---|
|F|37b1cf72990885962d8449d8bd5ec7e303329350|7ea6df2dbe34e64a34f468c343bb12922d3e2829|新tol-only DF3・sector4入力、状態検証|
|FP|de72e5d4e48177734c0ddd79dffac14222410658|af724928689be3592b391fab3a14538bf4752b76|native DFとfull wrapper、PR2候補各n1の診断|
|D|cf99e4452df96291d7e9d37d76d2e91be5cbc413|7b202a1a569c6839c4c7a5dfb18b6ff51ec3f3eb|D70の保存signal再利用・D_run02の受入済み費用|
|A|de72e5d4e48177734c0ddd79dffac14222410658|1d2f67ec285956ef4090ea26520d64b11e925f9c|A66、新n8、528 trajectory・1056軸|
|S_STATS|e0b959e8a617f7934d05c3e249832ccc1c43909f|829d697ae7905b3ea90a61da9d2d4da15d8b757a|保存D/A summaryに基づく標本数設計|
|P|ce1f511c4f7429190adcee58cde86be8a87a2fb1|f2d1a86c431f835db700f348ce618d8604e7ee23|最終seal後の独立固定n80/12、2条件の有限点結果|

GitHub connectorで全6段階のSOURCE・完了commitを取得し、各完了directoryの公開ファイル105件のGit blob IDを固定ローカルGit bytesと照合した。各execution bindingのSOURCE hash、artifact manifest、結果hash、保存終了監査とnative exit0・同一親wait4/reapを確認した。[証拠CSV](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/stage_evidence.csv)は結果・監査の固定pathとSHAを示す。

Fの古いnative_completion_auditにwait4欄がない点は、同じ完了snapshotのnative_exitで確認する。Pの公開confirmation_resultはprivate原resultのprojectionであり、両者の異なるSHAを混同しない。公開Git bytesの照合、保存監査の読取、private証拠の保全照合を、科学SOURCEの再実行や別担当による独立科学レビューと呼ばない。

旧D_run01はユーザーの切断後に最終native exit/resultを確認できなかったSTOPとして保持する。旧partial費用は採用せず、旧56 compile予約も返還していない。正式受入のDは別契約D_run02であり、過去STOPを成功へ書き換えない。F/FP/D/A/S_STATS/PのSOURCE・raw・grant・ledger・marker・STOPは変更しない。

## 研究資料への反映

H2は**旧v1対象外の追加系列**。v1本文・採択資料は不変とする。公開済みv2は別branchの固定snapshot `1f001cfc4e8e0531279a26613be29a52c7c0640c` で文書採択されているが、そのH2現在欄はS_STATS完了時点の履歴である。本作業は既存草案 `c0808b25` と採択版の双方を固定参照し、FからP完了・結果正式受入までを追記する**H2改訂案**を別文書・別manifestにした。この改訂案を正式採択したとは扱わない。

H4/H6/H8の[公開主表](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/1923fce8f7461513d1bc47b5061e87cd70917e87/artifacts/resource_applicability/track_a_h468_paper_display_methods/2026-10-19/main_table.csv)6条件は、候補・費用・SE・判定・監査状態を一切変更せずsnapshot保全する。追加2条件との8行表は別の研究比較資料であり、論文主表を8行へ差し替えない。H4粗精度のNOT_REPRODUCED、H6粗精度のREPRODUCEDかつPF低費用、H8の公開詳細PR監査不足をそのまま表示する。

既存Figure2A/2Bの34ファイル、H4 legacy DF12、Mechanism32/64軸、H8_CONFIRM_1/3のcohort関係は保全する。Mechanism因子をH2へ移送せず、同じH8観測を重複した独立証拠と数えない。既存Methodsの「H2はこの論文統合の対象外」は元の主表scopeとして維持し、H2の本文採否を先取りしない。

[paper manifest更新案](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/paper_manifest_update_DRAFT.json)は固定baseのpath/SHA、追加案、適用規則を保持する。paper dataにはPR/PFの4行、CEMにはH2の2 claim、証拠台帳には保存費用・SE・n・方向判定の10行を追加する案とし、元1576/14/316行の正本を変更していない。全てDRAFTであり、H2結果自体の正式受入とは別に論文資料への適用判断を残す。

## 構造検証・制約・次の判断

検証コードは [integrate_h2_research_documents.py](../../scripts/integrate_h2_research_documents.py)。stdlibとGit metadataだけを使い、[構造検証](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/structural_validation.json)と[保全記録](../../artifacts/resource_applicability/track_a_h2_results_integration/2026-10-19/preservation.json)に、固定公開blob/SOURCE/結果hash、既存6行の完全保持、H2保存token転記、CSV schema/IDの非衝突、v1・原図34ファイル、H2 private領域、既存manifest全項目・入口の旧本文保持を記録する。科学moduleのimport、NPZ decode、数値解析、sampling、compile、新grant発行は行わない。

母平均優位、未観測希少eventの費用・分散、完全な入力手順同一性、厳密bias、実power、全域最適、FT資源・energy/RPE総費用は未保証である。H2のdense sector4とH4 dense sector36、H6/H8の記録上のiterative solver・正規化・残差gateの相違を保持し、分子サイズの因果効果や漸近scalingを主張しない。

- **ユーザー**：H2の本文／補遺への掲載と、paper manifest・CEM・台帳追加案の適用範囲を判断する。
- **ユーザー**：今回のH2 v2追記案を別途採択するか判断する。既存v2採択の撤回・再採択は本作業に含めない。
- **Codex（追加指示後）**：採用された文書更新だけを反映する。新科学解析、詳細科学レビュー、最終evidence freezeは今回実施しない。

**本作業完了後はmandatory STOP。**
