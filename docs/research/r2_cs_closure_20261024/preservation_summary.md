# 既存資産の保全と非干渉

本作業の書込先はこの新しい終了記録ディレクトリだけである。C0.6を基点とする専用branch/worktreeで、merge/rebaseや旧成果の書換えを行わない。新statusはR2-CS専用schemaに属し、他研究の状態へ伝播しない。

## 資産を取り出す入口

[decision_record.json](decision_record.json) の`preserved_assets`は、以下の8群を`fixed_references`のcommit/path/SHA/URLへ結ぶ。元の式・コード・結果を再複製せず、既存の固定資料へ戻る。

|資産ID|保全内容|主な固定参照ID|
|---|---|---|
|SAME_R_LAW|同じR=qrの条件付きlaw共有。signal・N・Gの不変性へ拡張しない|PR2_SIGNAL_CODE、C05_structural_hypotheses_md|
|FINITE_RTE|係数・normalization・signed time・整数allocation|RTE_CODE、ALLOCATION、SCHEDULE|
|G6_P5|既存数理・Green局所query・P5少数群・生成SOURCE|G6_NOTE、G6_CORE、P5_CODE|
|EXISTING_CERTIFICATES|上下界・誤差保証・十分予算policyの成立条件|W2B_BUDGET、W2C_CONDITIONS、C06_candidate_propositions_md|
|INFORMATION_ACCESS|情報の取得経路・費用・安価さの限界|C06_information_access_matrix_csv、C06_REPORT_md|
|REUSE_CATALOGS|既存PR/PFコードと固定SOURCEの索引|C05_CATALOG、C06_source_catalog_json|
|NEGATIVE_AND_COUNTEREXAMPLE|R3 STOP、Track Bの否定/同値/保留の区別、P_U反例|R3、PU、C05_CROSSWALK_CSV|
|NO_NEW_SELECTOR_CONNECTION|新しい選択則へ接続できなかった理由|C06_candidate_propositions_md、C06_prior_art_delta_md、C06_decision_packet_md|

これらは工学的・研究上の再利用資産であり、今後の作業や実行の承認ではない。

## 保全対象と確認範囲

- 固定C0/C0.5/C0.6と必須参照のSHA・Git上の所在を確認する。内容の再解析・再採点はしない。
- 旧P1 DRAFT、C0/C0.5/C0.6のSTOP・HOLDを、固定Git原本と本worktreeのbyte照合で確認する。
- 基点からの差分が新ディレクトリだけで、既存tracked fileが変更されていないことを確認する。既存SOURCE・結果・grant・ledger・marker・第1研究資料を追加差分に含めない。
- 他worktreeのファイル・index・環境・processを変更しない。他チャットの通常のHEAD/status変更は凍結対象とせず、全共有環境を一括監査したとは主張しない。
- 公開内容は文書、固定された公開参照metadata、終了status、最小の文書管理検査に限る。private raw、seed、grant実体、内部絶対path、認証情報を含めない。

検査は [verification.json](verification.json)、再実行入口は `python3 -B docs/research/r2_cs_closure_20261024/technical/check_closure.py`。既存のgit/digest/check管理helperを再利用し、旧科学pipeline・旧人工数理fixture・実データは実行しない。これは独立科学レビューや数学的証明の再認証ではない。

## 最終状態の参照順序

R2-CSの現行方針は新しいdecision recordとSTOPから読む。旧C0〜C0.6は当時の提案・未採択・HOLDという履歴を保つ。新記録による正式終了/P1予定外という方針更新は、旧科学SOURCEや実行権限を変更しない。

**保全対象のTrack A、PR-2、G6/P5、他Track B、R3、他PR/PF研究はそれぞれの従来状態を維持する。次の作業なし、mandatory STOP。**
