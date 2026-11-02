# Track A共通実験プロトコル v1 DRAFT：統合報告

2026-10-19。**DRAFT_NOT_ADOPTED、静的整理と文書作成のみ、local commit後mandatory STOP。** 正式採択・公開push・新科学計算・既存契約への適用は行わない。

## 完成資料

1. [共通実験プロトコル](track_a_pr_resource_evaluation_protocol_v1_DRAFT.md)：共通入力/会計条件、Core/Extended/Guarantee、主図/主表の採用・例外・非再現・STOPの規則案。
2. [要件別充足表](../../track_a_protocol_compliance_matrix.csv)：4系列、22要件、88 metadata行。COMPLETEを記載scopeに限定し、targetとの相違/未照合、cohort、実測/model、根拠commit/SOURCE、最小次作業を別列化。
3. [差異・例外・最小追加検証](../../track_a_protocol_gap_and_exception_report.md)：legacy DF12、F/FP、H6 P、H8 profile、H4機構scope、既知説明欠測を区別。
4. [採択判断表](../../track_a_protocol_adoption_decisions.md)：AD01–AD12は全てユーザー判断待ち。候補数、n、model許容差、native閾値等を勝手に正式基準へ固定していない。
5. [取得根拠catalog](../../artifacts/resource_applicability/track_a_protocol_v1_draft/2026-10-19/evidence_catalog.json)と[構造検査](../../scripts/check_track_a_protocol_draft.py)、[人工テスト](../../tests/test_track_a_protocol_draft.py)。

## 主な整理結果

H4 legacy DF12のD登録104と独立Pは完了している。H4 tol-only DF10のF/FP完了とは別系列で、DF政策共通化案にそのまま適合するとはしない。H4主比較準備のCodex BとH6新A独立確認準備のCodex Aは、ユーザー指示に記された状態として保持し、未取得の準備branch/SOURCEや完了結果を推測しない。

H8 n48の2候補確認は固定両端に対して完了している。SOURCE上はsymmetric-directionalであり、ordinary ancilla branchの意味論guardと矛盾しない。一方で原profile bytes/旧compiler出力の同一性とprivate終了証拠の今回の再監査は未確認のまま。

Mechanism core32/32と後続説明63/64を、それぞれの固定scopeの完了証拠として継承する。説明欠測はA82_OLD_PR.PRで、新A/n48の欠測ではない。H4 MechanismはPRE_D_P・旧subset16で、新D/Pへ接続済みとはしない。今回G、比、shot、SE、最良候補を算出し直していない。

## 起点・取得範囲・SOURCE

作業branch：`track-a-pr-resource-protocol-v1-draft-20261019`。独立worktreeの起点は論文準備 `0a8ac1d2d03c5a9aef2acab0295226f8c88b38ce`。指定9commitのGit objectを全て実取得した。H4 D/PやMechanism結果は起点treeにない場合も各固定commitから読んでおり、勝手にmerge/cherry-pickしていない。

科学SOURCEの主要対応（報告/metadataからの転記）：H4 D=`8a82b9272ee24896e7d7cbb7400d9b8c12d8f804`、H4 P=`73ec1964501f14390d037a378c3149110e273635`、H4 F=`dfef38fc8107dd76f5d71dcdeca31f8d3dc39f64`/FP=`64e0fc91cd273cff02ae8cffb729ee8e40c302a9`、新A会計=`8601b921342fc1cc83780a830e1f65bf64fe9e93`、H8 P=`745a674546f2cb6258526fbfde1d9a00601689f8`、core=`5c256067db29a326bcaa0a0e244b38f6b87bd138`、説明=`b6716396acc5760fde7cf7cc6f2481b183a08f10`。会計SOURCEと原取得SOURCEを同一視しない。

本作業には新しい科学実行SOURCEはない。文書・metadata checkerの版はこの報告を含むlocal commitで識別する。原科学SOURCEの全closureやprivate raw/運用witnessを再検証したとは主張しない。GitHub公開・remote SHA最新一致の検査は今回していない。

## 技術検証と保全

構造検査は固定Git blobのSHA/bytes、9commitの取得、evidence参照・commit対応、全系列×要件の一意行、DRAFT/未承認、cohort区分、model/legacy表示を検査する。人工テストは欠測行/重複行/未取得根拠/commit誤結合/unionを独立Pへ改称/modelを実測へ改称/legacyをtol-onlyへ改称/実行フラグの混入を拒否する。費用恒等式のテストは小さい人工分数だけで、保存科学値に適用しない。結果は専用bundleのvalidation_summaryへ保存する。

新sampling/compile/signal/SCF/DF/state生成、科学module実行、保存値の数値再解析/新統計/再rankingは0。既存SOURCE・raw・grant・ledger・marker・STOP、Codex A/B、保留m5 pilotは変更しない。無関係なworktreeを凍結するgateも追加しない。AIサブエージェント0。既知ファイルの独立読取をprocessで並列化した。

## 次の担当と停止

**ユーザー**：DRAFTと採択判断AD01–AD12を確認し、共通条件・最低検証・例外掲載・研究を一区切りにする範囲を正式採択するか判断する。文書採択と新しい科学計算の実行承認は別。

Codexの今回の権限はlocal commitまで。公開push・新科学計算・正式採択・Codex A/Bの既存契約変更・m5再開は行わず、mandatory STOPする。
