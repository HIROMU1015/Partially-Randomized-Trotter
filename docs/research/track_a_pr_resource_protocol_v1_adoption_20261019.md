# Track A共通実験プロトコル v1：文書正式化報告

2026-10-19。**ADOPTED_V1_DOCUMENT_ONLY**。ユーザー承認に従い文書基準を正式化した。新科学計算・保存値の数値再解析・公開push・既存科学契約への自動適用は0。local commit後にmandatory STOPする。

## 完成資料と採択範囲

- [正式版v1](track_a_pr_resource_evaluation_protocol_v1.md)：共通条件、Core/Extended/Guarantee、掲載・例外・非再現・停止規則。
- [充足表88行](../../track_a_protocol_compliance_matrix.csv)：4系列×22要件を維持し、採択方針と証拠の達成statusを分離。
- [採択判断表](../../track_a_protocol_adoption_decisions.md)：AD01/02/04/06/07/12を基本方針採択、AD05/08/09/10/11を条件付き採択、AD03の追加PFを保留。
- [差異・例外報告](../../track_a_protocol_gap_and_exception_report.md)：legacy、実測/model、技術閾値、説明未検証、掲載scopeを明示。
- [機械可読採択記録](../../artifacts/resource_applicability/track_a_protocol_v1/2026-10-19/adoption_record.json)、[固定根拠catalog](../../artifacts/resource_applicability/track_a_protocol_v1/2026-10-19/evidence_catalog.json)、[更新台帳](../../artifacts/resource_applicability/track_a_protocol_v1/2026-10-19/matrix_change_log.json)。

共通DF政策はtol-only1e-8・rank自動決定。H4 legacy DF12は独立系列として既存D/Pを保全し、tol-only10の費用や状態へ読み替えない。SCF/軌道/積分/DF/状態生成の全規則がH4/H6/H8で一致するという未確認statusも保持する。

独立PはH8型の事前固定・新cohort・新seed・標本非混合という手順を採択した。n=48を強制せず、個別の分散・比較差・希少event・目的に応じ事前設計する。PR不利・点順位非再現・tieも有効な結果として扱い、母平均保証に読み替えない。

既存S2/S4の比較条件を維持する。m5_best等の追加PF主比較採用は保留で、別pilotは再開していない。既存modelと直接実測を区別し、旧2%等の各契約の技術閾値を一律変更しない。掲載例外の方針採択は、全系Core完了や個別図表の最終採否の自動認定ではない。

## 追加証拠の反映

|対象|固定commitとSOURCE|今回反映した状態|未完了として維持する事項|
|---|---|---|---|
|H6独立確認Phase 0|6d290fc8fb656f4db148e2c9f91b757d80ed83c2、adapter 1ab88d7c13e7f635bc00bf93a8ecac3d0fb2fd46|技術準備完了、既存準備の未確認から固定証拠取得へ更新|最終n/seed registry/科学SOURCE/call数はnull、launch_ready=false。独立Pのsampling/compile、数値設計は未承認|
|A82原signal回収|bd41b3760ff89e428380f55585d78094a9330a56、signal c31ef424175c02379eafebe8fc6c7548c1d9aabe、D0解析 20f89bc32b161ee22e4a1ed709f9e3414c19633a|旧union n32のA82_OLD_PR.PRについて原signalとnormalization/signed_bias fieldの回収・metadata同一性の報告を確認|説明数値検証未完了。歴史結果63/64 sideを維持し、回収を数値PASSへ変更しない|

H6の準備はε=.05のp4/q2/r4/K2（PR不利方向）とε=.0005のp4/q8/r4/K2（PR有利方向）を対象とするが、今回新たに候補・nを選んだのではない。A82は新A/n48のp8ではなく、旧n32のp7/q2/r4/K2である。

追加A82報告内のH4 P未完了記載は、その資料の歴史的scopeとして保存した。別に固定取得済みのH4 P完了8a18b277…を取り消さず、主題ごとの証拠を継承する。H4機構表がPRE_D_Pであることも変更しない。

## 来歴・検証・保全

起点はDRAFT commit `9c54d89f6154525e3c7c814a4da5ecdc37453b8a`。新branchは `track-a-pr-resource-protocol-v1-adopted-20261019`、独立worktreeは同名の専用領域。旧DRAFT本文・旧bundle・旧checkerは変更せず、履歴として保持した。旧testは正式版treeで固定DRAFT commitからfixtureを読み出す最小変更だけを行い、旧10件の検査内容は維持した。旧DRAFTの再検査は元DRAFT checkoutで行い、現行の採択済みmatrixを旧DRAFT checkerへ渡さない。

指定追加2commitはmerge/cherry-pickせず固定Git objectで読んだ。既存25＋追加12＝37 fileをcatalogへ結合しSHA/bytesを検査する。これは文書/SOURCE/metadata取得の証拠であり、private raw/runtimeの再監査、remote SHA確認、科学実行の独立再現ではない。新しい科学SOURCEはなく、文書・metadata checkerの版は本報告を含むlocal commitで識別する。

[構造checker](../../scripts/check_track_a_protocol_v1.py)と[人工テスト](../../tests/test_track_a_protocol_v1.py)は採択partition、12判断と表の一致、88行と固定根拠、既存status・実測/model・legacy区分の維持、H6準備とA82回収の承認境界を検査する。科学moduleをimportせず、保存費用・比・SE・shot・順位を再計算しない。新18件＋旧DRAFT回帰10件の人工構造テスト28件、37固定Git objectのSHA/bytes、88行と参照、local link、diff検査はPASS。元DRAFT worktreeのcheckerも25file/88行でPASSした。検査結果とfile inventoryは専用bundleに記録する。これは作者自身の構造・技術検査であり、別reviewerによる独立レビューではない。

既存SOURCE、raw結果、grant、ledger、marker、STOP、Codex A/Bの実装/契約を変更しない。検査対象に無関係worktreeのHEAD/statusをSTOP条件として追加していない。既存worktree/既存科学領域へのfile書込・他jobへのsignal・共有設定変更は行っていない。新branch/worktree登録に必要なGit metadataと、人工testの一時directoryだけを使用した。AIサブエージェント0。独立した読取と構造検査processのみ必要に応じ並列化した。

## 次の担当とmandatory STOP

**ユーザー**：正式版の公開の要否を判断する。必要ならH6の限定した数値設計、A82の限定説明検証、個別論文panelの採否をそれぞれ別判断する。これらは提案する次作業であり、実行済み/実行中ではない。

**Codex（今回）**：文書・構造テストを保存してlocal commitし、mandatory STOP。公開push・新科学計算・数値再解析・Codex A/Bへの自動適用・m5再開は行わない。
