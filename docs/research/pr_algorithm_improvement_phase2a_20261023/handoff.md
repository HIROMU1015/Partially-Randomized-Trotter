# Phase 2A 最終引継ぎ

**PRの改善を妨げる費用機構・強対照・反証材料を整理し、7課題をPhase 2Bへ提出できる形にした。科学的順位・採択は保留、レビュー自体は未実施。**

## 調査範囲と主な知見

前段階固定commit `a9bdcf1d44061daceffe896637e2c92fcb3353b4` の70研究・10候補を利用し、重要30単位の一次記録を対象箇所まで確認した。関連PF資料を含み、SOURCEの静的excerpt、保存結果／既存数学監査を確認した範囲は[索引](SOURCE_EVIDENCE_INDEX.json)に個別記載。全SOURCE/rawの再読、731 branch再走査ではない。第1環境`8dfdfa6460f9e7d25a9595132a672e8de2a5dc7e`と第2環境`096c293a85a0e756574245673ecf29d5245ddca1`の公開探索報告も取得した。

重要な改善余地として**判断を要する**のは、(1)norm節約が生成費を上回る表現構造、(2)同情報の既知split法を超える構造、(3)既存共有後の回路生成、(4)moment/rangeと古典費を払う安いsampling、(5)native辞書と全wrapperの交換条件、(6)異構成roundの共通energy接続、(7)接続性による費用構造。残る可能性の列挙であり、有効な算法が存在すると示した結論ではない。

過去研究の主な制約：small normは安い回路を意味しない。bias低下はshot/追加backboneで相殺される。強い対照を入れると選択一致・既知法への還元が残る例がある。H4 C1ではparent縮約後にもchild回転が残り、logical差が小さい。basis非control・同basis共有・step境界融合は実装済みで、その欠如を改善前提にできない。technical failureと性能negativeは分け、原因未確定をCAUSE_UNRESOLVEDとした。

## 7課題と依存する証拠

|課題|中心の問い|必要な既存証拠／まだ確定していない点|
|---|---|---|
|RQ-01 表現・軌道|normと実装費を同時に下げる構造は何か|REP-09強対照、H2四方式、N1/QC/THRIFTの限界。旧UWCの成功は根拠にしない|
|RQ-02 分割・grouping|非prefixで同情報の既知partition/BCHを超えるか|PR-2/PMの限定支持、P-D/B-F/BM/R3/P-Cの反証。新しい具体差が不足|
|RQ-03 回路生成|共通共有後に大きい費用が残るか|WP06b、P-A、M2 actual SOURCE、Gaussian closure。basis寄与は未分解|
|RQ-04 sampling実行可能性|古典生成費と有限confidenceを払って得するか|時間相関監査、P5/CTS、C1 summary/source、B3、M1-S。安い生成差は未確立|
|RQ-05 native機構|辞書・配分・合成配置の利益が全wrapperへ残るか|SP/S4/D系列/G9、C1制約。whole wrapper・多step・prep移送は未知|
|RQ-06 多round構成|構成を変えても同じE0を安く読めるか|WP/PM/M1-S/関連PF。共通energy bridge・増分利益が未確立|
|RQ-07 接続性|既知routing後にもPR特有の課題があるか|topology-free条件と既知FSWAP/Givens生成。費用寄与・独立差は未測定|

各課題の12項目・停止条件・固定URLは[OPEN_RESEARCH_PROBLEMS](OPEN_RESEARCH_PROBLEMS.md)、強い対照と前回10候補の8論点再評価は[対応表](STRONG_BASELINE_AND_OVERLAP_MATRIX.md)。統合・分離の根拠も記録した。既知PR/composite、SPRINT/GRADE、Pauli/DF・圧縮表現、compact BCH/cache、Euler/IS、P5/CTS、Gaussian/routingを名称だけで新提案にしない。

## Phase 2Bへ渡す判断

[PHASE2B_REVIEW_PREPARATION](PHASE2B_REVIEW_PREPARATION.md)に、科学的中心問題、既知の解決範囲、負の結果、価値を左右する仮説、根拠十分／不足、課題ごとの独立質問と必須SOURCE・結果の固定GitHub URLを収録した。[PRIMARY_SOURCE_GUIDE](PRIMARY_SOURCE_GUIDE.md)から直接取得できる。資料が揃うことは性能・新規性が証明されたことではなく、RQ-06/07等の前提不足そのものも判断対象。

次の判断は、どの問いを残すか、同情報既知法との独立差をどこに要求するか、最小識別証拠の取得に価値があるか。Codexは正式な順位・GO/NO-GOを採択していない。

## 公開・保全・STOP

branch：`research/pr-algorithm-improvement-phase2a-20261023`、base：`a9bdcf1d44061daceffe896637e2c92fcb3353b4`。payloadの固定commit・公開bytes/SHA256/Git blobは[PUBLICATION](PUBLICATION.json)、文書・参照検査は[VERIFICATION](VERIFICATION.json)。最終receipt-only HEADと最終再取得はチャットで報告する。変更範囲は新namespaceだけ。

UWC候補7件はFOUND_UNVERIFIED、修正canonical Cgs/元anchorは今回利用不可、旧DFはDO NOT USE / staleを維持。[EVIDENCE_LIMITATIONS](EVIDENCE_LIMITATIONS.md)を参照。追加環境探索・原本代替計算はしない。元SOURCE/raw/grant/ledger/marker/STOP・原稿・元dirty checkout・他branch/worktreeを変更しない。既存限定成果を外部再現やimmutable CIへ格上げしない。

新科学計算・raw再解析／再ranking・simulation・sampling・build/compile/synthesis・新算法設計/実装・新非自明証明・旧実験再開・新grant・詳細GPT/Claudeレビュー・正式新規性/科学GO/投稿GOは0。評価方法再設計と四次PF逆転原因は別チャットのまま。**今回のmandatory STOP。**

次担当は**ユーザー**：Phase 2B科学レビュー開始を明示承認する。承認後の**GPT**：研究全体の方向性を扱うLevel 1レビュー（ユーザー指示ではGPT-6 Pro推奨）で7課題の価値と選定を判断する。その後の**Codex**による設計・実装・科学実行は、選定とは別の対象限定指示・実行承認を要する。今回の完了から自動開始しない。
