# SOURCE・文献取得・保全・終了監査

2026-10-21 JST。文献調査の作業監査であり、科学証拠の最終freeze・独立科学レビュー・過去runの再認証ではない。

## 1. 作業範囲・環境

共通 `/home/AbeHiromu/projects/AGENTS.md` を明示的に読み、指定commitのAGENTS、PROJECT_MAP、研究概要を入口にした。sub-agentは0。過去のCPU/worker/RAM契約を新規上限として継承していない。本作業には科学workerがなく、計算資源・他processの制御は変更していない。

専用branch `research/pr-prior-art-novelty-20261021`、基点 `80b7a35c4dbf683f952293eeb55944bc10b9efc8`。専用worktreeのsparse checkoutを用い、旧作業ツリーのbranch/index/内容を切り替えない。新規writeはこの調査directoryと専用文献cache/一時readerに限定。Gitの追加worktree/新branch登録とlocal commitは依頼済み範囲。

通常sandboxのshell起動は `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` で失敗した。以後のshell読取・専用writeは同じ範囲のescalationを指定して実行した。自動承認reviewによる拒否は今回発生していない。H8過去文書の公開拒否は歴史的記録であり、今回の実行拒否とは別。

## 2. 固定SOURCE・証拠

[全固定path/blob/SHA](audit/source_inventory.json)。基点のimmutable Git objectを読む。指定3commitはGitHub connectorのread-only fetch_commitで返却SHAを確認し、[remote観測](audit/github_observations.json)へ記録した。この観測を全blob/全rawのremote照合とは表現しない。

|役割|固定commit/記録|扱い|
|---|---|---|
|共通プロトコルv2・概要・現行q感度採用|80b7a35c|現行条件と保存結論の入口。旧追記は時点履歴|
|H4/H6/H8独立表|同基点のmain_table_proposed.jsonと統合文書|保存方向・n・cohort・exact値を読取。新しい比/検定なし|
|H2独立P|v2 §10、result f2d1a86c、受入8fbe7d77への参照|主表6条件とは別に2条件を表示|
|H4 B0I|98cdb35c結果を参照する基点の結果文書、SOURCE924b779c|DF12 geometry系列。DF10 Pへ移送しない|
|H6独立P|3cccfcb2・採用877e7c28、SOURCEbf0c26da|事前PF優位方向のREPRODUCEDも保持|
|H8公開証拠packet|a6246f03|redacted receiptsの到達範囲を確認。実行runnerやレビューを起動しない|
|H8原profile補足|5f267e96|原profile回収状態へ更新。保存回路0/192の制約は保持|
|Mechanism|1ff6ac0b説明32比較/64軸への基点参照|恒等的会計と説明因子を新理論にしない|
|理論対compiled|f1575262、修正結果6e2b7eb6、counter SOURCEf01fa5fb、会計SOURCEda8eb64b|旧S4誤りを修正した後の結果を引用。科学SOURCEを実行しない|
|MC変動/正式補足|295b93d5 / bee9ae99、元MC defa5f6d、SOURCE78352b86|既存bootstrap/経験分布。新しい独立cohortに数えない|
|q価格100scope|80b7a35c / result3983fe4f|既存CEMと保存集計を読取。価格/shot/順位を再計算しない|
|finite-RTE/control/shot実装|rte.py、87b6b9f4 native builder、既存precision会計|source文字列を静的確認。import/関数実行なし|

SOURCE/raw/結果/grant/ledger/marker/旧STOPは変更しない。今回のSTOPは新規調査成果の終了記録だけに作成する。既存の詳細科学レビューや最終evidence freezeを開始しない。

## 3. 文献探索・取得の記録

[39論文catalog](audit/literature_catalog.json)、[取得URL/HTTP/bytes/SHA](audit/literature_acquisition.json)、[Crossref初回](audit/crossref.json)と[再取得](audit/crossref_retry.json)、[reader来歴](audit/reader_provenance.json)を保存した。初公開、使用版、誌版を分ける。

検索系列は、(a) P01の原論文/引用（Wan、qDRIFT、Composite、DF、RPE、truncation）、(b) Compositeの引用系列（SparSto、Jin–Li、Pocrnic、Kiss）、(c) 2025–2026年のprecision/variance/PR応用、(d) DF/FT/高次PF・明示回路比較。特定著者の系列だけで結論を決めず、各task・指標の相違を比較した。[主要query記録](audit/search_log.md)を参照。

重要論文はPDF本文の関連節・式・caption/Methodsを確認した。一方、背景文献の多くはabstract/導入だけの内容評価である。各文献の `reading_scope` を正本とし、「取得したPDFのページ数」を「読了ページ数」と呼ばない。全証明・数値実験を再現する監査はしない。

Crossref初回には429が発生した。取得間隔を空けた逐次再取得で書誌を回収し、失敗記録を削除しない。Pocrnic誌版PDFの直接取得429、Mehendale誌版PDF403、P01 HTMLの不在version URL/404、versioned Qiskit API取得失敗を未取得として残す。arXiv、誌版web、一次repositoryによる代替確認と、abstractのみの判断を区別する。

commit対象のcatalog/Crossref記録は書誌項目に絞り、abstract全文・reference list・所属/資金情報の長文を省いた。取得レスポンスの完全な保存物は管理外cacheへ移し、[変換来歴](audit/metadata_projection.json)にそのSHAと保存先を記録した。取得receiptのHTTP payload SHAと、この書誌projection自体のSHAは別である。

PDF readerはpypdf/fontToolsを作業専用一時directoryへ展開した。共有Python/OS/venvを変更していない。初回の文字抽出にfontTools不足、図formの一部抽出省略という警告があり、追加読取ではfontToolsを局所展開した。文字抽出のみから図のpixel・全数式の完全性を認定しない。P01のPDF screenshotは一部cache miss。原PDFと式番号/本文の照合を基準とし、図の全pixelを目視精査したとはしない。

取得PDF/HTML/抽出本文はGit管理外 `literature_cache/` に保存した。新規commitには文献本文PDF、第三者コードの全文、private科学rawを含めない。metadata/SHA・自分の比較要約とBibTeXを保存する。ローカルcacheの所在とSHAは取得記録から追跡可能。

## 4. 保全と終了確認

[構造確認script](audit/check_documents.py) は標準libraryだけで実行する。固定source bytes/blobの一致、新規write範囲、Markdown相対path、文献ID/Claim ID、CSV/JSON/BibTeX構造、取得cacheのSHAを確認する。科学module/runnerをimportせず、signal/統計/費用/比/順位を計算しない。文書の転記とhash照合は科学結果の再計算とは別。

[終了監査](audit/final_audit.json)と[成果物SHA](audit/artifact_manifest.json)が実際の確認結果を記録する。検査はprimary Codex自身の文書確認であり、別担当の独立科学レビューではない。他worktreeのHEAD/statusは凍結条件にせず、他chatの通常更新があっても停止・修正しない。

実施しなかったもの：新規signal/sampling/simulation/compile/科学統計/再ランキング、bootstrapの再実行、保存値の再会計、原稿本文・既存原稿の改訂、希少RTE調査への介入、進行中の4次PF研究への介入、外部送信、GitHub push、詳細レビューの新規依頼、最終evidence freeze。今回起動したのは文献取得・文字抽出・文書生成/構造確認用のprocessのみ。

全成果を専用local commitに保存して **MANDATORY STOP**。commit SHAはGit履歴と最終チャットで確認できる（自己参照commit SHAを保存内容へ埋め込まない）。

初回commitはGitの著者情報未設定により作成されなかった。共有・repositoryのconfigは変更せず、commit commandだけに `user.name=Codex`、`user.email=codex@local.invalid` を指定する。これは今回の担当を表すローカル識別子であり、利用者本人の著者資格や実在メールを推定したものではない。stage範囲・diff体裁を確認し、active commit hookは存在しなかった。

次担当：**ユーザー**が新規公開範囲と、必要な場合のみ後続研究判断を行う。**Codex**はSTOPを維持し、承認前にpush・原稿作成・追加科学処理へ進まない。
