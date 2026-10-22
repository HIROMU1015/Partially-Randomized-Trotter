# PR改善 Phase 2A：障害と研究課題の具体化

2026-10-23 JST。**7件の未解決研究課題と比較・反証材料を準備した。正式順位・科学GOは未採択、Phase 2Bレビューは未開始。** Hamiltonian分解へ固定せず、回路生成・sampling・native合成・複数round・接続性を含める。

基点：a9bdcf1d44061daceffe896637e2c92fcb3353b4、専用branch：`research/pr-algorithm-improvement-phase2a-20261023`。前段階の70研究・10候補・3 repository/731 branch inventoryを再利用し、重要30研究単位の一次文書／結果／既存数学監査／SOURCEの対象箇所を再確認した。全30の全SOURCE/rawを再読したものではない。新しいbranch全数調査は行わない。[確認範囲](STUDY_SCOPE_AND_TRIAGE.md)と[各ファイルの取得・読解範囲](SOURCE_EVIDENCE_INDEX.json)を分ける。

最初に[handoff](handoff.md)、[課題定義](OPEN_RESEARCH_PROBLEMS.md)、[Phase 2B準備](PHASE2B_REVIEW_PREPARATION.md)を読む。根拠は固定SOURCE・保存結果へ戻れるようにした。

|成果物|内容|
|---|---|
|[README.md](README.md)|入口・範囲|
|[BOTTLENECK_AND_LEVER_MAP.md](BOTTLENECK_AND_LEVER_MAP.md)|7層以上の費用とA/B/C/D区分|
|[PAST_IMPROVEMENTS_AND_FAILURE_MODES.md](PAST_IMPROVEMENTS_AND_FAILURE_MODES.md)|主要23項目の到達点・確認機構・原因不明|
|[OPEN_RESEARCH_PROBLEMS.md](OPEN_RESEARCH_PROBLEMS.md)|7課題の12項目定義|
|[STRONG_BASELINE_AND_OVERLAP_MATRIX.md](STRONG_BASELINE_AND_OVERLAP_MATRIX.md)|baselineと10候補の8論点再評価|
|[EVIDENCE_LIMITATIONS.md](EVIDENCE_LIMITATIONS.md)|旧8GAP・条件不一致・未実施検証|
|[PHASE2B_REVIEW_PREPARATION.md](PHASE2B_REVIEW_PREPARATION.md)|課題ごとのレビュー質問・必須固定URL|
|[SOURCE_EVIDENCE_INDEX.json](SOURCE_EVIDENCE_INDEX.json)|SOURCE/結果/監査のidentityと確認範囲|
|[PRIMARY_SOURCE_GUIDE.md](PRIMARY_SOURCE_GUIDE.md)|固定一次資料への読み口|
|[RESEARCH_PROBLEMS.json](RESEARCH_PROBLEMS.json)|課題と10候補の機械可読定義|
|[FAILURE_MODE_CATALOG.json](FAILURE_MODE_CATALOG.json)|23項目の機械可読整理|
|[STUDY_SCOPE_AND_TRIAGE.md](STUDY_SCOPE_AND_TRIAGE.md)|70研究の確認深度と課題対応|
|[STUDY_SCOPE.json](STUDY_SCOPE.json)|確認範囲の機械可読台帳|
|[verify_documents.py](verify_documents.py)|科学コードを実行しない文書検査|
|[VERIFICATION.json](VERIFICATION.json)|文書・参照・保全検査結果|
|[PUBLICATION.json](PUBLICATION.json)|公開payloadとHTTP bytes照合記録|
|[handoff.md](handoff.md)|ユーザー/GPT向け結果とSTOP|

## 実施・未実施

実施：固定GitHub資料の取得、既存Git object/cache再利用、SOURCE excerptの静的読解／AST syntax、保存済みsummary/既存判定の引用、費用機構と反証の整理、12項目の問い定義、文書・参照・公開・保全検査。

未実施：科学SOURCEのimport/実行/変更、新科学計算・raw再解析／再ranking、モデル/状態生成、simulation・sampling・build/compile/native合成、新しい具体的算法の設計/実装・非自明証明、旧実験retry/resume、旧STOP解除、新grant、詳細GPT/Claudeレビュー、正式新規性・科学GO・投稿GO。PR評価方法自体の再設計と四次PF逆転原因究明は別チャットのまま。

新資料はこの独立namespaceだけに追加する。研究決定・validation結果の変更ではないため既存概要・normative文書・manifest・原稿を更新しない。元dirty checkoutと他branch/worktree、既存SOURCE・raw・grant・ledger・marker・STOPを保全する。

共通運用指示2026-10-17を実取得し§8.4を確認した。ローカル原本と公開保存copyは同じSHA256。静的作業はCPU affinity32、約54GB available RAM、約349GiB空きfilesystemを起動時に確認。Git object共有・重複排除・独立GitHub読取の並行化を使い、科学計算のCPU/RAM/worker capは作っていない。短いmetadata/Git読取はI/O中心、GPU処理なし。quota変更・共有環境／他process操作なし。speedup率は測定・主張しない。

第1・第2環境の公開探索報告を固定commitで確認し、UWC候補7件は来歴未照合、旧DFはDO NOT USE / staleとした。追加の環境探索・原本回収・代替再計算は行わない。

公開のpayload commitとbytes/Git blob照合は[PUBLICATION](PUBLICATION.json)、その後のreceipt-only最終HEADと全ファイル再照合は最終チャットに示す。技術PASSは科学的再現・CI・研究採択ではない。**公開・検査終了後mandatory STOP。**

次担当：ユーザーがPhase 2Bの開始を明示承認した後、GPTが7課題の研究価値・独立差・選定を検討する。設計・実装・新科学実行は、その対象を別に指示／承認した後の作業。
