# C1成果総括・一時凍結・G6論文反映候補

採択済み方針は **C1追加研究を一時凍結し、G6数学・方法論研究を維持する**。状態は `C1_PAUSED_AFTER_EXPLORATORY_RESOURCE_ASSESSMENT`。恒久終了・G6の否定・投稿GOではない。今回の明示指示に基づく文書整理で、新しい科学レビュー・科学計算は行わない。[方針記録](research_state.json)。

|資料|内容|
|---|---|
|[C1成果総括](C1_RESEARCH_OUTCOMES.md)|入力・再現性、失敗/修正、限定correctness、whole-path、G6本体との境界|
|[資源観測と限界](C1_RESOURCE_EVIDENCE_AND_LIMITATIONS.md)|3方式の固定summary引用、費用層、実装差、UNKNOWN|
|[原稿挿入候補](G6_MANUSCRIPT_INSERTION_CANDIDATES.md)|英語補助節/limitations、supplement候補、不支持claim。正式改稿ではない|
|[凍結・再開条件](C1_PAUSE_AND_RESTART_CONDITIONS.md)|凍結範囲、保全、未採択の再開判断条件、承認境界|
|[handoff](handoff.md)|短い引継ぎ・次担当|
|[SOURCE inventory](source_inventory.json)|固定commit/path/bytes/SHA256/Git blob、証拠取得範囲|
|[literal引用](quoted_resource_values.json)|summary fieldを無加工転記。新統計・順位なし|
|[保全snapshot](preservation_snapshot.json) / [検証](preservation_verification.json)|既存C1/v1.1/private原本・Git tree非変更|
|[公開receipt](publication_verification.json)|payload公開HEADと全file byte/SHA照合。receipt自身は子commitで追加し最終HEADでも照合|

基点 `bb9059bacb697e1105e63c1b3826b0c34886ad15`、branch `track-b-c1-freeze-and-manuscript-synthesis-20261021`、追加先 `docs/research/track_b_c1_freeze_synthesis_20261021/`。旧原稿v1.1固定参照は `243eea8e840f3c472753f7935e510391626fd325`。既存namespace/原稿/索引/全体statusは変更せず、このnamespaceの新policy recordに今回の採択を記録する（今回のユーザー指定による編集範囲）。

新しいH4/保存raw再解析/統計/改善率/順位/N/回路/compile/synthesis/shot/新証明/文献調査/詳細新規性レビューは0。科学SOURCEをimport・実行していない。許可されたJSON/CSV表示・引用とGit/hash/リンクの管理検証だけを行う。主エージェントの編集・管理照合で、別AI/外部査読者による独立レビューではない。

既存資料の丸ごとcopy・rawの重複公開はせず、根拠を固定commitリンクへ戻す。旧P5 failureとP5R成功は別来歴、過去grant/marker/STOPを維持。私的レビュー原本本文を公開せず、他worktree/他job/共有環境に操作しない。公開・保全・clean照合後mandatory STOP。

非公開原本の照合詳細（絶対パス・hash）はローカル専用記録へ保持し、公開packetには含めない。自動承認レビューの公開拒否を受け、公開保全snapshotは件数・確認結果に限定した。
