# Track B：独立論文の中心的貢献に関する技術的事前調査

2026-10-18 JST。基点 `101bf3426fcd04cc0c01c0d202d0916268765d97`。ユーザーの指示に基づく資料取得・SOURCE静的読取り・既存方法の対応比較・主張設計であり、新しい重要科学レビュー、論文草稿、研究採択、投稿GOではない。

**中心に残る候補は、同じ有限Taylor operator meanに対する、形式returnの局所係数取得・normalizerを要求しない生成・有限bitの誤差会計を接続した構成である。ただし、同じ情報と最適化機会を与えた既知法に対する独立差と資源価値を同時に示す証拠は不足している。Track B単独の独立論文としての中心的貢献は、現在の保存証拠だけでは確立していない。**

| 方向 | 本調査の技術分類 | 理由 |
|---|---|---|
| A：新アルゴリズム | `A_INSUFFICIENT_EVIDENCE` | 局所生成・有限bit・P5専用生成は具体的な技術成果。Euler、Green再帰、ISは既知で、全結合の同費用での還元・非還元と実用的利益は未確定 |
| B：評価・設計方法論 | `B_DESCRIPTIVE_RESULTS_ONLY` | 保証と価格による順位反転、配分の限定的利益、失敗条件は記録済み。既知の統計費用×回路費用評価を越える一般的な選択則は未確立 |

これは「論文にならないことの証明」ではない。Aを一括して `A_KNOWN_METHOD_ONLY` と判定する根拠もない。狭い構成ノート・再現可能な資源診断としての価値と、独立した新方法の論文を支持する根拠を区別する。

## 資料の入口

- [Paper Claim Sheet](paper_claim_sheet.md)：A/Bそれぞれの仮題、主張案、現在の支持、不成立条件。約1〜2ページ相当の要約。
- [一次文献・方法差比較](prior_art_method_comparison.md)：数学的包含、生成・情報アクセス、保証、資源効果を別々に照合。
- [既存成果の利用可能性](results_usability_map.md)：G6–G10、D1–D3、S4、SP/R1、closureを固定証拠へ接続。
- [最小追加証拠の提案](minimal_evidence_proposal.md)：新契約・実行計画ではなく、論文主張の欠測と反証条件。
- [GPT・ユーザー向けハンドオフ](handoff.md)：次の判断、未解決事項、承認境界。
- [技術分類metadata](technical_classification.json)、[固定証拠索引](evidence_index.json)、[一次文献取得・読解範囲](literature_scope.json)、[SOURCE静的読取り](source_static_read_inventory.json)。
- [静的・保全検査](verification.json)、[公開照合receipt](publication_receipt.json)。これらは科学的correctnessの再検証ではない。

## 読み方と保全

〔既存〕は固定報告・既存証明・既存レビュー、〔一次〕は今回取得した論文の該当節、〔技術推論〕は方法対応から本調査で整理した判断を示す。既存レビュー中の派生集計は当該レビューの分析として引用し、本調査の新数値解析にはしない。数値は既存の文章・保存表に記載された値の引用であり、rawからの再計算・新ランキングは0件。

10/15のD＋E採択、Parent allocation現行探索のSTOP、各closureを維持する。研究④の[新テーマ選定保留](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/7eff8a33bef4ee1f29698ab10c6da46988dc8db5/docs/research/qc_iv_selection_hold_20261018/decision_record.md)も別系列として保持する。古い基点branchに後続他研究の状態をmergeしていない。本資料はTrack Bの追補であり、全研究の最新版を集約するbranchではない。

専用cloneで文書・metadataだけを追加し、親worktree、他研究、旧科学SOURCE・raw・契約・authorization・grant・ledger・marker・消費済みclaimは変更しない。新科学計算、SOURCE実装・実行、compile、最適化、実験契約・実行権限発行はすべて0件。公開・保全確認後 **mandatory STOP**。

原指示はCRLFを含む原bytesのまま保存した。標準diff whitespace検査はそのCRを警告するため、原指示以外の標準検査と、全ファイルを対象としたコマンド限定 `cr-at-eol` 検査を併記する。repository設定・原本・既存SOURCEを修正して警告を消したものではない。
