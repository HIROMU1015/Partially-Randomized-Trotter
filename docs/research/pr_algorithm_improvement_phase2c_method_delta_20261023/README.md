# Phase 2C-R1：独立改善機構の限定追加設計

2026-10-23 JST。起点 `0c94a3d973b0ba0d24b3d277ab8316829739b278`。対象はRQ-01＋RQ-03、RQ-02は補助。RQ-04～07は保留のまま。

**技術的分類：`NO_METHOD_DELTA_FOUND_WITHIN_SCOPE`。** 今回の一つの追加案「反可換Pauli対のinvolution化」は、既知unitary partitioningと一般involution-RTEの組合せへ整理できた。実際の実行単位はPhase 2Cから変わるが、独立した新生成機構、PR固有の追加構造、新しい取得法・保証を具体化できなかった。

これは今回の限定設計の結果であり、RQ-01/03全体の不可能性、性能negative、研究全体のNO-GOではない。元のPhase 2Cの `KNOWN_METHOD_APPLICATION_CANDIDATE` / `INDEPENDENT_METHOD_DELTA_UNESTABLISHED` / `NO_DEFENSIBLE_NEW_METHOD_YET` は書き換えていない。既知法の応用価値が実証されたという判定も行っていない。

| 資料 | 内容 |
|---|---|
| [PHASE2C_BASELINE_DECOMPOSITION.md](PHASE2C_BASELINE_DECOMPOSITION.md) | 既存8要素を5分類で分解。価格・選択と量子生成の差を区別 |
| [METHOD_DELTA_DESIGN.md](METHOD_DELTA_DESIGN.md) | 追加案一つの数式、入出力、手順、回路接続、不成立理由 |
| [STRONG_BASELINE_COMPARISON.md](STRONG_BASELINE_COMPARISON.md) | D1～D5の同情報比較と既知構成への対応 |
| [SEMANTICS_AND_OBLIGATIONS.md](SEMANTICS_AND_OBLIGATIONS.md) | PO-01～09の継承、phase・sector・有限mean・有限bit |
| [RESOURCE_MECHANISM_AND_COUNTEREVIDENCE.md](RESOURCE_MECHANISM_AND_COUNTEREVIDENCE.md) | λ低下と回転追加、共有、古典費、過去の反証の限界 |
| [NEXT_STAGE_DECISION.md](NEXT_STAGE_DECISION.md) | 技術的分類、継続を勧めない範囲、担当・承認 |
| [MINIMAL_NEXT_VALIDATION.md](MINIMAL_NEXT_VALIDATION.md) | 今回は新検証案を発行しない理由。旧F1も未着手 |
| [SOURCE_EVIDENCE_INDEX.json](SOURCE_EVIDENCE_INDEX.json) | 固定一次資料、identity、実際の静的確認範囲 |
| [DECISION_SPEC.json](DECISION_SPEC.json) | 分類・作業境界の機械可読記録 |
| [handoff.md](handoff.md) | ユーザー・GPTへの引継ぎ |
| [VERIFICATION.json](VERIFICATION.json) / [PUBLICATION.json](PUBLICATION.json) | 文書・出典・公開の管理検査。科学的検証ではない |

実施：Phase 2C固定資料、必要なPhase 2A節、指定ローカルPhase 2Bレビュー、関連SOURCE・一次文献の静的確認、概念設計、文書管理検査と公開。Phase 2Bのレビュー時点の採択待ちより後の、ユーザーの採択済み方針に従った。private review原本は転載していない。

未実施：新しい科学計算、raw再解析、Hamiltonian/state/DF生成、sampling、simulation、回路build/compile、native価格取得、アルゴリズム実装、非自明な新証明、grant準備・発行・再利用、詳細科学レビュー。70研究・731 branchの再調査もしていない。

公開先branchは `research/pr-algorithm-improvement-phase2c-method-delta-20261023`。新namespaceへの文書追加だけで、元Phase 2Cと科学SOURCE・結果・入力・review・grant・ledger・marker・旧STOP・論文を保全する。**公開確認後にmandatory STOP。**
