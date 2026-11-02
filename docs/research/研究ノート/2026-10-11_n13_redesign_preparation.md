# 2026-10-11 N1/N3再設計の準備

利用者は完了済みN1/N2/N3科学レビューを提示し、N1の誤差費用込みselectorと
N3の結合構造を残した係数保証を優先し、N2を固定SのL1修正に限定する方針を示した。
中心テーマは未採択で、「次の作業は」という質問をscience実行認可にしないとの明示条件を受けた。

[限定scope v1](../n13_redesign_limited_scope_v1_20261011.md)へ、旧44 factor候補/178 pairs、
既知二次PF係数bound、exact fallback、方向付き一外部軌道Schur条件、4 N3 inputs、固定S一件を具体化した。
N3は一体SPD係数solveと非負densityからBB†≤κ(A−L)を構成し、係数U/cから採否を検査する案。
真のactive最低値は証明中の記号だけで、selectorは参照しない。
5 modes成功、改善率、新規性を先に約束せず、未判定/改善なしも保存する。

旧公開ae743445から独立worktreeを作り、provided reviewをbyte同一copyした。
文書/source静的読解、候補件数metadata、input/hash、CPU/memory/device inventoryと準備資料の整合性だけを確認した。
science run、LP solve、対角化、compile、付録replay、scientific testsは0。
旧source/結果とroot dirty worktreeを保持し、新science module/runner/testは未実装。

wall/CPU-time capは置かず、候補/call/memory/output/log/bit幅上限を固定した。
科学実行前に実装・丸め条件・source-bound plan・資源/reference検証と別の実行指示が必要。
旧GPT科学レビューは完了、新結果が出た後だけGPTへ研究価値の判断を戻す。
`DESIGN_PREPARED_SCIENCE_NOT_AUTHORIZED`、中心仮説null、次science=false、mandatory STOP。
