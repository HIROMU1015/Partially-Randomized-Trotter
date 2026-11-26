# Repository guidance for Codex

## RomuTailのH4 RPE探索

- 利用者は探索の範囲や開始・停止・再開をCodexへ指示する。通常入口のStart・Resumeを実行する担当はCodexとする。利用者にコマンドやスクリプトの直接実行を依頼しない。回収・照合・容量整理の常設処理は新規計算の投入や再試行を行わない。
- 詳細は[H4 RPE分散探索の運用ルール](docs/research/rpe_distributed_operations.md)を参照する。

## H-chain最適化の分子間の改善共有（利用者指示、2026-11-27 JST）

- H4・H6・H8などのRPE最適化で中央制御・最適化器・ワーカーの修正を行う際は、実装前に他の分子が実際に使用中の中央制御・ワーカー・運用設定と、関連する最新版・作業ブランチを確認する。Gitの最新版と稼働中の固定版を区別し、他チャットで進行中の改善も取りこぼさないようにする。
- 対象の修正だけでなく、他分子で既に解決されている同種の問題、タスク配分・資源管理・回収・計算の高速化等で再利用できる改善を確認し、現在の修正に有用で互換性があるものは取り込む。逆に今回の修正が他分子にも有用かを確認し、適用可能な範囲でその実装にも統合・反映する。既存実装を重複して作らない。
- 毎回の共通ブランチへの強制統合や、全分子の稼働版の一律更新は不要。分子・計算環境・既存実行契約に固有の差分は保持し、適用可能性と実際の効果で判断する。稼働RUNの無断切替、固定科学SOURCE・入力・数値精度・探索条件の変更、計算ノードの解放、他チャットの未確定作業の上書きは行わない。
- 通常の作業報告に、確認した他分子の稼働版または確認できなかった理由、再利用・統合した改善、意図的に保留した差分を簡潔に記載する。専用の管理台帳や追加承認手続は作らない。

## Repository entry point

Before interpreting files by name, read `PROJECT_MAP.md`. It classifies the
library, runners, tests, evidence artifacts, research documents, presentation
material, and historical files.

Use these source-of-truth priorities:

1. current synthesis and normative research documents under `docs/research/`;
2. `VALIDATION_STATUS.md` and `artifacts/validation_manifest.json` for evidence status;
3. validation-specific documents and machine-readable artifacts for numerical claims;
4. dated research notes for decision history;
5. presentation plans, old protocols, notebooks, and implementation prompts only as context.

Do not treat `BentoSlide構成案*.md`, `Partial Randomized Study Protocol.md`,
`codex-inst.md`, `README_partial_randomized_pf.md`, or
`abe_trotter_project.ipynb` as the current project specification.

## Research-material tasks

When asked to explain this project or create slides, reports, summaries, or other
research-sharing material, start by reading
`docs/research/研究概要・現状.md` completely. It is the shortest current synthesis of
the background, objective, adopted assumptions, local validation results, and open
work.

Then read the documents linked from its evidence map that are relevant to the
requested material. In particular:

1. Use `docs/research/研究目的・研究課題.md` and the other main research documents
   for the normative research design.
2. Use `VALIDATION_STATUS.md` and `artifacts/validation_manifest.json` to determine
   reproducibility and evidence status.
3. Use the validation-specific documents and their machine-readable artifacts for
   numerical claims.
4. Treat `docs/research/研究ノート/` as chronological decision history, not as the
   current specification. A later entry may supersede an earlier tentative choice.

## Claim discipline

- Distinguish established theory, locally validated results, implementation-only
  capability, pending validation, and final scientific conclusions.
- Always state the Hamiltonian/model, geometry, basis, DF rank or rank policy,
  split `L_D`, delta window, and validation scope when quoting numerical results.
- Do not present dirty-worktree artifacts or local test results as immutable CI or
  externally reproduced evidence.
- Do not use the stale DF screening or prose-only UWC values as current scientific
  results.
- Do not claim that the final total-cost evaluation has been performed. The current
  priority is validating the approximations that will later feed that evaluation.
- Do not call `C_use` a rigorous upper bound. It is an empirical envelope over the
  explicitly executed delta window.
- Do not infer an H12 coefficient from H4 or H6. H12 remains undecided until the
  documented GPU runs are completed for the shortlisted `L_D` values.
- Keep QPE/RPE statistical error separate from deterministic PF coefficient `C`.

## Documentation updates

When a research decision or validation result changes:

1. update `docs/research/研究概要・現状.md`;
2. update the relevant normative or validation-specific document;
3. append the decision and reason to the dated research note rather than silently
   rewriting its earlier history; and
4. update the machine-readable manifest when the evidence inventory or status
   changes.

When adding a new validation path, keep the library module, runner, test,
validation document, and artifact directory discoverable through the indexes in
`PROJECT_MAP.md`, `scripts/README.md`, `src/trotterlib/README.md`, and
`docs/README.md`. Do not move evidence or rename public paths merely for tidiness
without updating all references and provenance records.
