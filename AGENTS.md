# Repository guidance for Codex

## No sub-agents (user instruction, 2026-10-15)

Work in the primary agent only. Do not spawn sub-agents or delegate implementation,
investigation, tests, or reviews to other AI agents. A request for an independent
review or parallel work does not authorize delegation; only a later explicit user
instruction permitting sub-agents can change this rule.

Perform technical checks in the primary agent and describe them accurately. Do not
present the author's own checks as an independent review by a separate reviewer.
This rule concerns AI-agent delegation; computational workers and parallel
scientific calculations remain subject to their separate execution authorization.

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


## H2 isolated resource work (user instruction, 2026-10-19)

- This worktree and branch are reserved for H2. Start the substantive H2 evaluation
  in a new Codex chat, separately from the ongoing H4 validation chat.
- First read `/home/AbeHiromu/projects/AGENTS.md` and
  `docs/research/track_a_h2_pr_resource_handoff_20261019.md`.
- Edit only this H2 worktree and its own private directories. Existing H4/H6/H8
  SOURCE, contracts, grants, ledgers, markers, STOPs, results, runtime, processes,
  branches, and worktrees are read-only references for this task.
- Reuse implementation from pinned Git objects or H2-owned snapshots. Do not run
  inherited H4 runners directly: they may bind H4-specific inputs and outputs.
  H2 adapters must bind H2 paths and a sealed H2 SOURCE/input/runtime closure.
- H2 public preparation/evidence root:
  `artifacts/resource_applicability/track_a_h2_pr_resource/2026-10-19/`.
  H2 private root: `.h2_pr_resource_private/2026-10-19/`.
  Reserve distinct stage/attempt namespaces before each approved run; do not
  pre-create scientific output, grants, ledgers, or one-shot markers at setup.
- Each new scientific stage needs explicit user approval bound to its fixed
  SOURCE, input or generation recipe, and scientific-call budget; then run once
  and mandatory STOP. No automatic retry or automatic next-stage execution.
- The requested terminal target is the final P seal. P scientific sampling and
  compile, detailed scientific review, and final paper evidence freeze are out
  of scope. No stage is authorized by this environment setup.
- Other worktrees' changing HEAD/status, temporary PSI/CPU contention, and
  observation delays are not blanket freeze or forced-stop conditions.
- Current shared AGENTS resource policy takes precedence over historical run
  limits. Scientific-call budgets remain explicit scope controls. Do not alter
  shared settings, environments, or other users' processes/files.
- Keep proposed clarification choices pending until the user adopts them. Setup
  does not authorize publication; push only within a verified approval scope.


## H2 current chat designation (user clarification, 2026-10-19)

The user stated: 「これがその新規チャットです」. This chat is the designated
independent H2 chat. Continue its static work here; do not ask for another chat.
Scientific stage execution still requires separate explicit approval. Current
F preparation: `docs/research/track_a_h2_phase0_f_preparation_20261019.md`.
