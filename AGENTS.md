# Repository guidance for Codex

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
