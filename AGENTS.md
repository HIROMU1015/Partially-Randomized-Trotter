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

Scale documentation updates to the research significance of a change:

1. Individual researchers maintain their fixed SOURCE, raw results, REPORTs,
   and relevant normative or validation-specific documents. Record research
   decisions and reasons in the dated research note, preserving earlier history.
2. Update `docs/research/研究概要・現状.md` and research-wide indexes only when
   the change affects the overall synthesis or discoverability; routine minor
   changes do not require a full documentation update.
3. Update the machine-readable manifest when the scientific evidence inventory
   or status changes. Preserve updates required by an applicable fixed contract.
4. Handle the integrated roadmap through the responsibility rules below;
   updating an individual REPORT does not require an immediate roadmap edit.

When adding a new validation path, keep the library module, runner, test,
validation document, and artifact directory discoverable through the applicable
indexes in `PROJECT_MAP.md`, `scripts/README.md`, `src/trotterlib/README.md`, and
`docs/README.md`, updating affected entries as needed. Do not move evidence or
rename public paths merely for tidiness without updating all references and
provenance records.

## Integrated research roadmap: impact reports and updates (2026-11-14)

On completing research, implementation, or validation work, assess whether the
outcome materially affects:

1. progress toward the research objectives;
2. implementation or validation progress;
3. scientific qualification or the scope of a result;
4. unresolved questions;
5. the next necessary validation or implementation;
6. dependencies between tasks or their owners; or
7. formal adoption, execution approval, or mandatory STOP.

For a material impact, briefly include in the final report or an existing REPORT:
the affected research issue, completed implementation or validation, new evidence
with fixed SOURCE and REPORT links, resolved issues, remaining questions, and
next work with dependencies and approval requirements. This is handoff information
for the integration owner, not permission to edit the roadmap directly. Do not
create a new management file or ledger solely for this handoff.

Minor bug fixes, artificial or regression test additions, confirmations of an
unchanged conclusion, documentation edits without research impact, and internal
refactoring that changes neither execution conditions nor scientific qualification
normally require no roadmap update. A brief "roadmap update unnecessary" in the
final report is optional.

Individual researchers, including A/B/C/D, must not directly edit
[`docs/research/RESEARCH_ROADMAP.md`](docs/research/RESEARCH_ROADMAP.md) unless
explicitly instructed. A Codex explicitly assigned as integration owner checks
the individual fixed REPORTs and SOURCE and updates only materially affected
parts under its section 12. Important changes may be batched; an immediate roadmap
edit is not a completion condition for each researcher. Avoid concurrent edits
to the same roadmap.

The roadmap is an entry point for overall progress, open questions, dependencies,
and direction. Individual REPORTs, raw results, and fixed SOURCE remain the
scientific source of truth. Do not merge distinct research series, physical
candidates, scientific contracts, or SOURCE without checking their conditions.
Roadmap edits do not change scientific qualification, formal adoption, execution
approval, or mandatory STOP, and authorize no new scientific computation, compile,
experiment, or review. Before new scientific computation, compile, or experiments,
apply the existing-results overlap check in section 10.5 of
`research_project_instructions_unified_2026-11-14.md`.
