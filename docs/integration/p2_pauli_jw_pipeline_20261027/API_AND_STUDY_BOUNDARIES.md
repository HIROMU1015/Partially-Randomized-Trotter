# P2 API and study boundaries

## Context and responsibility

PauliContext carries separate scientific SOURCE SHA, versioned SOURCE/STOP/result-index locators,
P2 CodeVersion, input/Hamiltonian/term-order/state/reference/norm identities, molecule/geometry/basis,
Pauli representation and dimensions/constant, logical method/candidate/prefix/q/r/K/S2,
effective CDF law/normalization, axes/preparation/control/phase, precision/epsilon,
compiler/estimand/metric/cost scope, seed address, exploration/confirmation role and cohort ID.
No DF rank or DF fragment partition is required. These are non-frozen entry records, not rewrites of old SOURCE schemas.

Adapter policy selects a small explicit set of existing module/attribute paths. `bind_core` resolves those normally
inside this checkout; it does not execute them or produce a scientific provider.
Core stage wrappers forward the distinct signatures, preserving rather than unifying their law, mean and account semantics.
The common trotterlib does not import the new research package.

|Logical entry|Existing config/law|Event/trajectory|Mean/target|Account|
|---|---|---|---|---|
|H4 HybridPR|`h4_pauli_pr_search.config(prefix,q,r,K)`; `law_descriptor(ham,cfg)`|`trajectory_seed(prefix,q,r,K,cohort,index)` then `central_trajectory(ham,cfg,seed=seed)`|`h4_pauli_pr_mean._mean_enclosure(law,raw,reference,bits)`|`h4_pauli_pr_search.account(target,prices,epsilon,descriptor)`|
|H6 HybridPR|`h6_hybrid_fairness_extension.config(cid)`; `law_descriptor(ham,cid)`|`trajectory(ham,cid,cohort,index)` owns seed generation|`mean_target(model,reference,law,cid)`|`account_target(target,prices,epsilon,law)`|
|H6 PureRTE|`pauli_rte_event_extension.config(r,K)`; `law_descriptor(ham,r,K)`|`trajectory(ham,molecule,r,K,cohort,index)` owns seed generation|`h6_mean(model,reference,law)`|`account_target(target,prices,epsilon,law)`|

Logical HybridPR/PureRTE map to the existing PR configuration with nonzero/zero prefix respectively.
H4 and H6 scopes are not interchangeable. The four H6 signatures are exactly the existing module's SIGNATURES;
the narrow H4 and pure-RTE entries do not add candidates to an old run's permission scope.
PF/B0/B0I remain existing kernel capabilities but are not connected adapters in this release.

## Preserved scientific distinctions

- Supplied terms/signs/constant go verbatim to the PauliHamiltonian constructor. No new sort, coefficient merging,
  cutoff, integral conversion, Hamiltonian/state generation or norm calculation is added.
- That constructor itself canonicalizes terms. Mocks do not verify the real canonicalization, hash, type invariants or operator equivalence;
  declared term-order/state/norm identifiers are checked, not authoritative real payload identity.
- Qiskit little-endian and descending-abs/label-tie order are explicit. Constant-once phase, central rotation control,
  +Im sine via Sdg before final H, raw norm without renormalization and state-preparation-excluded full-wrapper RZ scope stay separate fields.
- S2 is the product formula; K is maximum even Taylor event order, not a PF order. No law probabilities or K+1 polynomial is recomputed.
- Existing NumPy1.26.4/CDF53bit graph and original study trajectory/seed functions are referenced, never invoked by a real provider here.
  Exploration and confirmation roles/IDs are not merged or asserted mathematically independent.
- Compiler basis/opt/seed/version and absent topology/backend/layout/routing are explicit. The declared saved Qiskit1.3.0 profile is not an installed/runtime verification.
- Study accounts retain original epsilon conversion, normalization, guard, exact-integer, rho and cohort policies.
  Generic fixed_signal_shots and DF/RPE accounting are not interchangeable replacements.

## Stage data and qualification

`input → method_law → event_trajectory → signal_wrapper → shot_price → PipelineResult`.
Opaque Hamiltonian/model/state/law/body/wrapper/target objects remain in private in-memory payloads and are not serialized.
Every stage must return the complete context binding; SOURCE/input/state/norm/seed/compiler/cohort mismatches fail.
Explicit convention fields cannot be relabelled. Missing/UNQUALIFIED/FAILED/STOP halt the chain without retry.
A current STOP invokes no input/provider. Historical STOP locators remain historical; a fresh artificial fixture does not resume that study.

Synthetic records keep B/bias/normalization/N/G/SE/price/rare-event probability unacquired (`None`).
Zero, infinity and unobserved=zero substitutes are rejected. A successful route has only SYNTHETIC_END_TO_END_PASS,
with science_executed=false, numerical_equivalence_verified=false and reproduces_saved_results=false.
No real provider, loader, worker dispatch, retry/resume or statistical comparison is connected.

## Imports and output

Import/plan/synthetic need no numerical packages. Explicit real binding and optional output use normal trotterlib imports.
The original eager initializer is unchanged and may transitively import unrelated DF/phase modules;
new binding itself explicitly requests neither. This existing global behavior is not bypassed or certified.
Missing dependencies are visible and preserve their cause, with no fallback, alias loading or package installation.
Output resolves the original atomic writer before mkdir and requires a fresh directory. Existing output is not overwritten.

## Static compatibility versus science

Actual source signatures/dataclass fields are checked with AST and mock calls, not numerical execution.
Selected scientific SOURCE/base kernel blob matches: H4 PauliHamiltonian/PR-search/mean (3),
H6 Hybrid adapter/PauliHamiltonian (2), pure-RTE adapter/PauliHamiltonian (2).
Full dependency closure, real states/norms, effective-law/operator/control/phase equivalence,
compiled residual, statistical independence and qualified shots/G are unverified.
Existing dynamic snapshot names h4_b0_execution_private/hx_study/event_run and old launch contracts remain intact;
new normal-import wiring is not asserted to reproduce those frozen runs.
