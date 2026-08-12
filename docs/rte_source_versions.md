# RTE primary-source versions

Verification date: 2026-08-01 (Asia/Tokyo)

Primary implementation source:

- Günther et al., *Phase estimation with partially randomized time evolution*
- arXiv:2503.05647v2, last revised 2026-07-10 UTC
- 44 pages
- official page: <https://arxiv.org/abs/2503.05647>
- downloaded verification PDF SHA-256:
  `000068f58332b5f24c6f17bd3b08ea22de1908e1ef2242a4ffc6f644aaf9aa61`
- the downloaded v2 verification copy was temporary and is not retained in
  the repository

Repository legacy copy:

- file: `Phase estimation with partially randomized time evolution.pdf`
- arXiv:2503.05647v1, dated 2025-03-10
- 42 pages
- SHA-256:
  `63d273ee800a5368a45266130a093689144553c86903c60342d9ce16adb62c47`
- status: legacy reference only; do not use it as the primary basis for RTE
  equations or claims about v2

Equation routing for the current implementation:

- finite RTE: v2 Appendix A, Eqs. (A18)–(A31)
- partially randomized composition and normalization: v2 Appendix A,
  Eqs. (A32)–(A40)

## Versioned analytic cost baselines

The finite-RTE and RPE-accounting APIs do not implicitly select either of the
following simplified analytic cost baselines.  Callers that need a comparison
must select a model ID explicitly.

- `legacy_repo_model` preserves the historical repository expression
  `(280/9) * G_rand * gamma * (0.1*pi)^2`, including the repository-only
  convention `gamma=1` for qDRIFT and `gamma=2` for RTE.  The printed `280/9`
  can be traced to arXiv v1 Appendix E, Eq. (E5), but its derivation is
  unresolved: the adjacent v1 schedule
  `N_m=B^2(4+11(M-m))` gives `N_M=4B^2` and `D=11B^2`, which would produce
  `8(N_M/3+D/9)/B^2=184/9`, not `280/9`.  No primary-source definition of the
  repository's extra `gamma` multiplier has been confirmed.  This model is for
  numerical backward compatibility only.
- `pr_paper_v2_model` is the RTE-only expression in arXiv v2 Appendix E,
  Eq. (E22).  The corrected schedule `N_m=B^2(11+4(M-m))` gives
  `8(11/3+4/9)=296/9`.  It has no extra gamma multiplier and must not be used
  under a qDRIFT label.

The corresponding explicit API is `analytic_baseline_definition`,
`randomized_prefactor_b0_for_model`, and
`randomized_prefactor_B_for_model`.  Existing unversioned prefactor and
optimizer APIs remain pinned to the legacy behavior so that old analyses are
not silently reinterpreted.
