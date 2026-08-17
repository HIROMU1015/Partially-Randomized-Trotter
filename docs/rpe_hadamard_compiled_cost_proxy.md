# Large-q RPE Hadamard compiled-cost affine proxy

`rpe_hadamard_compiled_cost_proxy` fits and validates a deterministic
extrapolation model from a complete schema-v2 medium-q benchmark dataset.  Its
scope remains one measured Hadamard interrogation without state preparation:

```text
single_hadamard_interrogation_without_state_preparation
```

It does not execute a backend or quantum shots, model noise, reconstruct a
phase, aggregate a full RPE experiment, or connect its predictions to the
short-round provider or resource accounting.

## Fit model and partition isolation

For each axis `cosine | sine` and each of the six fixed compiled-cost metrics,
the module fits one model

$$
\widehat C_{a,\mu}(q_m)=s_{a,\mu}q_m+b_{a,\mu}.
$$

The coefficients are neither clamped nor rounded.  Prediction accepts only a
positive power-of-two $q_m$.  At least two distinct calibration values are
required.

`RPEHadamardCompiledCostProxyFitRequest` selects either `uniform` or
`inverse_variance` weighting.  Uniform fitting assigns every calibration point
weight one.  Inverse-variance fitting uses $1/\mathrm{SE}^2$ and rejects a
missing, nonfinite, or nonpositive standard error; it never falls back to
uniform weighting.

The public fit entry point extracts an immutable calibration-only snapshot.
Only that snapshot reaches the regression core.  Holdout means, standard
errors, and point fingerprints are absent from the fit specification,
calibration-subset fingerprint, coefficients, residuals, and fit fingerprint.
The full source-dataset fingerprint is retained separately for provenance, so
changing valid holdout data can change the proxy-record fingerprint but cannot
change the fitted model or fit fingerprint.  Fitting and validation do not
modify the benchmark dataset or its audit flags.

## Holdout validation

`RPEHadamardCompiledCostProxyValidationRequest` requires at least one holdout
$q_m$ and one explicit tolerance for every metric.  It checks that the proxy
and holdout dataset have identical DF, partial-S2, time-step, construction,
compiler/backend, and circuit-scope identities.  It applies the existing
coefficients without refitting.

For observed mean $C_{\mathrm{obs}}$ and optional standard error `SE`, an entry
passes only when the prediction is finite and nonnegative and

$$
\left|\widehat C-C_{\mathrm{obs}}\right|
\le
\tau_{\mathrm{abs}}
+\tau_{\mathrm{rel}}\left|C_{\mathrm{obs}}\right|
+z_{\mathrm{SE}}\,\mathrm{SE}.
$$

An absent standard error contributes zero.  When the observed mean is zero,
relative error is `null`.  A finite negative prediction is recorded unchanged
and fails.  A nonfinite prediction is represented by `predicted_cost=null`
plus `prediction_nonfinite_kind`, and also fails.  Overall validation passes
only if every axis/metric/holdout entry passes.

The result records only the tested holdout $q_m$ range and fixes
`accuracy_guaranteed_beyond_validated_range=false`.  It must not be interpreted
as evidence outside that range.

## Fingerprints and JSON

The proxy schema is
`rpe_hadamard_compiled_cost_affine_proxy_v1`; the validation schema is
`rpe_hadamard_compiled_cost_proxy_validation_v1`.  Both provide canonical JSON
round trips.

The following identities are separate:

- calibration-subset fingerprint;
- fit-specification fingerprint, including weighting;
- holdout-independent fit fingerprint;
- full proxy-record fingerprint, including source-dataset provenance;
- holdout-subset fingerprint;
- acceptance-policy fingerprint;
- validation fingerprint, covering the fit, holdout, policy, and results.

JSON loaders require canonical 64-character lowercase SHA-256 values.  Empty,
missing, malformed, or content-inconsistent fingerprints are rejected.
Coefficients, residual aggregates, record counts, validated ranges, and
pass/fail summaries are recomputed and checked during loading.
