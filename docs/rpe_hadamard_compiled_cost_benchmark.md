# Medium-q RPE Hadamard compiled-cost benchmark datasets

`rpe_hadamard_compiled_cost_benchmark` is a validation-only path for producing
reference costs at explicitly requested, bounded powers of two
$q_m=2^m$.  It is separate from the normal short-round API:

- `QiskitRPEHadamardInterrogationBuilder` still accepts only $q_m=1,2,4$;
- `DFRPEHadamardCompiledCostProvider` remains a short-round provider;
- no benchmark result is connected to RPE resource accounting.

The benchmark circuit builder delegates the wrapper gate sequence to the
ordinary builder after validating the benchmark repetition domain.  Each
trajectory is wrapped twice and the complete circuits are transpiled
independently:

```text
cosine: H -- controlled U_m -- H       -- ancilla Z measurement
sine:   H -- controlled U_m -- S-dagger -- H -- ancilla Z measurement
```

Both axes use the same already-controlled evolution trajectory.  The wrapper
does not resample or add a second control.  State preparation, backend
execution, and quantum shots are outside the recorded circuit scope
`single_hadamard_interrogation_without_state_preparation`.

## Generation API

Construct an `RPEHadamardCompiledCostBenchmarkRequest` with disjoint
`calibration_repetition_counts` and `holdout_repetition_counts`, then call
`generate_rpe_hadamard_compiled_cost_benchmark_dataset`.  Every repetition
count must be a positive power of two no larger than
`maximum_repetition_count`.  The result contains the dataset and the in-memory
paired estimates used to build its records.  `dataset.write_json(path)` and
`RPEHadamardCompiledCostBenchmarkDataset.read_json(path)` provide versioned
JSON persistence.

The generator creates two benchmark points for each requested $q_m$, one per
axis.  Point records contain $m$, $q_m$, $t_m=q_m\delta_{\rm time}$, $r_m$,
$K_m$, trajectory and circuit digests, compiler/backend context, all six cost
statistics, retained trajectory contributions, and explicit scope/status
flags.  Retained contributions include trajectory and step seeds, exact
probability when applicable, provenance and circuit-semantics fingerprints,
actual-circuit fingerprints, costs, and phase metadata.

The schema versions are:

```text
rpe_hadamard_compiled_cost_benchmark_dataset_v1
rpe_hadamard_compiled_cost_benchmark_point_v1
```

JSON floats round-trip without changing their Python values.  Fingerprints use
the exact hexadecimal float representation.  A point fingerprint separates
partition, axis, evolution and wrapper semantics, sampling provenance,
compiler/backend context, and evaluation configuration.  The dataset
fingerprint covers its configuration and a canonical sort of all point
records, so record enumeration order alone does not change it.

## Exact, Monte Carlo, and failures

Deterministic tails are evaluated exactly as one trajectory.  Randomized tails
use exactly the method requested:

- exact enumeration uses trajectory probabilities once;
- Monte Carlo uses the existing master-to-trajectory-to-step seed hierarchy
  and an unweighted mean, unbiased sample variance, and standard error.

There is no exact-to-Monte-Carlo fallback.  Monte Carlo point seeds are derived
from the user master seed, partition, and $q_m$; calibration and holdout use
disjoint seed series.  The same sampled trajectories feed both axes.

Repetition, trajectory/sample, build, transpile, planned-instruction,
untranspiled-size, and retained-record limits are explicit.  Preflight limits
are checked before circuit construction.  If a point cannot be generated, its
two axis records have `status="failed"` and a reason.  Such records are never
silently omitted: the dataset is `complete=false` unless every requested axis
point succeeds.

This module does not fit, select, or validate a large-$q$ proxy.  In particular,
it never labels a result `validated_long_circuit_proxy`, and holdout points are
recorded as unused for fitting.
