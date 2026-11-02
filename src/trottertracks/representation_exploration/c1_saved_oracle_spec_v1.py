"""Immutable saved-C1 scope. No Hamiltonian or PF imports."""
from dataclasses import dataclass

BASE_COMMIT = 'cc992b579290233ab5a7d8408d4a81c6d295f36d'
SCOPE = 'QC_C1_SAVED_ORACLE_POSTHOC_V1_20261014'
PREPARATION = 'artifacts/qc_c1_saved_oracle_preparation_v1/2026-10-14'
OUTPUT = 'artifacts/qc_c1_saved_oracle_diagnostic_v1/2026-10-14/attempt_1'
COMPARISONS = 'artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1/comparisons.json'
SEAL = 'artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1/selector_seal.json'
SCORES = ('empirical', 'joint', 'box')
LABELS = ('C1', 'best_single', 'strong_known_composition', 'uniform_order_mixture')
LIMITS = {'rows': 8064, 'contexts': 32, 'workers': 1, 'threads': 1,
          'memory_bytes': 256 * 2**20, 'output_bytes': 2 * 2**20,
          'log_bytes': 2**20, 'eigensolver_calls': 0, 'PF_calls': 0,
          'compile_calls': 0, 'attempts': 1, 'retry': 0, 'wall_time_cap': None}
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS')

OLD_SOURCE_LOCK = 'artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13/source_manifest.json'
SUMMARY = 'artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/saved_results_summary.json'
PRIOR_STOP = 'artifacts/qc_c1_macrocycle_execution_v1/2026-10-13/execution_stop.json'
SOURCE_FILES = tuple('src/trottertracks/representation_exploration/'+name for name in (
    'c1_saved_oracle_spec_v1.py','c1_saved_oracle_schema_v1.py','c1_saved_oracle_diagnostic_v1.py',
    'c1_saved_oracle_checker_v1.py','c1_saved_oracle_execution_v1.py','__init__.py')) + (
    'src/trottertracks/__init__.py','scripts/run_qc_c1_saved_oracle_diagnostic_v1.py',
    'scripts/verify_qc_c1_saved_oracle_diagnostic_v1.py','scripts/check_qc_c1_saved_oracle_preparation_v1.py',
    'tests/test_qc_c1_saved_oracle_preparation_v1.py',
    'docs/research/qc_c1_saved_oracle_diagnostic_contract_v1_20261014.md',
    PREPARATION+'/input_manifest.json',PREPARATION+'/runtime_lock.json',
    PREPARATION+'/diagnostic_contract.json',PREPARATION+'/diagnostic_output_schema.json')

@dataclass(frozen=True)
class Spec:
    inputs: tuple
    prices: tuple
    states: tuple
    targets: tuple
    h_levels: int
    single: tuple
    macro: tuple
    s4: tuple
    c2: tuple
    fixture: bool = False

    @property
    def methods(self):
        return self.single + self.macro + self.s4 + self.c2

FIXED = Spec(('S1_weak', 'S2_balanced', 'S3_signed', 'S4_strong'), (3, 6),
             ('00', '11'), (1e-5, 1e-6), 6,
             tuple(f'S{j}_m{m}' for j in range(3) for m in (1, 2, 3, 4)),
             tuple(f'M{j}{k}_{a}_{b}' for j in range(3) for k in range(3) if j != k
                   for a, b in ((1, 8), (1, 4), (1, 3), (3, 8))),
             tuple(f'S4_{j}' for j in range(3)), tuple(f'C2_{j}' for j in range(3)))

class InputError(ValueError):
    pass

class ApprovalError(ValueError):
    pass
