"""Declarative candidate. Scientific selections intentionally have no defaults."""
from itertools import product

BASE = '870af3f65e3e31e19d2b002c556216c7340c06f0'
ART = 'artifacts/track_b_m1s_v1_v0_preparation/2026-10-17/v1'
PACKAGE = 'src/trottertracks/algorithm_codesign/m1s_saved_signal_v1'
SOURCE_FILES = [PACKAGE + '/' + n for n in
                ['__init__.py', 'spec.py', 'control.py', 'bounds.py', 'analysis.py']]
SOURCE_FILES += ['scripts/tracks/algorithm_codesign/run_m1s_saved_signal_v1.py',
                 ART + '/result_schema.json']
INPUT = {
    'path': 'artifacts/track_b_m1s_finite_signal_v2_result/2026-10-16/completion_attempt_02/result.json',
    'commit': '025c5a90888b02b73d2815fbb10d6bd1d2823602',
    'bytes': 454422,
    'sha256': 'ff6c5192d8cca769cd61a1478cea01671d058a85a8f6a4cdcdc9baa672a56fe0',
    'schema': 'm1s_finite_signal_result_v2',
    'status': 'PILOT_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT',
}
METHODS = ['shared_pm', 'ordinary', 'independent_pm']
EVIDENCE = 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED'
FORECAST = 'IDEAL_SIGNAL_HYPOTHETICAL_SUFFICIENT_SHOT_FORECAST'
AUTH_PREFIX = 'artifacts/track_b_m1s_saved_signal_v1_execution_authorization/'
OUTPUT_PREFIX = 'artifacts/track_b_m1s_saved_signal_v1_result/'
SELECTIONS = {
    'accuracy_semantics': ['HIGH_PROBABILITY_CIRCULAR_PHASE_ERROR'],
    'confidence_scope': ['PER_METHOD_ROW', 'FAMILY_ALL_72_METHOD_ROWS'],
    'statistical_policy': ['FRESH_IID_HOEFFDING_EQUAL_AXES'],
    'coverage_acceptance': ['SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY'],
    'comparison_policy': ['COMMON_SUFFICIENT_DESIGNS_NO_OPTIMALITY'],
    'materiality_policy': ['DESCRIPTIVE_ONLY_NO_SCIENTIFIC_GO'],
}
RESOURCES = {
    'workers': 1, 'gpu': False, 'memory_bytes': 2147483648,
    'input_bytes': 1048576, 'result_bytes': 16777216, 'log_bytes': 16777216,
    'max_rational_bits': 65536, 'max_series_terms_per_call': 4096,
    'max_bound_steps': 1000000, 'max_shot_integer_bits': 4096,
    'max_arithmetic_bits': 256,
    'wall_time_seconds': None, 'cpu_time_seconds': None,
    'thread_environment': {k: '1' for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                         'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS']},
}


def keys():
    """Fixed identifiers only; never load or evaluate saved scientific values."""
    return list(product(['G3', 'G6', 'G9'], [2, 4], ['1/8', '1/16'], ['0', '1', '4'], METHODS))


def contrast_keys():
    return list(product(['G3', 'G6', 'G9'], [2, 4], ['1/8', '1/16'], ['1', '4'], METHODS[1:]))


def template():
    return {
        'schema': 'm1s_saved_signal_v1_contract', 'base_commit': BASE,
        'status': 'CANDIDATE_NOT_ADOPTED_NOT_EXECUTABLE',
        'adopted': False, 'coverage_model_adopted': False,
        'contract_grants_execution': False, 'mandatory_STOP': True,
        'research_direction_adopted': True,
        'direction': {'first': 'TASK_A_SAVED_RAW_H_SINGLE_TIME_PHASE',
                      'ultimate': 'TASK_B_MULTIROUND_ENERGY_RPE_NOT_IN_V1',
                      'route': 'UNCORRECTED_RAW_DIRECT',
                      'candidates': 'ALL_SAVED_CELLS_TWO_INTERNAL_COMPARATORS'},
        'model': {'molecule': 'linear_H4', 'distance_angstrom': '1.00',
                  'basis': 'STO-3G', 'modes': 8, 'df_rank': 12,
                  'splits': ['G3', 'G6', 'G9'], 'delta_Ha_inverse': ['1/8', '1/16'],
                  'R': [2, 4], 'T_Ha_inverse': ['1', '4'],
                  'state': 'SAVED_ORIGINAL_NORM2_QUOTIENT', 'signal_sign': '+iTH'},
        'input': dict(INPUT), 'grid': {'all_rows': 108, 'forecast_rows': 72,
                                      'saved_zero_controls': 36, 'paired_contrasts': 48},
        'science': {**{k: None for k in SELECTIONS}, 'gamma_rad': None,
                    'alpha_total': None, 'arithmetic_bits': None},
        'resources': dict(RESOURCES), 'source_identity': None,
        'evidence': EVIDENCE, 'forecast_kind': FORECAST,
        'bias_definition': 'min(pi,abs(saved_total_principal_bias)+saved_phase_radius)',
        'model_drift_added_again': False,
        'shot_formula': 'ceil_upper(4*log(2/alpha_axis)/(A_lower^2*sin(q)^2))',
        'visibility': 'max(0,saved_visibility-saved_visibility_error_radius)',
        'q': 'gamma-b_upper; require 0<gamma<pi/2 and q>0',
        'physical_shot_guarantee': False, 'scientific_GO': None,
        'native_costs': None, 'total_resources': None,
        'retry': False, 'one_shot': True,
    }
