"""Proposed limited decision gate. All adoption flags remain false."""
from itertools import product
BASE = 'b34572770e7c2105e16da003af050d789f490662'
ART = 'artifacts/track_b_m1s_gamma_pi3_preparation/2026-10-17/v1'
PACKAGE = 'src/trottertracks/algorithm_codesign/m1s_gamma_pi3_v1'
AUTH_PREFIX = 'artifacts/track_b_m1s_gamma_pi3_execution_authorization/'
OUTPUT_PREFIX = 'artifacts/track_b_m1s_gamma_pi3_result/'
EVIDENCE = 'CONDITIONAL_ENGINEERING_NOT_CERTIFIED'
FORECAST = 'SAVED_BOUNDARY_PI_OVER_THREE_INTERSECTION_ONLY_NOT_RPE_SUCCESS'
INPUT = {
 'path': 'artifacts/track_b_m1s_saved_signal_gamma_v2_result/2026-10-17/completion_attempt_02/result.json',
 'commit': BASE, 'bytes': 146764,
 'sha256': 'bb2d09e6963fdc8b8871170ba28e98c36f4e353f452e50870103a7e97bf2e55d',
 'schema': 'm1s_saved_signal_gamma_v1_result',
 'status': 'GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT',
 'provenance': {'S': '677dd6957e0e1913b70ca5f999458835ae8a2e23',
                'C': '2cb5160d9051a6568b8acc0cfb29fc169bdfca09',
                'A': '161ee8c3dc82c7fa19e0e1cfb79d71d3c236a03c'}
}
SOURCE_FILES = [PACKAGE+'/'+n for n in ['__init__.py','spec.py','control.py','compare.py','supervisor.py','runtime_compatibility.py']]
SOURCE_FILES += ['scripts/tracks/algorithm_codesign/run_m1s_gamma_pi3_v1.py', ART+'/result_schema.json']
RESOURCES = {
 'workers': 1, 'gpu': False, 'memory_bytes': 2147483648,
 'input_bytes': 1048576, 'result_bytes': 16777216, 'log_bytes': 16777216,
 'max_rational_bits': 65536, 'max_rational_text_chars': 25000,
 'max_fraction_fields': 1024, 'max_comparisons': 48,
 'wall_time_seconds': None, 'cpu_time_seconds': None,
 'thread_environment': {k: '1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS']}
}

def contrast_keys():
 return list(product(['G3','G6','G9'], [2,4], ['1/8','1/16'], ['1','4'], ['ordinary','independent_pm']))

def template():
 return {
  'schema': 'm1s_gamma_pi3_v1_contract', 'base_commit': BASE,
  'status': 'CANDIDATE_NOT_ADOPTED_NOT_EXECUTABLE',
  'decision_gate_adopted': False, 'adopted': False, 'coverage_model_adopted': False,
  'contract_grants_execution': False, 'source_identity': None,
  'input': INPUT, 'resources': RESOURCES,
  'science': {
   'target': 'GAMMA_STAR_STRICTLY_LESS_THAN_PI_OVER_THREE',
   'threshold': 'pi/3', 'unit': 'rad',
   'threshold_role': 'PROPOSED_CONSERVATIVE_NEAREST_CONSISTENT_BRANCH_ROUND_ANGLE_CEILING',
   'decoder_adopted_or_implemented': False,
   'boundary_interval': '[L,U] FROM SAVED V1; NO ROOT RECOMPUTATION',
   'pi_interval': '[p_lower,p_upper] FROM SAVED V1; NO PI RECOMPUTATION',
   'satisfied': '3*U < p_lower', 'not_satisfied': '3*L >= p_upper',
   'unresolved': 'NEITHER SUFFICIENT INTERVAL SEPARATION HOLDS',
   'shared_favourable_side': 'ABOVE_ROOT; REQUIRE SHARED_LOWER_BEFORE_AND_SHARED_HIGHER_AFTER',
   'all_splits_reported': ['G3','G6','G9'],
   'tested_splits': ['G3','G6'], 'tested_contrasts': 32,
   'G9': 'ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE_NOT_TESTED',
   'exact_comparison': 'BOUNDED_INTEGER_RATIONAL_CROSS_MULTIPLICATION_NO_FLOAT_TOLERANCE',
   'coverage': 'SAVED_V1_CONDITIONAL_ENGINEERING_ENCLOSURES_ONLY',
   'statistical_scope': 'NO_NEW_CONFIDENCE_CLAIM_NO_NEW_SHOTS',
   'interpretation': 'SAVED_CONTINUOUS_SUFFICIENT_DESIGN_INTERSECTION_ONLY',
   'automatic_candidate_selection': False, 'automatic_scientific_GO': False
  },
  'model': {'molecule':'linear_H4','distance_angstrom':'1.00','basis':'STO-3G','modes':8,'df_rank':12,
            'splits':['G3','G6','G9'],'delta_Ha_inverse':['1/8','1/16'],'R':[2,4],'T_Ha_inverse':['1','4']},
  'grid': {'saved_method_rows':108,'saved_nonzero_method_rows':72,'saved_zero_controls':36,'contrast_records':48,'tested_G3_G6':32,'G9_carried_not_tested':16},
  'evidence': EVIDENCE, 'forecast_kind': FORECAST,
  'one_shot': True, 'retry': False, 'mandatory_STOP': True,
  'absolute_shot_counts': None, 'native_costs': None, 'total_resources': None,
  'rpe_success_guaranteed': False, 'resource_improvement_guaranteed': False,
  'scientific_GO': None
 }
