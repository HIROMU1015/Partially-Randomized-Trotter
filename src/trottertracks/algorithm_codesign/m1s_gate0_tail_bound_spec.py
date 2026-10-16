"""Unadopted declarative design; no resolver, numerical imports or execution.

The proof applies conditionally to the frozen v4 coefficient pipeline.
This file cannot create a permit, evaluate a budget or execute Gate0.
"""

MODEL_SCHEMA = "m1s_gate0_tail_local_coefficient_model_v1"
EVIDENCE_CLASSIFICATION = "CONDITIONAL_ENGINEERING_NOT_CERTIFIED"
INTEGRATION_IMPLEMENTED = False
FORMAL_ADOPTION = False
COUNTS = {
    "n_modes": 8,
    "df_rank": 12,
    "complex_product_real_budget": 8,
    "phase_products_per_word": 8,
    "coefficient_products_per_word": 2,
    "fsum_padding": 32,
    "bilinear_group_max": 4,
    "one_body_group_max": 64,
    "one_body_support_max": 121,
    "square_group_max": 121,
    "multiply_leaf_gamma_index": 80,
    "bilinear_gamma_index": 116,
    "one_body_gamma_index": 220,
    "square_gamma_index": 673,
    "scaled_block_gamma_index": 681,
    "tail_gamma_base": 713,
    "tail_gamma_max_rank12": 725,
}
FORMULAS = {
    "G_l": "sum_pq (abs(Re(g_lpq))+abs(Im(g_lpq)))",
    "S_R": "sum_l_in_tail abs(lambda_l)*G_l**2",
    "alpha_R": "gamma_(713+r)*S_R",
    "lambda_error": "alpha_R+gamma_(M+32)*L",
    "ec": "alpha_R/lambda_lo+lambda_error*L/(lambda_lo*lambda_hat)",
    "propagation": "v4 law_limits algebra with alpha replaced by alpha_R; M unchanged",
}
FEASIBILITY_DOMAIN = {
    "S_R_over_lambda_hat_max": 8,
    "L_over_lambda_hat_max": 2,
    "M": 65536,
    "d": 256,
    "is_new_execution_gate": False,
}
