"""Conditional preregistered budgets. No evaluation without a validated permit.

Only frozen input scales and canonical-law coefficients enter this module.
No observed discrepancy, residual, W, v or rho is an input. Fraction arithmetic
keeps the budget calculation exact until an outward binary64 conversion.
The floating-point model is conditional, NOT certified by source inspection.
"""
from fractions import Fraction
import math

from .control import EVIDENCE_CLASSIFICATION, require_permit
from ..m1s_gate0_tail_bound_spec import COUNTS

FORMULA_IDS = {
    "eps_H": "reconstruction_F_v1",
    "eps_hermiticity_H": "hermiticity_F_v1",
    "eps_phase": "identity_ledger_F_v1",
    "eps_state_residual": "absolute_original_solver_acceptance_v1",
    "eps_imag_coeff": "coefficient_imaginary_L1_v1",
    "eps_imag_energy": "rayleigh_imaginary_v1",
    "W_error_budget": "W_forward_error_tail_local_v1",
    "v_error_budget": "variance_forward_error_tail_local_v1",
    "W_floor": "rho_resolution_floor_tail_local_v1",
}


def formula_policy():
    return {"schema": "m1s_gate0_conditional_roundoff_model_v5",
            "coverage_model_adopted": False,
            "tail_coefficient_model": "m1s_gate0_tail_local_coefficient_model_v1",
            "tail_gamma_base": COUNTS["tail_gamma_base"],
            "tail_scale": "sum_tail_abs_lambda_times_G_squared",
            "coefficient_bound_evidence": "AUTHOR_CONDITIONAL_DERIVATION_NOT_INDEPENDENTLY_REVIEWED",
            "oracle_quality_policy": "absolute_original_solver_acceptance_v1",
            "oracle_absolute_residual_quality_hartree": 1e-8,
            "evidence_classification": EVIDENCE_CLASSIFICATION,
            "assumptions": ["IEEE_BINARY64_ROUND_TO_NEAREST",
                "NO_UNACCOUNTED_OVERFLOW_UNDERFLOW_OR_FTZ",
                "COMPLEX_BLAS_GAMMA_8K_PLUS_32",
                "FSUM_NO_WORSE_THAN_GAMMA_M_PLUS_32_L1",
                "EXACT_HERMITIAN_INPUT_ONE_BODY_AND_FACTORS",
                "EXACT_SAVED_BINARY64_INPUT_IS_TARGET_NOT_EXACT_EIGENSTATE"]}


def is_formula(value, key):
    return (key in FORMULA_IDS and isinstance(value, dict)
            and value == {"formula_id": FORMULA_IDS[key]})


class _Budget:
    """Constructed only in the future permit-bound engine, never in static tests."""
    def __init__(self, permit, constant, one, lambdas, factors):
        require_permit(permit)
        self.permit = permit
        self.precision = permit.config["precision"]
        self.policy = permit.config["formula_policy"]
        if self.policy["coverage_model_adopted"] is not True:
            raise ValueError("Conditional roundoff model has not been adopted")
        self.u = Fraction(1, 2**53)
        self.C = permit.config["limits"]["max_product_contributions"]
        self.M = permit.config["limits"]["max_pauli_terms"]
        F = self.l1(one)
        gs = [self.l1(g) for g in factors]
        ls = [abs(self.q(x)) for x in lambdas]
        S = abs(self.q(constant)) + F + sum(l*g*g for l, g in zip(ls, gs))
        self.S = S
        # Each complex multiply uses <= 8 real operations; 24*C + 64
        # also dominates two coefficient-sum stages and the degree-four DAG.
        a = self.gamma(24*self.C + 64)*S
        self.imag = a*(1+self.gamma(self.M+32))
        self.alpha = 2*self.imag
        self.gdot = self.gamma(8*256+32)
        self.gdiv = self.gamma(8)
        self.gnorm = self.gamma(8*256*256+32)
        self.gmv = self.gamma(8*256+32)
        B_one = 16*self.gamma(8*64+32)*F
        B_ref = B_one
        for l, g in zip(ls, gs):
            ba = 16*self.gamma(8*64+32)*g
            sq = 2*g*ba + ba*ba + self.gmv*(16*g+ba)**2
            B_ref += l*(sq+self.gamma(8)*((16*g+ba)**2+sq))
        self.B_ref = B_ref+self.gamma(64)*(16*S+B_ref)
        move = self.gamma(8)*(2*S+2*self.alpha)
        B_map = 16*(self.alpha+move+self.gamma(8*self.M+32)*(S+2*self.alpha))
        self.B_map = B_map
        diff_round = self.gamma(32)*(64*S+2*B_map+2*self.B_ref)
        self.reconstruction = (1+self.gnorm)*(2*B_map+2*self.B_ref+diff_round)
        self.hermiticity = (1+self.gnorm)*(2*(B_map+self.B_ref)+diff_round)
        self.phase = self.reconstruction
        self.Yerr = self.B_ref+self.gmv*(16*S+self.B_ref)
        gd, gv = self.gdot, self.gdiv
        self.energy = ((self.Yerr+gd*(S+self.Yerr)+S*gd)
                       +gv*(1+gd)*(S+self.Yerr))/(1-gd)
        self.residual_roundoff = ((1+self.gnorm)*(self.Yerr+self.energy
            +self.gamma(32)*(2*S+self.Yerr+self.energy)))/(1-gd)

        # Separate raw-input tail models, all fixed BEFORE coefficient generation.
        # self.alpha and every global expression above remain the v4 global model.
        if len(gs) != 12 or len(ls) != 12:
            raise ValueError("Fixed rank12 tail model required")
        self.tail_block_scales = tuple(l*g*g for l, g in zip(ls, gs))
        self.tail_models = {}
        for count in (3, 6, 9):
            r = 12-count
            scale = sum(self.tail_block_scales[count:])
            index = COUNTS["tail_gamma_base"]+r
            self.tail_models[count] = (r, scale, index, self.gamma(index)*scale)

    def local_coefficient_ledger(self, split_count):
        require_permit(self.permit)
        if type(split_count) is not int or split_count not in (3, 6, 9):
            raise ValueError("Fixed G3/G6/G9 split required")
        r, scale, index, alpha_R = self.tail_models[split_count]
        return {"model": "m1s_gate0_tail_local_coefficient_model_v1",
            "coverage": EVIDENCE_CLASSIFICATION,
            "proof_status": "AUTHOR_CONDITIONAL_DERIVATION_NOT_INDEPENDENTLY_REVIEWED",
            "split": f"G{split_count}", "HR_df_indices": list(range(split_count, 12)),
            "tail_block_count": r, "gamma_index": index,
            "S_R_hartree_upper": self.up(scale), "alpha_R_hartree_upper": self.up(alpha_R),
            "input_dependencies": ["exact_saved_binary64_lambdas_and_factors", "fixed_split_count"],
            "resource_cap_or_observed_counter_used": False,
            "computed_before_coefficient_generation": True,
            "exact_eigenstate_distance_error_included": False}

    def local_coefficient_ledgers(self):
        require_permit(self.permit)
        return {f"G{count}": self.local_coefficient_ledger(count) for count in (3, 6, 9)}

    @staticmethod
    def q(value):
        x = float(value)
        if not math.isfinite(x):
            raise ValueError("Nonfinite budget input")
        return Fraction.from_float(x)

    def l1(self, matrix):
        return sum(abs(self.q(z.real))+abs(self.q(z.imag)) for row in matrix for z in row)

    def gamma(self, count):
        t = count*self.u
        if t >= 1:
            raise ValueError("Gamma denominator is nonpositive")
        return t/(1-t)

    @staticmethod
    def up(value):
        if value < 0:
            raise ValueError("Negative error budget")
        x = float(value)
        if Fraction.from_float(x) < value:
            x = math.nextafter(x, math.inf)
        if not math.isfinite(x):
            raise ValueError("Nonfinite resolved budget")
        return x

    def global_limits(self):
        quality = self.policy["oracle_absolute_residual_quality_hartree"]
        residual = self.q(quality)+self.residual_roundoff
        raw = {"eps_H": self.reconstruction, "eps_hermiticity_H": self.hermiticity,
               "eps_phase": self.phase, "eps_imag_coeff": self.imag,
               "eps_imag_energy": self.energy, "eps_state_residual": residual}
        limits = {}
        for key, value in raw.items():
            if is_formula(self.precision[key], key):
                if value is None:
                    raise ValueError("Oracle residual policy is incomplete")
                limits[key] = self.up(value)
        return limits, {"coverage": "CONDITIONAL_ENGINEERING_NOT_CERTIFIED",
            "oracle_quality_policy": self.policy["oracle_quality_policy"],
            "oracle_quality_component_hartree": quality,
            "oracle_residual_gate_limit_hartree": limits.get("eps_state_residual"),
            "old_solver_vs_new_reference": "DIFFERENT_PATHS_NOT_DIFFERENCE_CERTIFIED",
            "old_solver_residual_arithmetic_error": "NOT_CERTIFIED_NOT_TRANSFERRED",
            "old_residual_value": "NOT_USED_NOT_REEVALUATED",
            "quality_is_eigenstate_distance_certificate": False,
            "input_scale_H_l1_upper": self.up(self.S),
            "coefficient_l1_error_upper": self.up(self.alpha),
            "reference_F_error_upper": self.up(self.B_ref),
            "residual_arithmetic_error_upper": self.up(self.residual_roundoff),
            "resolved_global_limits": limits,
            "inputs_exclude": ["observed_validation_errors", "residual", "W", "v", "rho"],
            "exact_eigenstate_distance_error_included": False}

    def law_limits(self, coefficients, lam, precision, *, split_count):
        """Called before hpsi, Pauli expectations, W or v; no target metrics accepted."""
        require_permit(self.permit)
        local = self.local_coefficient_ledger(split_count)
        alpha_R = self.tail_models[split_count][3]
        for key in ("W_error_budget", "v_error_budget", "W_floor"):
            if not is_formula(precision[key], key):
                raise ValueError("v5 requires the fixed tail-local formula selectors")
        L = sum(abs(self.q(x)) for x in coefficients)
        lam = self.q(lam)
        ds = self.gamma(self.M+32)*L
        dl = alpha_R+ds
        lower = lam-dl
        if lower <= 0:
            return None, {"status": "LAMBDA_INTERVAL_CONTAINS_ZERO_PRECISION_UNRESOLVED",
                "tail_local_coefficient_error": local,
                "lambda_absolute_error_upper": self.up(dl),
                "global_coefficient_error_used_for_law": False}
        gd, gv = self.gdot, self.gdiv
        ratio = L/lam
        ec = alpha_R/lower+dl*L/(lower*lam)
        assembly = 16*self.gamma(8*self.M+32)*L
        y = ec+assembly/lam+self.gmv*(16*L+assembly)/lam
        y += gv*(1+self.gmv)*(16*L+assembly)/lam
        A = (1+y)**2
        quotient = (2*gd+gv*(1+gd))*A/(1-gd)
        Qup = (1+gv)*(1+gd)*A/(1-gd)
        ew = 2*y+y*y+quotient+self.u*(1+Qup)
        ep = ec+gv*ratio
        defect = (self.gamma(self.M+32)+gv)*ratio
        ea = (2*gd+gv*(1+gd))/(1-gd)
        gm = self.gamma(4*self.M+32)
        em = ep+(1+defect)*ea+gm*(1+defect)*(1+ea)
        am = (1+gm)*(1+defect)*(1+ea)
        ev = 4*ep+(1+defect)*(4*(ea+em)+(ea+em)**2)
        ev += self.gamma(16*self.M+32)*(1+defect)*(1+ea+am)**2
        values = {"W_error_budget": self.up(ew), "v_error_budget": self.up(ev)}
        if is_formula(precision["W_floor"], "W_floor"):
            rho_target = self.q(precision["eps_rho"])
            rho_roundoff = self.gamma(8)
            if rho_target <= rho_roundoff:
                return None, {"status": "RHO_TARGET_BELOW_ROUNDOFF_PRECISION_UNRESOLVED"}
            half_error_sum = (self.q(values["W_error_budget"])+self.q(values["v_error_budget"]))/2
            values["W_floor"] = self.up(half_error_sum*(1+rho_roundoff)/(rho_target-rho_roundoff))
        ledger = {"coverage": "CONDITIONAL_ENGINEERING_NOT_CERTIFIED",
            "resolved_law_limits": values, "lambda_absolute_error_upper": self.up(dl),
            "normalized_law_L1_error_upper": self.up(ep),
            "Pauli_expectation_error_upper": self.up(ea),
            "computed_before_W_v_rho": True,
            "tail_local_coefficient_error": local,
            "global_coefficient_error_used_for_law": False,
            "input_dependencies": ["frozen_tail_input_scales", "fixed_split_count", "canonical_coefficients", "lambda_sum"]}
        return values, ledger
