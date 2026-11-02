"""Nominal candidates from fixed HiGHS settings; certificates decide validity."""
from fractions import Fraction as F
import warnings
from .lp import dual_lower, make_farkas_problem, check_farkas_output
from .guard import TechnicalFailure, single_thread_environment

OPTIONS = {"presolve": True, "time_limit": 2., "threads": 1, "parallel": False,
           "dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9,
           "simplex_dual_edge_weight_strategy": "steepest-devex"}


def nominal(lp, guard, call_id, baseline, auxiliary=False, permit=None):
    if lp.domain != "SYNTHETIC":
        if permit is None:
            raise PermissionError("registered-domain solver requires separate authorization")
        permit.assert_active()
    lp.validate()
    single_thread_environment()
    with guard.lp_call(call_id, baseline, auxiliary):
        from scipy.optimize import linprog, OptimizeWarning
        with warnings.catch_warnings():
            # SciPy 1.16.2 forwards these two backend options verbatim.
            warnings.filterwarnings("ignore", "Unrecognized options detected", OptimizeWarning)
            result = linprog([float(v) for v in lp.c],
                             A_ub=[[float(v) for v in r] for r in lp.A] or None,
                             b_ub=[float(v) for v in lp.b] or None,
                             A_eq=[[float(v) for v in r] for r in lp.H] or None,
                             b_eq=[float(v) for v in lp.f] or None,
                             bounds=[(0, float(v)) for v in lp.upper],
                             method="highs-ds", options=OPTIONS.copy())
    output = {"status": int(result.status), "message": result.message}
    if result.status == 0:
        output["nominal_primal"] = [F.from_float(float(v)) for v in result.x]
        nu = [max(F(0), -F.from_float(float(v))) for v in result.ineqlin.marginals]
        u = [-F.from_float(float(v)) for v in result.eqlin.marginals]
        output["dual_certificate"] = dual_lower(lp, nu, u)
    return output


def evaluate(lp, guard, call_id, baseline, permit=None):
    result = nominal(lp, guard, call_id, baseline, permit=permit)
    if result["status"] == 0:
        return result
    if result["status"] != 2:
        raise TechnicalFailure("uncertified solver termination: "+result["message"],
                               {"main_solver": result})
    auxiliary = nominal(make_farkas_problem(lp), guard, call_id, baseline, True, permit)
    if auxiliary["status"] != 0:
        raise TechnicalFailure("Farkas acquisition failure", {"main_solver": result, "auxiliary_solver": auxiliary})
    cert = check_farkas_output(lp, auxiliary["nominal_primal"])
    if not cert["certified_infeasible"]:
        raise TechnicalFailure("Farkas exact ray verification failure", {
            "main_solver": result, "auxiliary_solver": auxiliary, "failed_certificate": cert})
    return {"status": "CERTIFIED_INFEASIBLE", "Farkas": cert,
            "full_auxiliary_primal": auxiliary["nominal_primal"], "nominal_status": 2}
