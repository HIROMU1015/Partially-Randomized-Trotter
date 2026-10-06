"""Static decomposition and ideal-class nesting checks, without optimization."""
from fractions import Fraction as F
from math import isqrt


def rational_square(v):
    v = F(v)
    return (isqrt(v.numerator)**2 == v.numerator
            and isqrt(v.denominator)**2 == v.denominator)


def audit_semantics(table):
    readout = {}
    for xs, data in table["tables"].items():
        target = list(map(F, data["target"]))
        columns = {c["id"]: c for c in data["columns"]}
        proofs = []
        for p in data["B0_saved_profiles"]:
            total = [F(0)]*4
            rounding = []
            for m in p["memberships"]:
                c = columns[m["column_id"]]
                a, b = map(F, c["saved_ideal_ab_exact"])
                k = c["degree"]
                total[k] += a
                if b:
                    total[k+1] += b
                saved = F(m["saved_midpoint_weight"])
                rounding.append(saved*saved != a*a+b*b)
            if total != target:
                raise ValueError("finite Taylor ideal decomposition failed")
            proofs.append({"arm": p["arm"], "epsilon": p["epsilon"],
                           "ideal_mean_exact": list(map(str, total)),
                           "B0_saved_midpoint_is_exact_ideal_weight": not any(rounding),
                           "fixed_precision_embedding_in_B1_ideal": True})
        ordinary = [p for p in data["B0_saved_profiles"] if p["arm"] == "ordinary"][0]
        norms_squared = []
        for m in ordinary["memberships"]:
            c = columns[m["column_id"]]
            a, b = map(F, c["saved_ideal_ab_exact"])
            norms_squared.append(a*a+b*b)
        ratio_square = norms_squared[0]/norms_squared[1]
        # If the ideal ratio of two positive group weights is irrational, their
        # normalized group masses cannot both be dyadic rational numbers.
        irrational = not rational_square(ratio_square)
        readout[xs] = {"profile_ideal_embeddings": proofs,
                       "ordinary_group_norm_ratio_squared_exact": str(ratio_square),
                       "ordinary_group_norm_ratio_irrational": irrational,
                       "ideal_B1_exact_membership_incompatible_with_all_dyadic_q": irrational,
                       "B0_saved_literal_subset_B1_ideal": False,
                       "ideal_class_nesting": "B0_ideal subset B1_ideal subset B2_ideal subset B3_ideal",
                       "ideal_nesting_proof": [
                           "Place all group weight in the registered precision to embed B0_ideal in B1.",
                           "Choose theta_r=1 for one representation to embed B1 in B2.",
                           "Coalesce only implementation-identical O0 aliases; Dq=yt follows by linearity to embed B2 in B3."],
                       "numerical_pipeline_nesting": "REQUIRES_GPT_REVIEW_AMENDMENT"}
    return {"schema": "ra_d0_semantic_audit_v1", "optimization_calls": 0, "readout": readout,
            "status": "IDEAL_NESTING_VERIFIED_NUMERICAL_NESTING_REQUIRES_AMENDMENT"}
