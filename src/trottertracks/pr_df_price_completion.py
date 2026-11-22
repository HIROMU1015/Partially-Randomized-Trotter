"""Complete a canonical PR-DF price table from measured same-step cohorts."""

from collections import defaultdict
from fractions import Fraction

from . import price_model_validation as model


AXES = ("X", "Y")


def exact(value):
    if isinstance(value, dict):
        return Fraction(value["numerator"], value["denominator"])
    return Fraction(value)


def encoded(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def feature_names(rounds):
    """A fixed outer count absorbs the qR term into the q coefficient."""
    if len({row["R_outer"] for row in rounds}) == 1:
        return ("one", "q")
    return ("one", "q", "qR")


def feature_values(round_):
    q = round_["q_PF"]
    return {"one": 1, "q": q, "qR": q * round_["R_outer"]}


def complete(config, measured, physical_ids, *, compiler, calibration_max_m):
    """Fit each needed step and axis, retaining direct prices and holdout roles.

    Only supplied, already verified finite cohorts are used. Missing support or
    nonpositive predictions raise an error and leave the configuration pending.
    The output is a conditional scenario for the optimizer's cost calculation.
    """
    spec = config["spec"]
    rounds = {row["m"]: row for row in config["rounds"]}
    if (spec["method"] != "PR" or spec["split"].get("tail_dictionary")
            or spec["profile"]["compiler"] != compiler or set(rounds) != set(physical_ids)
            or len(rounds) != len(config["rounds"])
            or type(calibration_max_m) is not int or not 0 <= calibration_max_m <= max(rounds)):
        raise ValueError("PR-DF price model needs its own compiler, schedule and calibration range")
    by_step = defaultdict(list)
    for row in rounds.values():
        by_step[exact(row["actual_delta_pf"])].append(row)
    prices = {}
    for row in measured:
        m, axis = key = row["m"], row["axis"]
        if (key in prices or m not in rounds or axis not in AXES
                or row.get("physical_id", physical_ids[m]) != physical_ids[m]
                or row["kind"] != "DIRECT_FINITE_COHORT" or exact(row["unit_price_RZ"]) <= 0):
            raise ValueError("PR-DF measured price belongs to a different round, axis or cohort")
        prices[key] = row
    preparations = {row["price_evidence"]["preparation_hash"] for row in prices.values()}
    if len(preparations) > 1 or any(not value for value in preparations):
        raise ValueError("PR-DF model cannot combine different preparations")
    fitted = []
    for step, group in sorted(by_step.items()):
        names = feature_names(group)
        for axis in AXES:
            missing = [row for row in group if (row["m"], axis) not in prices]
            if not missing:
                continue
            calibration = [row for row in group if row["m"] <= calibration_max_m and (row["m"], axis) in prices]
            if len(calibration) < len(names):
                raise ValueError(f"PR-DF {axis} step {step}: insufficient measured calibration rounds")
            route = {"system": spec["binding"]["system"], "configuration_id": config["id"],
                     "actual_step": encoded(step), "formula": spec["formula"], "L_D": spec["L_D"]}
            records = []
            for row in calibration:
                m = row["m"]
                receipt = prices[m, axis]
                evidence = receipt["price_evidence"]
                records.append(model.record(
                    cell_id=physical_ids[m], role="CALIBRATION", route=route, axis=axis,
                    features={name: feature_values(row)[name] for name in names},
                    rz_count=receipt["unit_price_RZ"], kind="DIRECT_COMPILED_FINITE_SAMPLE",
                    source=evidence, implementation={"lowering_id": spec["profile"]["control"],
                                                      "preparation_hash": evidence["preparation_hash"]},
                    profile=spec["profile"], n_paths=len(evidence["paths"]), law=spec["law"],
                ))
            frozen = model.fit(records, features=names, design_source={
                "policy": "PR_DF_SAME_ACTUAL_STEP_FINITE_COHORT",
                "maximum_calibration_round": calibration_max_m,
            })
            holdouts = []
            for row in group:
                m = row["m"]
                if m <= calibration_max_m or (m, axis) not in prices:
                    continue
                receipt = prices[m, axis]
                if receipt["price_evidence"]["preparation_hash"] != frozen["domain"]["implementation"]["preparation_hash"]:
                    raise ValueError("PR-DF descriptive comparison requires the same preparation")
                predicted = model.predict(frozen, feature_values(row))
                observed = exact(receipt["unit_price_RZ"])
                holdouts.append(dict(m=m, axis=axis, predicted=encoded(predicted), observed=encoded(observed),
                    signed_relative_error=encoded((predicted - observed) / observed), source=receipt["price_evidence"]))
            fitted.append(dict(route=route, axis=axis, calibration_rounds=[r["m"] for r in calibration],
                               outer_counts=sorted({r["R_outer"] for r in group}), fit=frozen,
                               descriptive_holdouts=holdouts))
            for row in missing:
                m = row["m"]
                price = model.predict(frozen, feature_values(row))
                if price <= 0:
                    raise ValueError(f"PR-DF model produced a nonpositive price at {m}/{axis}")
                prices[m, axis] = dict(m=m, axis=axis, physical_id=physical_ids[m], unit_price_RZ=encoded(price),
                    kind="CONDITIONAL_PR_DF_COMPILED_PRICE_MODEL", qualified_E_C=False,
                    price_evidence=dict(model_index=len(fitted) - 1, configuration_id=config["id"],
                        actual_step=encoded(step), compiler=compiler, direct_observation=False))
    return [prices[key] for key in sorted(prices)], fitted
