"""Future saved-interval adapter. Preparation tests never decode saved science."""
from fractions import Fraction as F
import re
from .control import require_permit
from .spec import INPUTS, EVIDENCE, contrast_keys, row_keys, old_status


def keyed(records, names, expected):
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("saved inventory length")
    result = {}
    for record in records:
        key = tuple(record[n] for n in names)
        if key in result:
            raise ValueError("duplicate saved identifier")
        result[key] = record
    if set(result) != set(expected):
        raise ValueError("saved inventory missing/extra")
    return result


class RationalReader:
    def __init__(self, resources):
        self.resources = resources
        self.fields = 0

    def parse(self, text):
        self.fields += 1
        if self.fields > self.resources["max_fraction_fields"]:
            raise ValueError("fraction field cap")
        if (type(text) is not str or len(text) > self.resources["max_rational_text_chars"]
                or re.fullmatch(r"-?[0-9]+/[1-9][0-9]*", text) is None):
            raise ValueError("bounded exact rational text required")
        value = F(text)
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > self.resources["max_rational_bits"]:
            raise ValueError("rational bit cap")
        return value


def prepare_saved(v1, pi3, permit):
    require_permit(permit)
    for name, raw in [("V1", v1), ("PI3", pi3)]:
        expected = INPUTS[name]
        if (raw.get("schema") != expected["schema"] or raw.get("status") != expected["status"]
                or raw.get("failure") is not None or raw.get("evidence") != EVIDENCE
                or raw.get("scientific_GO") is not None):
            raise ValueError("saved schema/status/evidence: " + name)
        if any(raw.get("provenance", {}).get(k) != v for k, v in expected["provenance"].items()):
            raise ValueError("saved provenance: " + name)
    if v1.get("physical_shot_guarantee") is not False:
        raise ValueError("V1 ideal-signal interpretation changed")
    rk = ["split", "R", "delta", "T", "method"]
    ck = ["split", "R", "delta", "T", "comparator"]
    rows = keyed(v1.get("rows"), rk, row_keys())
    comparisons = keyed(v1.get("comparisons"), ck, contrast_keys())
    gate = keyed(pi3.get("comparisons"), ck, contrast_keys())
    reader = RationalReader(permit.contract["resources"])
    pair = v1.get("arithmetic", {}).get("pi_interval")
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError("saved pi enclosure absent")
    pi = tuple(reader.parse(x) for x in pair)
    if not 3 < pi[0] <= pi[1] < 4:
        raise ValueError("saved pi enclosure/domain")
    cap = tuple(x / 3 for x in pi)
    jobs = {}
    metadata = []
    for key in contrast_keys():
        g, r, delta, t, comparator = key
        old = comparisons[key]
        expected = old_status(key)
        if gate[key].get("status") != expected:
            raise ValueError("fixed pi3 classification changed")
        metadata.append({"split": g, "R": r, "delta": delta, "T": t,
                         "comparator": comparator,
                         "old_pi3_gate_status": expected,
                         "old_absolute_contrast_status": old.get("saved_visibility_contrast_status"),
                         "new_stage_s_status": "NOT_EVALUATED"})
        if g == "G9":
            if (old.get("saved_visibility_contrast_status") != "UNRESOLVED_ABSOLUTE_CONTRAST"
                    or old.get("scientific_gain_claim") != "ABSOLUTE_CONTRAST_UNRESOLVED_NO_RELATIVE_RESCUE"):
                raise ValueError("G9 unresolved limitation changed")
            continue
        if (old.get("status") != "UNIQUE_TRADEOFF_BOUNDARY"
                or old.get("formula_applicability") != "REVIEW_TRADEOFF_DIRECT"
                or old.get("continuous_relation_before") != "SHARED_LOWER"
                or old.get("continuous_relation_after") != "SHARED_HIGHER"):
            raise ValueError("saved direct tradeoff orientation changed")
        if expected != "SATISFIED":
            continue
        if old.get("saved_visibility_contrast_status") != "RESOLVED_HIGHER_VISIBILITY":
            raise ValueError("eligible absolute contrast not resolved")
        boundary = old.get("boundary_interval_rad")
        if not isinstance(boundary, list) or len(boundary) != 2:
            raise ValueError("eligible saved boundary absent")
        lower, upper = [reader.parse(x) for x in boundary]
        if not 0 <= lower <= upper < cap[0]:
            raise ValueError("eligible interval not strictly below saved cap")
        cell = (g, r, delta, t)
        if cell not in jobs:
            methods = {}
            for method in ["shared_pm", "ordinary", "independent_pm"]:
                saved = rows[cell + (method,)]
                if (saved.get("status") != "READY_FOR_GAMMA_DOMAIN"
                        or saved.get("saved_evidence") != EVIDENCE
                        or saved.get("bounds", {}).get("model_drift_added_again") is not False):
                    raise ValueError("saved row bounds/coverage unavailable")
                fields = saved["bounds"]
                methods[method] = (reader.parse(fields["A_lower"]),
                                   reader.parse(fields["unclipped_bias_rad"]))
            jobs[cell] = methods
    if len(jobs) != 12:
        raise ValueError("expected 12 posthoc eligible cells")
    return cap, jobs, metadata, {"saved_pi_interval_rad": pair,
                                "fraction_fields_read": reader.fields,
                                "108_row_inventory_checked": True,
                                "zero_controls_reanalyzed": False}
