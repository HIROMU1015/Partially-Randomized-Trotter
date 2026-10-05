"""Static SP-1 native ledger and role/mask-independent exact fusion.

This module constructs angle/generator records, never a Hamiltonian or circuit.
Fusion precedes placement. No commuting reordering or angle periodicization is
performed. Mixed-role fused rotations have no automatically assigned placement.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F


@dataclass(frozen=True)
class Angle:
    pi: F = F(0)
    rad: F = F(0)

    def __post_init__(self):
        if any(isinstance(x, (float, bool)) for x in (self.pi, self.rad)):
            raise TypeError("angles require exact rational components")
        object.__setattr__(self, "pi", F(self.pi))
        object.__setattr__(self, "rad", F(self.rad))

    def __add__(self, other):
        return Angle(self.pi+other.pi, self.rad+other.rad)

    def scaled(self, factor):
        return Angle(self.pi*F(factor), self.rad*F(factor))

    def record(self):
        return {"pi": str(self.pi), "rad": str(self.rad)}

    @property
    def zero(self):
        return self.pi == self.rad == 0


def parse_angle(raw):
    unit, value = raw.split(":")
    if unit == "pi":
        return Angle(pi=F(value))
    if unit == "rad":
        return Angle(rad=F(value))
    raise ValueError("unknown exact angle unit")


@dataclass(frozen=True)
class Native:
    generator: str
    angle: Angle
    lineage: tuple

    def __post_init__(self):
        if len(self.generator) != 2 or any(p not in "IXYZ" for p in self.generator):
            raise ValueError("expected an ancilla/system Pauli word")
        if not isinstance(self.angle, Angle) or not self.lineage:
            raise ValueError("angle and nonempty provenance required")

    @property
    def roles(self):
        return frozenset(item["role"] for item in self.lineage)

    def record(self):
        return {"generator": self.generator, "angle": self.angle.record(),
                "roles": sorted(self.roles), "lineage": list(self.lineage)}


def lower_logical(logical):
    natives = []
    for position, gate in enumerate(logical):
        pauli = gate["pauli"]
        if pauli not in ("I", "X", "Y", "Z"):
            raise ValueError("unsupported system Pauli")
        angle = gate["angle"]
        for factor, (ancilla, sign) in enumerate((("I", F(1, 2)), ("Z", F(-1, 2)))):
            natives.append(Native(ancilla+pauli, angle.scaled(sign),
                ({"logical_position": position, "native_factor": factor, "role": gate["role"]},)))
    return tuple(natives)


def canonical_fusion(natives):
    stack, events = [], []
    candidate_count = cross_role = zero_deletions = 0
    for native in natives:
        if native.angle.zero:
            zero_deletions += 1
            events.append({"kind": "delete_zero", "lineage": list(native.lineage)})
            continue
        if stack and stack[-1].generator == native.generator:
            previous = stack.pop()
            candidate_count += 1
            roles = previous.roles | native.roles
            is_cross_role = "D" in roles and "R" in roles
            cross_role += int(is_cross_role)
            fused = Native(native.generator, previous.angle+native.angle,
                           previous.lineage+native.lineage)
            events.append({"kind": "adjacent_exact_addition", "generator": native.generator,
                           "cross_role": is_cross_role, "result": fused.record()})
            if fused.angle.zero:
                zero_deletions += 1
            else:
                stack.append(fused)
        else:
            stack.append(native)
    return tuple(stack), {"fusion_candidate_count": candidate_count,
        "actual_fusion_count": candidate_count,
        "cross_role_fusion_opportunities": cross_role,
        "zero_angle_deletions": zero_deletions, "events": events}


def selected(native, mask):
    if mask not in ("NONE", "D", "R", "DR"):
        raise ValueError("unknown placement")
    if "D" in native.roles and "R" in native.roles:
        raise ValueError("mixed-role fusion requires a new reviewed placement contract")
    return bool(native.roles & ({"D", "R"} if mask == "DR" else {mask}))


def registered_domain(contract):
    """Enumerate only the adopted literal domain; no mask/signal/cost evaluation."""
    domain = contract["domain"]
    if domain["sizes"] != [8, 16, 32, 64] or domain["templates"] != ["A", "B", "C"]:
        raise ValueError("registered SP-1 domain changed")
    specs = domain["template_specs"]
    wrappers = []
    for template in domain["templates"]:
        spec = specs[template]
        for n in domain["sizes"]:
            paths = []
            sigmas = (1, -1) if template == "C" else (1,)
            for sigma in sigmas:
                logical = []
                for position in range(n):
                    if template == "C":
                        gate = spec["block"][position % 4]
                        angle = parse_angle(gate["angle"])
                        if gate["role"] == "R":
                            angle = angle.scaled(sigma)
                    else:
                        gate = {"pauli": spec["pauli_cycle"][position % 2], "role": "D"}
                        angle = parse_angle(spec["angle_cycle"][position % len(spec["angle_cycle"])])
                    logical.append({"pauli": gate["pauli"], "role": gate["role"], "angle": angle})
                pre = lower_logical(logical)
                post, fusion = canonical_fusion(pre)
                if fusion["cross_role_fusion_opportunities"]:
                    raise ValueError("registered path has unexpected cross-role fusion; do not alter domain")
                paths.append({"path_id": f"{template}-n{n}-sigma{sigma:+d}",
                    "probability": F(1, len(sigmas)), "outer_weight": F(1),
                    "logical": tuple(logical), "pre": pre, "post": post, "fusion": fusion})
            wrappers.append({"wrapper_id": f"{template}-n{n}", "template": template,
                             "n": n, "paths": paths})
    return wrappers


def fusion_audit(contract):
    rows = []
    for wrapper in registered_domain(contract):
        for path in wrapper["paths"]:
            rows.append({"wrapper_id": wrapper["wrapper_id"], "path_id": path["path_id"],
                "probability": str(path["probability"]), "outer_weight": str(path["outer_weight"]),
                "pre_fusion_native_sequence": [g.record() for g in path["pre"]],
                **path["fusion"], "post_fusion_native_sequence": [g.record() for g in path["post"]]})
    return {"status": "STATIC_PRE_PLACEMENT_FUSION_AUDIT", "paths": rows,
        "registered_wrappers": 12, "registered_paths": len(rows),
        "cross_role_fusion_opportunities": sum(r["cross_role_fusion_opportunities"] for r in rows),
        "fusion_candidate_count": sum(r["fusion_candidate_count"] for r in rows),
        "actual_fusion_count": sum(r["actual_fusion_count"] for r in rows),
        "mask_dependent_fusion": False, "commuting_reordering": False,
        "resource_signal_evaluations": 0, "science_execution_authorized": False}
