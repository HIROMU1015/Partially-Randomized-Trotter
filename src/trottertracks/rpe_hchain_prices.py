"""Bind a complete H-chain price table to the common RPE optimizer inputs."""

import hashlib
import json
import math
from pathlib import Path
import re

from .pr_df_price_completion import exact


METHODS = {"Full PF": "FULL_PF", "B0I": "TRUNCATED_PF", "PR-DF": "PR_DF"}


def read_bound(path, expected):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("H-chain input differs from its saved SHA-256")
    return json.loads(payload)


def hydrate(problem, facts, candidates, source):
    """Add the requested canonical configurations; match the problem's H-chain identity."""
    path = str(Path(source["path"]).resolve())
    registry = read_bound(path, source["sha256"])
    signal_path = str(Path(registry["signal_packet"]).resolve())
    signal_sha = registry["signal_packet_sha256"]
    signals = read_bound(signal_path, signal_sha)
    system = problem["problem"]["system"]
    if not re.fullmatch(r"H[1-9][0-9]*", system):
        raise ValueError("H-chain inputs require a canonical H-chain system")
    n_atoms = int(system[1:])
    horizon = problem["rpe"]["M"]
    if (signals["status"] != "SIGNAL_PACKETS_READY" or signals["M"] != horizon
            or signals["compiler"] != registry["compiler"]
            or signals["target"] != problem["target"]["energy_ha"]):
        raise ValueError("H-chain signals differ from the selected problem")
    expected, routes = {}, {}
    for label, entry in registry["methods"].items():
        method = METHODS[label]
        scope = problem["methods"].get(method)
        if scope is None:
            continue
        cid = entry["configuration_id"]
        config_path = str(Path(entry["configuration"]).resolve())
        config_sha = entry["configuration_sha256"]
        config = read_bound(config_path, config_sha)
        spec = config["spec"]
        binding = spec["binding"]
        if (config["id"] != cid or spec["M"] != horizon or binding["system"] != system
                or binding["hamiltonian"] != facts["hamiltonian_hash"] or binding["state"] != facts["state_hash"]
                or binding["snapshot"] != facts["snapshot_sha256"] or binding["rank"] != facts["fragment_count"]
                or spec["profile"]["compiler"] != registry["compiler"] or exact(spec["h"]) != exact(signals["h"])):
            raise ValueError("H-chain priced configuration differs from the selected molecule or runtime")
        if scope.get("configuration_ids") is not None and cid not in scope["configuration_ids"]:
            continue
        raw_rounds = signals["methods"][label]
        if ([row["m"] for row in raw_rounds] != list(range(horizon + 1))
                or [row["m"] for row in config["rounds"]] != list(range(horizon + 1))):
            raise ValueError("H-chain signals require the canonical rounds in order")
        for round_, signal in zip(config["rounds"], raw_rounds, strict=True):
            raw = signal["signal"]["raw"]
            physical = signal.get("candidate")
            if (set(raw) != {"real", "imag"} or any(not math.isfinite(v) or abs(v) > 1 for v in raw.values())
                    or physical is not None and
                    (physical["system"] != system or physical["q"] != round_["q_PF"]
                     or physical["R_outer"] != round_["R_outer"] or physical["formula"] != spec["formula"]
                     or physical["prefix"] != spec["L_D"])):
                raise ValueError("H-chain signal does not match its priced round")
        axes = entry["axes"]
        if [(r["m"], r["axis"]) for r in axes] != [(m, a) for m in range(horizon + 1) for a in ("X", "Y")]:
            raise ValueError("H-chain candidate requires each X/Y price in round order")
        for row in axes:
            physical = raw_rounds[row["m"]].get("candidate")
            if (row["status"] != "ADMITTED" or not row.get("price_evidence")
                    or physical is not None and row["physical_id"] != physical["id"]):
                raise ValueError("H-chain price is missing matching physical-round evidence")
        prices = [exact(row["unit_price_RZ"]) for row in axes]
        if any(price <= 0 for price in prices):
            raise ValueError("H-chain compiled prices must be positive")
        pr = method == "PR_DF"
        if (spec["method"] != ("PR" if pr else "PF") or (pr and spec["split"].get("tail_dictionary"))
                or (not pr and ((spec["L_D"] == binding["rank"]) != (method == "FULL_PF")))):
            raise ValueError("H-chain price method differs from the canonical configuration")
        key = {"n_atoms": n_atoms, "L_D" if pr else "p": spec["L_D"], "formula": spec["formula"],
               "delta_over_h": str(exact(spec["delta_pf"]) / exact(spec["h"]))}
        if pr:
            limits = scope["L_D_range"]
            if not limits["min"] <= spec["L_D"] <= facts["fragment_count"] + limits["max_offset_from_fragment_count"]:
                continue
            if scope.get("L_D_values") is not None and spec["L_D"] not in scope["L_D_values"]:
                continue
            if scope.get("absolute_kappa_bounds") is not None:
                kappa = spec.get("kappa", {})
                if kappa.get("kind") == "FIXED_RTE_EVENTS_PER_OUTER":
                    continue
                lo, hi = map(exact, scope["absolute_kappa_bounds"])
                if kappa.get("value") is None or not lo <= exact(kappa["value"]) <= hi:
                    continue
        if scope.get("formulas") is not None and spec["formula"] not in scope["formulas"]:
            continue
        if not pr and scope.get("p_values") is not None and spec["L_D"] not in scope["p_values"]:
            continue
        if scope.get("delta_over_h_range") is not None:
            lo, hi = map(exact, scope["delta_over_h_range"])
            if not lo <= exact(key["delta_over_h"]) <= hi:
                continue
        if scope.get("requested_delta_pf_values") is not None and exact(spec["delta_pf"]) not in {
            exact(value) for value in scope["requested_delta_pf_values"]
        }:
            continue
        hits = [c for c in candidates if c["configuration_id"] == cid]
        if len(hits) > 1 or (hits and hits[0]["method"] != method):
            raise ValueError("H-chain price configuration belongs to a different optimizer candidate")
        if hits and hits[0].get("prices") is not None and list(hits[0]["prices"]) != prices:
            raise ValueError("H-chain registered prices conflict with an existing price vector")
        candidate = hits[0] if hits else dict(method=method, family="PR" if pr else "PF", caps=scope["caps"],
                                              configuration_id=cid, labels=[], saved_points=[])
        candidate.update(name=f"{system}:{method}:{cid[:8]}", key=key, M=horizon, ready=True, needs=[],
                         prices=prices, price_axes=axes, price_status="SOURCE_BOUND_HCHAIN_PRICES",
                         price_scope=registry["scope"], price_source=path, configuration_source=config_path,
                         signal_source=signal_path)
        if not hits:
            candidates.append(candidate)
        expected[cid] = dict(M=horizon, signal_source=signal_path)

        # Physics/evaluator execution stays in the selected route adapter.
        if "route" in source:
            route = dict(source["route"], registered_prices=path, registered_prices_sha256=source["sha256"],
                         configuration_ids=[cid], configurations={cid: dict(
                             configuration=config_path, configuration_sha256=config_sha,
                             signal_packet=signal_path, signal_packet_sha256=signal_sha,
                         )})
            routes[method] = [route]
    return expected, routes
