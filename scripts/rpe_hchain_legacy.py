"""Reuse reconciled H-chain predecessor inputs and Q3 in the common controller."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

from rpe_hchain_q3 import METHOD_LABELS, prepared_route, request_inputs
from trottertracks import rpe_method_optimizer as optimizer


def checked(settings, plan, *, verify_files=False, metadata_only=False):
    if "legacy" not in settings:
        return None
    fixed = settings["legacy"]
    root = Path(fixed["directory"]).resolve()
    raw = (root / "LEGACY.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != fixed["sha256"]:
        raise ValueError("Legacy H-chain handoff changed")
    history = json.loads(raw, parse_int=optimizer._outcome_int)
    if (history["schema"] != "RPE_HCHAIN_LEGACY_V1" or history["owner"] != plan["owner"]
            or history["plan_sha256"] != settings["plan"]["sha256"]):
        raise ValueError("Legacy H-chain handoff belongs to another fixed scope")
    if verify_files:
        # The PBS-only controller needs the fixed Q3 request and its bundled
        # candidate inputs. Large preparation/raw result files stay at the
        # compute site, where the normal child performs the full verification.
        needed = ({name for files in history["cases"].values() for name in files.values()} |
                  {folder + "/REQUEST.json" for folder in history["jobs"]} |
                  {"PLAN.json", "PURE_RTE_COST_SCREEN.json"}) if metadata_only else None
        for name, expected in history["files"].items():
            if needed is not None and name not in needed:
                continue
            relative = PurePosixPath(name)
            path = root / relative
            if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                    or not path.resolve().is_relative_to(root) or digest(path) != expected):
                raise ValueError(f"Legacy H-chain evidence changed: {name}")
        for name in history["files"]:
            if not metadata_only and name.startswith("prices/") and name.endswith(".json"):
                receipt = json.loads((root / name).read_bytes())["events"]
                if isinstance(receipt, dict):
                    sidecar = str(Path(name).with_suffix(".events.json.xz"))
                    if history["files"].get(sidecar) != receipt["sha256"]:
                        raise ValueError(f"Legacy event archive differs: {name}")
        candidate_ids = {case["id"] for case in plan["cases"]}
        charges = (*history["records"], *history["holds"])
        identities = {(row["configuration_id"], tuple(row["N_X"]), tuple(row["N_Y"])) for row in charges}
        if (len(identities) != len(charges) or len(charges) != sum(history["spent"].values())
                or any(row[0] not in candidate_ids for row in identities)):
            raise ValueError("Legacy H-chain Q3 charge inventory is inconsistent")
    return root, history


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_receipts(run, legacy, *, plan=None):
    root, history = legacy
    allowed = None
    if plan is not None:
        from run_rpe_hchain_preparation import planned_inputs
        allowed = {f"signals/{cid}.json" for cid in planned_inputs(plan, "signal")}
        for stage in ("compile", "audit"):
            for cid, cell in planned_inputs(plan, stage).items():
                for index in range(plan["scope"]["price_paths"] if cell["method"] == "PR" else 1):
                    allowed.add(f"prices/{cid}-{index}.json")
                    allowed.add(f"prices/{cid}-{index}.events.json.xz")
    for name, expected in history["files"].items():
        if not name.startswith(("signals/", "prices/")) or (allowed is not None and name not in allowed):
            continue
        target = run / name
        target.parent.mkdir(exist_ok=True)
        source = root / name
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        if target.is_symlink() or (not target.samefile(source) and digest(target) != expected):
            raise ValueError(f"Run already has different saved H-chain evidence: {name}")


def install(plan, legacy, rows, routes, expected):
    root, history = legacy
    # Index once: large H6 continuations contain thousands of launches.
    by_case = {}
    for folder in history["jobs"]:
        original = json.loads((root / folder / "REQUEST.json").read_bytes())
        by_case.setdefault(original["configuration_id"], []).append(original)
    for cid, files in history["cases"].items():
        if cid not in rows:
            continue
        paths = {key: root / relative for key, relative in files.items()}
        entry = dict(configuration_id=cid, configuration=str(paths["configuration"]),
                     configuration_sha256=history["files"][files["configuration"]],
                     signal_packet=str(paths["signal_packet"]),
                     signal_packet_sha256=history["files"][files["signal_packet"]])
        route = prepared_route(entry, paths["registered_prices"], owner=history.get("case_owners", {}).get(cid, plan["owner"]), target=plan["target"],
                               binding=plan["binding"], interpreter=sys.executable)
        row = rows[cid]
        method = row["method"]
        axes = json.loads(paths["registered_prices"].read_bytes())["methods"][METHOD_LABELS[method]]["axes"]
        # Verify at the current adapter boundary for every original allocation,
        # including the charged attempt with a held result.
        for original in by_case.get(cid, []):
            request_inputs(dict(original, route=route))
        row.update(ready=True, needs=[], prices=[optimizer.frac(axis["unit_price_RZ"]) for axis in axes],
                   signal_source=entry["signal_packet"], price_axes=axes, price_status="SOURCE_BOUND_HCHAIN_PRICES")
        expected[cid] = dict(M=row["M"], signal_source=entry["signal_packet"])
        routes[method].append(route)


def remaining(budgets, history):
    return {method: cap if cap == "UNLIMITED" else max(0, cap - history["spent"].get(method, 0))
            for method, cap in budgets.items()}
