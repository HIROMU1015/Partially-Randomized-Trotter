"""Emit a read-only archive of the existing H8 PBS continuation evidence."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


def export(parent, code, job_id, output):
    provider = code / "scripts/h6_fixed_run"
    sys.path[:0] = [str(code / "src"), str(provider)]
    from h8_q3_history import memory_stopped_parent, read_history
    from h6_price_pruning import path_count, price_cells
    from run_h6_scope import scope_problem

    plan_path = parent / "PLAN.json"
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    stopped = memory_stopped_parent(parent, job_id)
    legacy_cases, spent, unsettled, holds = read_history(plan, [parent], scope_problem(plan), memory_stopped=stopped)
    if not stopped or unsettled:
        raise ValueError("The predecessor still owns unresolved H8 Q3 attempts")

    cells = {cell["id"]: cell for case in plan["cases"] for cell in case["rounds"]}
    priced = price_cells(plan)
    paths = {plan_path, parent / "STOP_REQUESTED.json", parent / "PBS_PREPARATION_TEAM.json"}
    for path in (parent / "signals").glob("*.json"):
        row = json.loads(path.read_bytes())
        if row["owner"] != plan["owner"] or row["candidate"] != cells[path.stem]:
            raise ValueError(f"Foreign saved H8 signal: {path.name}")
        paths.add(path)
    for path in (parent / "prices").glob("*.json"):
        row = json.loads(path.read_bytes())
        cell = priced[row["candidate"]["id"]]
        index = row["index"]
        if (row["owner"] != plan["owner"] or row["candidate"] != cell or row["compiler"] != plan["compiler"]
                or row["cohort"] != plan["price_cohort"] or not 0 <= index < path_count(plan, cell)
                or path.stem != f"{cell['id']}-{index}"):
            raise ValueError(f"Foreign saved H8 price: {path.name}")
        if isinstance(row["events"], dict):
            sidecar = path.with_suffix(".events.json.xz")
            if row["events"]["file"] != sidecar.name or sidecar.stat().st_size != row["events"]["size_bytes"]:
                raise ValueError(f"Saved H8 price sidecar differs: {path.name}")
            paths.add(sidecar)
        paths.add(path)

    configs, records, jobs = {}, [], []
    for cid, (entry, route, axes) in legacy_cases.items():
        files = {role: Path(path) for role, path in (
            ("configuration", entry["configuration"]), ("signal_packet", entry["signal_packet"]),
            ("registered_prices", route["registered_prices"]))}
        configs[cid] = {role: str(path.relative_to(parent)) for role, path in files.items()}
        paths.update(files.values())
    for path in sorted((parent / "jobs").glob("*/*/REQUEST.json")):
        folder = path.parent
        if not ((folder / "LAUNCH.json").is_file() or (folder / "STARTING.json").is_file()):
            continue
        request = json.loads(path.read_bytes())
        jobs.append(str(folder.relative_to(parent)))
        paths.add(path)
        for name in ("LAUNCH.json", "STARTING.json", "H6_STARTED.json", "OUTCOME.json", "PROCESS_RETURN.json"):
            member = folder / name
            if member.is_file():
                paths.add(member)
        if (folder / "OUTCOME.json").is_file():
            outcome = json.loads((folder / "OUTCOME.json").read_bytes())
            records.append(dict(configuration_id=request["configuration_id"], N_X=request["N_X"], N_Y=request["N_Y"],
                                status=outcome["status"], MSE_lower=outcome["MSE_lower"], MSE_upper=outcome["MSE_upper"],
                                source="RECONCILED_H8_PBS_Q3", job=str(folder)))
    if len(jobs) != sum(spent.values()) or len(records) + len(holds) != len(jobs):
        raise ValueError("Legacy charged Q3 count differs from the verified history")

    metadata = dict(schema="RPE_HCHAIN_LEGACY_V1", source_job=job_id, source_run=str(parent),
                    plan_sha256=hashlib.sha256(plan_raw).hexdigest(), owner=plan["owner"],
                    spent=spent, holds=holds, records=records, cases=configs, jobs=jobs, files={})
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=1) as compressed, \
            tarfile.open(fileobj=compressed, mode="w|") as archive:
        for path in sorted(paths):
            if path.is_symlink() or not path.resolve().is_relative_to(parent):
                raise ValueError("Legacy path leaves the stopped run")
            relative = str(path.relative_to(parent))
            raw = path.read_bytes()
            metadata["files"][relative] = hashlib.sha256(raw).hexdigest()
            info = tarfile.TarInfo(relative)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
        body = json.dumps(metadata, separators=(",", ":")).encode()
        info = tarfile.TarInfo("LEGACY.json")
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)
    export(args.run.resolve(), args.code.resolve(), args.job, sys.stdout.buffer)


if __name__ == "__main__":
    main()
