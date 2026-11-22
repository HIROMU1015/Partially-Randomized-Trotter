"""Register remaining H6 prices from bound PBS receipts for the fixed Q3 test."""

import argparse
from fractions import Fraction
from pathlib import Path

from collect_h6_prices import AXES, COMPILER, read, sha
from register_full_m7_prices import write
from run_full_long import EXPECTED_IDS


METHODS = {"FULL_PF": ("Full PF", "FULL"), "TRUNCATED_PF": ("B0I", "B0I"),
           "PR_DF": ("PR-DF", "PRDF")}


def full_prices(root, saved, binding, preparation, submissions, *, use_closure=False, scoped=False):
    rows = list(saved)
    for m in range(8, 13):
        leaf = root / f"full-m{m}"
        result_path = leaf / "result.json"
        if (scoped or use_closure and m >= 10) and not result_path.is_file():
            continue
        start = read(leaf / "STARTED.json")
        result = read(result_path)
        if (start["scientific_binding"] != binding or start["preparation_hash"] != preparation
                or start["pbs_job_id"] != submissions[m] or start["candidate"] != result["candidate"]
                or result["candidate"]["id"] != EXPECTED_IDS[m]
                or result["status"] != "PASS_PRICE_ACQUIRED" or result["round"] != m
                or (leaf / "STOP.json").exists() or set(result["axes"]) != set(AXES)
                or len({receipt["fingerprint"] for receipt in result["axes"].values()}) != 2):
            raise ValueError(f"Full PF completion does not match the registered candidate at m={m}")
        for native, axis in AXES.items():
            receipt = result["axes"][native]
            circuit = leaf / f"{native}_compiled.qpy"
            price = receipt["metrics"]["rz_count"]
            if (receipt != read(leaf / f"{native}_price.json") or receipt["compiler"] != COMPILER
                    or receipt["qpy"] != circuit.name or type(price) is not int or price <= 0
                    or receipt["qpy_sha256"] != sha(circuit)):
                raise ValueError(f"Invalid Full PF compiled receipt at m={m}/{axis}")
            rows.append(dict(m=m, axis=axis, physical_id=EXPECTED_IDS[m], unit_price_RZ=price,
                kind="DIRECT_COMPILED_DETERMINISTIC", price_evidence=dict(path=str(result_path),
                    sha256=sha(result_path), selector=["axes", native, "metrics", "rz_count"],
                    pbs_job_id=start["pbs_job_id"])))
    return rows


def paired_prices(root, early, signal_sha, *, use_closure=False, only=None):
    packets = [early]
    for line in (root / "price-fast.submissions").read_text().splitlines():
        method, first, last, job = line.split()
        if only is not None and method != only:
            continue
        folder = root / f"price-{method}-m{first}-{last}"
        retry = root / "pr-m4-5-retry.submitted"
        if (use_closure or only == "PR-DF") and (method, first, last) == ("PR-DF", "4", "5") and retry.is_file():
            job = retry.read_text().strip()
            folder = root / "price-PR-DF-m4-5-retry1"
        result_file = folder / "VERIFIED.json"
        if (only is not None or use_closure and method == "B0I" and int(first) >= 10) and not result_file.is_file():
            continue
        result = read(result_file)
        if result["pbs_job_id"] != job:
            raise ValueError(f"Price receipt does not match submitted PBS job {job}")
        packets.append(result)
    b0i, pr = [], []
    for packet in packets:
        if (packet["status"] != "VERIFIED_STAGED_COMPILED_RECEIPTS" or packet["compiler"] != COMPILER
                or packet["signal_packet_sha256"] != signal_sha
                or packet["current_pr_preparation_hash"] != early["current_pr_preparation_hash"]):
            raise ValueError("Price receipt differs from the registered H6 signal or compiler")
        if only != "PR-DF":
            b0i.extend(packet["b0i_prices"])
        if only == "B0I":
            continue
        for cohort in packet["pr_finite_cohorts"]:
            paths = cohort["paths"]
            if (len(paths) != 8 or {p["index"] for p in paths} != set(range(8))
                    or cohort["population_expectation_qualified"] is not False):
                raise ValueError("Invalid finite PR cohort")
            price = Fraction(sum(path["compiled_rz"] for path in paths), len(paths))
            unit = dict(numerator=price.numerator, denominator=price.denominator)
            if unit != cohort["finite_sample_mean"]:
                raise ValueError("PR cohort mean differs from the verified paths")
            pr.append(dict(m=cohort["m"], axis=cohort["axis"], unit_price_RZ=unit,
                           kind="DIRECT_FINITE_COHORT", qualified_E_C=False,
                           price_evidence=dict(pbs_job_id=packet["pbs_job_id"], paths=paths,
                                               preparation_hash=packet["current_pr_preparation_hash"])))
    return b0i, pr


def merge_closure(root, label, direct, signal_sha):
    covered = {(row["m"], row["axis"]) for row in direct}
    missing = {(m, axis) for m in range(13) for axis in ("X", "Y")} - covered
    if not missing:
        return direct
    folder = root / f"closure-{label}"
    path = folder / "DERIVED_PRICES.json"
    audit = read(path)
    if (audit["method"] != label or audit["compiler"] != COMPILER
            or audit["signal_packet_sha256"] != signal_sha
            or audit["scope"] != "H6_FIXED_COMPILER_CLOSURE_CONDITIONAL"):
        raise ValueError(f"Conditional {label} price source differs from the accepted audit")
    qualified = {}
    for axis in ("X", "Y"):
        source = folder / f"{axis}.json"
        report = read(source)
        if not all(report[key] == "PASS" for key in ("A1", "A2", "A3", "A4", "A5")):
            continue
        for row in report["eligible_conditional_rows"]:
            key = row["m"], row["axis"]
            if (row["axis"] != axis or row["m"] not in (10, 11, 12) or key in qualified
                    or row["kind"] != "CONDITIONAL_FIXED_COMPILER_CLOSURE_DERIVATION"
                    or row["direct_observation"] is not False or row["conditional_use_only"] is not True):
                raise ValueError(f"Invalid conditional price for {label} {key}")
            if row not in audit["axes"]:
                raise ValueError(f"Conditional {label} report and summary disagree at {key}")
            qualified[key] = dict(row, price_evidence=dict(path=str(source), sha256=sha(source),
                                                           compiler=COMPILER))
    if not missing <= set(qualified):
        raise ValueError(f"{label} has prices without sufficient closure evidence: {sorted(missing - set(qualified))}")
    return direct + [qualified[key] for key in sorted(missing)]


def assemble(root, initial, signals_path, *, use_closure=False, method=None):
    signals = read(signals_path)
    signal_sha = sha(signals_path)
    selected = METHODS if method is None else {method: METHODS[method]}
    rows_by_method = {}
    if "FULL_PF" in selected:
        m7 = read(initial / "FULL-PRICES.json")["axes"]
        bound = read(root / "payload-bound/PRICE-INPUTS.json")
        submit = dict(line.split() for line in (root / "full-fast.submissions").read_text().splitlines())
        rows_by_method["Full PF"] = full_prices(
            root, m7, bound["scientific_binding"], bound["preparation_hash"]["19"],
            {int(m): job for m, job in submit.items()}, use_closure=use_closure, scoped=method is not None,
        )
    if set(selected) & {"TRUNCATED_PF", "PR_DF"}:
        b0i, pr = paired_prices(root, read(initial / "EARLY-PRICES.json"), signal_sha,
                                use_closure=use_closure, only=None if method is None else METHODS[method][0])
        rows_by_method.update({"B0I": b0i, "PR-DF": pr})
    methods = {}
    expected = [(m, axis) for m in range(13) for axis in ("X", "Y")]
    for label, name in selected.values():
        rows = rows_by_method[label]
        if use_closure and label in ("Full PF", "B0I"):
            rows = merge_closure(root, "Full" if label == "Full PF" else label, rows, signal_sha)
        config_path = initial / f"{name}-CONFIGURATION.json"
        config = read(config_path)
        rows.sort(key=lambda row: (row["m"], row["axis"]))
        keys = [(row["m"], row["axis"]) for row in rows]
        if keys != expected:
            missing = sorted(set(expected) - set(keys))
            raise ValueError(f"Incomplete or duplicate price evidence for {label}; missing {missing}")
        for row in rows:
            m = row["m"]
            case = signals["methods"][label][m].get("candidate")
            physical = EXPECTED_IDS[m] if label == "Full PF" else case["id"]
            if row.get("physical_id", physical) != physical:
                raise ValueError(f"Signal and price differ for {label} m={m}")
            row.update(status="ADMITTED", physical_id=physical)
        methods[label] = dict(configuration_id=config["id"], configuration=str(config_path),
                              configuration_sha256=sha(config_path), axes=rows)
    return dict(scope="H6_FIXED_REPRESENTATIVES_FINITE_PR_PRICE_SCENARIO", compiler=COMPILER,
                signal_packet_sha256=signal_sha, methods=methods)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("root", "initial", "signals", "output"):
        parser.add_argument(argument, type=Path)
    parser.add_argument("--deterministic-closure", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    write(args.output, assemble(args.root.resolve(), args.initial.resolve(), args.signals.resolve(),
                                use_closure=args.deterministic_closure, method=args.method))
