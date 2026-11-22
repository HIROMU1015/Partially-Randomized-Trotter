"""Exercise one-method dispatch and recovery without running scientific Q3."""

import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import pytest

import h6_q3_adapter
import run_h6_continuation as dispatch
import run_q3


ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ROOT / "recovered/SIGNALS.json"
CONFIGS = {"FULL_PF": "FULL", "TRUNCATED_PF": "B0I", "PR_DF": "PRDF"}
PASS = "MODEL_BOUND_MEETS_ADOPTED_THRESHOLD"


def mock_registry(tmp_path, methods):
    entries = {}
    for method in methods:
        config = ROOT / "initial-evidence" / f"{CONFIGS[method]}-CONFIGURATION.json"
        axes = [dict(m=m, axis=axis, unit_price_RZ=100 + 3 * m + (axis == "Y"),
                     status="ADMITTED", kind="SYNTHETIC", price_evidence={"fixture": True})
                for m in range(13) for axis in ("X", "Y")]
        entries[dispatch.METHODS[method]] = dict(
            configuration_id=run_q3.read(config)["id"], configuration=str(config),
            configuration_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), axes=axes,
        )
    path = tmp_path / "prices.json"
    run_q3.write(path, dict(scope="SYNTHETIC", compiler=run_q3.read(SIGNALS)["compiler"],
                            signal_packet_sha256=hashlib.sha256(SIGNALS.read_bytes()).hexdigest(),
                            methods=entries))
    return path


def test_method_can_dispatch_recover_and_ingest_independently(tmp_path, monkeypatch):
    monkeypatch.setenv("H6_ALLOCATED_CPUS", "4")
    launches = []
    run_process = dispatch.subprocess.run

    def mocked_worker(args, **kwargs):
        path = Path(args[args.index("--output") + 1])
        request = run_q3.read(path / "REQUEST.json")
        _, _, _, _, _, identity = h6_q3_adapter.request_inputs(request)
        if args[-1] == "dry-check":
            run_q3.write(path / "DRY_CHECK.json", dict(status="DRY_CHECK_PASS"))
        else:
            launches.append(path.name)
            run_q3.write(path / "OUTCOME.json", dict(
                status=PASS, identity=identity, request_id=request["request_id"],
                route=request["route"]["kind"], MSE_lower=0, MSE_upper=0,
                scientific_evaluation_calls=1, artificial=True,
            ))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(dispatch.subprocess, "run", mocked_worker)
    output = tmp_path / "outcomes"
    output.mkdir()
    full = tmp_path / "full"
    full.mkdir()
    path = mock_registry(full, ["FULL_PF"])
    with pytest.raises(ValueError, match="different signal packet or compiler"):
        list(dispatch.requests(path, SIGNALS, output, "PR_DF"))
    tasks = list(dispatch.requests(path, SIGNALS, output, "FULL_PF"))
    assert dispatch.run(output, tasks, dry_check=False) == 0
    assert dispatch.run(output, tasks, dry_check=False) == 0
    assert launches == ["FULL_PF"]
    assert not (output / "TRUNCATED_PF").exists()

    adoption = tmp_path / "adoption.json"
    ingest = [sys.executable, "-B", str(ROOT / "ingest_h6_continuation.py"),
              str(ROOT), str(ROOT / "initial-evidence/PROBLEM.json"), str(path), str(output),
              str(adoption), "--method", "FULL_PF"]
    run_process(ingest, check=True)
    run_process(ingest, check=True)
    report = run_q3.read(adoption)
    assert list(report["methods"]) == ["FULL_PF"]
    assert report["methods"]["FULL_PF"]["status"] == PASS
    assert report["methods"]["FULL_PF"]["source"] == "ARTIFICIAL_JOB"

    b0i = tmp_path / "b0i"
    b0i.mkdir()
    path = mock_registry(b0i, ["TRUNCATED_PF"])
    tasks = list(dispatch.requests(path, SIGNALS, output, "TRUNCATED_PF"))
    assert dispatch.run(output, tasks, dry_check=False) == 0
    assert launches == ["FULL_PF", "TRUNCATED_PF"]
    assert run_q3.read(output / "PROCESS_RETURN.json") == {"FULL_PF": 0, "TRUNCATED_PF": 0}


def test_missing_price_and_interrupted_attempt_do_not_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("H6_ALLOCATED_CPUS", "4")
    path = mock_registry(tmp_path, ["PR_DF"])
    output = tmp_path / "outcomes"
    output.mkdir()
    tasks = list(dispatch.requests(path, SIGNALS, output, "PR_DF"))
    job, request = tasks[0]
    job.mkdir()
    run_q3.write(job / "REQUEST.json", request)
    (job / "execute.log").touch()
    monkeypatch.setattr(dispatch.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected request launch"))
    assert dispatch.run(output, tasks, dry_check=False) == 2
    assert run_q3.read(output / "PROCESS_RETURN.json") == {"PR_DF": 3}

    registry = run_q3.read(path)
    registry["methods"]["PR-DF"]["axes"].pop()
    run_q3.write(path, registry, replace=True)
    with pytest.raises(ValueError, match="all 26"):
        list(dispatch.requests(path, SIGNALS, output, "PR_DF"))
    with pytest.raises(ValueError, match="changed after"):
        h6_q3_adapter.request_inputs(request)


def test_register_and_dry_check_b0i_without_other_price_inputs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setenv("H6_ALLOCATED_CPUS", "4")
    from finish_h6_registry import assemble

    config = ROOT / "initial-evidence/B0I-CONFIGURATION.json"
    (tmp_path / config.name).symlink_to(config)
    signals = run_q3.read(SIGNALS)
    signal_sha = hashlib.sha256(SIGNALS.read_bytes()).hexdigest()
    rows = [
        dict(m=m, axis=axis, physical_id=signals["methods"]["B0I"][m]["candidate"]["id"],
             unit_price_RZ=100 + m, price_evidence={"fixture": True}, kind="SYNTHETIC")
        for m in range(13) for axis in ("X", "Y")
    ]
    run_q3.write(tmp_path / "EARLY-PRICES.json", dict(
        status="VERIFIED_STAGED_COMPILED_RECEIPTS", compiler=signals["compiler"],
        signal_packet_sha256=signal_sha, current_pr_preparation_hash="fixture", b0i_prices=rows,
    ))
    (tmp_path / "price-fast.submissions").write_text("PR-DF 4 5 123.fixture\n")
    path = tmp_path / "prices.json"
    run_q3.write(path, assemble(tmp_path, tmp_path, SIGNALS, method="TRUNCATED_PF"))
    assert list(run_q3.read(path)["methods"]) == ["B0I"]
    destination = tmp_path / "outcomes"
    subprocess.run(
        [sys.executable, "-B", str(Path(dispatch.__file__)), "--registered-prices", str(path),
         "--signals", str(SIGNALS), "--output", str(destination),
         "--method", "TRUNCATED_PF", "--dry-check"],
        check=True,
    )
    assert run_q3.read(destination / "TRUNCATED_PF/DRY_CHECK.json")["status"] == "DRY_CHECK_PASS"
