"""A fixed H-chain plan yields the ordinary preparation graph and leaf outputs."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_rpe_hchain_preparation as hchain
from trottertracks.rpe_preparation_queue import normalize_preparation_request


def test_fixed_batches_reuse_signal_price_and_audit_provider(tmp_path):
    signal = dict(id="physical-signal", method="PR", R_outer=2)
    anchor = dict(id="physical-anchor", method="PR", R_outer=5, m=None)
    holdout = dict(id="physical-holdout", method="PR", R_outer=8, m=None)
    designs = {"model": dict(anchors=[anchor], holdout=holdout)}
    cases = [dict(rounds=[signal])]
    plan = dict(cases=cases, designs=list(designs.values()), scope=dict(price_paths=4),
                owner="owner", binding={"system": "H8"}, target=-1, compiler="fixed",
                price_cohort="retained-cohort")
    batches = [("signal", [[signal["id"], 0]]), ("audit", [[holdout["id"], 1]]),
               ("compile", [[anchor["id"], 0], [anchor["id"], 1]])]
    settings = dict(source=dict(commit="a"*40, files={}), assets=dict(controller_path="assets"),
                    budgets=dict(signal=1, audit=1, compile=1), approved=True)
    graph = hchain.preparation_graph(plan, "b"*64, batches, **settings)
    requests = [normalize_preparation_request(task["request"]) for task in graph["tasks"]]
    assert len({request["request_id"] for request in requests}) == len(batches)
    assert requests[1]["preparation"]["outputs"] == [
        "prices/physical-holdout-1.json", "prices/physical-holdout-1.events.json.xz"]
    calls = []
    state = SimpleNamespace(cases=cases, binding=plan["binding"], target=plan["target"], compiler=plan["compiler"])

    def price(cell, index, cohort):
        calls.append((cell, index, cohort))
        return {"rz": 10}

    provider = SimpleNamespace(startup=lambda *args: state, calibration_design=lambda *args: designs,
                              signal=lambda cell: {"candidate": cell["id"]}, price=price,
                              write=lambda path, data: path.write_text(str(data)),
                              write_price=lambda path, data: path.write_text(str(data)))
    for stage, targets in batches:
        hchain.run(plan, stage, targets, tmp_path, tmp_path, provider)
    assert calls == [(holdout, 1, "retained-cohort"), (anchor, 0, "retained-cohort"),
                     (anchor, 1, "retained-cohort")]
    assert (tmp_path / "signals/physical-signal.json").is_file()
    assert (tmp_path / "prices/physical-anchor-1.json").is_file()
    with pytest.raises(ValueError, match="more than one batch"):
        hchain.preparation_graph(plan, "b"*64, [*batches, batches[0]], **settings)
    with pytest.raises(ValueError, match="outside the specified plan and stage"):
        list(hchain.leaves(plan, "compile", [[holdout["id"], 1]]))
    assert len(calls) == 3


def test_publish_preparation_adopts_after_the_complete_price_pair(tmp_path):
    artifacts = {"prices/round-0.json": "receipt", "prices/round-0.events.json.xz": "events"}
    spec = dict(provider="HCHAIN_SCOPED", outputs=list(artifacts))
    calls = []
    executor = SimpleNamespace(
        store=SimpleNamespace(row=lambda job: {"request": {"preparation": spec}}),
        preparation_result=lambda *args, **kwargs: dict(status="PREPARATION_COMPLETE", artifacts=artifacts),
        artifact_cache=SimpleNamespace(materialize=lambda files, *args, **kwargs: calls.extend(files.values())),
        adopt_preparation=lambda job: calls.append(job))
    hchain.publish_preparation(executor, "job", tmp_path)
    assert calls == ["events", "receipt", "job"]
    calls.clear()
    spec["outputs"].append("extra-output")
    with pytest.raises(ValueError, match="output names differ"):
        hchain.publish_preparation(executor, "job", tmp_path)
    assert calls == []
