import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import run_rpe_hchain_round as hchain
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor, hchain_q3_probe_window
from trottertracks.rpe_host_profiles import digest, profile_core, select_host_profile


@pytest.mark.parametrize("policy", [None, {"schema": "INVALID", "jobs": {}}])
def test_hchain_initializes_only_missing_live_q3_policy(tmp_path, monkeypatch, policy):
    monkeypatch.setattr(hchain, "selected_plan", lambda *_: ({}, []))
    monkeypatch.setattr(hchain.scope, "provider_root", lambda _: tmp_path)
    provider = SimpleNamespace(select_cases=lambda plan, cases: plan)
    monkeypatch.setattr(hchain, "load_provider", lambda _: (provider, None))
    monkeypatch.setattr(hchain.legacy_io, "checked", lambda *a, **k: None)
    monkeypatch.setattr(hchain, "price_screen", lambda *_: None)
    configured = {"host_id": "pbs-fixture", "root": "/fixed/source", "interpreter": "/fixed/python"}
    distributed = {"live_q3_readiness": True, "pbs": {"remote_root": "/scratch/runs/",
                   "workers": [{"host_id": configured["host_id"], "profile": configured}]}}
    if policy is not None:
        distributed["host_eligibility"] = policy
    spec = {"hchain": {}, "methods": [], "science_approved": True,
            "executor": {"distributed": distributed}}
    frozen = copy.deepcopy(spec)
    request = {"route": {"kind": "hchain_scoped_q3"}, "method": "PR_DF"}
    live_check = Mock(return_value={"eligible_host_ids": [], "confirmed": "dry-check-called"})

    class CheckedExecutor(Exception):
        pass

    def make_executor(*args):
        runtime = args[-1]["executor"]["distributed"]
        namespace = digest({"controller_run": str(tmp_path.resolve())})[:20]
        deployed = dict(configured, prefix_cache_directory=f"/scratch/runs/{namespace}/cache/pbs-fixture/dp")
        bound = profile_core(runtime["pbs"]["workers"][0]["profile"])
        assignment = {"request": request, "host_execution_profile": bound,
                      "host_execution_profile_sha256": digest(bound)}
        assert select_host_profile(deployed, assignment) == deployed
        assert runtime["pbs_pinned_settings"] == frozen["executor"]["distributed"]["pbs"]
        fake = SimpleNamespace(distributed=runtime, source_manifest={},
                               store=SimpleNamespace(artificial=lambda _: False), _live_eligibility=live_check)
        row = {"request": request, "eligibility": {"execution_profiles": {"pbs-fixture": bound}}}
        assert DistributedExecutor._resume_eligibility(fake, row) is row["eligibility"]
        row["eligibility"] = {"execution_profiles": {"pbs-fixture": profile_core(configured)}}
        assert DistributedExecutor._resume_eligibility(fake, row) is None
        decision = DistributedExecutor._eligibility_for(fake, request, "synthetic-h8")
        if policy is None:
            assert decision["confirmed"] == "dry-check-called"
            live_check.assert_called_once()
            assert live_check.call_args.args[2]["eligible_host_ids"] == []
        else:
            live_check.assert_not_called()
            assert decision["reasons"] == ["MISSING_HOST_ELIGIBILITY_POLICY"]
        raise CheckedExecutor

    with pytest.raises(CheckedExecutor):
        hchain.run_round(tmp_path, tmp_path, {}, spec, [], make_executor=make_executor)
    assert spec == frozen


def test_h8_batch_splits_before_phase_cache_overflow(tmp_path, monkeypatch):
    cells = {str(i): {"id": str(i), "method": "B0I", "formula": "4th(m5_best)",
                     "prefix": 1, "q": 1, "T": float(i+1), "R_outer": None}
             for i in range(12)}
    monkeypatch.setattr(hchain, "planned_inputs", lambda _, stage: cells if stage == "signal" else {})
    worker = SimpleNamespace(h8_preparation_priority=lambda *args: lambda row: (0,))
    plan = {"owner": "fixture", "scope": {"system": "H8", "price_paths": 8}}
    batches = list(hchain.batches(plan, tmp_path, worker, {"signal": 16}, {"signal": "UNLIMITED"}))
    assert [len(batch) for batch in batches] == [5, 5, 2]
    assert {cell["id"] for batch in batches for _, cell, _ in batch} == set(cells)


def test_pbs_profile_binding_keeps_preparation_provider_through_round_start(tmp_path, monkeypatch):
    monkeypatch.setattr(hchain, "selected_plan", lambda *_: ({"owner": "fixture"}, []))
    monkeypatch.setattr(hchain.scope, "provider_root", lambda _: tmp_path)
    plan_provider = SimpleNamespace(select_cases=lambda plan, cases: plan)
    preparation_worker = SimpleNamespace(h8_preparation_priority=Mock())
    monkeypatch.setattr(hchain, "load_provider", lambda _: (plan_provider, preparation_worker))
    monkeypatch.setattr(hchain.legacy_io, "checked", lambda *a, **k: None)
    monkeypatch.setattr(hchain, "price_screen", lambda *_: None)
    priority = Mock(side_effect=lambda graph, executor, plan, run, worker: worker.h8_preparation_priority())
    monkeypatch.setattr(hchain, "preparation_priorities", priority)
    monkeypatch.setattr(hchain, "task_graph", lambda *a: {})
    monkeypatch.setattr(hchain, "write", lambda *a: None)
    pbs = {"remote_root": "/scratch/runs", "workers": [{"host_id": "pbs-fixture", "profile": {}}]}
    spec = {"hchain": {}, "methods": [], "science_approved": True,
            "executor": {"distributed": {"live_q3_readiness": True, "pbs": pbs,
                                         "preparation": {"budgets": {}}}}}
    executor = SimpleNamespace(configure_preparation_graph=Mock(return_value={}),
                               _user_stop_requested=lambda: True, finish=Mock())
    hchain.run_round(tmp_path, tmp_path, {}, spec, [], make_executor=lambda *a: (executor, None))
    preparation_worker.h8_preparation_priority.assert_called_once()
    executor.finish.assert_called_once()


def test_hchain_q3_has_one_check_even_with_a_full_preparation_window():
    def row(job_id, route):
        return {"job_id": job_id, "request": {"route": {"kind": route}}}

    q3 = row("q3", "hchain_scoped_q3")
    preparation = [row(n, "preparation") for n in range(64)]
    rows = preparation + [q3]
    assert hchain_q3_probe_window(rows, 8, 4) == [q3, *preparation[8:11]]
    assert hchain_q3_probe_window(preparation, 8, 4) == preparation[8:12]
    assert hchain_q3_probe_window(rows, 8, 0) == []


def test_hchain_q3_headroom_uses_same_reservation_and_only_when_runnable(monkeypatch):
    from trottertracks import rpe_pbs

    host = "pbs-fixture"
    q3 = SimpleNamespace(demand=True, status="QUEUED", rule={"kind": "hchain_scoped_q3"},
                         job=SimpleNamespace(name="q3"))
    request = {"route": q3.rule}
    decision = {"eligible_host_ids": [host]}
    rows = {"q3": {"request": request, "state": "QUEUED", "eligibility": decision}}
    pool = {"cpu_count": 96, "memory_bytes": 350 * rpe_pbs.GIB}
    monkeypatch.setattr(rpe_pbs, "resource_need", lambda *a: (8, 160 * rpe_pbs.GIB))
    executor = SimpleNamespace(_uses_pbs_task_resources=DistributedExecutor._uses_pbs_task_resources,
                               _fits_host_walltime=lambda *a: True, _fits_host_task_class=lambda *a: True)
    reserve = DistributedExecutor._hchain_q3_headroom
    assert reserve(executor, host, [q3], rows, [], pool, {}, {}) == (q3, 8, 160 * rpe_pbs.GIB)
    active = [{"host_id": host, "method": "PR_DF", "state": "RUNNING"}]
    assert reserve(executor, host, [q3], rows, active, pool, {}, {}) is None
    assert reserve(executor, "other-host", [q3], rows, [], pool, {}, {}) is None
    assert reserve(executor, host, [q3], rows, [], dict(pool, memory_bytes=160 * rpe_pbs.GIB), {}, {}) is None
