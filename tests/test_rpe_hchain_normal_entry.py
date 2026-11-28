"""Use the ordinary H-chain entry and two synthetic PBS transports."""
import base64
from collections import Counter
from fractions import Fraction
import hashlib
from pathlib import Path
import sys
from threading import Condition
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_rpe_method_optimizer as cli
import run_rpe_hchain_round as hchain
import run_rpe_pbs_storage_worker as launcher
import rpe_pbs_gateway as gateway_wire
from trottertracks import rpe_hchain_scope as scope
from trottertracks import rpe_local0_search as central
from trottertracks import rpe_method_optimizer as optimizer
from trottertracks import rpe_pbs as pbs
from trottertracks import rpe_q3_coordinator as coordinator
from trottertracks import rpe_worker_storage as storage
from trottertracks.rpe_optimizer_concurrent import ConcurrentExecutor, DistributedExecutor
from trottertracks.rpe_preparation_artifacts import ArtifactCache
from test_rpe_hchain_q3_route import prepared


HOSTS = ("pbs-a", "pbs-b")
GIB = 1024 ** 3


def pool(tmp_path, **extra):
    limit = dict(limit_bytes=50 * GIB, roots=[str(tmp_path)], sample_path=str(tmp_path / "storage.json"))
    return dict(ssh_alias="ARTIFICIAL_ONLY", remote_root=str(tmp_path / "pbs"), transport_source="transport",
                transport_root=str(ROOT), python=sys.executable,
                workers=[dict(host_id=host, profile=dict(host_id=host, root=str(ROOT), interpreter=sys.executable,
                         dp_workers=1, slots=1, slot_policy=dict(max_slots=1), storage_limit=limit,
                         route_views={"cbc471c913072de45c25a0ed22d51dff8a9b7034": str(ROOT)})) for host in HOSTS],
                resources=dict(ncpus=96, mem="350gb", walltime="72:00:00", queue="workq"),
                allocation_sizing=dict(per_task_resources=True, q3_startup_memory_bytes_per_task=GIB),
                **extra)


def transport(settings):
    return dict(pbs=settings, pbs_dispatch_gateway=dict(path="/fixed/gateway", sha256="bound",
                split_exchange=True, compute_storage_sampling=True, bounded_results=True, defer_q3_evidence=True,
                selective_harvest=True), pbs_gateway_storage_launcher=dict(path="/fixed/worker", sha256="bound"),
                pbs_gateway_completion_workers=2, port=0)


def fixed_scope(tmp_path):
    plan = dict(owner="fixture", binding=dict(system="H8", rank=28, hamiltonian="artificial"), target=-1.0, compiler="fixed",
                scope=dict(system="H8", M=0, methods=dict(FULL_PF={}), scaled_shots={}, price_paths=1),
                cases=[dict(id=host, method="FULL_PF", width_over_h=1, formula="2nd", prefix=28, rounds=[dict(
                    id=host+"-physical", method="FULL", formula="2nd", q=1, prefix=28, R_outer=None)]) for host in HOSTS],
                designs=[])
    path = tmp_path / "PLAN.json"
    central.write(path, plan)
    return plan, dict(plan=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                      pbs_nodes=2, batch_sizes={"signal": 1},
                      provider_source=dict(commit="fixture", files={}), assets={"controller_path": "fixture"})


def test_normal_start_resume_pin_the_same_two_worker_round(tmp_path, monkeypatch):
    _, hchain_options = fixed_scope(tmp_path)
    round_path = tmp_path / "template.json"
    central.write(round_path, dict(max_new_q3_per_method={"FULL_PF": 0}, executor=dict(scheduler="concurrent",
        distributed=transport(pool(tmp_path, task_seconds=dict(q3=60)))) ))
    config = dict(hchain_profiles={"H8": dict(hchain=hchain_options, entry="method", round=str(round_path))},
                  readiness_metadata={"H8": {}}, state_directory=str(tmp_path / "central"),
                  code_root=str(ROOT), code_source="fixed", computation_source="fixed", python=sys.executable)
    monkeypatch.setattr(central, "fixed_implementation", lambda config: {"source": "fixed"})
    monkeypatch.setattr(scope, "provider_root", lambda settings: ROOT)
    calls = []

    def round_call(root, run, problem, spec, extra, **kwargs):
        calls.append((run, problem, spec))

    monkeypatch.setattr(hchain, "run_round", round_call)

    def launch(run, problem, spec):
        cli.round_mode(SimpleNamespace(round=run/"ROUND.json", run_dir=run), problem, [])
        marker = dict(invocation_id=str(len(calls)), pid=0)
        central.write(run / "LAUNCH.json", marker)
        central.write(run / "EXIT.json", dict(marker, returncode=0))

    gateway = central.Gateway(config, launch=launch)
    request = dict(scope=dict(system="H8"), instruction="Synthetic H8 only")
    preview = gateway.handle(dict(request, action="preview"))
    assert preview["applied"]["execution_hosts"] == list(HOSTS)
    started = gateway.handle(dict(request, action="start"))
    assert started["status"] == "STARTED"
    assert started["preflight"]["status"] == "PASS"
    run = Path(started["run_directory"])
    gateway.handle(dict(action="stop", run_id=started["run_id"]))
    central.write(round_path, {"new_template": "belongs to a different future Start"})
    resumed = gateway.handle(dict(action="resume", run_id=started["run_id"], instruction=request["instruction"]))
    assert resumed["status"] == "STARTED"
    assert calls[0] == calls[1]
    assert calls[1][0] == run
    assert calls[1][2]["executor"]["distributed"]["pbs"]["allocation_sizing"]["fixed_node_resources"]
    assert len(list((run / "stop-history").glob("*.json"))) == 1


@pytest.mark.parametrize('screened',[False,True])
def test_new_round_feeds_collected_inputs_back_to_the_common_optimizer(tmp_path, monkeypatch,screened):
    plan, options = fixed_scope(tmp_path)
    if screened:options['screened_ready_streaming']=True
    run = tmp_path / "run"
    run.mkdir()
    cfg = run / "round.json"
    distributed = transport(pool(tmp_path, task_seconds=dict(signal=60, q3=60)))
    central.write(cfg, dict(executor=dict(scheduler="concurrent", distributed=distributed),
                           max_new_q3_per_method={"FULL_PF": "UNLIMITED"}))
    problem, spec, _ = scope.resolve(dict(hchain=options, round=str(cfg)), dict(action="start",
        scope=dict(system="H8", preparation_budgets=dict(signal="UNLIMITED"))))
    calls = []
    executor = SimpleNamespace(
        _user_stop_requested=lambda: False, state=SimpleNamespace(cv=Condition(), failure=None),
        artifact_cache=ArtifactCache(run/"cas"), source_manifest={"commit":"fixture","files":{}}, finish=lambda: None)
    provider = SimpleNamespace(
        select_cases=lambda plan, cases: dict(plan, cases=cases), price_models=lambda *args, **kwargs: ({}, []),
        ready_batch=lambda plan, run, methods, installed, **kwargs: dict(cases=[
            case for case in plan["cases"] if case["id"] not in installed
            and (run / "signals" / (case["rounds"][0]["id"] + ".json")).is_file()]))
    worker = SimpleNamespace(h8_preparation_priority=lambda *args: lambda job: (0, 1))
    from threading import Event
    import time
    active=Event();received=[];screen_updates=[]
    if screened:
        executor.supports_nonblocking_prefetch=True
        def update(*_):
            assert not active.is_set(), 'screen must not change during active selector'
            screen_updates.append(1)
            return set()
        screen=SimpleNamespace(update=update,state={'skip_price_cells':set()})
        monkeypatch.setattr(hchain,'price_screen',lambda *_a,**_k:screen)
    monkeypatch.setattr(scope, "provider_root", lambda settings: ROOT)
    monkeypatch.setattr(hchain, "load_provider", lambda path: (provider, worker))
    monkeypatch.setattr(hchain, "prepared_case", lambda provider, run, plan, case, models, **kwargs: (
        [dict(unit_price_RZ=1)] * 2, dict(signal_packet=case["id"]), dict(kind="hchain_scoped_q3")))

    def register(graph,**kwargs):
        assert len(graph["tasks"]) == 2
        mapping = {}
        for task in graph["tasks"]:
            prep = task["request"]["preparation"]
            packet = executor.artifact_cache.path(prep["inputs"]["PLAN.json"])
            assert central.load(packet)["scope"] == plan["scope"]
            mapping[task["name"]] = prep["outputs"][0]
        return mapping

    executor.configure_preparation_graph = register
    executor.ready_preparations = lambda jobs, **kwargs: (
        sorted(jobs)[:1] if not screened or not received or active.is_set() else [])

    def receive(ex, job, run):
        received.append(job)
        central.write(run / job, {"status": "synthetic"})
        return {"status": "PREPARATION_COMPLETE"}

    def optimize(root, problem, *, context, **kwargs):
        ready = tuple(row["configuration_id"] for row in context["candidates"] if row["ready"])
        calls.append(ready)
        if screened:
            source=kwargs['ready_source']
            active.set()
            try:
                deadline=time.monotonic()+5
                view=Mock();trial=Mock()
                view.for_method.return_value=trial
                view.routes={'FULL_PF':[]};view.expected={}
                trial.route.return_value={'kind':'fixture'}
                trial.refresh_execution_hold.return_value=None
                trial.expected_for.return_value={}
                pending,take=source.subscribe('FULL_PF',{},[],view)
                while not pending():
                    if time.monotonic()>deadline:pytest.fail('late READY never reached active selector')
                    with executor.state.cv:executor.state.cv.wait(.02)
                late,_=take()
                assert [row['configuration_id'] for row in late]==[HOSTS[1]]
                return {'FULL_PF':{'stop_reason':'SCALED_SHOT_GRID_COMPLETE'}}
            finally:active.clear()
        evaluator = ConcurrentExecutor(root, run / "toy-optimizer", {"FULL_PF": []}, {}, problem,
                                       budgets={"FULL_PF": 0}, science_approved=False)
        try:
            return optimizer.optimize(root, problem, context=context, executor=evaluator,
                                      max_new=0, science_approved=False)
        finally:
            evaluator.finish()

    monkeypatch.setattr(hchain, "publish_preparation", receive)
    hchain.run_round(ROOT, run, problem, spec, [], make_executor=lambda *args: (executor, optimize))
    assert calls == ([(HOSTS[0],)] if screened else [(HOSTS[0],), HOSTS])
    if screened:assert len(screen_updates)>=2
    assert set(central.load(run/"ROUND_RESULT.json")["ready"]) == set(HOSTS)


def test_two_allocations_sample_and_recover_their_own_artifacts(tmp_path, monkeypatch):
    settings = pool(tmp_path, task_seconds=dict(q3=60))
    profile = transport(settings)
    assert scope.pbs_profiles(profile, 2) == list(HOSTS)
    server = SimpleNamespace(server_address=("127.0.0.1", 1), shutdown=lambda: None, server_close=lambda: None)
    monkeypatch.setattr(coordinator, "serve", lambda *args: (server, SimpleNamespace(join=lambda: None)))
    monkeypatch.setattr(pbs.PbsManager, "start", lambda self: None)
    routes = {"PR_DF": [prepared(tmp_path, host) for host in HOSTS]}
    method_problem = dict(accuracy=dict(epsilon_E_decimal="0.00159360010199040"), target=dict(energy_ha=-1.0),
                          rpe=dict(M=0))
    source = cli.head()
    executor = DistributedExecutor(ROOT, tmp_path / "run", routes, {host: dict(M=0) for host in HOSTS},
                                   method_problem, optimizer_source=source, max_concurrent=2, budgets={"PR_DF": 2},
                                   distributed=profile)
    manager = executor.pbs_manager
    # Stub only the cluster transport. The state, chunk reader and ownership checks are real.
    remote = Mock()
    remote.storage_report.side_effect = AssertionError("login-node scan")
    manager.gateway = remote
    manager.evidence_gateway = remote
    observed = Counter()

    def rpc(message):
        if message["action"] == "exchange":
            return dict(allocations=[], requests={})
        if message["action"] == "read_bulk_chunk":
            observed[message["token"]] += 1
            return gateway_wire.read_bulk_chunk(tmp_path / message["token"], message)
        return {}

    remote.call.side_effect = rpc
    try:
        method = executor.for_method("PR_DF")
        proposals = [dict(configuration_id=host, vector=(1, 1), cost=Fraction(20), origin=dict(kind="fixture"))
                     for host in HOSTS]
        policy = {}
        for host, proposal in zip(HOSTS, proposals, strict=True):
            job = method.job_dir(proposal)
            request = method.request(proposal, job, method.route(host))
            fingerprint = dict(host_id=host, fixture=True)
            row = dict(method="PR_DF", request_sha256=coordinator.digest(request), input_hashes=[],
                       source_manifest=executor.source_manifest, source_manifest_sha256=coordinator.digest(executor.source_manifest),
                       eligible_host_ids=[host], hosts={host: dict(status="PASS",
                           current_host_fingerprint_sha256=coordinator.digest(fingerprint))})
            row["decision_sha256"] = coordinator.digest(row)
            policy[job.name] = row
        executor.distributed["host_eligibility"] = dict(schema="RPE_Q3_HOST_ELIGIBILITY_V1", jobs=policy)
        tasks = [method._enqueue(proposal, demand=True) for proposal in proposals]
        manager.cycle()
        allocations = {record["spec"]["profile"]["host_id"]: record["spec"]
                       for record in manager.state["allocations"].values()}
        assert list(allocations) == list(HOSTS)
        tokens = []
        for host, task, proposal in zip(HOSTS, tasks, proposals, strict=True):
            allocation = allocations[host]
            token = allocation["token"]
            tokens.append(token)
            assert allocation["resources"] == settings["resources"]
            assert allocation["profile"]["pbs"]["compute_storage_sampling"] is True
            assert allocation["profile"]["pbs"]["defer_q3_evidence"] is True
            sample_root = tmp_path / token
            sample_root.mkdir()
            sample = sample_root / "STORAGE_USAGE_GATEWAY.json"
            limit = dict(allocation["profile"]["storage_limit"], sample_path=str(sample))
            measured = storage.report(limit, spools=(), refresh=True)
            assert measured["accepting_new_jobs"] is True
            assert launcher.GatewayStorage(sample, max_age=60, local_clock=True).report(limit) == measured
            resources = dict(cpu_count=96, memory_bytes=350 * GIB)
            offer = dict(accepting=True, remaining_seconds=72 * 3600, resources=resources, job_id=token + ".fixture")
            manager.allocations.append(dict(token=token, host_id=host, status="R", job_id=offer["job_id"], worker=offer))
            polled = manager._dispatch(token, dict(request_id=token + "-poll", endpoint="/v1/poll",
                payload=dict(host_id=host, slots=1, source_sha=source,
                    resource_report=dict(pbs=dict(offer, resource_pool=resources), storage=measured),
                    host_fingerprints={task.job.name: dict(host_id=host, fixture=True)})))
            assert polled["ok"], polled
            assignment, = polled["result"]["assignments"]
            identity = dict(configuration_id=host, M=0, contract_matches=True, allocation=[dict(m=0, N_X=1, N_Y=1)])
            root = tmp_path / (host + "-result")
            root.mkdir()
            central.write(root / "OUTCOME.json", dict(request_id=assignment["job_id"],
                route="hchain_scoped_q3", status=optimizer.PASS, identity=identity, MSE_lower=0, MSE_upper=0))
            central.write(root / "PROCESS_RETURN.json", dict(returncode=0, signal=None))
            owner = {key: assignment[key] for key in ("job_id", "attempt_id", "request_sha256")}
            data = (host.encode() + b"x" * 1024**2)
            central.write(sample_root / "PROFILE.json", allocation["profile"])
            attempt = Path(allocation["profile"]["work_dir"]) / owner["job_id"] / owner["attempt_id"]
            central.write(attempt / "ASSIGNMENT.json", assignment)
            (attempt / "HCHAIN_RESULT.json").write_bytes(data)
            files = {path.name: base64.b64encode(path.read_bytes()).decode() for path in root.iterdir()}
            response = manager._complete_with_deferred_evidence(token, dict(request_id=token, payload=dict(
                owner, host_id=host, files=files, pbs_evidence={"HCHAIN_RESULT.json": len(data)})))
            assert response["result"]["evidence_pending"] is True
            marker = manager.evidence_directory / owner["job_id"] / (owner["attempt_id"] + ".json")
            manager._archive_evidence(marker, pbs.read(marker))
            assert method.answer(proposal)["status"] == optimizer.PASS
            assert (task.job / "HCHAIN_RESULT.json").read_bytes() == data
        assert len(set(tokens)) == 2
        # Both independently owned artifacts fit the upgraded bounded chunk.
        assert list(observed.values()) == [1, 1]
        remote.storage_report.assert_not_called()
    finally:
        manager.close = lambda **kwargs: manager.lock.close()
        with executor.state.cv:
            for task in executor.state.tasks.values():
                if task.status == "RUNNING":
                    task.status = "DONE"
                    task.event.set()
            executor.state.active = 0
            executor.state.cv.notify_all()
        executor.finish()


import pytest


@pytest.mark.parametrize("system", ["H6", "H8"])
def test_hchain_independent_parent_and_child_use_exact_cases(tmp_path, monkeypatch, system):
    from trottertracks import rpe_pbs_offline as offline

    plan, options = fixed_scope(tmp_path)
    plan["cases"][1]["width_over_h"] = 2
    plan["scope"]["system"] = plan["binding"]["system"] = system
    central.write(Path(options["plan"]["path"]), plan)
    options["plan"]["sha256"] = hashlib.sha256(Path(options["plan"]["path"]).read_bytes()).hexdigest()
    options["pbs_offline_search"] = dict(mode="independent", source="a"*40,
        code_root=str(ROOT), interpreter=sys.executable, pbs_share=1,
        memory_bytes_per_task=GIB)
    run = tmp_path/"run"; run.mkdir()
    cfg = run/"template.json"
    distributed = transport(pool(tmp_path, task_seconds=dict(signal=60, q3=60)))
    central.write(cfg, dict(round_id="fixed", executor=dict(scheduler="concurrent", distributed=distributed),
                            max_new_q3_per_method={"FULL_PF": 3}))
    policy = {"pbs_offline_search": options.pop("pbs_offline_search")}
    problem, spec, canonical = scope.resolve(dict(hchain=options, round=str(cfg), release_worker_policy=policy),
        dict(action="start", scope=dict(system=system, preparation_budgets=dict(signal=5, compile=0, audit=0))))
    for name, data in (("ROUND.json", spec), ("PROBLEM.json", problem),
                       ("APPLIED_SCOPE.json", dict(run_id="search-test", scope=canonical))):
        central.write(run/name, data)
    monkeypatch.setattr(scope,'provider_root',lambda _:ROOT)
    monkeypatch.setattr(hchain,'load_provider',lambda *_:(
        SimpleNamespace(select_cases=lambda p,c:dict(p,cases=c)),
        SimpleNamespace(h8_preparation_priority=lambda *args:lambda job:(0,1))))
    partition = hchain.partition_for_run(run, spec)
    assert hchain.partition_for_run(run, spec) == partition
    assert len(partition["parts"]) == len(offline.pbs_units(partition)) == 2
    assert all(block["owner"] == "pbs" for block in partition["parts"])
    assert sum(block["q3_budget"] for block in partition["parts"]) == 3
    assert sum(block["preparation_budgets"]["signal"] for block in partition["parts"]) == 5
    profile = distributed["pbs"]["workers"][0]["profile"]
    children = []
    for part in offline.pbs_units(partition):
        child_problem, child = offline.part_documents(partition, part, profile, tmp_path/part["id"])
        assert child["executor"]["distributed"]["execution_hosts"] == ["local"]
        assert "pbs" not in child["executor"]["distributed"]
        assert "pbs_offline_search" not in child["hchain"]
        second_profile = dict(profile, host_id="pbs-b", storage_limit=dict(profile["storage_limit"],
            sample_path=str(tmp_path/"replacement-sampler.json")))
        assert offline.part_documents(partition, part, second_profile, tmp_path/part["id"]) == (child_problem, child)
        _, cases = hchain.selected_plan(child["hchain"], child_problem, child["methods"])
        children.extend(case["id"] for case in cases)
    assert Counter(children) == Counter(HOSTS)
    manager = SimpleNamespace(independent_finished=lambda: True, detached=False)
    executor = SimpleNamespace(pbs_manager=manager, _user_stop_requested=lambda: False, finish=Mock())
    factory = Mock(return_value=(executor, None))
    hchain.run_round(ROOT, run, problem, spec, [], make_executor=factory)
    assert factory.call_args.args[-1]["executor"]["distributed"]["pbs_offline_plan"] == partition
    assert not (run/"distributed/PREPARATION_GRAPH.json").exists()
    assert central.load(run/"SEARCH_TERMINATION.json")["reason"] == "HCHAIN_SCOPE_COMPLETE"
    executor.finish.assert_called_once()


def test_hchain_partition_completion_never_marks_unknown_or_missing_methods(tmp_path):
    for results in ({}, {"FULL_PF": {"stop_reason": "UNKNOWN_RESULT"}},
                    {"FULL_PF": {"stop_reason": "WAITING_FOR_INPUTS"}}):
        hchain.finish_block(tmp_path, results, ["FULL_PF"])
        assert not (tmp_path/"SEARCH_TERMINATION.json").exists()
    hchain.finish_block(tmp_path, {"FULL_PF": {"stop_reason": "SCALED_SHOT_GRID_COMPLETE"}}, ["FULL_PF"])
    assert central.load(tmp_path/"SEARCH_TERMINATION.json")["reason"] == "HCHAIN_SCOPE_COMPLETE"


def test_hchain_partition_charges_legacy_attempts_only_once(tmp_path):
    from trottertracks.rpe_pbs_offline import prepare_plan, restrict_round

    plan, opts = fixed_scope(tmp_path)
    plan["cases"][1]["width_over_h"] = 2
    central.write(Path(opts["plan"]["path"]), plan)
    opts["plan"]["sha256"] = hashlib.sha256(Path(opts["plan"]["path"]).read_bytes()).hexdigest()
    legacy = tmp_path/"legacy"; legacy.mkdir()
    central.write(legacy/"LEGACY.json", dict(schema="RPE_HCHAIN_LEGACY_V1", owner=plan["owner"],
        plan_sha256=opts["plan"]["sha256"], records=[{"configuration_id":HOSTS[0]}],
        holds=[{"configuration_id":HOSTS[1]}], spent={"FULL_PF":2}))
    opts["legacy"] = dict(directory=str(legacy), sha256=hashlib.sha256((legacy/"LEGACY.json").read_bytes()).hexdigest())
    run = tmp_path/"run"; run.mkdir()
    spec = dict(methods=["FULL_PF"], hchain=opts, science_approved=True, round_id="fixed",
                max_new_q3_per_method={"FULL_PF":5},
                executor=dict(distributed=transport(pool(tmp_path))))
    for name,value in (("ROUND.json",spec),("PROBLEM.json",{}),("APPLIED_SCOPE.json",dict(run_id="test"))):
        central.write(run/name,value)
    partition=prepare_plan(run,dict(mode="independent",source="a"*40,code_root=str(ROOT),interpreter=sys.executable))
    assert sum(p["q3_budget"]-1 for p in partition["parts"]) == 3
    assert sum(restrict_round(partition,[p])["max_new_q3_per_method"]["FULL_PF"] for p in partition["parts"]) == 5


def test_hchain_partition_keeps_only_source_bound_rte_estimates(tmp_path):
    original = {"cases":[{"id":"a"},{"id":"b"}]}
    part = {"cases":[{"id":"b"}]}
    path = tmp_path/"provider.py"; path.touch()
    forecast_path = tmp_path/"forecast.json"; forecast_path.touch()
    forecast = {"comparison_id":"a,b","estimates":[{"configuration_id":"b","estimated_total_RZ":123}],
                "incumbent":{"configuration_id":"a","total_RZ":456}}
    provider=SimpleNamespace(__file__=str(path),read=lambda path:forecast,
        pure_rte=SimpleNamespace(FORECAST=forecast_path.name,
            comparison_id=lambda plan:','.join(case["id"] for case in plan["cases"])),
        build_price_pruning=lambda plan,run,forecast:forecast)
    observed=hchain.price_screen(provider,part,tmp_path,{"saved_price_screen":True},full_plan=original)
    assert observed["comparison_id"]=="b" and observed["original_comparison_id"]=="a,b"
    assert observed["estimates"]==forecast["estimates"] and observed["incumbent"] is None
    assert hchain.price_screen(provider,part,tmp_path,{"saved_price_screen":True},full_plan=part) is None


def test_pbs_only_metadata_preflight_keeps_raw_inputs_on_compute_site(tmp_path):
    import rpe_hchain_legacy as legacy
    plan, options = fixed_scope(tmp_path)
    root=tmp_path/"legacy-local";root.mkdir()
    raw=b'fixed plan metadata'
    (root/"PLAN.json").write_bytes(raw)
    checksum=hashlib.sha256(raw).hexdigest()
    record=dict(schema="RPE_HCHAIN_LEGACY_V1",owner=plan["owner"],plan_sha256=options["plan"]["sha256"],
                records=[],holds=[],spent={},cases={},jobs=[],
                files={"PLAN.json":checksum,"prices/not-local.json":"d"*64})
    central.write(root/"LEGACY.json",record)
    options["legacy"]=dict(directory=str(root),sha256=hashlib.sha256((root/"LEGACY.json").read_bytes()).hexdigest())
    assert legacy.checked(options,plan,verify_files=True,metadata_only=True)[1]==record
    with pytest.raises((ValueError,OSError)):
        legacy.checked(options,plan,verify_files=True)


def test_legacy_q3_rss_seed_matches_runtime_and_preserves_larger_live_peak(tmp_path):
    import rpe_hchain_legacy as legacy
    from rpe_hchain_q3 import RISK_COMMIT
    from trottertracks import rpe_q3_resources as resources

    runtime={"interpreter":"/fixed/python","dp_workers":7,"task_cpu_count":8}
    request=dict(method="FULL_PF",configuration_id="a",N_X=[8,4],N_Y=[8,4],
                 route=dict(kind="h6_df19_scoped_q3",view_commit=RISK_COMMIT,
                            interpreter=runtime["interpreter"],allocated_cpus=8))
    folder="jobs/FULL_PF/a";path=tmp_path/folder;path.mkdir(parents=True)
    sample=dict(method="FULL_PF",shots=[8,8,4,4],status="FINISHED",usable=True,
                returncode=0,outcome_status="PASS",sampled_peak_rss_bytes=1024**3)
    central.write(path/"REQUEST.json",request);central.write(path/"Q3_MEMORY.json",sample)
    history=dict(source_job="test",records=[dict(configuration_id="a",N_X=[8,4],N_Y=[8,4],status="PASS")],
                 jobs=[folder],files={f"{folder}/{name}":legacy.digest(path/name) for name in ("REQUEST.json","Q3_MEMORY.json")})
    seed=legacy.resource_peaks((tmp_path,history),runtime)
    kind=resources.measured_resource_class(request,runtime=runtime,preparation_detail=True)
    assert seed["counts"]=={kind:1}
    assert seed["task_resource_peaks"][kind]["dp_workers"]==8
    old=dict(task_resource_peaks={kind:dict(peak_rss_bytes=2*1024**3)},isolated_resource_class=kind)
    merged=legacy.seed_resource_state(old,seed)
    assert merged["task_resource_peaks"][kind]["peak_rss_bytes"]==2*1024**3
    assert "isolated_resource_class" not in merged and old["isolated_resource_class"]==kind
    assert not legacy.resource_peaks((tmp_path,history),dict(runtime,dp_workers=3))["counts"]


def test_compile_pair_priority_prefers_small_circuit_even_if_other_candidate_nearer():
    small=dict(q=1,prefix=5,R_outer=3)
    large=dict(q=25,prefix=10,R_outer=100)
    assert hchain.preparation_order("price",small,(1,99)) < hchain.preparation_order("price",large,(0,1))
    assert hchain.preparation_order("signal",large,(0,1)) < hchain.preparation_order("signal",small,(1,99))


def test_dispatch_priorities_use_saved_cost_only_after_small_compile():
    from fractions import Fraction
    small = dict(q=1, prefix=5, R_outer=3)
    large = dict(q=25, prefix=10, R_outer=100)
    huge = dict(q=50, prefix=10, R_outer=100)
    order = hchain.dispatch_preparation_order
    assert order("price", small, (1,99)) < order("price", huge, (0,1), Fraction(1))
    assert order("price", huge, (0,1), Fraction(1)) < order("price", large, (0,1), Fraction(2))
    assert order("price", large, (0,1)) < order("price", huge, (0,1))


def test_saved_approximate_cost_ranks_direct_and_shared_price_dependencies():
    from fractions import Fraction
    worker = SimpleNamespace(group_key=lambda c:c['group'], design_key=lambda d:d['group'])
    a=dict(id='a',group='direct');b=dict(id='b',group='direct');shared=dict(id='probe',group='fit');hold=dict(id='audit',group='fit')
    plan=dict(cases=[dict(id='chosen',rounds=[a,dict(id='round',group='fit')]),dict(id='excluded',rounds=[b])],
              designs=[dict(group='direct',mode='DIRECT',anchors=[a,b]),dict(group='fit',mode='FIT',anchors=[shared],holdout=hold)])
    screen=SimpleNamespace(approximation=SimpleNamespace(estimates={'chosen':{'estimated_total_RZ':{'numerator':7,'denominator':2}},'excluded':{'estimated_total_RZ':1}}),state={'excluded':{'excluded':{}}})
    assert hchain.approximate_price_priorities(plan,worker,screen)=={k:Fraction(7,2) for k in ('a','probe','audit')}
    assert hchain.approximate_price_priorities(plan,worker,None)=={}


def test_new_local_only_scope_needs_no_pbs_gateway(tmp_path):
    _, options = fixed_scope(tmp_path)
    options['pbs_nodes'] = 0
    options['pbs_offline_search'] = {'mode': 'independent'}
    round_path = tmp_path / 'template.json'
    central.write(round_path, dict(max_new_q3_per_method={'FULL_PF': 0}, executor=dict(
        scheduler='concurrent', distributed=dict(local_worker=dict(host_id='local', slots=1)))))
    _, spec, canonical = scope.resolve(dict(hchain=options, round=str(round_path)),
                                      dict(scope=dict(system='H8'), action='preview'))
    assert canonical['execution_hosts'] == ['local']
    assert spec['executor']['distributed']['execution_hosts'] == ['local']
    assert 'pbs_offline_search' not in spec['hchain']


def test_hchain_disposition_applies_shared_queue_policy(tmp_path):
    _, options = fixed_scope(tmp_path)
    round_path = tmp_path/'template.json'
    central.write(round_path, dict(max_new_q3_per_method={'FULL_PF':0}, executor=dict(
        scheduler='concurrent', distributed=transport(pool(tmp_path)))))
    policy = dict(controller_parallel=True, ready_queue_waves=3, controller_queue_first=True,
                  pbs_collection_workers=2, calibration_workers=5, pbs_hold_node=True,
                  pbs_ready_buffer=True,pbs_q3_memory_backfill=True,pbs_standby_waves=2)
    _, spec, _ = scope.resolve(dict(hchain=options, round=str(round_path), release_worker_policy=policy),
                               dict(scope=dict(system='H8'), action='preview'))
    assert spec['search_overrides']['controller_parallel'] is True
    assert spec['executor']['distributed']['ready_queue_waves'] == 3
    assert spec['executor']['distributed']['controller_queue_first'] is True
    assert spec['executor']['distributed']['pbs_hold_node'] is True
    assert spec['executor']['distributed']['pbs_ready_buffer'] is True
    assert spec['executor']['distributed']['pbs_q3_memory_backfill'] is True
    assert spec['executor']['distributed']['pbs_standby_waves']==2
    assert spec['hchain']['worker_policy_disposition']['not_applied'] == ['calibration_workers']


def test_legacy_digest_remains_content_bound_on_file_change(tmp_path):
    import rpe_hchain_legacy as legacy
    path = tmp_path/'legacy.json'
    path.write_bytes(b'first')
    assert legacy.digest(path) == hashlib.sha256(b'first').hexdigest()
    path.write_bytes(b'other')
    assert legacy.digest(path) == hashlib.sha256(b'other').hexdigest()


def test_hchain_ready_source_delivers_only_new_candidate_and_its_route():
    from threading import Condition
    from unittest.mock import Mock
    executor = SimpleNamespace(state=SimpleNamespace(cv=Condition()))
    rows = {'first':{'configuration_id':'first','method':'FULL_PF','ready':True},
            'late':{'configuration_id':'late','method':'FULL_PF','ready':False}}
    source = hchain.HChainReadySource(executor, rows)
    view = Mock()
    trial = Mock()
    view.for_method.return_value = trial
    view.routes = {'FULL_PF':[{'configuration_ids':['first']}]}
    view.expected = {'first':{'binding':'fixed'}}
    trial.route.return_value = {'kind':'fixture','configuration_ids':['late']}
    trial.refresh_execution_hold.return_value = None
    trial.expected_for.return_value = {'binding':'late-fixed'}
    pending, take = source.subscribe('FULL_PF', {}, [], view)
    assert not pending()
    rows['late']['ready'] = True
    source.publish(rows, {'FULL_PF':[trial.route.return_value]}, {'late':trial.expected_for.return_value}, set())
    assert pending()
    added, saved = take()
    assert [row['configuration_id'] for row in added] == ['late']
    assert saved is None
    assert view.expected == {'first':{'binding':'fixed'}, 'late':{'binding':'late-fixed'}}
    assert not pending()
    assert take() == ([],None)


def test_selected_legacy_receipt_closure_skips_only_unselected_large_receipts(tmp_path):
    import rpe_hchain_legacy as legacy
    import hashlib
    plan,options=fixed_scope(tmp_path)
    root=tmp_path/'legacy-closure';root.mkdir()
    selected=dict(plan,cases=plan['cases'][:1],designs=[])
    receipt=next(name for name in legacy.receipt_names(selected) if name.startswith('signals/'))
    path=root/receipt;path.parent.mkdir();path.write_bytes(b'fixture')
    record=dict(schema='RPE_HCHAIN_LEGACY_V1',owner=plan['owner'],plan_sha256=options['plan']['sha256'],
                records=[],holds=[],spent={},cases={},jobs=[],files={receipt:hashlib.sha256(b'fixture').hexdigest(),
                'prices/not-selected.events.json.xz':'d'*64})
    central.write(root/'LEGACY.json',record)
    options['legacy']=dict(directory=str(root),sha256=hashlib.sha256((root/'LEGACY.json').read_bytes()).hexdigest())
    assert legacy.checked(options,plan,verify_files=True,receipt_plan=selected)[1]==record
    with pytest.raises((ValueError,OSError)):
        legacy.checked(options,plan,verify_files=True)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='evidence changed'):
        legacy.checked(options,plan,verify_files=True,receipt_plan=selected)


def test_candidate_local_io_hold_does_not_abort_independent_hchain_ready(tmp_path,monkeypatch):
    cases=[{'id':'bad','method':'FULL_PF'},{'id':'good','method':'FULL_PF'}]
    plan={'cases':cases}
    def select(plan,cases):return dict(plan,cases=list(cases))
    def ready_batch(plan,*args,**kwargs):
        if any(case['id']=='bad' for case in plan['cases']):raise OSError('fixture missing input')
        return plan
    provider=SimpleNamespace(select_cases=select,ready_batch=ready_batch,price_models=lambda *_a,**_kw:({},{}))
    monkeypatch.setattr(hchain,'prepared_case',lambda *_a,**_kw:([{'unit_price_RZ':1}],{'signal_packet':'fixture'},{'fixture':True}))
    rows={case['id']:{'ready':False,'method':case['method'],'M':1} for case in cases}
    routes={'FULL_PF':[]};expected={};spec={'methods':['FULL_PF'],'max_new_q3_per_method':{'FULL_PF':'UNLIMITED'}}
    assert hchain.install_ready(provider,plan,tmp_path,rows,routes,expected,spec,final=False)=={'FULL_PF'}
    assert rows['good']['ready'] and not rows['bad']['ready']
    assert 'missing input' in rows['bad']['preparation_hold']
    # A finite method must not bypass its all-ready gate by narrowing to one case.
    rows={case['id']:{'ready':False,'method':case['method'],'M':1} for case in cases}
    spec['max_new_q3_per_method']['FULL_PF']=2
    assert not hchain.install_ready(provider,plan,tmp_path,rows,{'FULL_PF':[]},{},spec,final=False)
    assert not rows['good']['ready']


def test_selected_legacy_jobs_bind_small_requests_and_skip_other_raw(tmp_path):
    import rpe_hchain_legacy as legacy
    plan,options=fixed_scope(tmp_path)
    assert len(plan['cases']) >= 2
    selected=dict(plan,cases=plan['cases'][:1],designs=[])
    root=tmp_path/'legacy-jobs';root.mkdir()
    files={}
    jobs=[]
    for i,case in enumerate(plan['cases'][:2]):
        folder='jobs/method/job-'+str(i)
        jobs.append(folder)
        path=root/folder/'REQUEST.json'
        path.parent.mkdir(parents=True)
        central.write(path,dict(configuration_id=case['id']))
        files[folder+'/REQUEST.json']=hashlib.sha256(path.read_bytes()).hexdigest()
        path=root/folder/'OUTCOME.json'
        if i==0:
            path.write_bytes(b'fixture')
            files[folder+'/OUTCOME.json']=hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            files[folder+'/OUTCOME.json']='e'*64
    record=dict(schema='RPE_HCHAIN_LEGACY_V1',owner=plan['owner'],plan_sha256=options['plan']['sha256'],
                records=[],holds=[],spent={},cases={},jobs=jobs,files=files)
    central.write(root/'LEGACY.json',record)
    options['legacy']=dict(directory=str(root),sha256=hashlib.sha256((root/'LEGACY.json').read_bytes()).hexdigest())
    assert legacy.checked(options,plan,verify_files=True,receipt_plan=selected)[1]['jobs']==jobs[:1]
    with pytest.raises((ValueError,OSError)):
        legacy.checked(options,plan,verify_files=True)
    (root/jobs[1]/'REQUEST.json').write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):
        legacy.checked(options,plan,verify_files=True,receipt_plan=selected)


def test_legacy_run_cache_reuses_only_stable_content_bound_identity(tmp_path,monkeypatch):
    import rpe_hchain_legacy as legacy
    plan,options=fixed_scope(tmp_path)
    root=tmp_path/'legacy-cache';root.mkdir()
    path=root/'fixture.json';path.write_bytes(b'fixed')
    record=dict(schema='RPE_HCHAIN_LEGACY_V1',owner=plan['owner'],plan_sha256=options['plan']['sha256'],
                records=[],holds=[],spent={},cases={},jobs=[],
                files={'fixture.json':hashlib.sha256(path.read_bytes()).hexdigest()})
    central.write(root/'LEGACY.json',record)
    options['legacy']=dict(directory=str(root),sha256=hashlib.sha256((root/'LEGACY.json').read_bytes()).hexdigest())
    cache=tmp_path/'LEGACY_HASH_CACHE.json'
    import time
    real=time.time_ns
    monkeypatch.setattr(legacy.time,'time_ns',lambda:real()+2_000_000_000)
    assert legacy.checked(options,plan,verify_files=True,cache_path=cache)[1]==record
    assert cache.is_file()
    legacy._verified_digest.cache_clear()
    with monkeypatch.context() as m:
        m.setattr(legacy,'digest',lambda path:pytest.fail('stable cache should avoid reading'))
        assert legacy.checked(options,plan,verify_files=True,cache_path=cache)[1]==record
    path.write_bytes(b'other')
    with pytest.raises(ValueError,match='evidence changed'):
        legacy.checked(options,plan,verify_files=True,cache_path=cache)


def test_screened_ready_streaming_preserves_default_and_finite_wave_order():
    executor=SimpleNamespace(supports_nonblocking_prefetch=True)
    spec=dict(methods=['FULL_PF'],max_new_q3_per_method={'FULL_PF':'UNLIMITED'})
    assert hchain.stream_ready_in_wave(None,{},spec,executor)
    assert not hchain.stream_ready_in_wave(object(),{},spec,executor)
    settings={'screened_ready_streaming':True}
    assert hchain.stream_ready_in_wave(object(),settings,spec,executor)
    spec['max_new_q3_per_method']['FULL_PF']=2
    assert not hchain.stream_ready_in_wave(object(),settings,spec,executor)


def test_signal_batch_policy_reuses_fixed_h8_and_accepts_explicit_provider_policy():
    assert hchain.signal_batch_policy({'scope':{'system':'H8'}},SimpleNamespace())==(
        {'2nd':1,'4th':2,'4th(m5_best)':6},32)
    provider=SimpleNamespace(SIGNAL_BATCH_POLICY={'phase_keys':{'synthetic-formula':2},'phase_key_limit':8})
    assert hchain.signal_batch_policy({'scope':{'system':'Hsynthetic'}},provider)==({'synthetic-formula':2},8)
    provider.SIGNAL_BATCH_POLICY['phase_key_limit']=1
    with pytest.raises(ValueError,match='Invalid provider'):
        hchain.signal_batch_policy({'scope':{'system':'Hsynthetic'}},provider)


def test_exhausted_inputs_are_operational_terminal_not_scientific_completion(tmp_path):
    from trottertracks.rpe_q3_worker import read
    import run_rpe_hchain_round as hchain
    hchain.finish_block(tmp_path,{'FULL_PF':{'stop_reason':'WAITING_FOR_INPUTS'}},['FULL_PF'],
                        exhausted={'PREPARATION_FAILED':1})
    saved=read(tmp_path/'SEARCH_TERMINATION.json')
    assert saved['reason']=='HCHAIN_INPUTS_EXHAUSTED' and not saved['scientific_scope_complete']


def test_fixed_startup_declaration_checks_profile_and_manifest_without_provider(tmp_path):
    from trottertracks.rpe_hchain_scope import verify_startup_contract
    import pytest
    settings=dict(openblas_coretype='HASWELL',provider_source={'files':{'worker.py':'sha'}},assets={'files':{'input':'sha'}})
    plan={'scope':{'system':'H8'},'verified_saved_preparations':{'p':'sha'},
          'provider_contract':{'openblas_coretype':'HASWELL','startup_source_files':['worker.py'],'startup_asset_files':['input']}}
    profiles={'local':{'preparation_profiles':[{'provider':'HCHAIN_SCOPED','profile':{'openblas_coretype':'HASWELL'}}]}}
    verify_startup_contract(settings,plan,{'preparation_profiles':profiles})
    with pytest.raises(ValueError,match='OpenBLAS'):
        verify_startup_contract(dict(settings,openblas_coretype='NEHALEM'),plan,{'preparation_profiles':profiles})
    with pytest.raises(ValueError,match='file declaration'):
        verify_startup_contract(dict(settings,assets={'files':{}}),plan,{'preparation_profiles':profiles})


def test_late_ready_held_route_can_be_reannounced_without_new_input(monkeypatch):
    from threading import Condition
    from unittest.mock import Mock
    clock=[10.]
    monkeypatch.setattr(hchain.time,'monotonic',lambda:clock[0])
    executor=SimpleNamespace(state=SimpleNamespace(cv=Condition()))
    rows={'late':{'configuration_id':'late','method':'FULL_PF','ready':False}}
    source=hchain.HChainReadySource(executor,rows)
    view=Mock();trial=Mock();view.for_method.return_value=trial
    view.expected={};view.routes={'FULL_PF':[]}
    trial.route.return_value={'kind':'fixture'};trial.refresh_execution_hold.return_value={'hold':True}
    trial.expected_for.return_value={}
    rows['late']['ready']=True
    source.publish(rows,{}, {}, set())
    pending,take=source.subscribe('FULL_PF',{},[],view)
    assert pending() and take()==([],None)
    assert not source.retry_due()
    clock[0]+=1.1;trial.refresh_execution_hold.return_value=None
    assert source.retry_due()
    source.publish(rows,{}, {},set())
    assert [r['configuration_id'] for r in take()[0]]==['late']
    assert not source.retry_due()


def test_fixed_startup_checks_saved_export_at_controller_without_provider(tmp_path):
    from trottertracks.rpe_hchain_scope import verify_startup_contract
    import hashlib,json
    path=tmp_path/'export.json';path.write_text(json.dumps({'p:2':{'hash':'fixture'}}))
    settings={'assets':{'controller_path':str(tmp_path),'files':{'export.json':hashlib.sha256(path.read_bytes()).hexdigest()}}}
    plan={'scope':{'system':'FAKE'},'verified_saved_preparations':{'2':{'saved':'fixture'}},
          'provider_contract':{'saved_preparation_export':{'path':'export.json','key_prefix':'p:','hash_field':'hash'}}}
    verify_startup_contract(settings,plan,{})
    path.write_text('{}')
    with pytest.raises(ValueError,match='export hash differs'):verify_startup_contract(settings,plan,{})
