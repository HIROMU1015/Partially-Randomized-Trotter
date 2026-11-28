"""Synthetic H-chain input transfer and collection through the common Q3 adapter."""
import base64
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import rpe_hchain_q3 as hchain
from trottertracks import rpe_method_optimizer as optimizer
from trottertracks import rpe_q3_coordinator as transport
from trottertracks import rpe_q3_worker as worker
from trottertracks.rpe_optimizer_concurrent import DistributedExecutor


def write(path, data):
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def prepared(tmp_path, cid):
    inputs = tmp_path / cid
    inputs.mkdir()
    config, signals, prices = (inputs / name for name in ("configuration", "signals", "prices"))
    binding = dict(system="H8", rank=28, hamiltonian="synthetic")
    config_sha = write(config, dict(id=cid, spec=dict(binding=binding, M=0, formula="S2", L_D=4),
                                   rounds=[dict(m=0, q_PF=1, R_outer=1, actual_delta_pf=0.1)]))
    signals_sha = write(signals, dict(M=0, h=0.1, target=-1.0, status="SIGNAL_PACKETS_READY",
                                    scope_owner="fixture", run_id="fixture", compiler="test",
                                    methods={"PR-DF": [dict(m=0, candidate=dict(q=1, R_outer=1,
                                        formula="S2", prefix=4), signal=dict(raw=dict(real=1.0, imag=0.0)))]}))
    entry = dict(configuration=str(config), configuration_sha256=config_sha,
                 configuration_id=cid, signal_packet=str(signals), signal_packet_sha256=signals_sha)
    axes = [dict(m=0, axis=axis, kind="DIRECT_COMPILED_RZ", price_evidence={"fixture": True},
                 status="ADMITTED", unit_price_RZ=10) for axis in ("X", "Y")]
    write(prices, dict(compiler="test", signal_packet_sha256=signals_sha,
                       methods={"PR-DF": dict(entry, axes=axes)}))
    route = hchain.prepared_route(entry, prices, owner="fixture", target=-1.0, binding=binding,
                                  interpreter=sys.executable)
    shutil.rmtree(inputs)
    return route


def test_prepared_q3_uses_shared_assignment_and_completion_on_two_workers(tmp_path, monkeypatch):
    cids = ("early", "later")
    routes = {"PR_DF": [prepared(tmp_path, cid) for cid in cids]}
    expected = {cid: dict(M=0) for cid in cids}
    problem = dict(accuracy=dict(epsilon_E_decimal=hchain.EPSILON), target=dict(energy_ha=-1.0), rpe=dict(M=0))
    source = worker.head(ROOT)
    server = SimpleNamespace(server_address=("127.0.0.1", 1), shutdown=lambda: None, server_close=lambda: None)
    monkeypatch.setattr(transport, "serve", lambda *args: (server, SimpleNamespace(join=lambda: None)))
    executor = DistributedExecutor(ROOT, tmp_path / "run", routes, expected, problem,
                                   optimizer_source=source, max_concurrent=2, budgets={"PR_DF": 2},
                                   distributed=dict(port=0, affinity_wait_seconds=0))
    try:
        method = executor.for_method("PR_DF")
        proposals = [dict(configuration_id=cid, vector=(1, 1), cost=Fraction(20),
                          origin=dict(kind="RPE_SCALED_SHOTS_V1", M=0)) for cid in cids]
        eligibility, fingerprints = {}, {}
        for proposal, host in zip(proposals, ("pbs-a", "pbs-b"), strict=True):
            job = method.job_dir(proposal)
            request = method.request(proposal, job, method.route(proposal["configuration_id"]))
            fingerprint = dict(host_id=host, fixture=True)
            record = dict(method="PR_DF", request_sha256=transport.digest(request), input_hashes=[],
                          source_manifest=executor.source_manifest,
                          source_manifest_sha256=transport.digest(executor.source_manifest), eligible_host_ids=[host],
                          hosts={host: dict(status="PASS", current_host_fingerprint_sha256=transport.digest(fingerprint))})
            record["decision_sha256"] = transport.digest(record)
            eligibility[job.name] = record
            fingerprints[host] = {job.name: fingerprint}
        executor.distributed["host_eligibility"] = dict(schema="RPE_Q3_HOST_ELIGIBILITY_V1", jobs=eligibility)
        for proposal in proposals:
            assert method._enqueue(proposal, demand=True)
        for host, proposal in zip(("pbs-a", "pbs-b"), proposals, strict=True):
            assignment, = executor.rpc("/v1/poll", dict(host_id=host, slots=1, source_sha=source,
                                                         host_fingerprints=fingerprints[host]))["assignments"]
            assert assignment["request"]["configuration_id"] == proposal["configuration_id"]
            profile = dict(host_id=host, root=str(ROOT), interpreter=sys.executable,
                           route_views={hchain.RISK_COMMIT: str(ROOT)})
            # The synthetic worker skips the real risk checkout. Its only task
            # is reconstructing the result envelope from synthetic evidence.
            request = assignment["request"]
            name, *_, provenance, identity = hchain.request_inputs(request)
            out = tmp_path / host
            out.mkdir()
            request_path = out / "request.json"
            write(request_path, request)
            proof_path = out / "HCHAIN_RESULT.json"
            proof = dict(request_id=request["request_id"], source=provenance, identity=identity,
                         result=dict(method=name, status=optimizer.PASS, MSE_lower=0, MSE_upper=0))
            write(proof_path, proof)
            command = [profile["interpreter"], "-B", str(ROOT / "scripts/run_rpe_optimizer_q3_job.py"),
                       "--request", str(request_path), "--output", str(out), "--mode", "collect"]
            subprocess.run(command, check=True)
            write(out / "PROCESS_RETURN.json", {"returncode": 0, "signal": None})
            completion = dict(host_id=host, job_id=assignment["job_id"], attempt_id=assignment["attempt_id"],
                              request_sha256=assignment["request_sha256"], files={
                                  filename: base64.b64encode((out / filename).read_bytes()).decode()
                                  for filename in ("OUTCOME.json", "HCHAIN_RESULT.json", "PROCESS_RETURN.json")})
            assert executor.rpc("/v1/complete", completion) == {"ok": True, "duplicate": False}
            assert executor.rpc("/v1/complete", completion) == {"ok": True, "duplicate": True}
            assert method.answer(proposal)["status"] == optimizer.PASS
    finally:
        with executor.state.cv:
            for task in executor.state.tasks.values():
                if task.status == "RUNNING":
                    task.status = "DONE"
                    task.event.set()
            executor.state.active = 0
            executor.state.cv.notify_all()
        executor.finish()


def test_packed_input_binds_cost_and_signal(tmp_path):
    route = prepared(tmp_path, "candidate")
    request = dict(route=route, configuration_id="candidate", method="PR_DF", cost="21",
                   N_X=[1], N_Y=[1], expected=dict(configuration_id="candidate"))
    with pytest.raises(ValueError, match="do not reproduce"):
        hchain.request_inputs(request)
    request["cost"] = "20"
    route["configurations"]["candidate"]["signal_packet"]["json"] += " "
    with pytest.raises(ValueError, match="signal_packet bytes differ"):
        hchain.request_inputs(request)


def test_hchain_proof_compact_json_preserves_exact_integer_values(tmp_path):
    from rpe_hchain_q3 import write
    import json
    data={'integers':[10**1000,10**700,0],'nested':{'exact':True}}
    pretty=tmp_path/'pretty.json';compact=tmp_path/'compact.json'
    write(pretty,data);write(compact,data,compact=True)
    assert json.loads(compact.read_text())==json.loads(pretty.read_text())==data
    assert compact.stat().st_size<pretty.stat().st_size
