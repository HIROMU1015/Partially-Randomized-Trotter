#!/usr/bin/env python3
"""Fixed P1 only. Existing target/law/reference bytes are read, never evaluated."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/pauli_pure_rte_mechanism_validation/2026-11-01/bias"
HYBRID = "00958ae1945c780718b2deef07d5dab1096a5cd8"
RTE = "10cb3e5cca7834ebf75301bd91bd093ca51d4a6b"
HBASE = "artifacts/h6_hybrid_fairness_extension/2026-10-27/"
RBASE = "artifacts/pauli_rte_event_extension/2026-10-26/"
IPATH = "artifacts/pauli_h2_h6_comparison/2026-10-25/H6/inputs/"
MODEL_HASH = "6c66a2eeb4d99cf5bbd9137bbcb067275536be3201d5cc047225266dc5f58681"
REFERENCE_HASH = "9956466a5cf3286af9960b792b455bb8bacc2532a34ac7a28b470475f92fcb7a"
HAM_HASH = "37b2747eaafacef8b4b470e5f4d6e195ac0602dc8e59953e3b76152d4576c847"
CANDIDATES = dict(H16=("PR_p0078_q8_r16_K2", 78, 16),
                  H32=("PR_p0078_q8_r32_K2", 78, 32),
                  H64=("PR_p0078_q8_r64_K2", 78, 64),
                  R64=("H6_p000_q8_r64_K2", 0, 64))
FILES = ("scripts/run_h6_pauli_p1_bias.py",
         "src/trotterlib/pauli_mechanism_bias.py",
         "src/trotterlib/pauli_signal_adapter.py", "src/trotterlib/rte.py",
         "tests/test_pauli_mechanism_bias.py")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def gitbytes(ref, path):
    return subprocess.check_output(["git", "show", ref+":"+path], cwd=ROOT)


def save(name, payload):
    path = OUT/name
    temporary = path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False)+"\n")
    temporary.replace(path)


def archive_members(commit, base, namespace, wanted):
    records = json.loads(gitbytes(commit, base+"data_archives.json"))
    record = next(x for x in records if x["namespace"] == namespace)
    chunks = []
    for part in record["parts"]:
        raw = gitbytes(commit, base+part["path"])
        if sha(raw) != part["sha256"]:
            raise ValueError("ARCHIVE_PART_IDENTITY")
        chunks.append(raw)
    raw = b"".join(chunks)
    if sha(raw) != record["archive_sha256"]:
        raise ValueError("ARCHIVE_IDENTITY")
    mapping = {record["members"][k]["archive_member"]:k for k in wanted}
    found, provenance = {}, {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r|xz") as archive:
        for member in archive:
            if member.name not in mapping:
                continue
            key = mapping[member.name]
            content = archive.extractfile(member).read()
            if sha(content) != record["members"][key]["sha256"]:
                raise ValueError("SAVED_MEMBER_IDENTITY:"+key)
            found[key] = json.loads(content)
            provenance[key] = dict(commit=commit, index=base+"data_archives.json",
                                   member=key, sha256=sha(content))
            if len(found) == len(wanted):
                break
    if set(found) != set(wanted):
        raise ValueError("MISSING_SAVED_MEMBER")
    return found, provenance


def inputs():
    mraw, rraw = (gitbytes(RTE, IPATH+n) for n in ("model.json", "reference.json"))
    if sha(mraw) != MODEL_HASH or sha(rraw) != REFERENCE_HASH:
        raise ValueError("FIXED_INPUT_IDENTITY")
    model, reference = json.loads(mraw), json.loads(rraw)
    # Check historical SOURCE maps without loading any old scientific runner.
    hs = json.loads(gitbytes(HYBRID, HBASE+"SOURCE.json"))
    rs = json.loads(gitbytes(RTE, RBASE+"SOURCE_e755ecf581600192fd2745d7066e410cebd933d3.json"))
    for source in (hs, rs):
        for path, digest in source["code_sha256"].items():
            if sha(gitbytes(source["commit"], path)) != digest:
                raise ValueError("HISTORICAL_SOURCE_IDENTITY:"+path)
    hp = [f"H6/batch01/cases/{cid}/{name}.json"
          for cid, prefix, r in list(CANDIDATES.values())[:3]
          for name in ("spec", "target")]
    h, hprov = archive_members(HYBRID, HBASE, "H6", hp)
    rp = [f"cases/{CANDIDATES['R64'][0]}/{name}.json"
          for name in ("law", "target")]
    pure, pprov = archive_members(RTE, RBASE, "cases", rp)
    laws, targets = {}, {}
    for tag, (cid, prefix, r) in CANDIDATES.items():
        if prefix:
            laws[tag] = h[f"H6/batch01/cases/{cid}/spec.json"]["law"]
            targets[tag] = h[f"H6/batch01/cases/{cid}/target.json"]
        else:
            laws[tag] = pure[rp[0]]
            targets[tag] = pure[rp[1]]
        law, target = laws[tag], targets[tag]
        expected = dict(L_D=prefix, q=8, r=r, K=2, T=.8)
        if any(law["config"][k] != v for k, v in expected.items()):
            raise ValueError("SAVED_LAW_CONFIG:"+tag)
        if law["Hamiltonian_hash"] != HAM_HASH:
            raise ValueError("SAVED_LAW_HAMILTONIAN")
        if (target["raw_norm_exact"] != reference["raw_norm_exact"] or
                target["state_renormalized"] or target["B_applied_count"] != 1):
            raise ValueError("SAVED_NORM_OR_SCALE")
    if (model["Hamiltonian_hash"] != HAM_HASH or reference["Hamiltonian_hash"] != HAM_HASH
            or model["n_qubits"] != 12 or model["nonidentity_terms"] != 918
            or model["all_Z_prefix"] != 78 or model["state_renormalized"]):
        raise ValueError("FIXED_H6_MODEL")
    retained = model["hamiltonian"]["terms"][:78]
    if not all(set(label) <= set("IZ") for label, coefficient in retained):
        raise ValueError("RETAINED_COMMUTATIVITY_STRUCTURE")
    for tag in ("H16", "H32", "H64"):
        if [(x["label"], x["coefficient"]) for x in laws[tag]["retained"]] != [tuple(x) for x in retained]:
            raise ValueError("RETAINED_DESCRIPTOR_JOIN")
    provenance = dict(model=dict(commit=RTE, path=IPATH+"model.json", sha256=sha(mraw)),
                      reference=dict(commit=RTE, path=IPATH+"reference.json", sha256=sha(rraw)),
                      saved_members={**hprov, **pprov},
                      historical_SOURCE=dict(Hybrid=hs["commit"], RTE=rs["commit"]),
                      raw_norm_exact=reference["raw_norm_exact"],
                      saved_state_hash=model["state_hash"],
                      Hamiltonian_hash=HAM_HASH, retained_I_Z_only=True,
                      saved_targets_re_evaluated=0, saved_laws_re_evaluated=0)
    return model, reference, targets, provenance


def modules():
    # Follow existing numerical-worker practice: skip unrelated package __init__.
    package = types.ModuleType("p1_study")
    package.__path__ = [str(ROOT/"src/trotterlib")]
    sys.modules[package.__name__] = package
    from p1_study import pauli_mechanism_bias as kernel
    from p1_study.pauli_signal_adapter import PauliHamiltonian, array_hash
    return kernel, PauliHamiltonian, array_hash


def execute():
    # A fresh directory prevents accidental scientific retries; no approval ledger.
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counts = dict(major_signals_started=0, consistency_started=0, completed=0,
                  new_reference=0, saved_target_evaluations=0, law_evaluations=0,
                  RTE_sampling=0, circuits_built=0, compile=0, N_G_SE_ranking=0)
    signals = {}
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        codes = {p:sha((ROOT/p).read_bytes()) for p in FILES}
        if any(sha(gitbytes(commit,p)) != h for p,h in codes.items()):
            raise ValueError("SCIENTIFIC_SOURCE_NOT_COMMITTED_OR_CHANGED")
        model, reference, targets, provenance = inputs()
        runtime = {n:importlib.metadata.version(n) for n in ("numpy","scipy","qiskit")}
        if runtime != dict(numpy="1.26.4", scipy="1.14.1", qiskit="1.3.0"):
            raise ValueError("PLANNED_RUNTIME")
        source = dict(commit=commit, code_sha256=codes, inputs=provenance,
                      runtime=runtime, python=sys.version, candidates=CANDIDATES,
                      scientific_scope="P1_ONLY_MAJOR6_PLUS_PREPLANNED_CHECK2",
                      execution_UTC=datetime.now(timezone.utc).isoformat(),
                      affinity_CPUs=len(os.sched_getaffinity(0)),
                      meminfo_start={line.split(":")[0]:line.split(":")[1].strip()
                         for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith(("MemTotal:","MemAvailable:"))},
                      uncertainty_policy="1e-10 per new signal axis, empirical diagnostic resolution, not a rigorous bound")
        save("source_and_inputs.json", source)
        kernel, PauliHamiltonian, array_hash = modules()
        import numpy as np
        h = model["hamiltonian"]
        ham = PauliHamiltonian(model["n_qubits"], h["constant"],
                               tuple(map(tuple,h["terms"])), json.dumps(h["metadata"]))
        state = np.array([complex(x,y) for x,y in model["state"]],dtype=complex)
        if ham.hamiltonian_hash != HAM_HASH or array_hash(state) != model["state_hash"]:
            raise ValueError("IN_MEMORY_INPUT_IDENTITY")
        ops = {p:kernel.split_operators(ham,p) for p in (78,0)}
        norm = reference["raw_norm_exact"]

        def job(name, mode, prefix, r, subdivisions=1):
            kind = "major_signals_started" if subdivisions==1 else "consistency_started"
            counts[kind] += 1
            save("execution_status.json", dict(counts=counts, current=name, state="RUNNING"))
            row = kernel.propagate(ops[prefix],state,norm,mode=mode,T=.8,q=8,r=r,K=2,
                                   tail_subdivisions=subdivisions)
            signals[name] = row
            counts["completed"] += 1
            save("new_signals.json", signals)
            return row

        b = job("B_Hybrid", "B", 78, 16)
        c = job("C_Hybrid", "C", 78, 16)
        checks = dict(B_C_absolute=abs(kernel.value(b)-kernel.value(c)),
                      L0_B_C_A="EXACT_STRUCTURAL_IDENTITY_HD_ZERO; saved A reused, no full-H propagation",
                      no_sector_projection=True, retained_terms_commute=True)
        if checks["B_C_absolute"] > kernel.CONSISTENCY_ATOL:
            raise ValueError("B_C_INCONSISTENT")
        for mode, original in (("B",b),("C",c)):
            check = job(mode+"_tail_semigroup",mode,78,16,2)
            error = abs(kernel.value(check)-kernel.value(original))
            checks[mode+"_tail_semigroup_absolute"] = error
            if error > kernel.CONSISTENCY_ATOL:
                raise ValueError("TAIL_ACTION_INCONSISTENT")
        # Four independent finite means; no selection, top-up, or new law.
        # Counters/checkpoints stay in the parent; workers return only signals.
        def polynomial(item):
            tag, (_,prefix,r) = item
            return tag,kernel.propagate(ops[prefix],state,norm,mode="Dpoly",
                                        T=.8,q=8,r=r,K=2)
        counts["major_signals_started"] += 4
        save("execution_status.json",dict(counts=counts,state="RUNNING_DPOLY"))
        with ThreadPoolExecutor(max_workers=min(4,len(os.sched_getaffinity(0)))) as pool:
            for tag, row in pool.map(polynomial,CANDIDATES.items()):
                signals["Dpoly_"+tag] = row
                counts["completed"] += 1
                save("new_signals.json",signals)
        a = kernel.enclosure(reference["physical_reference"])
        rows = {}
        for tag in CANDIDATES:
            # L0 B=C=A is an algebraic alias, not another numerical reference.
            bb, cc = (a,a) if tag=="R64" else (b,c)
            poly = signals["Dpoly_"+tag]
            grid = kernel.enclosure(targets[tag]["physical_corrected_mean"])
            terms = dict(outer=kernel.difference(bb,a),
                         hold=kernel.difference(cc,bb),
                         trunc=kernel.difference(poly,cc),
                         grid=kernel.difference(grid,poly))
            import math
            total = kernel.value(grid)-kernel.value(a)
            summed = complex(math.fsum(x["real"] for x in terms.values()),
                             math.fsum(x["imag"] for x in terms.values()))
            closure = total-summed
            if abs(closure)>kernel.CONSISTENCY_ATOL:
                raise ValueError("COMPLEX_CLOSURE")
            rows[tag] = dict(candidate=CANDIDATES[tag][0],differences=terms,
                             closure_residual=dict(real=closure.real,imag=closure.imag,absolute=abs(closure)),
                             closure_kind="ALGEBRAIC_TELESCOPING_NOT_INDEPENDENT_ACCURACY_PROOF",
                             structural_zero_terms=["outer","hold"] if tag=="R64" else ["hold"])
        save("signals_and_signed_differences.json",dict(rows=rows,saved_A_Dgrid="REFERENCES_ONLY_IN_SOURCE_AND_INPUTS; not new signal evaluations"))
        save("numerical_checks.json",dict(checks=checks,consistency_atol=kernel.CONSISTENCY_ATOL,
             new_axis_resolution=kernel.DIAGNOSTIC_AXIS_RESOLUTION,
             uncertainty_not_certified=True,phase_alignment=False,
             actual_compiled_rho="NOT_EVALUATED",synthetic_tests="tests/test_pauli_mechanism_bias.py"))
        unresolved = [(tag,name) for tag,row in rows.items()
                      for name,d in row["differences"].items()
                      if d["status"]=="UNRESOLVED" and name not in row["structural_zero_terms"]]
        status = "STOP_UNRESOLVED_COMPONENTS" if unresolved else "P1_NUMERICAL_DIAGNOSTIC_COMPLETE"
        stop = dict(status=status,unresolved_components=unresolved,counts=counts,
                    wall_seconds=time.monotonic()-started,automatic_retry=False,
                    further_science_started=False,mandatory_STOP=True)
        save("execution_status.json",stop)
        save("STOP.json",stop)
        return stop
    except BaseException as error:
        stop = dict(status="STOP_FAILURE",reason=str(error),traceback=traceback.format_exc(),
                    counts=counts,wall_seconds=time.monotonic()-started,
                    automatic_retry=False,mandatory_STOP=True)
        save("STOP.json",stop)
        save("execution_status.json",stop)
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("inspect","execute"))
    args=parser.parse_args()
    if args.command=="inspect":
        model,reference,targets,provenance=inputs()
        print(json.dumps(provenance,ensure_ascii=False,indent=2))
    else:
        print(json.dumps(execute(),ensure_ascii=False,indent=2))
