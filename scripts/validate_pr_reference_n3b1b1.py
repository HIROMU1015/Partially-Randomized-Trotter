"""Strict certificate bindings and finite diagnostics for N3-B1b-1.

Does not call acquire/strict_guard, build a physical PF, sample trajectories,
compile native circuits or calculate RPE totals.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import runpy
import sys
import time

import run_pr_reference_n3b1b1 as R

ROOT, OUT = R.ROOT, R.OUT
NS = PROVIDER = None


def initialize(parameters, budget, basic, cache):
    global NS, PROVIDER
    NS = runpy.run_path(str(ROOT/"scripts/verify_pr_reference_n3b1b0_runtime.py"),
                       run_name="n3b1b1_validation_import")
    NS["mp"].mp.dps = NS["mp"].iv.dps = 100
    PROVIDER = R.provider_module().DictionaryProvider(parameters, budget, basic, cache)


def diagnostic(task):
    import numpy as np
    from qiskit.quantum_info import Operator
    if task["kind"] == "basic":
        r = task["record"]
        result = NS["word_test"]("j="+str(task["j"]), r["sequence"],
                                 float.fromhex(r["theta_hex"]))
        assert all(x["passed"] for x in result["fixtures"])
        return dict(kind="basic", j=task["j"], record_key=r["key"], result=result)
    if task["kind"] == "composite":
        theta = float.fromhex(task["theta_hex"])
        positive = PROVIDER.provide(theta)
        expected = NS["matrix"](positive.positive_sequence)
        adj = "".join(NS["adj_map"][g] for g in reversed(positive.positive_sequence))
        assert NS["Exact"].from_gates(
            (positive.positive_sequence+adj).replace("t","TTTTTTT")) == NS["Exact"].identity()
        fixtures = []
        for sign in (1,-1):
            value = sign*theta
            circuit, cert = PROVIDER.emit(value, NS["lower_native"])
            assert cert.signed_sequence() == (adj if value<0 else positive.positive_sequence)
            target_word = expected.conj().T if value<0 else expected
            actual = Operator(circuit).data
            observed = float(np.linalg.norm(actual-target_word,2))
            names = [x.operation.name for x in circuit.data]
            wanted = [dict(H="h",S="s",T="t",t="tdg",X="x")[g]
                      for g in reversed(positive.positive_sequence) if g!="W"]
            inverse_names = dict(h="h",s="sdg",t="tdg",tdg="t",x="x")
            if value<0:
                wanted = [inverse_names[g] for g in reversed(wanted)]
            assert names == wanted
            assert sum(n in ("t","tdg") for n in names)==cert.T
            assert sum(n=="tdg" for n in names)==cert.Tdagger
            ideal = np.diag(np.exp(np.array([-.5j,.5j])*value))
            # Diagnostic only. The actual strict certificate is Fraction
            # rounding + saved outward basic guards + the SOURCE bound.
            rz_observed = float(np.linalg.norm(actual-ideal,"fro"))
            assert observed < 1e-10 and cert.strict_upper < F(1,10**11)
            fixtures.append(dict(certificate=cert.summary(), names_match=True,
                emitted_T_matches=True, emitted_Tdagger_matches=True,
                strict_word_operator_diagnostic=observed,
                target_RZ_Frobenius_diagnostic=rz_observed,
                emitted_phase_hex=float(circuit.global_phase).hex(), passed=True))
        return dict(kind="composite", theta_hex=theta.hex(), fixtures=fixtures,
                    exact_complete_adjoint_identity=True)
    # Upper-bound fixture: all 43 ACTUAL words once, although its ideal grid
    # angle is outside the scientific domain. Never call provider there.
    words = [PROVIDER.basic[j]["sequence"] for j in range(-36,7)]
    whole = "".join(words)
    outside = float(F(2**43-1,2**36))
    try:
        PROVIDER.provide(outside)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-domain all-ones angle was accepted")
    result = NS["word_test"]("artificial_43_actual_chunks", whole)
    assert all(x["passed"] for x in result["fixtures"])
    return dict(kind="artificial_43_chunks", basic_indices=list(range(-36,7)),
        physical_target=False, public_provider_rejected_outside_angle=True,
        sequence_sha256=hashlib.sha256(whole.encode()).hexdigest(), result=result)


def main():
    started = time.monotonic()
    manifest = json.loads((OUT/"BASIC_WORDS_MANIFEST.json").read_text())
    parameters = R.read_base(R.B0+"FIXED_PARAMETERS.json")
    budget = R.read_base(R.B0+"ERROR_BUDGET.json")
    if manifest["status"] != "ACQUIRED":
        R.save(OUT/"COMPOSITE_CORRECTNESS.json", dict(status="UNQUALIFIED",
            reason="Incomplete acquisition", missing=[r["theta_hex"] for r in manifest["rows"]
                                                     if r["status"]!="PASS"]))
        R.save(OUT/"T_PRICE_BOUND.json", dict(status="UNQUALIFIED",
            B_dict=None, B_cache=None, B_P=None, partial_prices=[
                dict(theta_hex=r["theta_hex"],T=json.loads((OUT/r["record"]).read_text())["T"])
                for r in manifest["rows"] if r["status"]=="PASS"]))
        raise RuntimeError("Acquisition STOP remains in force; no provider validation")
    funcs = R.old_functions()
    validate = funcs["validate_record"]
    basic = {}
    for row in manifest["rows"]:
        raw = (OUT/row["record"]).read_bytes()
        assert R.sha(raw)==row["record_sha256"]
        r = json.loads(raw)
        validate(r)
        assert r["key"]==row["key"] and r["theta_hex"]==row["theta_hex"]
        assert r["precision_exponent"]==-13 and F(r["strict_error_upper"])<=F(1,10**13)
        assert r["guard_method"]=="exact cyclotomic word matrix + outward interval strict Frobenius"
        basic[row["j"]]=r
        row.update(T=r["T"], Tdagger=r["Tdagger"], global_W=r["global_W"],
                   strict_error_upper=r["strict_error_upper"], guard_qualified=True)
    assert len(basic)==43
    assert {p.name for p in (OUT/"synthesis").glob("*.json")} == {
        r["key"]+".json" for r in manifest["rows"]}
    cache = R.retained_cache(validate)
    mod = R.provider_module()
    provider = mod.DictionaryProvider(parameters,budget,basic,cache)

    # Reference Fraction.__round__, not a second floating multiplication.
    ties = [0.0,-0.0,2.0**-37,3*2.0**-37,15*2.0**-37]
    angles = ties + [-x for x in ties] + [provider.limit,-provider.limit]
    angles += [math.nextafter(x,direction) for x in ties
               for direction in (-math.inf,math.inf)]
    rounding = []
    for theta in angles:
        n = mod.round_index(theta)
        want = round(F.from_float(theta)*2**36)
        assert n==want and mod.round_index(-theta)==-n
        if abs(theta)<=provider.limit:
            cert = provider.provide(theta)
            assert cert.strict_upper<provider.epsilon
        rounding.append(dict(theta_hex=theta.hex(), index=n, reference=want, passed=True))
    rejections = []
    for theta in [float("nan"),float("inf"),float("-inf"),
                  math.nextafter(provider.limit,math.inf),
                  -math.nextafter(provider.limit,math.inf)]:
        try:
            provider.provide(theta)
        except ValueError:
            rejections.append(dict(input_repr=repr(theta),rejected=True))
        else:
            raise AssertionError("Invalid target accepted")
    priorities = []
    for key,r in sorted(cache.items()):
        for sign in (1,-1):
            theta=sign*float.fromhex(key)
            cert=provider.provide(theta)
            assert cert.route=="fixed_C0_priority"
            assert cert.positive_sequence==r["sequence"]
            assert cert.T==r["T"] and cert.strict_upper<provider.epsilon
            assert cert.Tdagger==(r["T"]-r["Tdagger"] if theta<0 else r["Tdagger"])
        priorities.append(dict(key=key, sequence_sha256=r["sequence_sha256"],passed=True))
    immutable = False
    try:
        provider.cache[next(iter(provider.cache))]["T"]=0
    except TypeError:
        immutable=True
    assert immutable
    # Prospective internal-key overlap must not overwrite old C0 selection.
    assert not set(cache).intersection(r["theta_hex"] for r in basic.values())

    valid = [0.0,2.0**-37,3*2.0**-37,2.0**-20,0.123456789,1.0,
             float(F(2**42-1,2**36)),math.nextafter(provider.limit,0.0),provider.limit]
    tasks = [dict(kind="basic",j=j,record=r) for j,r in sorted(basic.items())]
    tasks += [dict(kind="composite",theta_hex=x.hex()) for x in valid]
    tasks.append(dict(kind="artificial_43_chunks"))
    R.save(OUT/"VALIDATION_INPUTS.json",dict(
        valid_composite_angles=[x.hex() for x in valid],
        rounding_inputs=[x.hex() for x in angles],
        rejections=[repr(x) for x in [float("nan"),float("inf"),float("-inf"),
                       math.nextafter(provider.limit,math.inf),
                       -math.nextafter(provider.limit,math.inf)]],
        artificial_43_chunk_fixture="all indices ascending, +/- exact word; not a physical target",
        tolerance=1e-10, tolerance_is_strict_certificate=False,
        no_new_guard=True))
    slots = max(1,math.floor(len(os.sched_getaffinity(0))-os.getloadavg()[0]))
    workers = min(len(tasks),len(os.sched_getaffinity(0)),slots)
    threads = max(1,len(os.sched_getaffinity(0))//workers)
    # Per-process library threading derived from the independent task count;
    # no shared/system settings or permanent resource limits are changed.
    os.environ["OPENBLAS_NUM_THREADS"]=str(threads)
    os.environ["OMP_NUM_THREADS"]=str(threads)
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    initialize(parameters,budget,basic,cache)
    runtime = NS["runtime_checks"]()
    initial_env=json.loads((OUT/"ENVIRONMENT.json").read_text())
    assert runtime["acceleration_binary"]["sha256"]==initial_env["acceleration_binary"]["sha256"]
    with ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,initargs=(parameters,budget,basic,cache)) as executor:
        results=list(executor.map(diagnostic,tasks))
    actual_sum=sum((F(r["strict_error_upper"]) for r in basic.values()),F(0))
    allocation=43*F(1,10**13)
    Q=F(budget["quantization_Frobenius"]["upper"])
    uniform=Q+allocation+mod.EMITTER_UPPER
    actual=Q+actual_sum+mod.EMITTER_UPPER
    assert actual<=uniform<F(budget["epsilon"])
    assert mod.EMITTER_UPPER<F(budget["emitter_remaining"]["lower"])
    R.save(OUT/"STRICT_ERROR_BUDGET.json",dict(status="PASS",
        Q_upper=str(Q), basic_guard_allocated_sum=str(allocation),
        actual_basic_guard_sum=str(actual_sum), emitter_upper=str(mod.EMITTER_UPPER),
        allocated_uniform_upper=str(uniform), actual_guard_uniform_upper=str(actual),
        epsilon=budget["epsilon"], saved_emitter_residual_lower=budget["emitter_remaining"]["lower"],
        all_43_saved_outward_guards_qualified=True,
        actual_guard_calls="Executed by unchanged acquire during acquisition; not repeated here",
        C0_guard_plus_emitter_qualified=299, C0_emitter_upper=str(mod.CACHE_EMITTER_UPPER),
        whole_dictionary_domain="finite binary64 |theta| <= saved L, exact ties-even and ascending fixed word order",
        emitter_scope="unchanged lower_native, inplace joins, no QPY/copy/post-control",
        operator_tolerance_is_certificate=False, full_CT_RPE_qualified=False))
    prices=[dict(j=j,key=r["theta_hex"],record_key=r["key"],T=r["T"],Tdagger=r["Tdagger"])
            for j,r in sorted(basic.items())]
    B_dict=sum(r["T"] for r in basic.values())
    B_cache=max(r["T"] for r in cache.values())
    B_P=max(B_dict,B_cache)
    threshold=R.read_base("artifacts/pr_reference_n1_numeric/2026-11-08/RESULTS.json")["exact"]
    assert (threshold["K"],threshold["d0"],threshold["d2"],threshold["V_threshold"])==(160222,106,114,173790)
    R.save(OUT/"T_PRICE_BOUND.json",dict(status="FINITE_POLICY_BOUND_EVALUATED",
        basic_prices=prices,B_dict=B_dict,B_cache=B_cache,B_P=B_P,
        V_threshold=173790,strict_sufficient_inequality=B_P<173790,
        gap_to_threshold=173790-B_P,Tdagger_is_subset=True,
        C0_prices_and_words_unchanged=True,
        N1_fixed_values=threshold,
        qualification="Vacuum cap and N1 sufficient inequality only; compiler condition C/common policy and other-role qualifications remain necessary",
        full_support_CT_RPE_or_global_T_advantage_qualified=False,
        vacuum_law_or_expected_distribution_generated=False,
        whole_18round_total_T_evaluated=False))
    R.save(OUT/"COMPOSITE_CORRECTNESS.json",dict(status="PASS",
        rounding=rounding, invalid_targets=rejections, C0_priority=priorities,
        diagnostics=results, finite_tests_are_not_the_whole_domain_proof=True,
        full_domain_proof="Exact rounding + <=43 fixed bits + every basic outward guard + unitary telescoping + qualified emitter bound",
        dictionary_online_backend_calls=0, QPY_calls=0, production_PF_builds_or_compiles=0))
    R.save(OUT/"CHECKS.json",dict(status="PASS", basic_words=43,
        basic_record_metadata_and_hashes=True, guard_qualification=True,
        exact_W_and_whole_actual_adjoint=True, written_order="ascending j",
        rounding_ties_boundaries_endpoints=True, artificial_43_chunks=True,
        all_C0_priority_and_headroom=True, cache_records_immutable=immutable,
        runtime_C_phi_reverified=True, runtime_binary_sha256=runtime["acceleration_binary"]["sha256"],
        strict_budget=True, T_prices_from_actual_sequences=True,
        scientific_fixed_parameters_unchanged=True, new_guard_calls_during_validation=0,
        key_acquisition_retries=0,
        diagnostics_workers=workers, BLAS_threads_per_process=threads,
        diagnostic_tasks=len(tasks),wall_seconds=time.monotonic()-started))
    R.save(OUT/"BASIC_WORDS_MANIFEST.json",manifest)
    R.save(OUT/"SOURCE.json",dict(base_commit=R.BASE,
        acquisition_scope_source_hashes=json.loads((OUT/"SCOPE.json").read_text())["source_hashes"],
        acquisition_source_snapshot_sha256=R.sha((OUT/"ACQUISITION_SOURCE.py").read_bytes()),
        provider_source="src/trottertracks/pr_reference_dyadic_dictionary.py",
        provider_source_sha256=R.sha((ROOT/"src/trottertracks/pr_reference_dyadic_dictionary.py").read_bytes()),
        validation_source=str(Path(__file__).relative_to(ROOT)),
        validation_source_sha256=R.sha(Path(__file__).read_bytes()),
        reused_lower_native_and_word_matrix=NS["PROVENANCE"],
        runtime_qualification_base="fbe7ac8b53acd5c795e94939c5c5e400ced5679a"))
    print(json.dumps(dict(status="PASS",basic_words=43,B_dict=B_dict,B_cache=B_cache,
        B_P=B_P,threshold=173790,strict_inequality=B_P<173790,
        allocated_uniform_error=str(uniform),wall_seconds=time.monotonic()-started)),flush=True)


if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        import traceback
        R.save(OUT/"VALIDATION_FAILURE.json",dict(status="UNQUALIFIED",error=repr(exc),
                                                traceback=traceback.format_exc()))
        raise
