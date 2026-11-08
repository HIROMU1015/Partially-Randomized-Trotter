"""Approved N5-B2 central slice: four logical Q+G bounds, eight indices.

Stdlib and exact Fractions only. No production provider, word construction,
guard, synthesis, quantum module, compiler, price or RPE evaluator is loaded.
"""
import ast
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction as F
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = "50013be49f03cb609fc6e43ed4dfcf43251f7823"
OUT = ROOT / "artifacts/pr_reference_n5b2_numeric/2026-11-08"
SCOPE = "artifacts/pr_reference_n5b2_static/2026-11-08/NEXT_NUMERIC_INPUTS.json"
D3 = "artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/"
POLICY = "src/trottertracks/pr_reference_dyadic_dictionary.py"
REFERENCES = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    REFERENCES[path] = dict(bytes=len(raw), sha256=sha(raw))
    return raw


def read(path):
    return json.loads(blob(path))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False) + "\n")


def value(x):
    with localcontext() as ctx:
        ctx.prec = 24
        ctx.rounding = ROUND_CEILING
        upper_display = str(Decimal(x.numerator) / Decimal(x.denominator))
    return dict(exact_fraction=str(x), upper_decimal_24sig=upper_display)


def reused_components():
    raw = blob(POLICY)
    assert (ROOT/POLICY).read_bytes() == raw
    parsed = ast.parse(raw)
    constants = [n for n in parsed.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in ("B", "EXPONENTS") for t in n.targets)]
    rounding = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "round_index")
    parent = next(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == "DictionaryProvider")
    provide = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == "provide")
    wanted = ("indices", "selected", "delta", "q_upper", "g_upper")
    assignments = [n for n in provide.body if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in wanted]
    assert tuple(n.targets[0].id for n in assignments) == wanted
    wrapper = ast.FunctionDef(name="saved_components", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=x) for x in ("self", "theta", "n")],
        kwonlyargs=[], kw_defaults=[], defaults=[]), decorator_list=[],
        body=assignments + [ast.Return(value=ast.Tuple(
            elts=[ast.Name(id=x, ctx=ast.Load()) for x in ("indices", "delta", "q_upper", "g_upper")],
            ctx=ast.Load()))])
    namespace = dict(F=F, math=math)
    exec(compile(ast.fix_missing_locations(ast.Module(body=constants+[rounding,wrapper], type_ignores=[])),
                 BASE + ":" + POLICY + ":selected_pure_parts", "exec"), namespace)
    assert namespace["B"] == 36 and namespace["EXPONENTS"] == tuple(range(-36,7))
    return namespace, dict(path=POLICY, round_index_source=ast.get_source_segment(raw.decode(), rounding),
        unchanged_component_assignments=[ast.get_source_segment(raw.decode(), n) for n in assignments],
        skipped="provide domain/cache/construction/T/W aggregation/WordCertificate/emit; entire module not imported")


def saved_guards(scope):
    manifest = read(scope["basic_record_manifest"])
    archive = blob(scope["basic_archive"])
    basic, rows = {}, []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:xz") as t:
        for m in manifest["rows"]:
            raw = t.extractfile(m["record"]).read()
            assert sha(raw) == m["record_sha256"]
            r = json.loads(raw)
            assert r["key"] == m["key"] == sha(r["spec"].encode())
            assert r["theta_hex"] == m["theta_hex"] and r["spec"] == m["spec"]
            assert r["precision_exponent"] == m["precision_exponent"] == -13
            assert r["status"] == m["status"] == "PASS" and m["guard_qualified"]
            assert m["up_to_phase"] is False
            assert sha(r["sequence"].encode()) == r["sequence_sha256"]
            assert r["guard_method"] == "exact cyclotomic word matrix + outward interval strict Frobenius"
            assert r["strict_error_upper"] == m["strict_error_upper"]
            g = F(r["strict_error_upper"])
            assert 0 <= g <= F(1, 10**13)
            assert m["j"] not in basic
            basic[m["j"]] = dict(guard=g)
            rows.append(dict(j=m["j"], record=m["record"], key=r["key"],
                record_sha256=sha(raw), sequence_sha256=r["sequence_sha256"], saved_guard=str(g)))
    assert set(basic) == set(range(-36,7)) and len(rows) == 43
    return basic, rows


def main():
    started = time.monotonic()
    assert not OUT.exists(), "Fresh output required; preserve completed scope and STOP"
    scope = read(SCOPE)
    assert scope["fixed_b"] == 36 and scope["fixed_basic_exponents"] == [-36,6]
    assert scope["fixed_m17_condition"]["m"] == 17
    expected = {
        "Bprime": ["0x1.7e4088e024c5fp-12", "0x1.fdab627b8d37ap-14"],
        "reference": ["0x1.7e4088e024c5fp-11", "0x1.fdab627b8d37ap-13"]}
    assert len(scope["central_slice"]) == 2
    assert {r["arm"]:r["positive_literals"] for r in scope["central_slice"]} == expected
    old = read(D3 + "INPUTS.json")
    params, budget = old["fixed_parameters"], old["error_budget"]
    assert (params["b"], params["j_min"], params["j_max"], params["M_b"]) == (36,-36,6,43)
    assert budget["sqrt2"]["exact_square_enclosure_passed"]
    basic, manifest_rows = saved_guards(scope)
    ns, reuse = reused_components()
    context = SimpleNamespace(basic=basic, sqrt2_upper=F(budget["sqrt2"]["upper"]))
    limit = float.fromhex(params["L_hex"])
    assert (sys.float_info.radix, sys.float_info.mant_dig) == (2,53)
    save("INPUTS.json", dict(scope_commit=BASE, fixed_scope=scope,
        dictionary_parameters=params, saved_sqrt2=budget["sqrt2"], basic_records=manifest_rows,
        calculation="exact rational Q, G and Q+G; no emitted certificate or RPE reaccounting"))
    save("ENVIRONMENT.json", dict(python=sys.version, executable=sys.executable,
        hostname=platform.node(), affinity=len(os.sched_getaffinity(0)),
        memory_available=next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')),
        numerical_arithmetic="Exact Python Fraction/integer; binary64 inputs read from saved hex; directed 24-significant-digit Decimal displays only",
        resource_strategy="Four tiny dependent component checks; no resource caps, quantum imports or worker pool",
        shared_settings_changed=False))
    rows, comparisons = [], []
    index_calls = 0
    for group in scope["central_slice"]:
        arm = group["arm"]
        assert group["axes"] == ["X", "Y"]
        for key in group["positive_literals"]:
            theta = float.fromhex(key)
            assert theta.hex() == key and theta > 0 and theta <= limit
            negative_key = "-" + key
            assert negative_key in group["signed_literals"]
            negative = float.fromhex(negative_key)
            assert negative == -theta and F.from_float(negative) == -F.from_float(theta)
            n = ns["round_index"](theta)
            nn = ns["round_index"](negative)
            index_calls += 2
            assert nn == -n and 0 < n <= int(params["N_b"]) and n.bit_length() <= 43
            indices, delta, Q, G = ns["saved_components"](context, theta, n)
            assert sum(1 << (j+36) for j in indices) == n
            assert delta <= F(1, 2**37) and (delta != F(1,2**37) or n % 2 == 0)
            logical = Q + G
            assert logical - Q == G and Q >= 0 and G >= 0
            rows.append(dict(arm=arm, positive_literal=key, negative_literal=negative_key,
                exact_binary64_target=str(F.from_float(theta)), rounded_index=n, negative_rounded_index=nn,
                selected_basic_indices=list(indices), delta=value(delta), Q=value(Q), G=value(G),
                logical_component_upper=value(logical),
                larger_component="quantization" if Q > G else "selected_saved_guards" if G > Q else "equal",
                checks=dict(domain=True, negative_symmetry=True, index_reconstruction=True,
                    nearest_ties_to_even=True, component_identity=True)))
            for axis in group["axes"]:
                saved_cap = scope["fixed_m17_condition"]["arms"][arm]["axis"][axis]["uniform_gate_error_strict_upper"]
                cap = F.from_float(saved_cap)
                relation = "ABOVE_CAP" if logical > cap else "STRICTLY_BELOW_CAP" if logical < cap else "EQUAL_CAP"
                comparisons.append(dict(arm=arm, axis=axis, positive_literal=key,
                    saved_cap_binary64_hex=saved_cap.hex(), saved_cap_exact=str(cap),
                    saved_cap_display=saved_cap, relation=relation,
                    sufficient_allocation_not_necessary_precision=True))
    assert len(rows) == 4 and len(comparisons) == index_calls == 8
    save("RESULTS.json", dict(status="FIXED_CENTRAL_LOGICAL_COMPONENTS_EVALUATED",
        rows=rows, cap_comparisons=comparisons,
        cap_relation_counts={s:sum(c["relation"] == s for c in comparisons)
            for s in ("ABOVE_CAP", "EQUAL_CAP", "STRICTLY_BELOW_CAP")},
        bound_not_actual_error_or_lower_bound=True, emitted_strict_certificate_generated=False,
        m17_RPE="UNQUALIFIED", no_precision_or_policy_adopted=True))
    save("SOURCE_INPUTS.json", dict(scope_commit=BASE, fixed_files=REFERENCES,
        reuse=reuse, helper="scripts/evaluate_pr_reference_n5b2_central.py",
        helper_sha256=sha(Path(__file__).read_bytes()),
        production_provider_or_scientific_module_imported=False))
    save("CHECKS.json", dict(status="PASS_FIXED_NUMERIC_SCOPE", positive_component_cases=4,
        reused_round_index_calls=index_calls, axis_cap_comparisons=8, saved_basic_records=43,
        all_domain_symmetry_reconstruction_nearest_and_component_checks=True,
        new_guard_synthesis_word_generation_provider_emitter_consumer_compile_Operator_sampling_RPE_T_calls=0,
        precision_domain_policy_changed=False, elapsed_seconds=time.monotonic()-started))
    save("STOP.json", dict(status="MANDATORY_STOP", reason="Approved fixed central slice and checks complete",
        m17_RPE="UNQUALIFIED", new_precision_or_policy=False, next_owner="GPT/user",
        automatic_followup_execution=False))
    print(json.dumps(dict(status="FIXED_CENTRAL_LOGICAL_COMPONENTS_EVALUATED",
        positive_cases=len(rows), index_calls=index_calls, comparisons=len(comparisons),
        relations=[c["relation"] for c in comparisons], elapsed_seconds=time.monotonic()-started)))


if __name__ == "__main__":
    main()
