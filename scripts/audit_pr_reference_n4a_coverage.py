"""Read-only N4-A metadata audit; never imports a scientific module.

No provider/word generation, guard, circuit, optimizer, RPE or price runner.
Route labels below are metadata annotations, not a replacement provider.
"""
from __future__ import annotations
import ast
import hashlib
import io
import json
import lzma
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
BASE = "002ba1141fcdbfb9bd59b8173fb594da9a89388b"
OUT = ROOT/"artifacts/pr_reference_n4a_coverage/2026-11-08"
D3 = "artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/"
C0 = "artifacts/pr_reference_ct4/2026-11-07/"
CT0 = "artifacts/pr_reference_ct0/2026-11-07/"
REFERENCES = {}


def data(path):
    raw = subprocess.check_output(["git","show",BASE+":"+path],cwd=ROOT)
    REFERENCES[path] = dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    return raw


def read(path):
    raw = data(path)
    return json.loads(lzma.decompress(raw) if path.endswith(".xz") else raw)


def members(raw):
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:xz") as tar:
        return {m.name:tar.extractfile(m).read() for m in tar if m.isfile()}


def save(name, value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def positive_function():
    # Reuse the exact old literal normalization only. No scientific imports.
    raw = data("src/trottertracks/pr_reference_ct0.py")
    node = next(n for n in ast.parse(raw).body
                if isinstance(n,ast.FunctionDef) and n.name=="positive")
    ns=dict(math=math)
    exec(compile(ast.Module(body=[node],type_ignores=[]),"fixed_CT0_positive","exec"),ns)
    return ns["positive"]


def route(key, cache, limit, positive):
    pos = positive(key)
    if pos in cache:
        return dict(positive_key=pos,classification="C0_FIXED_WORD",
                    existing_record_key=cache[pos]["key"],
                    existing_sequence_sha256=cache[pos]["sequence_sha256"],
                    new_role_certificate=None,provider_executed=False)
    return dict(positive_key=pos,
        classification="DICTIONARY_DOMAIN_COVERED" if abs(float.fromhex(key))<=limit
                       else "OUTSIDE_DICTIONARY_DOMAIN",
        candidate_only=True,new_role_certificate=None,provider_executed=False)


def main():
    positive = positive_function()
    fixed = read(D3+"INPUTS.json")["fixed_parameters"]
    limit = float.fromhex(fixed["L_hex"])
    oldcheck = read(D3+"CHECKS.json")
    assert oldcheck["status"]=="PASS" and oldcheck["basic_words"]==43
    manifest = read(D3+"BASIC_WORDS_MANIFEST.json")
    assert len(manifest["rows"])==43
    assert [(r["j"],r["theta_hex"]) for r in manifest["rows"]]==[
        (r["j"],r["positive_binary64_key"]) for r in fixed["basic_keys"]]
    info = read(D3+"RECORD_ARCHIVE.json")
    raw = data(D3+"basic_records.tar.xz")
    assert hashlib.sha256(raw).hexdigest()==info["sha256"]
    arc = members(raw)
    for r in manifest["rows"]:
        b=arc[r["record"]]
        assert hashlib.sha256(b).hexdigest()==r["record_sha256"]
        v=json.loads(b)
        assert v["theta_hex"]==r["theta_hex"]
        assert hashlib.sha256(v["sequence"].encode()).hexdigest()==v["sequence_sha256"]
        assert v["sequence_sha256"] and r["guard_qualified"] and r["status"]=="PASS"
    c0_members=members(data(C0+"strict_records.tar.xz"))
    adopted=read(C0+"KEY_REUSE_AND_SYNTHESIS.json")
    keys={r["key"] for r in adopted["records"]}
    cache={}
    for b in c0_members.values():
        v=json.loads(b)
        assert v["key"] in keys and v["precision_exponent"]==-11
        assert hashlib.sha256(v["sequence"].encode()).hexdigest()==v["sequence_sha256"]
        cache[v["theta_hex"]]=v
    assert len(cache)==len(keys)==299
    old_headroom=read(D3+"C0_STRICT_HEADROOM.json")
    assert old_headroom["status"]=="PASS"
    assert all(r["passed"] and r["in_vacuum_domain"] for r in old_headroom["rows"])
    assert {(r["record_key"],r["sequence_sha256"]) for r in old_headroom["rows"]} == {
        (r["key"],r["sequence_sha256"]) for r in cache.values()}

    coverage=read(CT0+"KEY_COVERAGE.json.xz")
    phase=read(CT0+"PHASE_SUPPORT.json")
    grouped=defaultdict(lambda:dict(keys=set(),symbolic=set(),axes=set(),signed=False))
    for r in coverage["role_rows"]:
        if r["m"]!=7 or r["arm"] not in ("Bprime","reference"):
            continue
        g=grouped[(r["arm"],r["role"])]
        g["keys"].update(r["numeric_keys"])
        g["symbolic"].update(r["symbolic_keys"])
        g["axes"].add(r["axis"])
        g["signed"] |= r["signed_correction"]
    inv=coverage["inverse_D_supplement"]
    g=grouped[("reference","D_actual_SOURCE_inverse_supplement")]
    g["keys"].update(inv["mapping"])
    g["axes"].update(("X","Y"))

    event=next(r for r in phase["event_phase"] if r["r"]==1)
    grouped[("Bprime","RTE_event_sign_phase")]["keys"].update(event["keys"])
    vacuum=next(r for r in phase["vacuum_phase"] if r["m"]==7)
    assert (vacuum["q"],vacuum["r"])==(128,1)
    assert float(vacuum["absolute_actual_binary_angle_upper"]).hex()==fixed["L_hex"]

    key_routes={}
    rows=[]
    for (arm,role),g in sorted(grouped.items()):
        pos_keys=sorted({positive(k) for k in g["keys"]})
        for key in pos_keys:
            key_routes[key]=route(key,cache,limit,positive)
        histogram={label:sum(key_routes[k]["classification"]==label for k in pos_keys)
                   for label in ("C0_FIXED_WORD","DICTIONARY_DOMAIN_COVERED","OUTSIDE_DICTIONARY_DOMAIN")}
        statuses=["SOURCE_ROLE_COVERAGE_IDENTIFIED","CONDITIONAL_ON_COMPILER_CONTEXT",
                  "ADDITIONAL_ROLE_QUALIFICATION_REQUIRED","EMITTER_CONTEXT_UNQUALIFIED"]
        if histogram["C0_FIXED_WORD"]:statuses.append("C0_PRIORITY_PRESERVED")
        if histogram["DICTIONARY_DOMAIN_COVERED"]:statuses.append("DICTIONARY_DOMAIN_COVERED")
        if histogram["OUTSIDE_DICTIONARY_DOMAIN"]:statuses.append("OUTSIDE_DICTIONARY_DOMAIN")
        if not g["keys"] and not g["symbolic"]:
            statuses=["SOURCE_ROLE_COVERAGE_IDENTIFIED","NOT_APPLICABLE_NO_RZ_IN_THIS_ROLE"]
        if arm=="reference" and role=="scalar_or_vacuum_correction":
            statuses.append("DICTIONARY_DOMAIN_COVERED")
        rows.append(dict(arm=arm,role=role,axes=sorted(g["axes"]),
            signed_saved_numeric_keys=sorted(g["keys"]),positive_saved_keys=pos_keys,
            symbolic_keys=sorted(g["symbolic"]),literal_route_counts=histogram,
            signed_correction_bookkeeping=g["signed"],statuses=statuses,
            scope="Saved registered numeric superset / inverse-D supplement, conditional on the fixed compiler templates; not trajectory enumeration"))
    missing=sorted(k for k,v in key_routes.items() if v["classification"]=="DICTIONARY_DOMAIN_COVERED")
    outside=sorted(k for k,v in key_routes.items() if v["classification"]=="OUTSIDE_DICTIONARY_DOMAIN")
    frame_rows=[r for r in rows if r["role"]=="RTE_basis_frames"]
    assert len(frame_rows)==2 and frame_rows[0]["positive_saved_keys"]==frame_rows[1]["positive_saved_keys"]
    special={}
    for name,key in [("RTE_frame","0x1.08ba32d420f2cp+2"),
                     ("Bprime_event_b2","0x1.921fb54442d18p+2")]:
        assert key not in cache and key in key_routes
        special[name]=dict(route=key_routes[key],
            saved_roles=[dict(arm=r["arm"],role=r["role"]) for r in rows
                         if key in r["positive_saved_keys"]],
            compiler_condition="CT-5B condition C; actual numeric post-unroll template",
            emitter_condition="Unchanged lower_native on a qualified unitary segment; actual adjoint/inplace/no post-control",
            new_scientific_execution_required_for_role_certificate=True)
    special["Bprime_event_b2"]["canonical_reachability"]="NOT_ESTABLISHED_HERE: order2 plus exactly one negative product with nonzero finite-bit cells; saved superset is sufficient for candidate coverage"
    special["Bprime_event_b2"]["reachability_classification"]="SOURCE_SUPPORT_UNKNOWN_FOR_CANONICAL_B2_ONLY"
    assert event["superset_not_reachability_claim"]

    # Pure mock metadata checks, not real provider execution or certificates.
    mock={"0x1.0000000000000p+0":dict(key="MOCK",sequence_sha256="MOCK")}
    a=route("0x1.0000000000000p+0",mock,limit,positive)
    b=route("-0x1.0000000000000p+0",mock,limit,positive)
    assert a==b and a["classification"]=="C0_FIXED_WORD"
    assert route("0x1.0000000000001p+0",mock,limit,positive)["classification"]=="DICTIONARY_DOMAIN_COVERED"
    assert route("0x1.0000000000000p+8",mock,limit,positive)["classification"]=="OUTSIDE_DICTIONARY_DOMAIN"
    assert positive("0x0.0p+0")!=positive("0x1.921fb54442d18p+2")

    # Locate only the frozen representatives needed to define a later local
    # test. Do not rerun the completed four-context compiler audit.
    basis_id="df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2"
    targets=[]
    for path,member in [
        ("artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz","catalogue/rare1_Y1.json"),
        ("artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz","catalogue_boundary/reference_ccx1_rare1_Y1.json")]:
        batch=json.loads(members(data(path))[member])
        records={r["name"]:r for r in batch["records"]}
        selected=[]
        for name in ("basis:"+basis_id,"basis_inverse:"+basis_id,"p_symbolic"):
            record=records[name]
            literal_keys=sorted({positive(float(params[0]).hex()) for op,qs,params in record["lowering"]
                                 if op=="rz" and isinstance(params[0],(int,float))})
            selected.append(dict(name=name,n=record["n"],positive_numeric_keys=literal_keys,
                global_phase_metadata_present=("global_phase" in record or "global_phase_hex" in record),
                key_set_contained_in_saved_frame_inventory=set(literal_keys)<=set(frame_rows[0]["positive_saved_keys"]),
                record_sha256=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),
                symbolic_literal_template=record["lowering"] if name=="p_symbolic" else None))
        targets.append(dict(archive=path,member=member,request=batch["request"],records=selected))
    native=read(C0+"NATIVE_INPUT_CHECKS.json")
    save("INTEGRATION_TARGETS.json",dict(status="FROZEN_SAVED_TARGETS_ONLY",basis_id=basis_id,
        targets=targets,whole_native_metadata=[dict(name=r["name"],qubits=r["qubits"],
            operations=r["operations"],global_phase_hex=r["global_phase_hex"],
            native_qpy_sha256=r["native_qpy_sha256"]) for r in native["rows"]],
        missing_local_global_phase_rule="Do not fill omitted SOURCE/global phase with zero. Compare native gate products as local replacements, and retain the actual whole-native phase separately.",
        provider_calls=0,compile_calls=0,new_certificates=0))

    source_paths=[
        "src/trottertracks/pr_reference_dyadic_dictionary.py",
        "src/trottertracks/pr_reference_rotation_v3.py",
        "src/trottertracks/pr_reference_ct4.py",
        "src/trottertracks/pr_reference_interference_pilot.py",
        "src/trottertracks/pr_reference_transfer_qualification.py",
        "src/trottertracks/pr_reference_transfer_error.py",
        "src/trottertracks/pr_reference_bprime_accounting.py",
        "src/trottertracks/track_c_compiler_a.py",
        "src/trottertracks/track_c2_b_verify.py",
        "src/trottertracks/track_c_h4_synthesis.py",
        "src/trotterlib/df_rte_qiskit.py",
        "src/trotterlib/df_rte_tail.py",
        "src/trotterlib/rte.py",
        "src/trotterlib/df_partial_s2.py"]
    for path in source_paths:data(path)
    published_prices=read(D3+"T_PRICE_BOUND.json")
    assert published_prices["B_P"]==6072 and published_prices["V_threshold"]==173790
    old_strict=read(D3+"STRICT_ERROR_BUDGET.json")
    assert old_strict["status"]=="PASS"
    save("ROLE_COVERAGE.json",dict(status="SOURCE_ROLE_COVERAGE_IDENTIFIED",
        base_commit=BASE,roles=rows,literal_routes=key_routes,
        non_C0_domain_candidate_keys=missing,outside_dictionary_keys=outside,
        special_missing_keys=special,
        vacuum=dict(saved_bound=vacuum,domain_key=fixed["L_hex"],
                    classification="DICTIONARY_DOMAIN_COVERED",
                    local_provider="Previously qualified N3-B1b-1; not rerun",
                    exact_reachable_key_set="not enumerated; interval enclosure suffices"),
        full_support_policy_status="COMMON_POLICY_STATICALLY_COMPATIBLE_CONDITIONAL_ON_COMPILER_CONTEXT",
        actual_role_certificates_generated=0,full_support_CT_RPE="UNQUALIFIED"))
    save("CHECKS.json",dict(status="PASS",
        metadata_only=True,C0_records_preserved=299,basic_records_preserved=43,
        archive_and_word_hashes_match=True,fixed_vacuum_domain_matches=True,
        mock_sign_priority_adjacent_and_outside_cases=True,no_2pi_identification=True,
        paired_saved_frame_key_sets_equal=True,
        provider_instantiations=0,provider_calls=0,new_role_certificates=0,
        synthesis_calls=0,strict_guard_calls=0,circuit_or_compiler_calls=0,
        scientific_price_aggregations=0,RPE_reanalysis_calls=0,
        missing_candidate_keys=missing,outside_keys=outside))
    save("SOURCE_REFERENCES.json",dict(base_commit=BASE,files=REFERENCES,
        audit_source=str(Path(__file__).relative_to(ROOT)),
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime=dict(host=platform.node(),python=sys.version,affinity=len(os.sched_getaffinity(0))),
        no_scientific_modules_imported=True))
    print(json.dumps(dict(status="STATIC_AUDIT_COMPLETE",role_rows=len(rows),
        non_C0_domain_candidate_count=len(missing),outside=outside,provider_calls=0)))


if __name__=="__main__":
    main()
