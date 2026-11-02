"""Phase-T proof interfaces and exact symbolic lemmas; no scientific execution.

This module does not issue certificates or call the legacy shot accountant.
Saved claims, checked bytes and formal lemmas are distinct objects. Real-H4
numerical/compiled-program proof adapters remain unavailable in this SOURCE.
Only the Python standard library is used. Matrix, sampling and compiler APIs
are deliberately absent. Word products use matrix order (rightmost acts first).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

SCHEMA = "H4_certificate_gate_technical_bundle_v1"
C_SOURCE = "e80d7510880f8db993b777c627bb8e95b05a070d"
ORIGINAL_SOURCE = "0c329b30a81accb35c212be8defa66fee0dd0f12"
CASES = ("pauli_jw_PF", "pauli_jw_PR", "df_native_PF", "df_native_PR")
EPSILONS = ("0.05", "0.01", "0.005", "0.001", "0.0005")
AXES = ("cosine", "sine")
CLAIMS = frozenset({"VERIFIED", "EVIDENCE_EXISTS_UNCHECKED", "MISSING_EVIDENCE",
                    "FAILED_CONDITION", "ASSUMPTION_DEPENDENT", "UNKNOWN"})
GATE_SCOPES = {
    "numerical_enclosure": "ENCODED_MODEL",
    "reference_encoded_full_H": "ENCODED_MODEL",
    "bias_encoded_saved": "ENCODED_MODEL",
    "normalization_nominal_log_B": "NOMINAL_NORMALIZATION",
    "measurement_binary_range": "BINARY_MEASUREMENT",
    "encoded_event_norm": "ENCODED_EVENT",
    "sampling_law": "IMPLEMENTED_LAW",
    "wrapper_transfer": "LITERAL_WRAPPER",
    "compiler_transfer": "COMPILED_PROGRAM",
    "conditional_statistics": "CONDITIONAL_STATISTICS",
    "provenance": "PROVENANCE",
    "saved_B_to_shot_upper_binding": "MEASUREMENT_SCALE_BINDING",
    "certificate_acceptance_implementation": "ACCEPTANCE_IMPLEMENTATION",
    "final_combined_error_budget": "COMBINED_ERROR",
}
EXISTING_ASSUMPTIONS = frozenset({"IID_UNIFORM_WORDS", "FRESH_INDEPENDENT_SHOTS"})
RULE_GATES = {
    "BINARY_READOUT": "measurement_binary_range",
    "WORD_UNITARITY": "encoded_event_norm",
    "CONTROLLED_WORD": "wrapper_transfer",
    "BASIS_CONTROL": "wrapper_transfer",
    "HADAMARD_WRAPPER": "wrapper_transfer",
    "WORD_EQUALITY": "compiler_transfer",
}


class GateValidationError(ValueError):
    """Invalid structure, binding or unsupported proof rule; never infer PASS."""


def require(condition, code):
    if not condition:
        raise GateValidationError(code)


def keys(value, expected, code):
    require(type(value) is dict and set(value) == set(expected), code)


def identifier(value):
    require(type(value) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_:.-]*", value),
            "SYMBOL_IDENTIFIER")
    return value


def hex_digest(value, length=64):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % length, value),
            "DIGEST_FORMAT")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_json_bytes(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    def nonfinite(_):
        raise GateValidationError("NONFINITE_JSON_CONSTANT")

    def finite_float(text):
        value = float(text)
        require(math.isfinite(value), "NONFINITE_JSON_NUMBER")
        return value

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite,
                      parse_float=finite_float)


def checked_file(root, relative, expected):
    """Read only the named regular file; reject traversal and any symlink."""
    hex_digest(expected)
    require(type(relative) is str, "RELATIVE_PATH_REQUIRED")
    p = PurePosixPath(relative)
    require(not p.is_absolute() and p.parts and str(p) == relative
            and all(x not in {".", ".."} for x in p.parts), "UNSAFE_EVIDENCE_PATH")
    cursor = Path(root).resolve()
    for part in p.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "SYMLINK_EVIDENCE")
    require(cursor.is_file(), "MISSING_EVIDENCE_FILE")
    raw = cursor.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "EVIDENCE_HASH_MISMATCH")
    return raw


def identity_word(n):
    return dict(n=n, phase_quarters=0, phase_symbols={}, factors=[])


def normal_form(word):
    """Exact free-unitary word reduction; no rounded matrix is an atom.

    Exact H/Pauli definitions are built in. UNITARY_SYMBOL and real RZ/phase
    parameters carry explicit hypotheses. This is not a proof that saved
    binary64 matrices implement any of these symbols.
    """
    keys(word, {"n", "phase_quarters", "phase_symbols", "factors"}, "WORD_SCHEMA")
    n = word["n"]
    require(type(n) is int and n > 0, "WORD_DIMENSION")
    require(type(word["phase_quarters"]) is int, "EXACT_PHASE_REQUIRED")
    require(type(word["phase_symbols"]) is dict and type(word["factors"]) is list,
            "WORD_COMPONENTS")
    require(all(type(k) is str for k in word["phase_symbols"]), "SYMBOL_IDENTIFIER")
    conditions, phases, stack = set(), [], []
    for name, coefficient in sorted(word["phase_symbols"].items()):
        identifier(name)
        require(type(coefficient) is int, "EXACT_PHASE_COEFFICIENT")
        if coefficient:
            phases.append((name, coefficient))
            conditions.add("REAL_PHASE:" + name)
    for atom in word["factors"]:
        require(type(atom) is dict and "kind" in atom, "ATOM_SCHEMA")
        kind = atom["kind"]
        require(type(atom.get("inverse")) is bool, "EXACT_ADJOINT_FLAG")
        inverse = atom["inverse"]
        if kind == "PAULI":
            keys(atom, {"kind", "label", "inverse"}, "PAULI_SCHEMA")
            label = atom["label"]
            require(type(label) is str and len(label) == n and set(label) <= set("IXYZ"),
                    "PAULI_LABEL")
            if set(label) == {"I"}:
                continue
            token = (kind, label, False)
        elif kind == "H":
            keys(atom, {"kind", "wire", "inverse"}, "H_SCHEMA")
            wire = atom["wire"]
            require(type(wire) is int and 0 <= wire < n, "WIRE_RANGE")
            token = (kind, wire, False)
        elif kind == "RZ":
            keys(atom, {"kind", "wire", "angle", "inverse"}, "RZ_SCHEMA")
            wire = atom["wire"]
            require(type(wire) is int and 0 <= wire < n, "WIRE_RANGE")
            angle = identifier(atom["angle"])
            conditions.add("REAL_ANGLE:" + angle)
            token = (kind, wire, angle, inverse)
        elif kind == "UNITARY_SYMBOL":
            keys(atom, {"kind", "name", "inverse"}, "UNITARY_SYMBOL_SCHEMA")
            name = identifier(atom["name"])
            conditions.add("EXACT_UNITARY_SYMBOL:" + name)
            token = (kind, name, inverse)
        else:
            raise GateValidationError("UNSUPPORTED_ATOM_NO_MATRIX_UNITARITY_INFERENCE")
        self_adjoint = kind in {"PAULI", "H"}
        if stack and token[:-1] == stack[-1][:-1] and (self_adjoint or token[-1] != stack[-1][-1]):
            stack.pop()
        else:
            stack.append(token)
    return (n, word["phase_quarters"] % 4, tuple(phases), tuple(stack)), conditions


def adjoint_word(word):
    normal_form(word)
    factors = [dict(a, inverse=not a["inverse"]) for a in reversed(word["factors"])]
    return dict(n=word["n"], phase_quarters=-word["phase_quarters"],
                phase_symbols={k: -v for k, v in word["phase_symbols"].items()}, factors=factors)


def product(*words):
    require(bool(words), "EMPTY_PRODUCT")
    normal_form(words[0])
    result = identity_word(words[0]["n"])
    for word in words:
        normal_form(word)
        require(word["n"] == result["n"], "PRODUCT_DIMENSION_MISMATCH")
        result["phase_quarters"] += word["phase_quarters"]
        for name, value in word["phase_symbols"].items():
            result["phase_symbols"][name] = result["phase_symbols"].get(name, 0) + value
        result["factors"].extend(word["factors"])
    return result


def equal_words(left, right):
    a, ac = normal_form(left)
    b, bc = normal_form(right)
    require(a == b, "SYMBOLIC_EQUALITY_NOT_DERIVED")
    return ac | bc


def check_symbolic_rule(proof):
    """Check a closed set of formal algebra lemmas; never bind them to H4."""
    keys(proof, {"rule", "data"}, "SYMBOLIC_PROOF_SCHEMA")
    rule, data = proof["rule"], proof["data"]
    require(type(rule) is str and rule in RULE_GATES, "UNSUPPORTED_PROOF_RULE")
    conditions = set()
    if rule == "BINARY_READOUT":
        keys(data, {"scale_symbol", "bit_values", "estimator"}, "READOUT_SCHEMA")
        scale = identifier(data["scale_symbol"])
        require(type(data["bit_values"]) is list and len(data["bit_values"]) == 2
                and all(type(b) is int for b in data["bit_values"])
                and data["bit_values"] == [0, 1], "BINARY_BITS_REQUIRED")
        require(data["estimator"] == "B*(1-2*b)", "READOUT_ESTIMATOR")
        conditions.add("FINITE_NONNEGATIVE_SCALE:" + scale)
        conclusion = "OUTCOME_SUPPORT_MINUS_PLUS_SCALE"
    elif rule == "WORD_UNITARITY":
        keys(data, {"word"}, "UNITARITY_SCHEMA")
        _, conditions = normal_form(data["word"])
        equal_words(product(adjoint_word(data["word"]), data["word"]),
                    identity_word(data["word"]["n"]))
        conclusion = "FORMAL_WORD_UNITARY_NORM_ONE"
    elif rule == "WORD_EQUALITY":
        keys(data, {"before", "after"}, "EQUALITY_SCHEMA")
        conditions = equal_words(data["before"], data["after"])
        conclusion = "EXACT_FORMAL_WORD_EQUALITY"
    elif rule == "CONTROLLED_WORD":
        keys(data, {"evolution", "zero_branch", "one_branch"}, "CONTROL_SCHEMA")
        normal_form(data["evolution"])
        conditions |= equal_words(data["zero_branch"], identity_word(data["evolution"]["n"]))
        conditions |= equal_words(data["one_branch"], data["evolution"])
        conclusion = "CONTROLLED_DIAG_IDENTITY_EVOLUTION_WITH_PHASE"
    elif rule == "BASIS_CONTROL":
        keys(data, {"basis", "diagonal", "zero_branch", "one_branch"}, "BASIS_SCHEMA")
        basis, diagonal = data["basis"], data["diagonal"]
        conditions |= equal_words(data["zero_branch"],
                                  product(adjoint_word(basis), identity_word(basis["n"]), basis))
        conditions |= equal_words(data["one_branch"], product(adjoint_word(basis), diagonal, basis))
        conclusion = "BASIS_CONTROL_BRANCH_CORRESPONDENCE"
    else:
        keys(data, {"axis", "operations"}, "WRAPPER_SCHEMA")
        require(data["axis"] in AXES, "WRAPPER_AXIS")
        expected = (["H", "CONTROLLED_U", "H"] if data["axis"] == "cosine"
                    else ["H", "CONTROLLED_U", "SDG", "H"])
        require(data["operations"] == expected, "WRAPPER_PHASE_OR_ORDER")
        conditions |= {"NORMALIZED_PREPARED_STATE", "EXACT_CONTROLLED_U", "IDEAL_BORN_BINARY_READOUT"}
        conclusion = ("P0_MINUS_P1_REAL_EXPECTATION" if data["axis"] == "cosine"
                      else "P0_MINUS_P1_IMAG_EXPECTATION")
    return dict(status="FORMAL_LEMMA_CHECKED", conclusion=conclusion,
                gate=RULE_GATES[rule], assumptions=sorted(conditions),
                binding_to_saved_H4="NOT_PROVED", formal_certificate_accepted=False)


def validate_target(target):
    keys(target, {"case", "epsilon_sig", "axis", "C_SOURCE", "original_SOURCE",
                  "C_result_sha256", "C_seal_sha256", "audit_matrix_sha256",
                  "compiler_target", "compiled_program_sha256"}, "TARGET_SCHEMA")
    require(target["case"] in CASES and target["epsilon_sig"] in EPSILONS
            and target["axis"] in AXES, "TARGET_AXIS_NOT_REGISTERED")
    for name in ("C_SOURCE", "original_SOURCE"):
        hex_digest(target[name], 40)
    require(target["C_SOURCE"] == C_SOURCE and target["original_SOURCE"] == ORIGINAL_SOURCE,
            "FIXED_SCIENTIFIC_SOURCE_MISMATCH")
    for name in ("C_result_sha256", "C_seal_sha256", "audit_matrix_sha256"):
        hex_digest(target[name])
    require(target["compiler_target"] in {"UNRESOLVED", "HISTORICAL_PROGRAM", "NEW_PROGRAM"},
            "COMPILER_TARGET")
    if target["compiled_program_sha256"] is not None:
        hex_digest(target["compiled_program_sha256"])
        require(target["compiler_target"] != "UNRESOLVED", "PROGRAM_WITH_UNRESOLVED_TARGET")


def validate_bundle(bundle, root):
    """Validate integrity/scope and check optional lemmas, with explicit blockers.

    No genuine numerical/compiled-program proof verifier is registered in T.
    Consequently a lemma or a digest can never be relabelled as such a proof.
    The result is a technical preflight, not an acceptance or shot-accounting API.
    """
    keys(bundle, {"schema", "domain", "target", "assumptions", "evidence", "gates",
                  "execution_authorized", "formal_certificate_accepted", "N_G_authorized"},
         "BUNDLE_SCHEMA")
    require(bundle["schema"] == SCHEMA, "BUNDLE_VERSION")
    require(bundle["domain"] in {"SAVED_H4_CANDIDATE", "ARTIFICIAL"}, "BUNDLE_DOMAIN")
    require(all(bundle[k] is False for k in ("execution_authorized", "formal_certificate_accepted",
                                            "N_G_authorized")), "TECHNICAL_SCOPE_ONLY")
    validate_target(bundle["target"])
    assumptions = bundle["assumptions"]
    require(type(assumptions) is list and all(type(a) is str for a in assumptions)
            and len(set(assumptions)) == len(assumptions)
            and set(assumptions) <= EXISTING_ASSUMPTIONS, "EXISTING_ASSUMPTIONS_ONLY")
    required = {"FRESH_INDEPENDENT_SHOTS"}
    if bundle["target"]["case"].endswith("_PR"):
        required.add("IID_UNIFORM_WORDS")
    require(required <= set(assumptions), "CONDITIONAL_ASSUMPTIONS_NOT_RECORDED")
    require(type(bundle["evidence"]) is list, "EVIDENCE_LIST")
    refs = {}
    for ref in bundle["evidence"]:
        keys(ref, {"id", "path", "sha256", "source_commit"}, "EVIDENCE_REFERENCE_SCHEMA")
        name = identifier(ref["id"])
        require(name not in refs, "DUPLICATE_EVIDENCE_ID")
        hex_digest(ref["source_commit"], 40)
        checked_file(root, ref["path"], ref["sha256"])
        refs[name] = ref
    require(bool(refs), "EVIDENCE_REQUIRED")
    for ref_id, target_key in (("audit_matrix", "audit_matrix_sha256"),
                               ("C_result", "C_result_sha256"), ("C_seal", "C_seal_sha256")):
        require(ref_id in refs and refs[ref_id]["sha256"] == bundle["target"][target_key],
                "TARGET_EVIDENCE_BINDING_MISMATCH")
    require(refs["C_result"]["source_commit"] == bundle["target"]["C_SOURCE"]
            and refs["C_seal"]["source_commit"] == bundle["target"]["C_SOURCE"],
            "SOURCE_EVIDENCE_BINDING_MISMATCH")
    program_sha = bundle["target"]["compiled_program_sha256"]
    if program_sha is not None:
        require("compiled_program" in refs and refs["compiled_program"]["sha256"] == program_sha,
                "POSTCOMPILED_PROGRAM_BYTES_REQUIRED")
    keys(bundle["gates"], GATE_SCOPES, "COMPLETE_GATE_SET_REQUIRED")
    results = {}
    for gate, scope in GATE_SCOPES.items():
        entry = bundle["gates"][gate]
        keys(entry, {"claim", "scope", "evidence_ids", "lemma"}, "GATE_ENTRY_SCHEMA")
        require(type(entry["claim"]) is str and entry["claim"] in CLAIMS, "CLAIM_VOCABULARY")
        require(entry["scope"] == scope, "GATE_SCOPE_MISMATCH")
        ids = entry["evidence_ids"]
        require(type(ids) is list and all(type(x) is str for x in ids)
                and len(set(ids)) == len(ids) and set(ids) <= set(refs), "UNBOUND_EVIDENCE_ID")
        if entry["claim"] in {"VERIFIED", "ASSUMPTION_DEPENDENT", "EVIDENCE_EXISTS_UNCHECKED"}:
            require(bool(ids), "POSITIVE_CLAIM_NEEDS_EVIDENCE")
        lemma = entry["lemma"]
        checked = None
        if lemma is not None:
            keys(lemma, {"target_digest", "proof"}, "LEMMA_ENVELOPE_SCHEMA")
            require(lemma["target_digest"] == digest(bundle["target"]), "PROOF_TARGET_MISMATCH")
            checked = check_symbolic_rule(lemma["proof"])
            require(checked["gate"] == gate, "LEMMA_GATE_SCOPE_MISMATCH")
        blocker = ("FAILED_SAVED_CLAIM_REQUIRES_RESOLUTION" if entry["claim"] == "FAILED_CONDITION"
                   else "H4_BINDING_AND_GATE_PROOF_ADAPTER_MISSING" if checked
                   else "REQUIRED_GATE_PROOF_NOT_ATTACHED")
        results[gate] = dict(saved_claim=entry["claim"], scope=scope,
                             evidence_hashes_checked=len(ids), checked_lemma=checked,
                             proof_gate_satisfied=False, blocker=blocker)
    return dict(status="TECHNICAL_PREFLIGHT_COMPLETE_SCIENTIFIC_ACCEPTANCE_BLOCKED",
                schema=SCHEMA, target=bundle["target"], target_digest=digest(bundle["target"]),
                assumptions=assumptions, integrity_checked=True, gates=results,
                all_gate_proofs_verified=all(x["proof_gate_satisfied"] for x in results.values()),
                missing_proof_adapters=list(GATE_SCOPES),
                unconditional_PCG64_independence_required=False,
                formal_certificate_accepted=False, scientific_execution_authorized=False,
                N_G_authorized=False, N=None, G=None, mandatory_STOP=True)


def bundles_from_saved_audit(root, inventory_path):
    """Join only pre-existing SHA/schema/claims; do not evaluate saved numbers."""
    inventory_file = Path(root) / inventory_path
    inventory = read_json_bytes(inventory_file.read_bytes())
    require(inventory.get("kind") == "STATIC_AUDIT_DELIVERY_BYTE_INVENTORY", "AUDIT_INVENTORY_KIND")
    audit = "artifacts/non_df_h4_certificate_gate_audit/2026-10-23/"
    matrix_path = audit + "certificate_gate_matrix.json"
    catalog_path = audit + "gate_catalog.json"
    for name in (matrix_path, catalog_path):
        require(name in inventory["files"], "AUDIT_INVENTORY_INCOMPLETE")
    matrix_raw = checked_file(root, matrix_path, inventory["files"][matrix_path])
    matrix = read_json_bytes(matrix_raw)
    catalog = read_json_bytes(checked_file(root, catalog_path, inventory["files"][catalog_path]))
    require({g["id"] for g in catalog["gates"]} == set(GATE_SCOPES)
            and len(catalog["gates"]) == len(GATE_SCOPES), "AUDIT_GATE_CATALOG")
    require(matrix.get("schema") == "H4_existing_evidence_certificate_gate_audit_v1"
            and matrix.get("axis_count") == 40 and type(matrix.get("rows")) is list
            and len(matrix["rows"]) == 40, "AUDIT_MATRIX_SCHEMA")
    require(matrix.get("formal_certificate_accepted") is False
            and matrix.get("formal_certificate_generated") is False, "AUDIT_ACCEPTANCE_STATE")
    expected = {(c, e, a) for c in CASES for e in EPSILONS for a in AXES}
    seen, bundles = set(), []
    seal_path = "artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/seal.json"
    seal_sha = hashlib.sha256((Path(root) / seal_path).read_bytes()).hexdigest()
    require(seal_sha == read_json_bytes(checked_file(root, audit + "SOURCE.json",
                inventory["files"][audit + "SOURCE.json"]))["fixed_C_seal_SHA256"], "C_SEAL_HASH")
    source_commit = matrix["C_SOURCE"]
    original_source = read_json_bytes(checked_file(root, audit + "SOURCE.json",
                inventory["files"][audit + "SOURCE.json"]))["original_scientific_SOURCE"]
    for row in matrix["rows"]:
        key = (row["case"], str(row["epsilon_sig"]), row["axis"])
        require(key in expected and key not in seen, "AUDIT_AXIS_DUPLICATE_OR_UNKNOWN")
        seen.add(key)
        require(row["formal_certificate_accepted"] is False and row["old_N"] is None
                and row["old_G"] is None and row["status"] == "TARGET_MET"
                and row["bits"] == 64, "SAVED_STATE_CHANGED")
        checked_file(root, row["fixed_C_result"], row["fixed_C_result_SHA256"])
        keys(row["gates"], GATE_SCOPES, "AUDIT_GATE_SET")
        target = dict(case=key[0], epsilon_sig=key[1], axis=key[2], C_SOURCE=source_commit,
                      original_SOURCE=original_source, C_result_sha256=row["fixed_C_result_SHA256"],
                      C_seal_sha256=seal_sha, audit_matrix_sha256=inventory["files"][matrix_path],
                      compiler_target="UNRESOLVED", compiled_program_sha256=None)
        refs = [dict(id="audit_matrix", path=matrix_path,
                     sha256=inventory["files"][matrix_path], source_commit=inventory["audit_SOURCE"]),
                dict(id="C_result", path=row["fixed_C_result"],
                     sha256=row["fixed_C_result_SHA256"], source_commit=source_commit),
                dict(id="C_seal", path=seal_path, sha256=seal_sha, source_commit=source_commit)]
        assumptions = ["FRESH_INDEPENDENT_SHOTS"]
        if key[0].endswith("_PR"):
            assumptions.append("IID_UNIFORM_WORDS")
        bundles.append(dict(schema=SCHEMA, domain="SAVED_H4_CANDIDATE", target=target,
                            assumptions=assumptions, evidence=refs,
                            gates={g: dict(claim=row["gates"][g], scope=s,
                                           evidence_ids=["audit_matrix", "C_result", "C_seal"], lemma=None)
                                   for g, s in GATE_SCOPES.items()},
                            execution_authorized=False, formal_certificate_accepted=False,
                            N_G_authorized=False))
    require(seen == expected, "AUDIT_AXIS_COVERAGE")
    return bundles
