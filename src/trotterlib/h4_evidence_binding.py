"""Read-only bindings of saved H4 evidence to phase-T gate propositions.

No matrix, interval, probability, error-bound or shot arithmetic is performed.
Rational endpoints are syntax-checked and copied, never compared or combined.
Binding integrity, scoped algebra and scientific acceptance are separate states.
The phase-T SOURCE remains unchanged; its checker is imported directly.
"""
import importlib.util
from pathlib import Path
import re

_spec = importlib.util.spec_from_file_location("h4_phase_T_frozen", Path(__file__).with_name("h4_certificate_gates.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
require = T.require
SCHEMA = "H4_saved_evidence_binding_v1"
FIXED_COMMITS = {
    "base": "36d0c12de9cec2a4fbb3abc0d8ab59c7f890cb2a",
    "T": "a66cefb497cb83c13158c9d4ab926202f791309f",
    "C": T.C_SOURCE,
    "original": T.ORIGINAL_SOURCE,
    "results": "86b8b002704aced8ca239414080f53793de81658",
}
PROPOSITIONS = {
    "numerical_enclosure": "The saved encoded-model allowance endpoint meets the unchanged design target.",
    "reference_encoded_full_H": "Saved reference encloses psi-dagger exp(-i T H_encoded) psi in full Fock space.",
    "bias_encoded_saved": "Saved same-ray physical bias and semantic allowance enclose error relative to saved signed bias.",
    "normalization_nominal_log_B": "Finite K4 nominal normalization and saved upper log_B refer to the same tail.",
    "measurement_binary_range": "A bit b in {0,1}, recorded as B_saved*(1-2*b), has support {-B_saved,+B_saved}.",
    "encoded_event_norm": "Exact signed-Pauli components have norm one; rounded DF conjugations need separate transfer.",
    "sampling_law": "Saved effective component/order masses describe the implementation under IID uniform words.",
    "wrapper_transfer": "Encoded expectation transfers, with controlled phase, to literal cosine/sine wrapper.",
    "compiler_transfer": "The historical postcompiled program preserves the literal wrapper's measurement semantics.",
    "conditional_statistics": "The bounded-outcome sufficient condition is conditional on fresh independent shots and PR word IID.",
    "provenance": "Saved result, model, preparation, state, input, code and runtime have a consistent recorded identity.",
    "saved_B_to_shot_upper_binding": "The fixed saved measurement scale is covered by the unchanged shot upper bound.",
    "certificate_acceptance_implementation": "All scoped proof obligations are validated before any later authorized acceptance.",
    "final_combined_error_budget": "C error plus additional transfer terms fits the unchanged budget without double counting.",
}


SOURCE_ROLES = {
    "numerical_enclosure": ["C_ball_source"], "reference_encoded_full_H": ["C_ball_source"],
    "bias_encoded_saved": ["C_ball_source", "C_export_source"],
    "normalization_nominal_log_B": ["C_ball_source"],
    "measurement_binary_range": ["old_wrapper_source", "old_account_source"],
    "encoded_event_norm": ["C_ball_source", "C_export_source"],
    "sampling_law": ["C_ball_source", "C_export_source", "old_rte_source"],
    "wrapper_transfer": ["old_wrapper_source", "old_df_source", "old_evaluator_source"],
    "compiler_transfer": ["old_compiler_source", "old_evaluator_source"],
    "conditional_statistics": ["old_account_source"], "provenance": ["C_export_source"],
    "saved_B_to_shot_upper_binding": ["C_ball_source", "old_account_source"],
    "certificate_acceptance_implementation": ["T_gate_source"],
    "final_combined_error_budget": ["C_ball_source", "old_account_source"],
}


def validate_manifest(manifest):
    T.keys(manifest, {"schema", "domain", "commits", "files", "execution_authorized", "grant_valid",
                      "dispatch_enabled", "formal_certificate_accepted", "N_G_authorized"}, "BINDING_MANIFEST_SCHEMA")
    require(manifest["schema"] == SCHEMA and manifest["domain"] == "SAVED_H4", "SAVED_H4_DOMAIN_REQUIRED")
    require(manifest["commits"] == FIXED_COMMITS, "BINDING_FIXED_SOURCE")
    require(all(manifest[k] is False for k in ("execution_authorized", "grant_valid", "dispatch_enabled",
                                               "formal_certificate_accepted", "N_G_authorized")), "BINDING_NO_AUTHORITY")
    require(type(manifest["files"]) is dict, "BINDING_FILES")
    paths = set()
    for role, ref in manifest["files"].items():
        T.identifier(role)
        T.keys(ref, {"path", "sha256", "source_commit", "domain"}, "BINDING_REFERENCE_SCHEMA")
        require(ref["domain"] == "SAVED_H4", "ARTIFICIAL_EVIDENCE_REJECTED")
        T.hex_digest(ref["sha256"])
        T.hex_digest(ref["source_commit"], 40)
        require(ref["path"] not in paths, "DUPLICATE_EVIDENCE_PATH")
        paths.add(ref["path"])


class EvidenceStore:
    def __init__(self, root, manifest):
        validate_manifest(manifest)
        self.root, self.manifest = Path(root), manifest
        self.raw, self.json = {}, {}
        for role, ref in manifest["files"].items():
            self.raw[role] = T.checked_file(root, ref["path"], ref["sha256"])

    def document(self, role):
        require(role in self.raw, "MISSING_EVIDENCE_ROLE:" + role)
        if role not in self.json:
            self.json[role] = T.read_json_bytes(self.raw[role])
        return self.json[role]

    def fact(self, role, pointer):
        require(type(pointer) is list and all(type(k) in (str, int) for k in pointer), "FACT_POINTER_SCHEMA")
        value = self.document(role)
        for key in pointer:
            if type(key) is int:
                require(type(value) is list and 0 <= key < len(value), "FACT_POINTER_LIST")
            else:
                require(type(value) is dict and key in value, "FACT_POINTER_KEY")
            value = value[key]
        ref = self.manifest["files"][role]
        return dict(role=role, path=ref["path"], sha256=ref["sha256"], source_commit=ref["source_commit"],
                    pointer=pointer, value=value, evidence_domain="SAVED_H4", value_digest=T.digest(value))


def endpoint_tokens(value):
    """Check serialization only, not signs, ordering, width or any inequality."""
    T.keys(value, {"lower", "upper"}, "ENDPOINT_PAIR_SCHEMA")
    for token in value.values():
        require(type(token) is str and re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", token),
                "EXACT_ENDPOINT_TOKEN")


def complex_tokens(value):
    T.keys(value, {"real", "imag"}, "COMPLEX_ENCLOSURE_SCHEMA")
    for axis in value.values():
        endpoint_tokens(axis)


def validate_error_terms(terms):
    """Validate coverage accounting, without computing a new scientific bound."""
    require(type(terms) is list and terms, "ERROR_TERMS_REQUIRED")
    ids, primitives = set(), set()
    for term in terms:
        T.keys(term, {"id", "primitive_terms", "fact", "newly_evaluated"}, "ERROR_TERM_SCHEMA")
        require(type(term["id"]) is str and term["id"] not in ids, "DUPLICATE_ERROR_TERM")
        ids.add(term["id"])
        covered = term["primitive_terms"]
        require(type(covered) is list and covered and all(type(x) is str for x in covered)
                and len(set(covered)) == len(covered), "ERROR_TERM_COVERAGE")
        require(not (set(covered) & primitives), "DOUBLE_COUNTED_ERROR_TERM")
        primitives.update(covered)
        require(term["newly_evaluated"] is False, "NEW_BOUND_EVALUATION_FORBIDDEN")
        require(type(term["fact"]) is dict, "MISSING_ERROR_FACT")
    return sorted(primitives)


def _locate(rows, case, epsilon, axis):
    matches = [i for i, row in enumerate(rows) if row["case"] == case
               and str(row["epsilon_sig"]) == epsilon and row["axis"] == axis]
    require(len(matches) == 1, "MISSING_OR_DUPLICATE_AXIS_PROOF")
    return matches[0]


def inspect_pass_proofs(proofs, required_passes):
    """Check explicit rule coverage, never infer whole-compiler equivalence."""
    require(type(proofs) is list and type(required_passes) is list
            and all(type(x) is str for x in required_passes), "PASS_PROOF_SCHEMA")
    seen = set()
    checked = []
    for proof in proofs:
        T.keys(proof, {"pass_instance", "rule", "proof"}, "PASS_PROOF_SCHEMA")
        require(type(proof["pass_instance"]) is str and proof["pass_instance"] not in seen,
                "DUPLICATE_PASS_PROOF")
        seen.add(proof["pass_instance"])
        require(proof["rule"] == "WORD_EQUALITY", "UNSUPPORTED_COMPILER_PASS_PROOF")
        result = T.check_symbolic_rule(proof["proof"])
        require(result["gate"] == "compiler_transfer", "COMPILER_LEMMA_SCOPE")
        checked.append(result)
    require(seen <= set(required_passes), "PASS_NOT_IN_FIXED_COVERAGE")
    return dict(checked_lemmas=checked, missing_pass_instances=sorted(set(required_passes)-seen),
                pass_to_historical_IR_binding="NOT_PROVED", proof_status="MISSING_EVIDENCE",
                whole_compiler_equivalence_proved=False)


def _typed_pauli_components(case, n):
    labels = []
    for block in case["blocks"]:
        label = block["label"]
        require(type(label) is str and len(label) == n and set(label) <= set("IXYZ"), "PAULI_BLOCK_LABEL")
    for component in case["components"]:
        require(type(component["sign"]) is int and component["sign"] in (-1, 1), "PAULI_SIGN")
        label = component["label"]
        require(type(label) is str and len(label) == n and set(label) <= set("IXYZ"), "PAULI_COMPONENT_LABEL")
        labels.append(label)
    return dict(blocks=len(case["blocks"]), components=len(labels),
                proof="Tensor Pauli matrices square to identity; signed components remain exact involutions.",
                scope_limit="Does not prove rounded gate angles or literal/compiled evolution transfer.")


def model_binding(store, representation, method):
    role = representation + "_model"
    model = store.document(role)
    require(model["schema"] == "h4_frozen_C_export_v1" and model["representation"] == representation,
            "ENCODED_MODEL_SCHEMA")
    require(model["old_scientific_SOURCE"] == T.ORIGINAL_SOURCE and model["qubit_order"] == "qiskit_little_endian",
            "MODEL_SOURCE_OR_QUBIT_ORDER")
    model_content = {k: v for k, v in model.items() if k != "binding_hash"}
    require(T.digest(model_content) == model["binding_hash"], "MODEL_BINDING_HASH")
    case = model["cases"][method]
    old = store.document(representation + "_" + method + "_old")
    require(case["metadata"] == case["expected_metadata"] == old["model"], "PREPARATION_METADATA_MISMATCH")
    require(case["saved_normalization"] == old["normalization"], "NORMALIZATION_BINDING_MISMATCH")
    require(model["state_hash"] == old["state"]["hash"], "STATE_PREPARATION_BINDING")
    require(old["provenance"]["source"]["SOURCE"] == T.ORIGINAL_SOURCE, "OLD_RESULT_SOURCE")
    require(case["config"] == dict(T=.8, q=8, r=4, K=4, method=method), "FIXED_EVOLUTION_CONTRACT")
    require(case["correspondence_scope"] == "encoded native local matrices and fixed law; literal compiled-wrapper transfer not certified",
            "ENCODED_LITERAL_SCOPE_CHANGED")
    # These are joins of saved declarations, not a proof that a matrix is unitary.
    require(case["exact_event_range_proved"] is (representation == "pauli_jw"), "RANGE_POLICY_CHANGED")
    return model, case, old


def _gate(store, gate, row, target, facts, status, obligations, lemma=None, assumptions=None):
    require(gate in T.GATE_SCOPES and status in T.CLAIMS, "GATE_BINDING_SCHEMA")
    return dict(gate=gate, scope=T.GATE_SCOPES[gate], proposition=PROPOSITIONS[gate],
                target_digest=T.digest(target), target=target, saved_claim=row["gates"][gate],
                binding_status="VERIFIED", proof_status=status, facts=facts,
                source_bindings=[dict(role=role, **store.manifest["files"][role]) for role in SOURCE_ROLES[gate]],
                lemma=lemma, assumptions=assumptions or [], obligations=obligations,
                numerical_inequality_evaluated=False, formal_certificate_accepted=False)


def produce_bindings(store):
    """Join saved facts and definitions; intentionally no numerical comparisons."""
    seal, contract = store.document("C_seal"), store.document("C_contract")
    require(seal["SOURCE"] == contract["C_production_implementation_SOURCE"] == T.C_SOURCE, "C_SOURCE_BINDING")
    require(seal["historical_scientific_SOURCE"] == T.ORIGINAL_SOURCE, "C_ORIGINAL_SOURCE_BINDING")
    require(contract["technical_seal_sha256"] == store.manifest["files"]["C_seal"]["sha256"], "CONTRACT_SEAL_BINDING")
    require(contract["input_manifest_sha256"] == seal["input_manifest_sha256"], "INPUT_MANIFEST_BINDING")
    for name, path in seal["routing"].items():
        require(contract[name] == path, "C_INPUT_ROUTING_BINDING")
    matrix = store.document("T_matrix")
    expected = {(c, e, a) for c in T.CASES for e in T.EPSILONS for a in T.AXES}
    actual = [(x["case"], str(x["epsilon_sig"]), x["axis"]) for x in matrix["rows"]]
    require(len(actual) == 40 and set(actual) == expected, "T_MATRIX_AXIS_COVERAGE")
    require(matrix["C_SOURCE"] == T.C_SOURCE, "MATRIX_C_SOURCE")
    axes_table = store.document("C_axis_outcomes")["rows"]
    require(len(axes_table) == 40, "SAVED_AXES_TABLE_COUNT")
    all_rows = []
    model_cache = {}
    for saved in matrix["rows"]:
        case_name, epsilon, axis = saved["case"], str(saved["epsilon_sig"]), saved["axis"]
        representation, method = case_name.rsplit("_", 1)
        if case_name not in model_cache:
            model_cache[case_name] = model_binding(store, representation, method)
        model, case, old = model_cache[case_name]
        result_role, model_role = representation + "_result", representation + "_model"
        result = store.document(result_role)
        result_index = _locate(result["axes"], case_name, epsilon, axis)
        table_index = _locate(axes_table, case_name, epsilon, axis)
        out = result["axes"][result_index]
        exported_index = _locate([dict(x, case=case_name) for x in case["saved_axes"]], case_name, epsilon, axis)
        exported = case["saved_axes"][exported_index]
        require(exported["target_exact"] == out["target_exact"], "SAVED_TARGET_BINDING")
        bias_name = "real" if axis == "cosine" else "imag"
        require(exported["signed_bias"] == old["signal"]["signed_bias"][bias_name], "SIGNED_BIAS_BINDING")
        require(out["formal_certificate_accepted"] is False and out["old_N"] is None and out["old_G"] is None,
                "SAVED_ACCEPTANCE_OR_NG_CHANGED")
        require(result["formal_certificate_accepted"] is False, "RESULT_ACCEPTANCE_CHANGED")
        row_table = axes_table[table_index]
        require(row_table["saved_result_SHA256"] == store.manifest["files"][result_role]["sha256"], "RESULT_TABLE_HASH")
        require(row_table["total_error_upper"] == out["total_error_upper"]["upper"]
                and row_table["bias_distance_upper"] == out["bias_distance"]["upper"]
                and row_table["target_exact"] == out["target_exact"], "SAVED_ENDPOINT_COPY_MISMATCH")
        for value in (out["bias_distance"], out["total_error_upper"], result["state_norm"], result["state_norm_defect"]):
            endpoint_tokens(value)
        for value in (result["reference"], result["implemented_law"][method]["same_ray_physical_bias"]):
            complex_tokens(value)
        target = dict(case=case_name, epsilon_sig=epsilon, axis=axis, C_SOURCE=T.C_SOURCE,
                      original_SOURCE=T.ORIGINAL_SOURCE, C_result_sha256=store.manifest["files"][result_role]["sha256"],
                      C_seal_sha256=store.manifest["files"]["C_seal"]["sha256"],
                      audit_matrix_sha256=store.manifest["files"]["T_matrix"]["sha256"],
                      compiler_target="HISTORICAL_PROGRAM", compiled_program_sha256=None)
        T.validate_target(target)
        f = store.fact
        prefix = ["cases", method]
        binding = dict(model_sha256=store.manifest["files"][model_role]["sha256"],
                       model_binding_hash=model["binding_hash"], state_hash=model["state_hash"],
                       hamiltonian_hash=case["metadata"]["hamiltonian_hash"],
                       preparation_hash=case["metadata"].get("preparation_hash"),
                       input_manifest_sha256=seal["input_manifest_sha256"],
                       runtime_identity_fact=f("C_seal", ["legacy_python_executable_sha256"]),
                       input_files_fact=f("C_seal", ["input_files"]),
                       qubit_order=model["qubit_order"], scope=case["correspondence_scope"])
        numerical_facts = [f(result_role, ["axes", result_index]), f(model_role, prefix + ["saved_axes", exported_index])]
        gates = {}
        def add(gate, facts, status, obligations, lemma=None, assumptions=None):
            gates[gate] = _gate(store, gate, saved, target, facts, status, obligations, lemma, assumptions)
        add("numerical_enclosure", numerical_facts, "EVIDENCE_EXISTS_UNCHECKED",
            ["Saved exact endpoint tokens and target are bound; numerical ordering and valid enclosure proof are not re-evaluated here."])
        add("reference_encoded_full_H", [f(result_role, ["reference"]), f(result_role, ["operator_snapshot_residual_upper"]),
            f(model_role, ["state_hash"]), f(model_role, ["representation"])], "EVIDENCE_EXISTS_UNCHECKED",
            ["Reuse existing full-H certificate computation; new numerical verifier not run.", "Normalized preparation-to-physical state transfer remains separate."])
        add("bias_encoded_saved", [f(result_role, ["implemented_law", method, "same_ray_physical_bias"]),
            f(result_role, ["implemented_law", method, "state_normalization_difference"]),
            f(result_role, ["implemented_law", method, "circuit_semantic_allowance"]),
            f(model_role, prefix + ["saved_axes", exported_index, "signed_bias"])], "EVIDENCE_EXISTS_UNCHECKED",
            ["The saved bias, normalized ray and semantic term are connected, not newly recomputed or certified."])
        pf = method == "PF"
        normalization = result["normalization"][method]
        if pf:
            require(normalization["B"] == "1" and normalization["log_B"] == "0"
                    and normalization["deterministic_structure"] is True, "PF_NORMALIZATION_STRUCTURE")
        add("normalization_nominal_log_B", [f(result_role, ["normalization", method]), f(model_role, prefix + ["saved_normalization"])],
            "VERIFIED" if pf else "EVIDENCE_EXISTS_UNCHECKED",
            [] if pf else ["PR saved log upper inclusion is recorded but not re-evaluated from endpoints.", "Nominal B versus saved readout scale binding remains separate."])
        readout = T.check_symbolic_rule(dict(rule="BINARY_READOUT", data=dict(scale_symbol="B_saved", bit_values=[0, 1], estimator="B*(1-2*b)")))
        add("measurement_binary_range", [f(model_role, prefix + ["saved_normalization"])], "VERIFIED", [],
            lemma=readout, assumptions=["Ideal Born binary readout; fixed finite nonnegative saved scale; not a DF matrix norm claim."])
        if representation == "pauli_jw":
            lemma = _typed_pauli_components(case, model["n_qubits"])
            add("encoded_event_norm", [f(model_role, prefix + ["exact_event_range_proved"])], "VERIFIED", [], lemma=lemma)
        else:
            add("encoded_event_norm", [f(model_role, prefix + ["exact_event_range_proved"])], "MISSING_EVIDENCE",
                ["148 rounded local matrix hashes need ideal-gate transfer, inverse and 11-basis composition; no exact unitarity is assumed."])
        add("sampling_law", [f(result_role, ["normalization", method])], "VERIFIED" if pf else "ASSUMPTION_DEPENDENT",
            [] if pf else ["Effective mass records are bound; no CDF/mass computation or IID proof is performed."],
            assumptions=[] if pf else ["IID_UNIFORM_WORDS"])
        wrapper_lemma = T.check_symbolic_rule(dict(rule="HADAMARD_WRAPPER", data=dict(axis=axis,
            operations=["H", "CONTROLLED_U"] + (["SDG"] if axis == "sine" else []) + ["H"])))
        add("wrapper_transfer", [f(model_role, prefix + ["correspondence_scope"]),
            f(result_role, ["implemented_law", method, "circuit_semantic_allowance"])], "MISSING_EVIDENCE",
            ["Prove controlled zero/one branch including pre-control global phase and DF identity/sign phases.",
             "Bind literal basis inverse/order and all event-family parameters with a uniform transfer bound.",
             "Abstract wrapper identity alone does not bind literal implementation."], lemma=wrapper_lemma)
        add("compiler_transfer", [f(case_name + "_old", ["compiler"]), f("compiler_audit", ["serialization_scope"])],
            "MISSING_EVIDENCE", ["No historical postcompiled program/pass receipt in the inspected saved namespace.",
             "Prove all fixed-compiler synthesis/rewrite paths uniformly or recover actual program; word cancellation alone is insufficient."])
        add("conditional_statistics", [f("C_contract", ["acceptance", "statistics"])], "ASSUMPTION_DEPENDENT", [],
            assumptions=["FRESH_INDEPENDENT_SHOTS"] + ([] if pf else ["IID_UNIFORM_WORDS"]))
        add("provenance", [f("C_seal", ["SOURCE"]), f("C_seal", ["input_manifest_sha256"]),
            f(model_role, prefix + ["metadata", "hamiltonian_hash"])], "VERIFIED", [],
            assumptions=["Scope: byte/declaration joins of saved SOURCE/input/preparation/runtime identities, not mathematical enclosure validation or original native completion."])
        add("saved_B_to_shot_upper_binding", [f(model_role, prefix + ["saved_normalization"])],
            "VERIFIED" if pf else "EVIDENCE_EXISTS_UNCHECKED", [] if pf else ["Connect fixed B_saved scale to existing log_B upper without changing B or computing a new upper here."])
        add("certificate_acceptance_implementation", [f("T_source", ["technical_SOURCE"])], "MISSING_EVIDENCE",
            ["No acceptance authority is granted. Remaining numerical, literal and compiled proof obligations need a later verifier and approval."])
        # Keep the pre-existing total intact. Semantic allowance is already inside it.
        terms = [dict(id="saved_C_total", primitive_terms=["C_bias_distance", "C_circuit_semantic_allowance"],
                      fact=f(result_role, ["axes", result_index, "total_error_upper"]), newly_evaluated=False)]
        validate_error_terms(terms)
        add("final_combined_error_budget", [term["fact"] for term in terms], "EVIDENCE_EXISTS_UNCHECKED",
            ["Add only uncovered future transfer errors; do not add C_circuit_semantic_allowance again.",
             "No addition, target comparison, headroom/guard evaluation or N/G is performed."])
        require(set(gates) == set(T.GATE_SCOPES), "GATE_COVERAGE")
        all_rows.append(dict(target=target, target_digest=T.digest(target), identity=binding, gates=gates,
                             error_terms=terms, all_scientific_gates_verified=False,
                             formal_certificate_accepted=False, N=None, G=None))
    return dict(schema=SCHEMA, domain="SAVED_H4", kind="SAVED_PROPOSITION_BINDINGS_NOT_CERTIFICATE",
                rows=all_rows, execution_authorized=False, grant_valid=False, dispatch_enabled=False,
                formal_certificate_accepted=False, N_G_authorized=False, numerical_reanalysis=False,
                new_numerical_bounds=0, mandatory_STOP=True)


def verify_bindings(store, output):
    """Reproduce only the structural binding records and reject altered proofs."""
    require(output == produce_bindings(store), "BINDING_RECORD_OR_SCOPE_OR_TARGET_MISMATCH")
    for row in output["rows"]:
        require(row["formal_certificate_accepted"] is False and row["N"] is None and row["G"] is None,
                "NO_ACCEPTANCE_OR_NG")
        validate_error_terms(row["error_terms"])
        for gate in row["gates"].values():
            require(gate["target_digest"] == row["target_digest"], "CROSS_AXIS_PROOF_REUSE")
            for fact in gate["facts"]:
                require(fact == store.fact(fact["role"], fact["pointer"]), "FACT_WITNESS_MISMATCH")
    return dict(status="PASS_STATIC_BINDINGS_40_AXES_14_GATES", axes=40, gates_per_axis=14,
                numeric_tokens_compared_or_evaluated=False, scientific_certificate_accepted=False,
                scientific_calls=0, mandatory_STOP=True)
