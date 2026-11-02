"""Pure JSON contract checks. No scientific imports, builders, compilers or IO."""
import hashlib
import json
import math

IDENTITY_FIELDS = (
    "geometry", "hamiltonian_sha256", "df_sha256", "state_sha256",
    "candidate_template", "candidate_fingerprint", "axis", "trajectory_seed",
    "trajectory_index", "compiler_fingerprint", "environment_fingerprint",
    "source_commit", "wrapper_semantics",
)
COMMON_FIELDS = tuple(k for k in IDENTITY_FIELDS
                      if k not in {"axis", "trajectory_seed", "trajectory_index", "candidate_fingerprint"})
CACHE_FIELDS = tuple(k for k in IDENTITY_FIELDS
                     if k not in {"trajectory_seed", "trajectory_index"})
METRICS = ("rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bytes(value):
    """UTF-8, sorted keys, compact separators; numbers use integer/tagged hex."""
    def walk(v):
        if v is None or type(v) in (str, bool, int):
            return
        if isinstance(v, list):
            for child in v:
                walk(child)
        elif isinstance(v, dict):
            require(all(type(k) is str for k in v), "JSON key must be string")
            if set(v) in ({"real64_hex"}, {"complex128_hex"}):
                check_numeric(v)
            for child in v.values():
                walk(child)
        else:
            raise ValueError("Use exact integer or tagged IEEE754 encoding")
    walk(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def real64(value):
    require(type(value) is float and math.isfinite(value), "finite binary64 required")
    return {"real64_hex": value.hex()}


def complex128(value):
    require(type(value) is complex, "complex128 required")
    return {"complex128_hex": [real64(value.real)["real64_hex"],
                               real64(value.imag)["real64_hex"]]}


def check_numeric(value):
    require(isinstance(value, dict), "numeric parameter must be tagged")
    if set(value) == {"real64_hex"}:
        parts = [value["real64_hex"]]
    elif set(value) == {"complex128_hex"}:
        parts = value["complex128_hex"]
        require(isinstance(parts, list) and len(parts) == 2, "complex needs two components")
    else:
        raise ValueError("unsupported or symbolic parameter")
    for part in parts:
        require(type(part) is str, "hex string required")
        number = float.fromhex(part)
        require(math.isfinite(number) and number.hex() == part, "nonfinite/noncanonical hex")


def circuit_fingerprint(circuit):
    """Hashes a DECLARATIVE JSON fixture; never builds or executes a circuit."""
    require(set(circuit) == {"format", "axis", "qubits", "clbits",
                            "global_phase", "instructions"}, "circuit fields")
    require(circuit["format"] == "ordered-numerical-full-circuit-v1", "circuit version")
    require(circuit["axis"] in {"cosine", "sine"}, "measurement axis")
    for key in ("qubits", "clbits"):
        require(isinstance(circuit[key], list), "ordered bit identities required")
        require(all(type(v) is str and v for v in circuit[key]), "bit identity string")
        require(len(set(circuit[key])) == len(circuit[key]), "duplicate bit identities")
    require(len(circuit["qubits"]) == 9 and len(circuit["clbits"]) == 1, "wrapper bit scope")
    check_numeric(circuit["global_phase"])
    require(set(circuit["global_phase"]) == {"real64_hex"}, "real global phase")
    require(isinstance(circuit["instructions"], list) and circuit["instructions"], "instruction sequence")
    for inst in circuit["instructions"]:
        require(set(inst) == {"name", "operation_identity", "qargs", "cargs",
                             "params", "definition", "condition", "control_state"}, "instruction fields")
        require(type(inst["name"]) is str and inst["name"], "operation name")
        require(type(inst["operation_identity"]) is str and inst["operation_identity"], "operation identity")
        for key, bits in (("qargs", circuit["qubits"]), ("cargs", circuit["clbits"])):
            require(isinstance(inst[key], list), "operand order required")
            require(all(type(i) is int and 0 <= i < len(bits) for i in inst[key]), "operand index")
        require(isinstance(inst["params"], list), "parameter sequence")
        for param in inst["params"]:
            check_numeric(param)
        # Custom instructions must carry their complete canonical definition.
        require(inst["name"] in {"h", "rz", "cx", "sdg", "measure"} or
                isinstance(inst["definition"], dict), "opaque custom instruction")
    require(any(i["name"] == "measure" and i["cargs"] == [0]
                for i in circuit["instructions"]), "measurement instruction required")
    return fingerprint({"domain": "h4-full-circuit-v1", "circuit": circuit})


def candidate_fingerprint(identity):
    return fingerprint({"domain": "h4-candidate-v1",
                        "identity": {k: identity[k] for k in COMMON_FIELDS}})


def trajectory_seed(identity, master_seed):
    require(type(master_seed) is int and 0 <= master_seed < 2**64, "master seed")
    payload = {"domain": "h4-trajectory-v1", "master_seed": master_seed,
               "identity": {k: identity[k] for k in COMMON_FIELDS},
               "trajectory_index": identity["trajectory_index"]}
    return int.from_bytes(hashlib.sha256(canonical_bytes(payload)).digest()[:8], "big")


def wrapper_key(identity):
    return fingerprint({"domain": "h4-wrapper-v1",
                        "identity": {k: identity[k] for k in IDENTITY_FIELDS}})


def record_digest(record):
    # The digest is outside the record, avoiding self-reference.
    return fingerprint({"domain": "h4-completion-record-v1", "record": record})


def validate_identity(identity, master_seed):
    require(identity["candidate_fingerprint"] == candidate_fingerprint(identity), "candidate binding")
    t = identity["candidate_template"]
    tid = f'{t["method"]}-rank{t["L_D"]}-q{t["q"]}-r{t["r"]}-K{t["K"]}'
    require(t["template_id"] == tid, "template ID")
    require(t["q"] in {1, 2, 4, 8}, "q scope")
    if t["method"] in {"B0", "B1"}:
        require(t["r"] == t["K"] == 0, "deterministic parameters")
        require(t["L_D"] in ({3, 4, 5, 6, 9} if t["method"] == "B0" else {12}), "baseline rank")
        require(identity["trajectory_seed"] is None and identity["trajectory_index"] is None,
                "deterministic seed/index must both be null")
    else:
        require(t["method"] in {"B2", "B3"}, "method scope")
        require(t["L_D"] in ({3, 6, 9} if t["method"] == "B2" else {0}), "random rank")
        regular = t["r"] in {1, 2, 4, 8, 16, 32} and t["K"] in {2, 4}
        require(regular or tid in {"B2-rank3-q1-r64-K2", "B3-rank0-q8-r64-K2"}, "r/K scope")
        require(type(identity["trajectory_index"]) is int and
                0 <= identity["trajectory_index"] < 32, "trajectory index scope")
        require(identity["trajectory_seed"] == trajectory_seed(identity, master_seed), "seed binding")


def validate_record(record, expected_identity, circuit, records, ledger,
                    numerical_registry, master_seed, schema_validator):
    """Schema plus independent expected identity, numerical registry and ledger.

    All arguments are in-memory artificial JSON in this preparation bundle.
    Future production must bind the independent registries to a sealed run.
    """
    schema_validator.validate(record)
    validate_identity(record, master_seed)
    require({k: record[k] for k in IDENTITY_FIELDS} == expected_identity, "expected identity mismatch")
    require(record["wrapper_key"] == wrapper_key(record), "wrapper key binding")
    random = record["candidate_template"]["method"] in {"B2", "B3"}
    require(record["sample_weight"] == {"numerator": 1, "denominator": 32 if random else 1},
            "preserve logical sample weight")
    if record["status"] != "COMPLETE":
        require(not record["cache_reuse"], "incomplete record cannot be reused")
        return
    numerical = circuit_fingerprint(circuit)
    require(circuit["axis"] == record["axis"], "circuit/record axis")
    require(record["numerical_circuit_fingerprint"] == numerical, "numerical circuit binding")
    require(numerical_registry.get(record["wrapper_key"]) == numerical, "independent numerical registry")

    def completion(r):
        schema_validator.validate(r)
        require(r["status"] == "COMPLETE", "owner not COMPLETE")
        require(r["wrapper_key"] == wrapper_key(r), "owner wrapper key")
        validate_identity(r, master_seed)
        entry = ledger["entries"].get(r["wrapper_key"])
        require(entry is not None and entry["status"] == "COMPLETE", "completion ledger missing")
        require(entry["record_sha256"] == record_digest(r), "external record digest mismatch")
        require(numerical_registry.get(r["wrapper_key"]) == r["numerical_circuit_fingerprint"],
                "owner numerical registry mismatch")
        if not r["cache_reuse"]:
            rid = r["actual_transpile_invocation_id"]
            reservation = ledger["reservations"].get(rid)
            require(reservation == {"status": "COMPLETE", "wrapper_key": r["wrapper_key"]},
                    "missing/wrong compile reservation")
    completion(record)
    if record["cache_reuse"]:
        owner = records.get(record["cache_owner_wrapper_key"])
        require(owner is not None, "cache owner missing")
        completion(owner)
        require(not owner["cache_reuse"], "cache owner must own an actual compile")
        require(owner["wrapper_key"] != record["wrapper_key"], "self reuse")
        require(all(owner[k] == record[k] for k in CACHE_FIELDS), "cross-identity reuse")
        require(owner["numerical_circuit_fingerprint"] == numerical, "owner circuit mismatch")
        require(owner["metrics"] == record["metrics"], "reused metric mismatch")


def validate_accounting(records, ledger):
    keys = [r["wrapper_key"] for r in records]
    require(len(keys) == len(set(keys)), "duplicate logical wrapper key")
    require(len(keys) <= 74784 and len(ledger["reservations"]) <= 74784, "wrapper/compile budget")
    invocations = [r["actual_transpile_invocation_id"] for r in records
                   if r["status"] == "COMPLETE" and not r["cache_reuse"]]
    require(len(invocations) == len(set(invocations)), "duplicate compile invocation")
    require(all(r["status"] == "COMPLETE" for r in ledger["reservations"].values()),
            "unresolved reservation: mandatory STOP, no automatic retry/resume")


def validate_seed_uniqueness(identities):
    seen = {}
    for identity in identities:
        if identity["trajectory_seed"] is None:
            continue
        slot = fingerprint({"identity": {k: identity[k] for k in COMMON_FIELDS},
                            "index": identity["trajectory_index"]})
        seed = identity["trajectory_seed"]
        require(seed not in seen or seen[seed] == slot, "duplicate seed across trajectories")
        seen[seed] = slot
