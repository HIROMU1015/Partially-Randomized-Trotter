"""Static exact-literal record lookup; no guard/T arithmetic or selection.

Saved guard and T fields are copied as evidence, never ranked or summed.
The small weighted-template helper is exercised only on artificial inputs.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import io
import json
import lzma
from pathlib import Path
import subprocess
import tarfile

BASE = "800554fbb64c21e584bb29a1a8f52490ab4163e0"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz"
V1 = "artifacts/pr_reference_post_v0_v1_budget/2026-11-08/RESULTS.json"
N3 = "artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/"
B3 = "artifacts/pr_reference_n5b2_numeric/2026-11-08/INPUTS.json"
N4 = "artifacts/pr_reference_n4b_consumer/2026-11-08/ROLE_WORD_CERTIFICATES.json"
METHOD = "exact cyclotomic word matrix + outward interval strict Frobenius"
REFERENCES = {}


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    REFERENCES[path] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return raw


def read(path):
    return json.loads(blob(path))


def archive_metadata(bundle, targets):
    raw = blob(bundle["path"])
    assert hashlib.sha256(raw).hexdigest() == bundle["sha256"]
    rows = []
    count = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            record = json.load(archive.extractfile(member))
            count += 1
            if record["theta_hex"] not in targets:
                continue
            # Metadata/provenance matching only, not validate_record's numeric
            # guard/price checks, word generation or an acquisition invocation.
            assert record["key"] == hashlib.sha256(record["spec"].encode()).hexdigest()
            assert record["spec"].startswith(record["theta_hex"] + ":")
            rows.append(dict(archive=bundle["path"], member=member.name,
                theta_hex=record["theta_hex"], key=record["key"], spec=record["spec"],
                precision_exponent=record["precision_exponent"], status=record["status"],
                sequence_sha256=record["sequence_sha256"],
                strict_error_upper_saved=record["strict_error_upper"],
                guard_method=record.get("guard_method"),
                guard_method_documented=record.get("guard_method") == METHOD,
                strict_spec=":strict:" in record["spec"],
                existing_T_quoted=record["T"], existing_Tdagger_quoted=record["Tdagger"],
                existing_global_W_quoted=record["global_W"],
                actual_adjoint_rule="complete positive word, including W; no new negative synthesis"))
    assert count == bundle["records"]
    return rows


def index_matches(target_roles, records, current):
    by_literal = defaultdict(list)
    for record in records:
        by_literal[record["theta_hex"]].append(record)
    entries = []
    for key, roles in sorted(target_roles.items()):
        selected = current.get(key)
        matches = by_literal.get(key, [])
        alternate = bool(selected) and any(
            r["sequence_sha256"] != selected["sequence_sha256"] and
            r["status"] == "PASS" and r["strict_spec"] and r["guard_method_documented"]
            for r in matches)
        entries.append(dict(target_literal=key, roles=roles, current_word=selected,
            exact_literal_records=matches, distinct_documented_alternative_word=alternate,
            numeric_guard_or_T_comparison_performed=False))
    return entries


def weighted_template_max(context_counts, guards):
    """Prepared arithmetic primitive. NOT called by the saved-metadata audit."""
    if not context_counts:
        return None
    values = []
    for counts in context_counts:
        if any(not isinstance(n, int) or n < 0 for n in counts.values()):
            raise ValueError("Nonnegative physical multiplicities required")
        if any(k not in guards for k, n in counts.items() if n):
            return None
        if any(guards[k] < 0 for k, n in counts.items() if n):
            raise ValueError("Negative norm bound")
        values.append(sum((n * guards[k] for k, n in counts.items() if n), 0))
    return max(values)


def main():
    assert not OUT.exists(), "Preserve prior static evidence"
    roles = read(V1)["saved_noncentral_role_envelopes"]
    target_roles = defaultdict(list)
    for role in roles:
        if role["count_group"] in ("D", "frames"):
            for key in role["saved_positive_keys"]:
                target_roles[key].append(dict(arm=role["arm"], role=role["count_group"]))
    inventory = read(N3 + "INPUTS.json")["record_inventory"]
    targets = set(target_roles)
    # Exactly the three already listed archives, not a new broad filesystem scan.
    with ThreadPoolExecutor(max_workers=len(inventory)) as pool:
        bundles = list(pool.map(lambda bundle: archive_metadata(bundle, targets), inventory))
    records = [r for bundle in bundles for r in bundle]
    C0 = "artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz"
    current = {r["theta_hex"]: dict(kind="C0_fixed_priority", sequence_sha256=r["sequence_sha256"],
                record_key=r["key"], strict_error_upper_saved=r["strict_error_upper_saved"],
                existing_T_quoted=r["existing_T_quoted"])
               for r in records if r["archive"] == C0}
    certificates = read(N4)
    for r in certificates["rows"]:
        key = r["positive_literal_hex"]
        if key not in targets:
            continue
        assert key not in current
        p = r["positive"]
        current[key] = dict(kind="N4_85_composite", sequence_sha256=p["positive_sequence_sha256"],
            Q_saved=p["quantization_upper"], G_saved=p["word_error_upper"],
            emitter_saved=p["emitter_upper"], strict_upper_saved=p["strict_operator_Frobenius_upper"],
            existing_T_quoted=p["T"], actual_adjoint_saved=r["negative"]["negative_whole_adjoint"])
    basic = read(N3 + "BASIC_WORDS_MANIFEST.json")
    saved_basic = {r["j"]: r for r in read(B3)["basic_records"]}
    basic_metadata = [dict(j=r["j"], theta_hex=r["theta_hex"], key=r["key"],
        guard_saved=r["strict_error_upper"], guard_qualified_saved=r["guard_qualified"],
        existing_T_quoted=r["T"], sequence_sha256=saved_basic[r["j"]]["sequence_sha256"],
        record=r["record"], record_sha256=r["record_sha256"]) for r in basic["rows"]]
    entries = index_matches(target_roles, records, current)
    assert all(e["current_word"] is not None for e in entries)
    summary = []
    for arm in ("Bprime", "reference"):
        for role in ("D", "frames"):
            subset = [e for e in entries if dict(arm=arm, role=role) in e["roles"]]
            summary.append(dict(arm=arm, role=role, target_literals=len(subset),
                with_raw_record_metadata=sum(bool(e["exact_literal_records"]) for e in subset),
                with_distinct_documented_alternative_word=sum(e["distinct_documented_alternative_word"] for e in subset)))
    result = dict(status="STATIC_LITERAL_RECORD_METADATA_LOOKUP_COMPLETE", base_commit=BASE,
        summary=summary, entries=entries, basic_dictionary_metadata=basic_metadata,
        source_inputs=REFERENCES, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numeric_guard_or_T_comparisons=0, new_guard_synthesis_compile_RPE_T_evaluations=0,
        current_policy_changed=False, m17_RPE="UNQUALIFIED", STOP="MANDATORY_STOP_STATIC_SCOPE")
    OUT.parent.mkdir(parents=True)
    OUT.write_bytes(lzma.compress(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False).encode()))
    print(json.dumps(dict(status=result["status"], summary=summary, basic_metadata=len(basic_metadata),
                         numeric_guard_or_T_comparisons=0)))


if __name__ == "__main__":
    main()
