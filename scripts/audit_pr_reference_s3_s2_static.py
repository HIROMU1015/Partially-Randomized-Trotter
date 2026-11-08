"""Static SOURCE/token and saved-frontier crosswalk; no guard/T arithmetic.

RZ words stay opaque exact sequence hashes, including W. This is an incomplete
syntactic proof system, not a matrix equality test or a new word constructor.
No synthesis/consumer/circuit/norm/RPE/optimizer module is imported.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import io
import json
import lzma
from pathlib import Path
import subprocess
import tarfile

BASE = "f0b4b143e9c01115cd010f4dd9f88354ba06c3b4"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/pr_reference_s3_s2_static/2026-11-08/STATIC_FINDINGS.json.xz"
META = "artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz"
S2 = "artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/RESULTS.json"
TR = "artifacts/pr_reference_transfer_qualification/2026-11-06/"
BP = "artifacts/pr_reference_bprime/2026-11-07/"
VR = "artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/"
FILES = {}
EMPTY_WORD_SHA = hashlib.sha256(b"").hexdigest()


@lru_cache(None)
def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    FILES[path] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw


def read(path):
    return json.loads(blob(path))


def archive(path):
    with tarfile.open(fileobj=io.BytesIO(blob(path)), mode="r:xz") as t:
        return [(m.name, json.load(t.extractfile(m))) for m in t if m.isfile()]


def reduce_tokens(tokens):
    """Only adjacent same-word powers, SX^4=I, X=SX^2 and CX^2=I.

    No phase fitting, arbitrary word equality, or disjoint-wire commuting.
    Each opaque word token contains the exact positive sequence's W phase.
    """
    stack = []
    for token in tokens:
        token = tuple(token)
        kind = token[0]
        if kind in ("word", "sx"):
            identity, exponent = token[:-1], token[-1]
            if stack and stack[-1][:-1] == identity:
                exponent += stack.pop()[-1]
            if kind == "sx":
                exponent %= 4
            if exponent:
                stack.append((*identity, exponent))
        elif kind == "cx":
            if stack and stack[-1] == token:
                stack.pop()
            else:
                stack.append(token)
        else:
            raise ValueError("Unsupported formal token")
    return stack


def tokens_from_lowering(lowering, words):
    tokens, missing = [], set()
    for op, wires, params in lowering:
        if op == "rz":
            value = float(params[0]); key = abs(value).hex()
            if key not in words:
                missing.add(key)
                continue
            sequence_hash = words[key]["sequence_sha256"]
            if sequence_hash != EMPTY_WORD_SHA:
                tokens.append(("word", wires[0], sequence_hash, -1 if value < 0 else 1))
        elif op in ("sx", "x"):
            tokens.append(("sx", wires[0], 1 if op == "sx" else 2))
        elif op == "cx":
            tokens.append(("cx", *wires))
        else:
            raise ValueError("Unsupported native instruction: " + op)
    return tokens, sorted(missing)


def pair_syntax(forward, inverse, words):
    a, ma = tokens_from_lowering(forward, words)
    c, mc = tokens_from_lowering(inverse, words)
    missing = sorted(set(ma + mc))
    if missing:
        return dict(status="CURRENT_WORD_METADATA_MISSING", missing_literals=missing)
    remaining = reduce_tokens(a + c)  # Chronological A then C: product C A.
    return dict(status="INSTRUCTION_PRODUCT_ADJOINT_PROVED" if not remaining else "NOT_PROVED_BY_LOCAL_REWRITES",
        missing_literals=[], unreduced_tokens=len(remaining),
        reduced_token_sha256=hashlib.sha256(json.dumps(remaining).encode()).hexdigest(),
        native_global_phase_relation="UNQUALIFIED_NOT_ZERO",
        strict_full_basis_eta="NOT_NUMERICALLY_EVALUATED",
        nonidentity_remainder_is_not_operator_inequality_proof=True)


def context_id(batch):
    return batch.get("key", json.dumps(batch.get("request", batch.get("context")), sort_keys=True))


def role_tag(name, tail):
    if name.startswith(("basis:", "basis_inverse:")):
        basis = name.split(":", 1)[1]
        return "SOURCE_RTE_frame" if basis in tail else "D_basis" if basis in ("D0", "D1", "D2", "D3") else "catalogue_only_basis"
    return "saved_D_template"


def main():
    assert not OUT.exists(), "Preserve existing static outputs"
    metadata_raw = blob(META)
    assert hashlib.sha256(metadata_raw).hexdigest() == "396f11326d2a5bc86316d0997c744a315b60a856ce5912a1fab064676f285027"
    meta = json.loads(lzma.decompress(metadata_raw))
    saved = read(S2)
    assert saved["status"] == "S1_S2_FINITE_SAVED_DIAGNOSTIC_COMPLETE"
    frontiers = {r["target_literal"]: r for r in saved["pareto"]}
    assert len(frontiers) == 375
    # Reuse saved IDs ONLY: no Pareto/dominance or numeric guard/T comparison.
    dominated = {k for k, r in frontiers.items() if "fixed_current" not in r["nondominated_ids"]}
    assert len(dominated) == 43
    meta_by_key = {e["target_literal"]: e for e in meta["entries"]}
    words = {}
    for member, r in archive("artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz"):
        words[r["theta_hex"]] = dict(sequence_sha256=r["sequence_sha256"],
            route="fixed_C0", record_key=r["key"], global_W_saved=r["global_W"],
            phase_scope="exact saved sequence including W")
    certificates = read("artifacts/pr_reference_n4b_consumer/2026-11-08/ROLE_WORD_CERTIFICATES.json")
    for r in certificates["rows"]:
        key = r["positive_literal_hex"]; assert key not in words
        words[key] = dict(sequence_sha256=r["positive"]["positive_sequence_sha256"],
            route="fixed_N4_composite", global_W_saved=r["positive"]["global_W_signed"],
            complete_actual_adjoint_saved=r["negative"]["negative_whole_adjoint"],
            phase_scope="exact saved composite including W")
    tail = tuple(read(TR + "STAGE1_STRUCTURE.json")["source_basis_probabilities"])
    groups = dict(Bprime=archive(BP + "Bprime_local_catalogue.tar.xz"),
                  reference=[r for r in archive(TR + "local_catalogue_boundary.tar.xz") if r[1]["request"][0] == "reference"])
    d_names = {"left_all", "right_all", "left_no0", "right_no0", "boundary_control", "block:0:full", *(f"block:{j}:half" for j in range(4))}
    pair_rows, hits = [], defaultdict(list)
    for arm, batches in groups.items():
        for member, batch in batches:
            by_name = {r["name"]: r for r in batch["records"]}
            for bid in (*tail, "D0", "D1", "D2", "D3"):
                f = by_name.get("basis:" + bid); c = by_name.get("basis_inverse:" + bid)
                if f is None or c is None:
                    pair_rows.append(dict(arm=arm, member=member, context=context_id(batch), basis_id=bid,
                                          status="SAVED_PAIR_METADATA_MISSING"))
                    continue
                assert f["n"] == c["n"]
                pair_rows.append(dict(arm=arm, member=member, context=context_id(batch), basis_id=bid,
                    role="SOURCE_RTE_frame" if bid in tail else "D_basis",
                    n=f["n"], forward_name=f["name"], inverse_name=c["name"],
                    post_unroll=batch["post_unroll"],
                    native_global_phase_fields_present=("global_phase" in f and "global_phase" in c),
                    **pair_syntax(f["lowering"], c["lowering"], words)))
            for name, rec in by_name.items():
                if name not in d_names and not name.startswith(("basis:", "basis_inverse:")):
                    continue
                for position, (op, wires, params) in enumerate(rec["lowering"]):
                    if op != "rz":
                        continue
                    literal = float(params[0]).hex(); key = abs(float(params[0])).hex()
                    if key in dominated:
                        hits[key].append(dict(arm=arm, archive_member=member, context=context_id(batch),
                            template=name, role=role_tag(name, tail), native_position=position,
                            wires=wires, signed_literal=literal))
    inverse_hits = []
    for member, batch in archive(VR + "stage_A_raw.tar.xz"):
        for rec in batch["records"]:
            for pos, (op, wires, params) in enumerate(rec["native"]["lowering"]):
                if op == "rz" and abs(float(params[0])).hex() in dominated:
                    inverse_hits.append(dict(member=member, context=batch["context"], template=rec["name"],
                        literal=float(params[0]).hex(), native_position=pos, wires=wires))
    candidates = []
    for key in sorted(dominated):
        entry = meta_by_key[key]
        lookup = {r["archive"] + ":" + r["member"]: r for r in entry["exact_literal_records"]}
        ids = frontiers[key]["nondominated_ids"]
        assert all(i in lookup for i in ids)
        candidates.append(dict(literal=key, current_word=entry["current_word"],
            nondominated_ids_saved=ids, candidate_metadata=[lookup[i] for i in ids],
            saved_template_hits=hits[key], policy_selected=False))
    d_frontiers = [dict(literal=e["target_literal"], roles=[r for r in e["roles"] if r["role"] == "D"],
                         saved_frontier=frontiers[e["target_literal"]])
                   for e in meta["entries"] if any(r["role"] == "D" for r in e["roles"])]
    assert all("fixed_current" in r["saved_frontier"]["nondominated_ids"] for r in d_frontiers)
    result = dict(status="STATIC_S3_AND_SAVED_S2_CROSSWALK_COMPLETE", base_commit=BASE,
        source_RTE_basis_ids=tail, pairs=pair_rows, candidates_43=candidates,
        actual_SOURCE_inverse_D_hits=inverse_hits, D_saved_frontiers_quoted=d_frontiers,
        summaries=dict(pair_statuses=dict(Counter(r["status"] for r in pair_rows)),
            target_literals=43, literals_with_SOURCE_frame_hits=sum(any(h["role"] == "SOURCE_RTE_frame" for h in hits[k]) for k in dominated),
            literals_with_D_hits=sum(any(h["role"] in ("D_basis", "saved_D_template") for h in hits[k]) for k in dominated)),
        no_operator_matrices_or_eta_guards_evaluated=True, S1_S2_reexecuted=False,
        guard_T_arithmetic_or_ranking=False, words_generated_or_selected=False,
        m17_RPE="UNQUALIFIED", STOP="MANDATORY_STOP_STATIC_SCOPE", source_inputs=FILES,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.parent.mkdir(parents=True)
    OUT.write_bytes(lzma.compress(json.dumps(result, indent=2, sort_keys=True).encode()))
    print(json.dumps(dict(status=result["status"], summaries=result["summaries"], actual_inverse_D_hits=len(inverse_hits))))


if __name__ == "__main__":
    main()
