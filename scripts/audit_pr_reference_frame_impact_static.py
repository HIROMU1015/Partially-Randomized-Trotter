"""Metadata/SOURCE crosswalk only. Never compare saved guard/T numbers."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import lzma
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "340670387dbe58bdda5606fdf222408774be1932"
OUT = ROOT / "artifacts/pr_reference_frame_impact/2026-11-08/STATIC_CROSSWALK.json.xz"
S1 = "artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/RESULTS.json"
S3 = "artifacts/pr_reference_s3_s2_static/2026-11-08/STATIC_FINDINGS.json.xz"
META = "artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz"
INPUTS = {}


@lru_cache(None)
def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    INPUTS[path] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw


def read(path):
    raw = blob(path)
    return json.loads(lzma.decompress(raw) if path.endswith(".xz") else raw)


def crosswalk(s3, s1, meta):
    """Literal/ID membership, raw-value quotation, and instruction counts only."""
    entries = {r["target_literal"]:r for r in meta["entries"]}
    tail = tuple(s3["source_RTE_basis_ids"])
    affected = set()
    candidates, grouped = [], defaultdict(list)
    for candidate in s3["candidates_43"]:
        key = candidate["literal"]
        contexts = defaultdict(list)
        for h in candidate["saved_template_hits"]:
            contexts[(h["arm"], h["archive_member"], h["context"], h["template"], h["role"])].append(h)
        rows = []
        for (arm, member, context, template, role), hits in sorted(contexts.items()):
            mapping = "b" if arm == "Bprime" else "ref"
            r = dict(arm=arm, member=member, context=context, template=template, role=role,
                physical_native_occurrences=len(hits),
                sign_counts=dict(Counter(h["signed_literal"] for h in hits)),
                positions_wires_signs=hits, S1_mapping=mapping,
                S1_template_max_quoted=s1["template_weights"][mapping].get(template),
                S1_numeric_key_contribution="NOT_EVALUATED")
            rows.append(r)
            if role == "SOURCE_RTE_frame":
                bid = template.split(":",1)[1]
                assert bid in tail
                affected.add(bid)
                grouped[(arm, bid, template, context)].append(key)
        candidates.append(dict(literal=key, current_word=entries[key]["current_word"],
            roles_saved_metadata=entries[key]["roles"],
            nondominated_ids_saved=candidate["nondominated_ids_saved"],
            candidate_metadata=candidate["candidate_metadata"], contexts=rows,
            selected_policy=False))
    basis_rows = [dict(arm=arm, basis_id=bid, template=template, context=context,
        literal_keys=sorted(set(keys))) for (arm,bid,template,context),keys in sorted(grouped.items())]
    return dict(candidates_43=candidates, affected_SOURCE_basis_ids=sorted(affected),
        unaffected_SOURCE_basis_ids=sorted(set(tail)-affected),
        SOURCE_template_context_bindings=basis_rows,
        all_SOURCE_basis_ids=tail,
        saved_S1_rows_quoted=[r for r in s1["rows"] if r["variant"]=="weighted_SOURCE_RTE_basis"],
        saved_S1_argmax_witness="NOT_SAVED; new numeric comparisons required, NOT evaluated",
        guard_T_numeric_comparison=False, guard_T_numeric_aggregation=False,
        S1_S2_B1_reexecuted=False, word_policy_applied=False)


def main():
    assert not OUT.exists(), "Preserve prior static output"
    s3, s1, meta = read(S3), read(S1), read(META)
    assert s3["status"] == "STATIC_S3_AND_SAVED_S2_CROSSWALK_COMPLETE"
    assert s1["status"] == "S1_SCOPED_COVERAGE_ONE_SHOT_COMPLETE"
    assert len(s3["candidates_43"]) == 43
    # The saved S1 schema contains scalars, not a winning basis/context ID.
    assert set(s1["template_weights"]) == {"b", "ref", "inv"}
    assert all(v is None or isinstance(v,str) for a in s1["template_weights"].values() for v in a.values())
    out = crosswalk(s3,s1,meta)
    out.update(status="FRAME_IMPACT_STATIC_PREPARATION_COMPLETE", base_commit=BASE,
        SOURCE_inputs=INPUTS, runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numerical_runner_executed=False, STOP="MANDATORY_STOP_BEFORE_SCIENTIFIC_NUMERICS")
    OUT.parent.mkdir(parents=True)
    OUT.write_bytes(lzma.compress(json.dumps(out,sort_keys=True,indent=2).encode()))
    print(json.dumps(dict(status=out["status"],literal_keys=len(out["candidates_43"]),
        affected_basis=out["affected_SOURCE_basis_ids"],unaffected_basis=out["unaffected_SOURCE_basis_ids"],
        template_context_bindings=len(out["SOURCE_template_context_bindings"]),
        argmax=out["saved_S1_argmax_witness"])))


if __name__ == "__main__": main()
