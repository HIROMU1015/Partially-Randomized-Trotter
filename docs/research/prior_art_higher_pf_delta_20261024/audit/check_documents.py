#!/usr/bin/env python3
"""Check documentary delta only. No scientific imports, analysis, or figures.

Reuse the fixed 2026-10-21 checker helpers; adapt its branch/catalog/write-scope
checks to this small delta. Do not re-audit all old publication caches/runtimes.
"""
import csv
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT.parents[2]
PREFIX = ROOT.relative_to(WORKTREE).as_posix() + "/"
BASE = "09e71910ee49e0d9d3a2f2f410249f6623022c0d"
OLD = "501b4232b3e9ef9778e917f0a86f1d6c1d05f189"
OLD_DIR = "docs/research/prior_art_novelty_20261021/"
old_source = subprocess.check_output(["git", "show", OLD + ":" + OLD_DIR + "audit/check_documents.py"], cwd=WORKTREE)
# Load helpers without running the previous checker's old-branch/cache main().
ns = {"__name__": "reused_document_check_helpers", "__file__": str(WORKTREE / OLD_DIR / "audit/check_documents.py")}
exec(compile(old_source, "<fixed-document-checker-501b4232>", "exec"), ns)
git, digest, check = ns["git"], ns["digest"], ns["check"]


def read_json(name):
    return json.loads((ROOT / name).read_text(), parse_float=str)


def rows(name):
    return list(csv.DictReader(io.StringIO((ROOT / name).read_text())))


def main():
    changes = [x for x in git("diff", BASE, "--name-only", "-z").decode().split("\0") if x]
    new = [x for x in git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0") if x]
    check("new_documentary_directory_only", all(x.startswith(PREFIX) for x in changes + new),
          "Existing SOURCE/results/adoption/manuscript/figures/grants/STOP not edited")
    # No existing tracked path from the base can have been rewritten.
    base_paths = set(git("ls-tree", "-r", "--name-only", BASE).decode().splitlines())
    check("no_existing_tracked_file_changed", not (set(changes) & base_paths), sorted(set(changes) & base_paths))

    catalog = read_json("audit/source_catalog.json")
    errors = []
    for key, ref in catalog.items():
        original = git("show", ref["commit"] + ":" + ref["path"])
        local = WORKTREE / ref["path"]
        if digest(original) != ref["sha256"] or (local.exists() and local.read_bytes() != original):
            errors.append(key)
    check("fixed_research_reference_binding", not errors, {"references": len(catalog), "mismatch": errors})

    literature = read_json("audit/literature_delta_catalog.json")
    paper_ids = [r["paper_id"] for r in literature]
    old_catalog = json.loads(git("show", OLD + ":" + OLD_DIR + "audit/literature_catalog.json"))
    old_by_id = {r["paper_id"]: r for r in old_catalog}
    check("old_bibliographic_identity_reused", all(
        r["title"] == old_by_id[r["paper_id"]]["metadata"]["citation_title"][0]
        and r["arxiv"] == old_by_id[r["paper_id"]]["id"] + old_by_id[r["paper_id"]]["version_fetched"]
        and r["authors"] == old_by_id[r["paper_id"]]["metadata"]["citation_author"]
        for r in literature), paper_ids)
    comparisons = rows("papers_comparison.csv")
    check("ten_papers_twelve_comparison_dimensions", len(comparisons) == 10
          and [r["paper_id"] for r in comparisons] == paper_ids
          and all(len(r) == 13 and all(r.values()) for r in comparisons), "Bibliography is separate; all 12 dimensions populated")
    claims = rows("claim_prior_art_matrix_delta.csv")
    old_claims = {r["claim_id"] for r in csv.DictReader(io.StringIO(git("show", OLD + ":" + OLD_DIR + "claim_prior_art_matrix.csv").decode()))}
    check("C12_C15_C16_delta_not_old_row_rewrite", {"C12", "C15", "C16"}.issubset({r["old_claim_id"] for r in claims})
          and all(not r["old_claim_id"] or r["old_claim_id"] in old_claims for r in claims), "Old rows remain at the fixed old commit")
    check("candidate_not_final_novelty_adoption", all(r["novelty_status"] in
          ["CANDIDATE_NOT_FORMALLY_ADOPTED_AS_NOVELTY", "UNSUPPORTED_NOT_ADOPTED"] for r in claims), len(claims))
    check("matrix_references_resolve", all(set(r["fixed_evidence"].split(";")) <= set(catalog) for r in claims)
          and all(set(re.findall(r"\bP\d{2}\b", r["direct_prior_art"])) <= set(paper_ids) for r in claims), "Fixed source and literature IDs")

    scope = read_json("audit/scope_record.json")
    original_ref = catalog["ORIGINAL"]
    original = json.loads(git("show", original_ref["commit"] + ":" + original_ref["path"]), parse_float=str)
    check("saved_ratios_and_policy_literal_only", scope["saved_ratios"] == {
          s: original["results"][s]["comparison"]["ratio_Y4_over_S2"] for s in ["H4", "H6"]}
          and scope["evidence_policy_saved"] == original["policy"], "No G/SE/CI/statistical recalculation")
    cem_ref = catalog["CEM"]
    adopted_cem = list(csv.DictReader(io.StringIO(git("show", cem_ref["commit"] + ":" + cem_ref["path"]).decode())))
    check("adopted_CEM_claim_ids_join", set(scope["adopted_claim_ids"]) <= {r["id"] for r in adopted_cem}, scope["adopted_claim_ids"])
    check("scope_not_escalated", scope["new_scientific_calls"] == scope["new_numerical_reanalysis"]
          == scope["new_figures"] == scope["new_grants"] == 0 and not scope["formal_research_direction_adopted"]
          and not scope["detailed_GPT_Claude_review_started"] and not scope["final_evidence_freeze"]
          and scope["old_strict_COMPARE"] == "BLOCKED" and scope["mandatory_STOP"], "Unchanged scientific authority and STOP")
    policy = read_json("audit/policy_version.json")
    check("unacquired_formal_instruction_not_claimed_read", not policy["user_reported_formal_original_acquired"]
          and not policy["local_formal_file_changed"] and not policy["local_10_3_present"], "Current user-supplied principle distinguished from unavailable original")

    bib = (ROOT / "references.bib").read_text()
    keys = re.findall(r"^@\w+\{([^,]+),", bib, re.M)
    depth, balanced = 0, True
    for ch in re.sub(r"\\[{}]", "", bib):
        depth += (ch == "{") - (ch == "}")
        balanced = balanced and depth >= 0
    check("reused_BibTeX_structure", balanced and depth == 0 and len(keys) == len(set(keys))
          and all(p + "_arxiv" in keys for p in paper_ids), {"entries": len(keys), "balance": depth})
    link_errors = []
    generated = {"audit/validation_summary.json", "audit/artifact_manifest.json"}
    for file in ROOT.rglob("*.md"):
        for href in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", file.read_text()):
            url = urlsplit(href.strip("<>"))
            if url.scheme or url.netloc:
                continue
            target = file.parent / unquote(url.path)
            if not target.exists() and target.relative_to(ROOT).as_posix() not in generated:
                link_errors.append({"source": file.name, "target": href})
    check("relative_document_links", not link_errors, link_errors)
    all_text = "\n".join(p.read_text() for p in ROOT.rglob("*.md"))
    check("key_scientific_distinctions_explicit", all(t in all_text for t in [
        "m5_best", "同一q", "異なるq", "探索", "独立", "NOT_CERTIFIED", "母平均", "P=0", "K4", "Figure2"]),
        "Author checks wording and fixed-source roles; not an independent scientific review")
    private_pattern = re.compile("/" + "home/" + r"[^\s\"<>]+|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
    private_hits = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()
                    and private_pattern.search(p.read_text())]
    check("public_document_scope_no_private_paths_or_keys", not private_hits, private_hits)
    passed = all(r["pass"] for r in ns["checks"])
    summary = {"status": "PASS" if passed else "FAIL", "scope": "AUTHOR_DOCUMENTARY_CHECKS_ONLY",
               "reused_checker": {"commit": OLD, "path": OLD_DIR + "audit/check_documents.py",
                                  "functions": ["git", "digest", "check"],
                                  "adaptation": "Small delta catalog and scope; old whole-cache/main checks not rerun"},
               "checks": ns["checks"], "new_scientific_calls": 0, "new_numerical_reanalysis": 0,
               "independent_scientific_review": False, "mandatory_STOP": True}
    (ROOT / "audit/validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    inventory = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p.read_bytes())} for p in sorted(ROOT.rglob("*"))
                 if p.is_file() and p.name != "artifact_manifest.json" and "__pycache__" not in p.parts]
    (ROOT / "audit/artifact_manifest.json").write_text(json.dumps({"files": inventory}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "checks": len(ns["checks"]),
                      "failed": [c for c in ns["checks"] if not c["pass"]]}, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
