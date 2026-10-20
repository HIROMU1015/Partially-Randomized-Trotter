#!/usr/bin/env python3
"""Check documentary artifacts only; never import or execute scientific code.

Reads fixed Git blobs, metadata, text, and cached publication bytes. Writes only
this report's final audit and artifact manifest. It does not recalculate any
scientific value or certify scientific correctness, novelty, or reproducibility.
"""

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT.parents[2]
BASE = "80b7a35c4dbf683f952293eeb55944bc10b9efc8"
BRANCH = "research/pr-prior-art-novelty-20261021"
PREFIX = ROOT.relative_to(WORKTREE).as_posix() + "/"
CACHE = ROOT / "literature_cache"
checks = []


def git(*args):
    return subprocess.check_output(["git", "-C", str(WORKTREE), *args])


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(name, condition, details):
    checks.append({"check": name, "pass": bool(condition), "details": details})


def write_json(name, value):
    (ROOT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    check("branch", git("branch", "--show-current").decode().strip() == BRANCH, BRANCH)
    check("base_is_ancestor", subprocess.run(
        ["git", "-C", str(WORKTREE), "merge-base", "--is-ancestor", BASE, "HEAD"],
        check=False, capture_output=True
    ).returncode == 0, BASE)

    changed = git("diff", BASE, "--name-only", "-z").decode().split("\0")
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all").decode().split("\0")
    status_paths = [s[3:] for s in status if s]
    out_of_scope = [s for s in changed + status_paths if s and not s.startswith(PREFIX)]
    check("write_scope", not out_of_scope, {"outside_scope": out_of_scope,
          "scope": PREFIX, "head_observed": git("rev-parse", "HEAD").decode().strip()})
    old_files = set(git("ls-tree", "-r", "--name-only", BASE).decode().splitlines())
    overwritten = sorted(set(status_paths) & old_files)
    check("no_existing_tracked_file_modified", not overwritten, overwritten)
    ignored = subprocess.run(["git", "-C", str(WORKTREE), "check-ignore", "-q",
                              str(CACHE / "guenther-v2.pdf")], capture_output=True)
    tracked_cache = git("ls-files", "--", PREFIX + "literature_cache/").decode().splitlines()
    check("publication_cache_excluded", ignored.returncode == 0 and not tracked_cache,
          {"ignored": ignored.returncode == 0, "tracked_cache": tracked_cache})

    inventory = read_json("audit/source_inventory.json")
    source_failures = []
    for item in inventory:
        obj = item["commit"] + ":" + item["path"]
        raw = git("show", obj)
        blob = git("rev-parse", obj).decode().strip()
        if (len(raw), digest(raw), blob) != (item["bytes"], item["sha256"], item["git_blob"]):
            source_failures.append(obj)
    check("fixed_source_blobs", not source_failures,
          {"items": len(inventory), "mismatch": source_failures})
    scope = read_json("audit/scope_record.json")
    check("scope_inventory_count", scope["source_items"] == len(inventory), len(inventory))

    catalog = read_json("audit/literature_catalog.json")
    expected_papers = [f"P{i:02d}" for i in range(1, 40)]
    expected_claims = [f"C{i:02d}" for i in range(1, 17)]
    check("literature_catalog_ids", [r["paper_id"] for r in catalog] == expected_papers,
          {"papers": len(catalog)})
    refs = (ROOT / "references.md").read_text(encoding="utf-8")
    check("reference_sections_and_anchors",
          re.findall(r"^## (P\d{2})\.", refs, re.M) == expected_papers and
          re.findall(r'<a id="(p\d{2})"></a>', refs) == [p.lower() for p in expected_papers],
          {"expected": len(expected_papers)})
    for name, key, expected in [("papers_comparison.csv", "paper", expected_papers),
                                ("claim_prior_art_matrix.csv", "claim_id", expected_claims)]:
        with (ROOT / name).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        md = (ROOT / name.replace(".csv", ".md")).read_text(encoding="utf-8")
        md_ids = re.findall(r"^\|\[?([PC]\d{2})(?:\]|\|)", md, re.M)
        check(name + "_ids", [r[key] for r in rows] == expected and
              md_ids == expected, len(rows))

    bib = (ROOT / "references.bib").read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,]+),", bib, re.M)
    depth = 0
    balanced = True
    for ch in re.sub(r"\\[{}]", "", bib):
        depth += (ch == "{") - (ch == "}")
        balanced = balanced and depth >= 0
    check("bibtex_structure", balanced and depth == 0 and len(keys) == len(set(keys))
          and all(p + "_arxiv" in keys for p in expected_papers),
          {"entries": len(keys), "brace_balance": depth, "unique_keys": len(set(keys))})
    check("reading_scopes_explicit", all(r.get("reading_scope") for r in catalog),
          "Acquired pages are distinct from pages read; no whole-paper-read claim.")

    whitespace_errors = []
    for file in sorted(ROOT.iterdir()):
        if file.suffix in {".md", ".csv"}:
            raw = file.read_bytes()
            if b"\r" in raw or any(line.rstrip() != line for line in raw.splitlines()):
                whitespace_errors.append(file.name)
    check("document_line_endings_and_whitespace", not whitespace_errors, whitespace_errors)

    json_failures = []
    json_files = []
    for file in sorted(ROOT.rglob("*.json")):
        if CACHE in file.parents or file.name in {"final_audit.json", "artifact_manifest.json"}:
            continue
        json_files.append(str(file.relative_to(ROOT)))
        try:
            json.loads(file.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            json_failures.append({"file": str(file), "error": str(exc)})
    check("json_structure", not json_failures, {"files": len(json_files), "errors": json_failures})

    # Generated audit files are valid future targets and are written below.
    generated_targets = {ROOT / "audit/final_audit.json", ROOT / "audit/artifact_manifest.json"}
    bad_links = []
    links_checked = 0
    bad_paper_ids = []
    for file in sorted(ROOT.rglob("*.md")):
        if CACHE in file.parents:
            continue
        body = file.read_text(encoding="utf-8")
        bad_paper_ids.extend(sorted(set(re.findall(r"\bP\d{2}\b", body)) - set(expected_papers)))
        for href in re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", body):
            href = href.strip("<>")
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            links_checked += 1
            target = (file.parent / unquote(url.path)).resolve() if url.path else file
            if not target.exists() and target not in generated_targets:
                bad_links.append({"from": str(file.relative_to(ROOT)), "target": href})
            elif url.fragment and target.name == "references.md":
                if f'id="{url.fragment}"' not in refs:
                    bad_links.append({"from": str(file.relative_to(ROOT)), "target": href})
    check("markdown_relative_links", not bad_links,
          {"checked": links_checked, "errors": bad_links})
    check("paper_id_references", not bad_paper_ids, sorted(set(bad_paper_ids)))

    acquisitions = read_json("audit/literature_acquisition.json")
    cache_failures = []
    cache_checked = 0
    for receipt in acquisitions:
        if not receipt.get("ok"):
            continue
        filename = receipt.get("file", receipt.get("local_file"))
        file = CACHE / filename
        cache_checked += 1
        if not file.exists() or digest(file.read_bytes()) != receipt["sha256"]:
            cache_failures.append(filename)
    zenodo = read_json("audit/supplementary_acquisition.json")["zenodo"]
    zfile = CACHE / "guenther-zenodo.zip"
    check("publication_cache_hashes", not cache_failures and zfile.exists() and
          digest(zfile.read_bytes()) == zenodo["sha256"],
          {"successful_receipts_checked": cache_checked, "mismatches": cache_failures,
           "zenodo_checked": True, "failed_acquisitions_retained": sum(not r.get("ok") for r in acquisitions)})
    projection_failures = []
    for item in read_json("audit/metadata_projection.json"):
        file = ROOT / item["raw_cache"]
        if not file.exists() or digest(file.read_bytes()) != item["raw_sha256"]:
            projection_failures.append(item["file"])
    check("metadata_raw_cache_hashes", not projection_failures, projection_failures)
    check("no_abstract_fulltext_in_bibliography_metadata",
          all("citation_abstract" not in r["metadata"] and
              "abstract" not in r.get("journal_metadata", {}) for r in catalog),
          "Only bibliographic fields retained; complete retrieved responses in ignored cache.")

    passed = all(c["pass"] for c in checks)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DOCUMENTARY_CHECKS_PASS" if passed else "DOCUMENTARY_CHECKS_FAIL",
        "scope": "Document structure, immutable blob/cache hashes and write scope only",
        "not_a_scientific_review_or_final_evidence_freeze": True,
        "checks": checks,
        "actions_not_performed": ["new scientific calculation", "sampling", "simulation", "compile",
                                  "statistical reanalysis", "stored-result reranking or re-accounting",
                                  "manuscript writing", "new detailed scientific review",
                                  "intervention in rare-event or separate S4 study", "push"],
        "subagents": 0,
        "termination": "MANDATORY STOP after the requested local commit; no automatic follow-up",
        "commit_identity": "Use Git history; no self-referential commit hash embedded"
    }
    write_json("audit/final_audit.json", result)
    manifest = []
    for file in sorted(ROOT.rglob("*")):
        if not file.is_file() or CACHE in file.parents or file.name == "artifact_manifest.json":
            continue
        raw = file.read_bytes()
        manifest.append({"path": str(file.relative_to(ROOT)), "bytes": len(raw), "sha256": digest(raw)})
    write_json("audit/artifact_manifest.json", {
        "scope": "New documentary deliverables only; excludes ignored cache and this manifest",
        "files": manifest
    })
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "failed": [c for c in checks if not c["pass"]],
                      "manifest_files": len(manifest)}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
