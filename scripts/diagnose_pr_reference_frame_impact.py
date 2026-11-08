"""PREPARED, NOT EXECUTED: saved-guard/T numerical diagnosis needs approval.

Fixed finite G/T/NT metadata substitutions only; no synthesis, circuit, guard,
Pareto recomputation, RPE or total-T optimization. Artificial tests call pure
functions with artificial numbers, never main() or saved scientific operands.
"""
from collections import Counter, defaultdict
from fractions import Fraction as F
from functools import lru_cache
import ast
import copy
import hashlib
import io
import json
import lzma
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = "340670387dbe58bdda5606fdf222408774be1932"
OUT = ROOT / "artifacts/pr_reference_frame_impact/2026-11-08/numeric"
S1 = "artifacts/pr_reference_noncentral_s1_coverage_fix/2026-11-08/science/run01/RESULTS.json"
S2 = "artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/RESULTS.json"
META = "artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz"
S3 = "artifacts/pr_reference_s3_s2_static/2026-11-08/STATIC_FINDINGS.json.xz"
LEGACY = "scripts/diagnose_pr_reference_noncentral_s1_coverage_fix.py"
POLICIES = ("G", "T", "NT")
INPUTS = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE+":"+path], cwd=ROOT)
    INPUTS[path] = dict(sha256=sha(raw), bytes=len(raw))
    return raw


def read(path):
    raw = blob(path)
    return json.loads(lzma.decompress(raw) if path.endswith(".xz") else raw)


def load_arithmetic():
    """Reuse coverage-corrected S1 arithmetic without its main/evaluate/STOP."""
    raw = blob(LEGACY)
    tree = ast.parse(raw)
    names = {"sha", "fixed_blob", "load_legacy", "required_templates", "scoped_parts"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    ns = dict(Counter=Counter, defaultdict=defaultdict, F=F, lru_cache=lru_cache,
        hashlib=hashlib, io=io, json=json, lzma=lzma, Path=Path, subprocess=subprocess,
        tarfile=tarfile, ROOT=ROOT, ast=ast, copy=copy)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ("SOURCE_COMMIT","OLD_SOURCE","OLD_HELPER","INPUT_BASE"):
                ns[node.targets[0].id] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),BASE+":"+LEGACY,"exec"),ns)
    legacy, arms, evidence = ns["load_legacy"]()
    # Resolve saved inputs at the current fixed base, not a mutable worktree.
    legacy["BASE"] = BASE
    return ns, legacy, arms, evidence


def choose_offers(frontier, mode):
    """Per-key finite rules, not a global optimizer or a Pareto recalculation."""
    current = next(r for r in frontier["offers"] if r["id"]=="fixed_current")
    ids = frontier["nondominated_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Missing/duplicate saved frontier IDs")
    lookup = {r["id"]:r for r in frontier["offers"]}
    if len(lookup)!=len(frontier["offers"]) or any(i not in lookup for i in ids):
        raise ValueError("Invalid saved offer references")
    offers = [lookup[i] for i in ids]
    for r in [current]+offers:
        if F(r["guard"])<0 or not isinstance(r["T"],int) or r["T"]<0:
            raise ValueError("Invalid saved guard/T")
    if mode == "NT":
        offers = [r for r in offers if r["T"]<=current["T"]] + [current]
    if mode in ("G","NT"):
        return min(offers,key=lambda r:(F(r["guard"]),r["T"],r["id"]))
    if mode == "T":
        return min(offers,key=lambda r:(r["T"],F(r["guard"]),r["id"]))
    raise ValueError("Unregistered finite policy")


def context_details(batches, guards, prices, weighted, target_keys):
    details = []
    for mapping, groups in batches.items():
        for batch in groups:
            context = batch.get("key",batch.get("context",batch.get("request")))
            if not isinstance(context,str): context=json.dumps(context,sort_keys=True)
            for rec in batch["records"]:
                native=rec["native"] if mapping=="inv" else rec
                counts=Counter(abs(float(p[0])).hex() for op,qs,p in native["lowering"] if op=="rz")
                guard=weighted([counts],guards)
                price=None if any(k not in prices for k in counts) else sum(n*prices[k] for k,n in counts.items())
                details.append(dict(mapping=mapping,template=rec["name"],context=context,
                    logical_guard=guard, local_template_T_from_saved_prices=price,
                    target_key_contributions={k:dict(multiplicity=counts[k],
                        guard=None if k not in guards else counts[k]*guards[k],
                        local_T=None if k not in prices else counts[k]*prices[k])
                        for k in sorted(set(counts)&set(target_keys))}))
    return details


def witnesses(weights, details, mapping, ids):
    """New scientific argmax evaluation: invoked only after future approval."""
    per_template={}
    for bid in ids:
        for stem in ("basis:","basis_inverse:"):
            name=stem+bid; value=weights[mapping].get(name)
            if value is None: return None
            contexts=[r for r in details if r["mapping"]==mapping and r["template"]==name]
            if not contexts or any(r["logical_guard"] is None for r in contexts): return None
            assert max(r["logical_guard"] for r in contexts)==value
            per_template[name]=dict(maximum=value,
                tied_contexts=[r["context"] for r in contexts if r["logical_guard"]==value])
    pairs={bid:weights[mapping]["basis:"+bid]+weights[mapping]["basis_inverse:"+bid] for bid in ids}
    if not pairs: raise ValueError("Empty SOURCE alphabet")
    maximum=max(pairs.values())
    return dict(pair_maximum=maximum,tied_basis_ids=[bid for bid in ids if pairs[bid]==maximum],
        per_basis_pair=pairs,per_template_context_witnesses=per_template,
        semantics="Separate context maxima for forward and inverse, then sum; NOT paired-context max")


def save(name,data):
    (OUT/name).write_text(json.dumps(data,sort_keys=True,indent=2,
        default=lambda v:str(v) if isinstance(v,F) else (_ for _ in ()).throw(TypeError(type(v).__name__)),
        allow_nan=False)+"\n")


def main():
    # This entrypoint is the unapproved science boundary in this preparation.
    assert not OUT.exists(), "Fresh diagnostic output required; preserve prior results"
    started=time.monotonic()
    before=sha(Path(__file__).read_bytes())
    s1,s2,meta,s3=read(S1),read(S2),read(META),read(S3)
    assert s1["status"]=="S1_SCOPED_COVERAGE_ONE_SHOT_COMPLETE"
    assert s2["status"]=="S1_S2_FINITE_SAVED_DIAGNOSTIC_COMPLETE"
    targets=tuple(sorted(r["literal"] for r in s3["candidates_43"]))
    assert len(targets)==43 and len(set(targets))==43
    saved_frontier={r["target_literal"]:r for r in s2["pareto"]}
    assert len(saved_frontier)==375
    assert all("fixed_current" not in saved_frontier[k]["nondominated_ids"] for k in targets)
    for r in s3["candidates_43"]:
        assert r["nondominated_ids_saved"]==saved_frontier[r["literal"]]["nondominated_ids"]
    control,legacy,arms,evidence=load_arithmetic()
    # Metadata frontier only; no word/provider/consumer is instantiated.
    c0path="artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz"
    c0=legacy["archive"](c0path)
    assert sha(legacy["blob"](c0path))==meta["source_inputs"][c0path]["sha256"]
    guards={r["theta_hex"]:F(r["strict_error_upper"]) for r in c0}
    prices={r["theta_hex"]:r["T"] for r in c0}
    for e in meta["entries"]:
        current=e["current_word"]
        if current["kind"]=="N4_85_composite":
            guards[e["target_literal"]]=F(current["Q_saved"])+F(current["G_saved"])
            prices[e["target_literal"]]=current["existing_T_quoted"]
    for key in targets:
        current=next(r for r in saved_frontier[key]["offers"] if r["id"]=="fixed_current")
        assert F(current["guard"])==guards[key] and current["T"]==prices[key]
    allowed_changes={"b":set(),"ref":set(),"inv":set()}
    for entry in s3["candidates_43"]:
        for hit in entry["saved_template_hits"]:
            if hit["role"]=="SOURCE_RTE_frame":
                allowed_changes["b" if hit["arm"]=="Bprime" else "ref"].add(hit["template"])
    assert all(name.startswith("basis:") for names in allowed_changes.values() for name in names)
    ids=tuple(s3["source_RTE_basis_ids"])
    alphabet={arm:ids for arm in ("Bprime","reference")}
    required={}
    for arm in alphabet: required.update(control["required_templates"](arm,ids))
    batches=dict(b=legacy["archive"]("artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz"),
        ref=[r for r in legacy["archive"]("artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz") if r["request"][0]=="reference"],
        inv=legacy["archive"]("artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/stage_A_raw.tar.xz"))
    batches={key:[dict(batch,records=[r for r in batch["records"] if r["name"] in required[key]])
        for batch in groups] for key,groups in batches.items()}
    saved_condition=legacy["read"]("artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json")["condition"]
    assert (saved_condition["m"],saved_condition["q"],saved_condition["r"],saved_condition["bits"])==(17,131072,32,88)
    assert {a:r["saved_shots_per_axis"] for a,r in saved_condition["arms"].items()}=={"Bprime":626,"reference":630}
    quoted=[r for r in s1["rows"] if r["variant"]=="weighted_SOURCE_RTE_basis"]
    assert len(quoted)==4
    baseline={arm:next(r for r in quoted if r["arm"]==arm and r["axis"]=="X") for arm in alphabet}
    for arm in alphabet:
        assert F.from_float(saved_condition["arms"][arm]["joint_synthesis_error_retaining_saved_N_inclusive_upper"])==F(baseline[arm]["saved_joint_budget"])
    OUT.mkdir(parents=True)
    save("INPUTS.json",dict(base_commit=BASE,runner_sha256=before,saved_inputs=INPUTS,
        legacy_inputs=legacy["INPUTS"],reused_arithmetic=evidence,targets=targets,
        policies=POLICIES,fixed_condition=saved_condition,
        baseline_aggregate_recomputed=False,frontier_recomputed=False))
    try:
        weighted=legacy["weighted_template_max"]
        current_details=context_details(batches,guards,prices,weighted,targets)
        saved_weights={m:{name:None if v is None else F(v) for name,v in values.items()}
            for m,values in s1["template_weights"].items()}
        current_witness={arm:witnesses(saved_weights,current_details,"b" if arm=="Bprime" else "ref",ids)
            for arm in alphabet}
        if any(v is None for v in current_witness.values()): raise ValueError("Current SOURCE coverage missing")
        outcomes=[];cache={}
        for mode in POLICIES:
            choices={k:choose_offers(saved_frontier[k],mode) for k in targets}
            signature=tuple(choices[k]["id"] for k in targets)
            if signature in cache:
                outcomes.append(dict(policy=mode,identical_to_policy=cache[signature]));continue
            cache[signature]=mode
            g,p=dict(guards),dict(prices)
            for k,offer in choices.items():g[k]=F(offer["guard"]);p[k]=offer["T"]
            weights={m:legacy["template_weights"](v,g,inverse=m=="inv") for m,v in batches.items()}
            for mapping,values in weights.items():
                for name,value in values.items():
                    if name not in allowed_changes[mapping]:
                        assert value==saved_weights[mapping][name],"Unexpected unchanged-template impact"
            details=context_details(batches,g,p,weighted,targets)
            rows=[]
            for arm in alphabet:
                case=control["scoped_parts"](131072,32,weights["b"],weights["ref"],weights["inv"],alphabet,arm,arms)
                if case["missing_templates"] or case["parts"] is None:raise ValueError("Required SOURCE coverage missing")
                parts=case["parts"]
                for role in ("D","products","CSWAP"):
                    assert parts[role]==F(baseline[arm]["parts"][role]),"Unexpected nonframe impact: stop scope"
                total=sum(parts.values(),F(0));budget=F(baseline[arm]["saved_joint_budget"])
                witness=witnesses(weights,details,"b" if arm=="Bprime" else "ref",ids)
                for axis in "XY":
                    rows.append(dict(arm=arm,axis=axis,parts=parts,noncentral_upper=total,
                        saved_joint_budget=budget,residual=budget-total,
                        frame_change_from_saved=parts["frames"]-F(baseline[arm]["parts"]["frames"]),
                        frame_witness=witness,m17_RPE="UNQUALIFIED",
                        central_domain_emitter_shell_terminal="UNQUALIFIED_NOT_ZERO"))
            outcomes.append(dict(policy=mode,choices={k:dict(record_id=v["id"],sequence_sha256=v["sequence_sha256"],
                guard=F(v["guard"]),T_saved=v["T"],guard_change=F(v["guard"])-guards[k],
                T_change_per_occurrence=v["T"]-prices[k]) for k,v in choices.items()},
                rows=rows,context_details=details,new_policy_adopted=False))
        assert sha(Path(__file__).read_bytes())==before
        save("RESULTS.json",dict(status="FINITE_FRAME_SAVED_DIAGNOSTIC_COMPLETE",baseline_rows_quoted=quoted,
            current_context_details=current_details,current_witness=current_witness,policies=outcomes,
            full_RPE="UNQUALIFIED",global_T="NOT_EVALUATED",frontier_recomputed=False,
            SOURCE=before,elapsed_seconds=time.monotonic()-started))
        save("STOP.json",dict(status="MANDATORY_STOP",reason="Fixed G/T/NT frame diagnostic complete; no expansion",next_owner="GPT/user"))
    except Exception as e:
        save("FAILURE.json",dict(type=type(e).__name__,message=str(e)))
        save("STOP.json",dict(status="MANDATORY_STOP",reason="Fixed diagnostic unqualified; preserve outputs",next_owner="GPT/user"))
        raise


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError("Existing diagnostic output preserved; no automatic rerun")
    try:
        main()
    except Exception as error:
        # Also preserve early input/spec failures; never replace an old STOP.
        if not (OUT/"STOP.json").exists():
            OUT.mkdir(parents=True,exist_ok=True)
            save("FAILURE.json",dict(type=type(error).__name__,message=str(error)))
            save("STOP.json",dict(status="MANDATORY_STOP",reason="Fixed input/preparation failure",next_owner="GPT/user"))
        raise
