#!/usr/bin/env python3
"""One saved q4-cohort descriptor pass. No circuits, prices or state actions."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
BASE="2f316fb229149117177367720a6eaa5aa8a94b76"
SELECTOR="a54d04059bfc61adef43aa2d40fb9c001b62378f"
FEATURE="64072ba81bb70db1df1b34e94adb08e78926f24b"
C6A="6b6667026ec8b670f610e41141d195883a9eb513"
C6A_PUBLIC="e9d4e2f602d5f9e014507ca7e391719bab017b88"
OLD="artifacts/track_d_c6a_paired_price/2026-11-06/run01"
PRIOR="artifacts/track_d_cut_h4_q4/2026-11-08/run01"
FEATURE_PATH="src/trottertracks/track_d_c7_4_models.py"
FEATURE_SHA="6045e9a1b750494e8863928d119dc5b88448ad259936d0f4870bedec1dbdc33f"
OUT="artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/run01"
TECH="artifacts/track_d_o5_q4_cohort_metadata/2026-11-08/technical"
FILES=("scripts/run_track_d_o5_q4_metadata.py","tests/check_track_d_o5_q4_metadata.py",
       "docs/research/track_d_o5_q4_cohort_metadata_20261108/PLAN.md",TECH+"/fixed_inputs.json",TECH+"/synthetic_tests.json")
REUSED={"src/trottertracks/track_d_cut_selection.py":SELECTOR,
        "src/trottertracks/track_d_cut_artificial.py":SELECTOR,
        "src/trottertracks/track_d_cut_h4_saved.py":BASE}
IDS=tuple(f"q4_sample{i:02d}" for i in range(16))


def sha(data):return hashlib.sha256(data).hexdigest()
def git(*args):return subprocess.check_output(["git","-C",str(ROOT),*args])
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def bootstrap():
    # Reuse the saved-q4 runner's namespace setup and immutable C7-4 loader.
    boot=load("o5_metadata_namespace",ROOT/"scripts/native_df_qiskit_validation_support.py")
    boot.package("trotterlib",ROOT/"src/trotterlib")
    boot.package("trotterlib.df_trotter",ROOT/"src/trotterlib/df_trotter")
    sys.path.insert(0,str(ROOT/"src"))
    from trottertracks import track_d_cut_h4_saved as saved
    from trottertracks import track_d_cut_selection as selector
    data=git("show",FEATURE+":"+FEATURE_PATH)
    if sha(data)!=FEATURE_SHA:raise ValueError("FIXED_FEATURE_SOURCE")
    with tempfile.TemporaryDirectory(prefix="o5-fixed-feature-") as temp:
        path=Path(temp)/"features.py";path.write_bytes(data)
        features=load("trottertracks._o5_q4_fixed_features",path)
    return saved,selector,features


def input_packet(original):
    """Read-only identity pass; no event decode or cut evaluation."""
    original=Path(original);bindings={r["sample_id"]:r for r in read(original/OLD/"event_bindings.json")}
    archives={r["sample_id"]:r for r in read(original/OLD/"payload_archives.json")["archives"]}
    names=[OLD+"/SOURCE.json",OLD+"/STOP.json",OLD+"/payload_archives.json",OLD+"/event_bindings.json"]
    names += [OLD+f"/paths/{sid}/payload_archive.json" for sid in IDS]
    public={}
    for name in names:
        data=(original/name).read_bytes()
        if data!=git("show",C6A_PUBLIC+":"+name):raise ValueError("C6A_PUBLIC_INDEX_CHANGED:"+name)
        public[name]=sha(data)
    if read(original/OLD/"SOURCE.json")["execution_source"]!=C6A:raise ValueError("C6A_SOURCE")
    if not read(original/OLD/"STOP.json")["mandatory_stop"]:raise ValueError("C6A_STOP")
    rows=[]
    for i,sid in enumerate(IDS):
        binding=bindings[sid];archive=archives[sid]
        manifest=read(original/OLD/f"paths/{sid}/payload_archive.json")
        if (binding["q"],binding["sample"],binding["cut_after_complete_step"],manifest["execution_source"])!=(4,i,2,C6A):
            raise ValueError("FIXED_COHORT_BINDING:"+sid)
        path=original/OLD/archive["path"]
        if not path.is_file():raise ValueError("MISSING_ORIGINAL:"+sid)
        if path.stat().st_uid!=os.getuid():raise ValueError("ORIGINAL_NOT_OWNED:"+sid)
        digest=sha(path.read_bytes())
        event=next(r for r in manifest["members"] if r["path"]=="events.json")
        if (digest,path.stat().st_size,len(manifest["members"]))!=(archive["sha256"],archive["bytes"],20):
            raise ValueError("ARCHIVE_HASH_SIZE_MEMBERS:"+sid)
        if manifest["sha256"]!=digest or event["sha256"]!=binding["record_sha256"]:
            raise ValueError("ARCHIVE_EVENT_INDEX:"+sid)
        rows.append(dict(sample_id=sid,archive_path=archive["path"],archive_sha256=digest,
            archive_bytes=archive["bytes"],event_sha256=event["sha256"],event_bytes=event["bytes"],members=20))
    prior={}
    for leaf in ("DECISION.json","SOURCE.json","STOP.json","INPUT_CHECK.json"):
        name=PRIOR+"/"+leaf;data=(ROOT/name).read_bytes()
        if data!=git("show",BASE+":"+name):raise ValueError("PRIOR_SAMPLE00_CHANGED")
        prior[name]=sha(data)
    return dict(C6a_source=C6A,C6a_publication=C6A_PUBLIC,cohort="track_d_c6a_q2_q4_paired_price_pilot_v1",
                public_indices=public,prior_sample00=prior,archives=rows)


def event_only(path,manifest):
    """Thin event-only reader: avoid the old full-payload reader's QPY/NPZ loads."""
    path=Path(path)
    if path.stat().st_size!=manifest["bytes"] or sha(path.read_bytes())!=manifest["sha256"]:
        raise ValueError("ARCHIVE_IDENTITY")
    wanted={r["path"]:r for r in manifest["members"]}
    with tarfile.open(path,"r:gz") as archive:
        members=archive.getmembers()
        if len(members)!=len(wanted) or len({m.name for m in members})!=len(members) or {m.name for m in members}!=set(wanted):
            raise ValueError("MEMBER_SET")
        member=archive.getmember("events.json")
        if not member.isfile():raise ValueError("EVENT_MEMBER_TYPE")
        data=archive.extractfile(member).read()
    if len(data)!=wanted["events.json"]["bytes"] or sha(data)!=wanted["events.json"]["sha256"]:
        raise ValueError("EVENT_MEMBER_IDENTITY")
    return data


def decision_row(sid,decision,features,reused):
    candidates=[dict(cut=c,score=score,left_load=left,right_load=right) for c,score,left,right in decision["candidates"]]
    scores={r["cut"]:r["score"] for r in candidates};best=min(scores.values())
    return dict(sample_id=sid,step_basis_runs=[f["basis_runs"] for f in features],
        step_support_work=[f["support_work"] for f in features],step_loads=list(decision["step_loads"]),
        candidates=candidates,selected_cut=decision["selected"],center_cut=2,cut_delta_from_center=decision["selected"]-2,
        minimum_score_tied_cuts=[c for c in scores if scores[c]==best],
        selection_tie=sum(v==best for v in scores.values())>1,
        any_candidate_score_tie=len(set(scores.values()))<3,
        best_noncentral_minus_center=min(scores[1],scores[3])-scores[2],
        improvement_vs_center=scores[2]-scores[decision["selected"]],
        step_load_range=max(decision["step_loads"])-min(decision["step_loads"]),
        center_half_load_imbalance=abs(sum(decision["step_loads"][:2])-sum(decision["step_loads"][2:])),
        decision_reused=reused,descriptor_sha256=decision["descriptor_sha256"])


def summarize(rows):
    counts=Counter(r["selected_cut"] for r in rows);noncentral=[r["sample_id"] for r in rows if r["selected_cut"]!=2]
    def describe(values):return dict(min=min(values),max=max(values),mean=statistics.mean(values))
    return dict(status="COMPLETE_FIXED_Q4_METADATA_COHORT",paths=len(rows),cuts={str(c):counts[c] for c in (1,2,3)},
        noncentral_count=len(noncentral),noncentral_fraction=len(noncentral)/len(rows),
        noncentral_ids=noncentral,next_B_candidate=min(noncentral) if noncentral else None,
        case="B" if noncentral else "A",selection_ties=sum(r["selection_tie"] for r in rows),
        nonwinning_or_winning_score_ties=sum(r["any_candidate_score_tie"] for r in rows),
        best_noncentral_minus_center=describe([r["best_noncentral_minus_center"] for r in rows]),
        step_load_range=describe([r["step_load_range"] for r in rows]),
        center_half_load_imbalance=describe([r["center_half_load_imbalance"] for r in rows]),
        decision_unit="whole path, same metadata for X/Y; n16, not n32",
        calibration_cohort=True,independent_holdout=False,law_support=False,
        compiled_depth_evaluated=False,score_is_compiled_depth=False,new_fit=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--original",type=Path,required=True);p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    if not args.execute:raise ValueError("EXPLICIT_METADATA_EXECUTION_REQUIRED")
    out=ROOT/OUT;out.mkdir(parents=True,exist_ok=False);local=out/"local";local.mkdir(mode=0o700);(local/".gitignore").write_text("*\n")
    os.environ["MPLCONFIGDIR"]=str(local/"matplotlib-cache")
    rows=[];calls=0;checked=0;feature_calls=0;success=preserved=False;source={}
    try:
        saved,selector,features=bootstrap();head=git("rev-parse","HEAD").decode().strip()
        for name in FILES:
            if (ROOT/name).read_bytes()!=git("show",head+":"+name):raise ValueError("UNFIXED_SOURCE:"+name)
        for name,commit in REUSED.items():
            if (ROOT/name).read_bytes()!=git("show",commit+":"+name):raise ValueError("FIXED_DEPENDENCY_CHANGED:"+name)
        packet=read(ROOT/TECH/"fixed_inputs.json")
        if packet!=input_packet(args.original):raise ValueError("FIXED_ORIGINALS_CHANGED")
        if read(ROOT/TECH/"synthetic_tests.json")["status"]!="PASS":raise ValueError("TESTS_REQUIRED")
        source=dict(execution_source=head,base=BASE,C6a_source=C6A,selector_source=SELECTOR,feature_source=FEATURE,
            feature_sha256=FEATURE_SHA,rule=selector.RULE,files={p:sha((ROOT/p).read_bytes()) for p in (*FILES,*REUSED)},
            input_manifest_sha256=sha((ROOT/TECH/"fixed_inputs.json").read_bytes()),
            paths=list(IDS),q=4,legal_cuts=[1,2,3],center=2,metadata_only=True,prices_accessed=False)
        write(out/"SOURCE.json",source)
        bindings={r["sample_id"]:r for r in read(args.original/OLD/"event_bindings.json")}
        prior=read(ROOT/PRIOR/"DECISION.json")
        for entry in packet["archives"]:
            sid=entry["sample_id"];manifest=read(args.original/OLD/f"paths/{sid}/payload_archive.json")
            data=event_only(args.original/OLD/entry["archive_path"],manifest)
            record=saved.checked_record(data,bindings[sid],C6A,features.ENDPOINT,features.PACKET,features.LAW);checked+=1
            observed=[]
            def capture(step):
                nonlocal feature_calls
                value=features.event_features(step);observed.append(value);feature_calls+=1;return value
            if sid=="q4_sample00":
                projected=selector.descriptors(record["ordered_events"])
                if (prior["rule"]!=selector.RULE or prior["q"]!=4 or prior["center"]!=2
                    or prior["event_sha256"]!=entry["event_sha256"] or selector.identity(projected)!=prior["descriptor_sha256"]):
                    raise ValueError("PRIOR_DECISION_INPUT_BINDING")
                # r_i/w_i were not separately saved; obtain only those missing
                # fields. Reuse the old decision, do not select this cut again.
                for i,step in enumerate(projected):capture(dict(q=1,sample_id=i,ordered_events=[step]))
                if [1+f["basis_runs"]+f["support_work"] for f in observed]!=prior["step_loads"]:
                    raise ValueError("PRIOR_STEP_LOADS")
                decision=prior
            else:
                calls+=1;decision=asdict(selector.select_cut(record["ordered_events"],capture))
            row=decision_row(sid,decision,observed,sid=="q4_sample00");row.update(archive_sha256=entry["archive_sha256"],event_sha256=entry["event_sha256"])
            rows.append(row);write(local/(sid+"_descriptor.json"),dict(decision=decision,step_features=observed))
            write(out/"paths.json",rows);print(sid,"selected",row["selected_cut"],"loads",row["step_loads"],flush=True)
        if len(rows)!=16 or checked!=16 or calls!=15:raise ValueError("COMPLETE_FIXED_COHORT_REQUIRED")
        write(out/"summary.json",summarize(rows));success=True
    except Exception as error:
        text=str(error).replace(str(args.original),"<original>").replace(str(ROOT),"<worktree>")
        write(out/"failure.json",dict(error_type=type(error).__name__,error=text))
    try:
        preserved=bool(source) and read(ROOT/TECH/"fixed_inputs.json")==input_packet(args.original)
        if source:
            preserved=preserved and all(sha((ROOT/p).read_bytes())==h for p,h in source["files"].items())
    except Exception:preserved=False
    write(out/"CALL_COUNTS.json",dict(metadata_paths=checked,new_cut_selections=calls,reused_decisions=int(bool(rows and rows[0]["decision_reused"])),
        fixed_event_features_calls=feature_calls,new_sampling=0,H4_build=0,raw_simulation=0,P0_compile=0,postcheck=0,
        QPY_load=0,NPZ_load=0,saved_price_reanalysis=0,model_fit=0,new_shots=0,full_RPE_update=0,retry=0))
    write(out/"PRIVATE_INDEX.json",dict(members=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
        for p in sorted(local.glob("*_descriptor.json"))],raw_payload_uploaded=False))
    status="METADATA_COMPLETE_MANDATORY_STOP" if success and preserved else "FAIL_MANDATORY_STOP"
    write(out/"PRESERVATION.json",dict(status="PASS" if preserved else "FAIL",original_SOURCE_archives_indices_STOP_unchanged=preserved,
        Q64_intervention=False,SERVER1_results_changed=False,private_events_uploaded=False))
    write(out/"STOP.json",dict(status=status,mandatory_stop=True,next_actor="user/GPT",next_scientific_stage_authorized=False))
    print(status,flush=True)
    return 0 if success and preserved else 1


if __name__=="__main__":raise SystemExit(main())
