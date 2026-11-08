"""Static inverse/D literal bindings. No guard/T arithmetic or word tests."""
from collections import Counter
from functools import lru_cache
import hashlib
import io
import json
import lzma
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
BASE="06895797e032bc236562da5ee541b33ce8eb5050"
OUT=ROOT/"artifacts/pr_reference_inverse_d/2026-11-08/STATIC_BINDINGS.json.xz"
META="artifacts/pr_reference_noncentral_reuse/2026-11-08/RECORD_METADATA.json.xz"
S2="artifacts/pr_reference_noncentral_saved_numeric/2026-11-08/RESULTS.json"
S3="artifacts/pr_reference_s3_s2_static/2026-11-08/STATIC_FINDINGS.json.xz"
GF="artifacts/pr_reference_frame_impact/2026-11-08/numeric/RESULTS.json"
BP="artifacts/pr_reference_bprime/2026-11-07/Bprime_local_catalogue.tar.xz"
REF="artifacts/pr_reference_transfer_qualification/2026-11-06/local_catalogue_boundary.tar.xz"
INV="artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/stage_A_raw.tar.xz"
D_NAMES=dict(b=("left_all","right_all","left_no0","right_no0","boundary_control"),
    ref=("block:0:full","block:0:half","block:1:half","block:2:half","block:3:half"))
D_NAMES["inv"]=D_NAMES["ref"]
INPUTS={}


@lru_cache(None)
def blob(path):
    raw=subprocess.check_output(["git","show",BASE+":"+path],cwd=ROOT)
    INPUTS[path]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return raw


def read(path):
    raw=blob(path)
    return json.loads(lzma.decompress(raw) if path.endswith(".xz") else raw)


def archive(path):
    with tarfile.open(fileobj=io.BytesIO(blob(path)),mode="r:xz") as t:
        return [(m.name,json.load(t.extractfile(m))) for m in t if m.isfile()]


def native_binding(mapping,archive_path,member,batch,rec,role):
    native=rec["native"] if mapping=="inv" else rec
    context=batch.get("key",batch.get("context",batch.get("request")))
    if not isinstance(context,str):context=json.dumps(context,sort_keys=True)
    rz=[dict(native_position=i,wires=qs,signed_literal=float(p[0]).hex(),
        positive_literal=abs(float(p[0])).hex()) for i,(op,qs,p) in enumerate(native["lowering"]) if op=="rz"]
    return dict(mapping=mapping,archive=archive_path,member=member,context=context,
        template=rec["name"],role=role,n=native["n"],ordered_RZs=rz,
        physical_key_multiplicities=dict(Counter(r["positive_literal"] for r in rz)),
        native_instruction_stream_sha256=hashlib.sha256(json.dumps(native["lowering"],sort_keys=True).encode()).hexdigest(),
        native_global_phase="NOT_SAVED_NOT_ZERO",
        SOURCE_global_phase_fields_quoted={k:rec[k] for k in ("forward_global_phase","inverse_global_phase") if k in rec},
        post_unroll_quoted=batch.get("post_unroll"))


def pool_metadata(entry,frontier,current_detail):
    """Use saved membership/sequence equality only; do not order guard or T."""
    retained="fixed_current" in frontier["nondominated_ids"]
    return dict(target_literal=entry["target_literal"],roles_quoted=entry["roles"],
        current_word=entry["current_word"],current_details_quoted=current_detail,
        exact_literal_records=entry["exact_literal_records"],
        nondominated_ids_saved=frontier["nondominated_ids"],current_retained_saved=retained,
        exact_backend_record_count=len(entry["exact_literal_records"]),
        frontier_distinct_sequence_count=len({r["sequence_sha256"] for r in frontier["offers"] if r["id"] in frontier["nondominated_ids"]}),
        smaller_guard_or_higher_T_direction="NOT_NEWLY_COMPARED",
        no_free_guard_T_dominance="Follows saved current-nondominated result, within the saved two-objective pool only" if retained else "See saved dominance result")


def main():
    assert not OUT.exists()
    meta={e["target_literal"]:e for e in read(META)["entries"]}
    s2={e["target_literal"]:e for e in read(S2)["pareto"]}
    s3=read(S3);ids=tuple(s3["source_RTE_basis_ids"])
    gf=read(GF);g=next(p for p in gf["policies"] if p["policy"]=="G")
    assert next(p for p in gf["policies"] if p["policy"]=="NT")["identical_to_policy"]=="G"
    c0={r["theta_hex"]:r for _,r in archive("artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz")}
    certs={r["positive_literal_hex"]:r for r in read("artifacts/pr_reference_n4b_consumer/2026-11-08/ROLE_WORD_CERTIFICATES.json")["rows"]}
    rows=[]
    for mapping,path in (("b",BP),("ref",REF),("inv",INV)):
        for member,batch in archive(path):
            if mapping=="ref" and batch["request"][0]!="reference":continue
            for rec in batch["records"]:
                name=rec["name"]
                is_frame=name.startswith(("basis:","basis_inverse:")) and name.split(":",1)[1] in ids
                if is_frame or name in D_NAMES[mapping]:
                    role=("frame_inverse" if name.startswith("basis_inverse:") else "frame_forward") if is_frame else "D"
                    rows.append(native_binding(mapping,path,member,batch,rec,role))
    inverse={k for r in rows if r["role"]=="frame_inverse" for k in r["physical_key_multiplicities"]}
    forward={k for r in rows if r["role"]=="frame_forward" for k in r["physical_key_multiplicities"]}
    dkeys={k for r in rows if r["role"]=="D" for k in r["physical_key_multiplicities"]}
    assert len(inverse)==109 and len(dkeys)==164
    assert not set(g["choices"]) & (inverse|dkeys)
    keys=sorted(inverse|forward|dkeys)
    pools=[]
    for key in keys:
        current=meta[key]["current_word"]
        if current["kind"]=="C0_fixed_priority":
            r=c0[key]
            detail={field:r[field] for field in ("key","spec","sequence_sha256","sequence","strict_error_upper","guard_method","T","Tdagger","global_W")}
        else:
            detail=certs[key]
        pools.append(pool_metadata(meta[key],s2[key],detail))
    basis_rows=[dict(basis_id=bid,basis_hash_suffix=bid[len("df-basis-"):],
        definition="DFBasisRegistry ordered_local_basis_operations_v1, 8 system qubits; frozen registry in prior common.pkl.xz",
        SOURCE_operations_not_materialized=True,
        records=[r for r in rows if r["role"].startswith("frame_") and r["template"].split(":",1)[1]==bid]) for bid in ids]
    result=dict(status="INVERSE_D_STATIC_PREPARATION_COMPLETE",base_commit=BASE,
        SOURCE_basis=basis_rows,D_template_records=[r for r in rows if r["role"]=="D"],key_pools=pools,
        fixed_keysets=dict(D1_inverse=sorted(inverse),D2_D=sorted(dkeys),
            inverse_D_shared=sorted(inverse&dkeys),inverse_forward_shared=sorted(inverse&forward)),
        quoted_G_forward_choices=g["choices"],quoted_G_rows=g["rows"],
        previous_S3_pair_statuses_quoted=s3["summaries"]["pair_statuses"],
        summaries=dict(inverse_keys=len(inverse),D_keys=len(dkeys),shared=len(inverse&dkeys),
            inverse_current_kinds=dict(Counter(meta[k]["current_word"]["kind"] for k in inverse)),
            inverse_no_exact_backend_records=sum(not meta[k]["exact_literal_records"] for k in inverse),
            inverse_current_frontier_retained=sum("fixed_current" in s2[k]["nondominated_ids"] for k in inverse),
            D_current_frontier_retained=sum("fixed_current" in s2[k]["nondominated_ids"] for k in dkeys)),
        science_numeric_reanalysis=False,guard_T_comparisons=False,weighted_template_called=False,
        previous_88_pair_check_reexecuted=False,new_word_policy_selected=False,
        SOURCE_inputs=INPUTS,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        STOP="MANDATORY_STOP_BEFORE_D1_D2_D3")
    OUT.parent.mkdir(parents=True)
    OUT.write_bytes(lzma.compress(json.dumps(result,sort_keys=True,indent=2).encode()))
    print(json.dumps(dict(status=result["status"],summaries=result["summaries"],shared_keys=result["fixed_keysets"]["inverse_D_shared"])))


if __name__=="__main__":main()
