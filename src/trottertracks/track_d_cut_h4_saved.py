"""Lossless saved-payload binding only; no sampling, circuits or prices."""
import hashlib
import json
from pathlib import Path
import tarfile


def sha(data):return hashlib.sha256(data).hexdigest()


def checked_payload(path,manifest):
    data=Path(path).read_bytes()
    if len(data)!=manifest["bytes"] or sha(data)!=manifest["sha256"]:
        raise ValueError("ARCHIVE_IDENTITY")
    wanted={row["path"]:row for row in manifest["members"]}
    if len(wanted)!=len(manifest["members"]):raise ValueError("MANIFEST_DUPLICATE")
    with tarfile.open(path,"r:gz") as tar:
        members=tar.getmembers()
        if len(members)!=len(wanted) or {m.name for m in members}!=set(wanted):
            raise ValueError("MEMBER_SET")
        out={}
        for member in members:
            if not member.isfile() or member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError("MEMBER_TYPE_OR_PATH")
            content=tar.extractfile(member).read();row=wanted[member.name]
            if len(content)!=row["bytes"] or sha(content)!=row["sha256"]:
                raise ValueError("MEMBER_IDENTITY:"+member.name)
            out[member.name]=content
    return out


def checked_record(data,binding,source,endpoint,packet,law):
    if sha(data)!=binding["record_sha256"]:raise ValueError("EVENT_BYTES")
    record=json.loads(data)
    for key in ("q","sample","sample_id","seed_domain","seeds","cut_after_complete_step"):
        if record[key]!=binding[key]:raise ValueError("EVENT_BINDING:"+key)
    if (record["execution_source"],record["endpoint"],record["input_packet_sha256"],record["law_sha256"])!=(source,endpoint,packet,law):
        raise ValueError("SOURCE_ENDPOINT_PACKET_LAW")
    if record["cohort"]!="track_d_c6a_q2_q4_paired_price_pilot_v1":raise ValueError("COHORT")
    if len(record["ordered_events"])!=4 or any(len(step)!=32 for step in record["ordered_events"]):
        raise ValueError("FOUR_COMPLETE_R32_STEPS")
    for step in record["ordered_events"]:
        for event in step:
            if event["taylor_order"]!=0 or event["product_component_ids"] or event["phase"]!={"real":1.0,"imag":0.0}:
                raise ValueError("FIXED_K0_EVENT")
    return record
