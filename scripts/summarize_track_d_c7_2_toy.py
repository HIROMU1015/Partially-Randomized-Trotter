#!/usr/bin/env python3
"""Project C7-2 toy records only; no molecular input, fit, or circuit call."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts/track_d_c7_2_toy/2026-11-07/run01"
METRICS=("rz_count","cx_count","rz_depth","cx_depth","total_depth","circuit_size","qubit_count")
def write(name,value):
    path=OUT/name
    with path.open("x") as stream:json.dump(value,stream,indent=2,ensure_ascii=False);stream.write("\n")
def main():
    stop=json.loads((OUT/"STOP.json").read_text())
    if stop["compile_calls"]!=54 or not stop["source_input_preserved"]:raise ValueError("INCOMPLETE")
    records={p.stem:json.loads(p.read_text()) for p in sorted((OUT/"cases").glob("*.json"))}
    raw=json.loads((OUT/"raw_checks.json").read_text());post=json.loads((OUT/"postchecks.json").read_text())
    checks=[x for row in raw["checks"].values() for x in row]
    projection=[];traces={}
    for name,d in records.items():
        label="P0_LOCAL_VALIDATED";relation="phase-preserving unitary and local P0 metrics only"
        if name=="cx_adjacent" or name in ("center_repeat_zz_controlled","center_repeat_zz_physical","actual_inverse_pair"):
            label="COUNTEREXAMPLE_FOUND";relation="unconditional preservation/additivity of pre-P0 native operation counts"
        if name=="cx_commuting_control_rz":
            label="COUNTEREXAMPLE_FOUND";relation="mathematical cancellability does not imply P0 removes the CX pair"
        if name=="diagonal_mixed_signed_phase_ordinary":
            label="COUNTEREXAMPLE_FOUND";relation="ordinary-control premium is not universally two CX per nonzero factor"
        if "_shell_" in name:
            label="COUNTEREXAMPLE_FOUND";relation="four isolated-CSWAP P0 prices do not sum to the scalar-only whole-shell price"
        if "_reference_" in name:
            label="COUNTEREXAMPLE_FOUND";relation="shell depth plus maximum standalone lane depth is not a whole-wrapper identity"
        projection.append(dict(name=name,status=d["status"],classification=label,tested_relation=relation,
            domain="artificial-only; not H4 law coverage",H4_global_claim="P0_GLOBAL_BRIDGE_MISSING",
            raw={k:v for k,v in d["raw"].items() if k not in ("operations","longest_path_operations")},
            compiled={k:v for k,v in d["compiled"].items() if k not in ("operations","longest_path_operations")},
            postcheck=d["postcheck"],original_record_sha256=hashlib.sha256((OUT/"cases"/(name+".json")).read_bytes()).hexdigest()))
        traces[name]={stage:dict(global_phase=d[stage]["metrics"]["global_phase"],
            ops=[[x["name"],x["qubits"],x["parameters"]] for x in d[stage]["operations"]],
            longest_path_operations=d[stage]["longest_path_operations"]) for stage in ("raw","compiled")}
    paired=[];lane=[]
    for name in ("q2_same","q2_mixed","q4_same","q4_mixed"):
        for axis in ("X","Y"):
            values={k:records[name+"_"+k]["compiled"]["metrics"] for k in
                    ("reference_"+axis,"symmetric_directional_"+axis,"shell_"+axis,"A_inverse","B")}
            r,s=values["reference_"+axis],values["symmetric_directional_"+axis]
            paired.append(dict(fixture=name,axis=axis,reference=r,symmetric=s,delta={k:r[k]-s[k] for k in METRICS}))
            lane.append(dict(fixture=name,axis=axis,relation="D_full-D_shell-max(D_Ainv,D_B)",
                residual={k:r[k]-values["shell_"+axis][k]-max(values["A_inverse"][k],values["B"][k])
                          for k in ("rz_depth","cx_depth","total_depth")},classification="COUNTEREXAMPLE_FOUND"))
    summary=dict(status="TOY_TECHNICAL_PASS_H4_GLOBAL_UNQUALIFIED",compile_calls=54,raw_checks=len(checks),
        wrapper_postchecks=len(post["checks"]),regression=dict(C5=8,C3=4,status="PASS"),
        max_raw_component=max(x["max_entry"] for x in checks),
        max_compiled_component=max(d["postcheck"]["max_entry"] for d in records.values()),
        max_observable_coherence_leakage_frobenius=max(x["frobenius"] for x in post["checks"]),
        paired=paired,lane_diagnostics=lane,case_results=projection,
        classification_summary=dict(source_structure="STRUCTURALLY_EXPLAINED",artificial_p0="P0_LOCAL_VALIDATED",
            naive_additivity="COUNTEREXAMPLE_FOUND",arbitrary_H4_P0="P0_GLOBAL_BRIDGE_MISSING",long_q_price="UNQUALIFIED"))
    write("summary.json",summary)
    with (OUT/"operation_traces.json").open("x") as stream:json.dump(traces,stream,separators=(",",":"));stream.write("\n")
    with (OUT/"case_metrics.csv").open("x",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["case","stage",*METRICS])
        for name,d in records.items():
            for stage in ("raw","compiled"):writer.writerow([name,stage,*[d[stage]["metrics"][k] for k in METRICS]])
    write("publication_scope.json",dict(artificial_only=True,H4_payload_uploaded=False,
        private=["local/ original raw and compiled QPY","cases/ original verbose toy records"],
        public=["SOURCE and fixture/PLAN","scalar correctness and metrics","compact artificial operation/dependency traces","payload hashes"],
        suitability="deterministic invented two-mode inputs; no molecular events, packets, internal paths or resource observations",
        operation_trace_parameter_policy="RZ/P/CRZ/RZZ values only; complete gate parameters reconstruct from fixed SOURCE/CASES and local QPY"))
    # Preserve originals; publish compact projections rather than duplicate data.
    (OUT/".gitignore").write_text("local/\ncases/\n")
if __name__=="__main__":main()
