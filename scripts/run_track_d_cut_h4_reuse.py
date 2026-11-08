#!/usr/bin/env python3
"""One central-cut reuse audit. No sampler, builder, compiler or state action."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
OLD="artifacts/track_d_c6a_paired_price/2026-11-06/run01"

def read(path):return json.loads(path.read_text())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    spec.loader.exec_module(m);return m
def write(path,value):
    with path.open("x") as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")

def main():
    p=argparse.ArgumentParser();p.add_argument("--run",required=True);a=p.parse_args();out=Path(a.run).resolve()
    if (out/"REUSE_START.json").exists() or (out/"STOP.json").exists() or (out/"STOP_FAILURE.json").exists():raise ValueError("OUTPUT_COLLISION_NO_RETRY")
    try:
        boot=load("cut_reuse_namespace",ROOT/"scripts/native_df_qiskit_validation_support.py")
        boot.package("trotterlib",ROOT/"src/trotterlib");boot.package("trotterlib.df_trotter",ROOT/"src/trotterlib/df_trotter")
        sys.path.insert(0,str(ROOT/"src"))
        from trottertracks.track_d_cut_h4_saved import sha
        from trottertracks import track_d_cut_h4_reuse as reuse
        from trottertracks.track_d_h4_reference import load_qpy,stored_events
        from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
        from trotterlib.track_d_minimal_reference import metrics
        decision=read(out/"DECISION.json");source=read(out/"SOURCE.json")
        if sha((out/"SOURCE.json").read_bytes())!=decision["source_manifest_sha256"]:raise ValueError("DECISION_SOURCE")
        if any(sha((ROOT/path).read_bytes())!=h for path,h in source["source_sha256"].items()):raise ValueError("SELECTION_SOURCE_CHANGED")
        if decision["selected"]!=2 or not decision["decision_before_price_access"]:raise ValueError("ONLY_CENTRAL_REUSE")
        old=ROOT/OLD;directory=old/"paths/q4_sample00";paths=["scripts/run_track_d_cut_h4_reuse.py","src/trottertracks/track_d_cut_h4_reuse.py","tests/check_track_d_cut_h4_reuse.py"]
        old_names=[old/"SOURCE.json",old/"prices.json",old/"result.json",old/"STOP.json",directory/"precompile_checks.json",directory/"paired_observables.json"]
        old_names += [directory/(m+"_"+axis+suffix) for m in ("symmetric_directional","reference") for axis in ("X","Y") for suffix in ("_check.json","_compile.json")]
        write(out/"SOURCE_REUSE.json",dict(rule_unchanged=True,decision_file_sha256=sha((out/"DECISION.json").read_bytes()),
            source_sha256={name:sha((ROOT/name).read_bytes()) for name in paths},
            reused_evidence_sha256={str(path.relative_to(ROOT)):sha(path.read_bytes()) for path in old_names},
            input_archive_sha256=source["archive_sha256"],operation="Saved-QPY structural inspection and old PASS/price qualification only"))
        write(out/"REUSE_START.json",dict(decision_frozen=True,selected=2,individual_price_access_started=True,new_compiles_permitted=0))
        local=out/"local";record=read(local/"events.json")
        if sha((local/"events.json").read_bytes())!=decision["event_sha256"]:raise ValueError("EVENT_CHANGED")
        # Reuse the saved event serializer's exact round-trip; no sampling.
        for step in record["ordered_events"]:stored_events(step)
        rows=[r for r in read(old/"prices.json") if r["sample_id"]=="q4_sample00"]
        reuse.check_request(record,decision,rows)
        profile=asdict(CompilerProfile());profile["basis_gates"]=list(profile["basis_gates"])
        raw=read(directory/"precompile_checks.json");paired=read(directory/"paired_observables.json")
        if raw["status"]!="PASS" or raw["input_basis_columns"]!=256 or raw["full_reference_Operator_constructed"]:raise ValueError("RAW_PASS_SCOPE")
        reuse.checked_errors(raw["checks"]);reuse.checked_errors(list(raw["paired_observables"].values()));reuse.checked_errors(list(paired.values()))
        if any(not raw["omitted_chi_negative_control"][axis]["omitted_phase_detected"] for axis in ("X","Y")):raise ValueError("ANALYTIC_CHI_PASS")
        audit=[]
        wanted=read(old/"paths/q4_sample00/payload_archive.json")["members"]
        for member in wanted:
            if sha((local/member["path"]).read_bytes())!=member["sha256"]:raise ValueError("PAYLOAD_CHANGED")
        for row in rows:
            name=row["method"]+"_"+row["axis"];check=read(directory/(name+"_check.json"));comp=read(directory/(name+"_compile.json"))
            if comp["status"]!="PASS" or comp["compile_calls"]!=1 or comp["compiler"]!=profile or row["compiler"]!=profile:raise ValueError("P0_SOURCE")
            if check["status"]!="PASS" or check["global_phase_fit"] is not False or check["arm"]!=row["method"] or check["axis"]!=row["axis"]:raise ValueError("OLD_POSTCHECK")
            reuse.checked_errors(check["checks"])
            wrappers={kind:load_qpy(local/"circuits"/(name+"_"+kind+".qpy")) for kind in ("raw","compiled")}
            width=17 if row["method"]=="reference" else 9
            for kind,qc in wrappers.items():
                if qc.parameters or qc.num_qubits!=width or qc.num_clbits!=1:raise ValueError("QPY_SCOPE")
                tail=qc.data[-1]
                if tail.operation.name!="measure" or qc.find_bit(tail.qubits[0]).index!=width-1 or qc.find_bit(tail.clbits[0]).index!=0:
                    raise ValueError("BIT_CONVENTION")
                if kind=="raw" and [i.operation.name for i in qc.data[-3:]]!=(["sdg","h","measure"] if row["axis"]=="Y" else [i.operation.name for i in qc.data[-3:-2]]+["h","measure"]):
                    raise ValueError("XY_READOUT_ORDER")
            measured=metrics(wrappers["compiled"])
            expected=dict(row["metrics"],qubit_count=row["peak_qubits"])
            if any(measured[k]!=v or check["metrics"][k]!=v for k,v in expected.items()):raise ValueError("SAVED_PRICE_QPY")
            audit.append(dict(method=row["method"],axis=row["axis"],status="QUALIFIED_REUSE",metrics=expected,
                raw_and_compiled_member_hashes="PASS",input_binding="PASS",old_raw_postcheck_paired_phase="PASS",
                canonical_semantics="Exact saved raw bytes, unchanged builder/source/policy, same request and cut; no rebuilt IR",new_gate_action=False))
        write(out/"REUSE_RESULT.json",dict(status="PASS",selected_equals_center=True,reused_wrappers=audit,old_raw_pass_reused=1,
            old_compiled_pass_reused=4,all256_input_evidence_reused=True,component_abs=1e-10,observable_frobenius=2e-10,rtol=0,phase_fit=False,
            vacuum_and_actual_inverse_and_chi="Existing exact-payload-bound C6a raw/paired PASS reused; not recalculated",
            readout="X:H; Y:Sdg then H; ancilla last -> bit0; bit0 +1, bit1 -1",private_raw_uploaded=False))
        comparisons=[]
        for axis in ("X","Y"):
            strong=next(r["metrics"] for r in audit if r["method"]=="symmetric_directional" and r["axis"]==axis)
            center=next(r["metrics"] for r in audit if r["method"]=="reference" and r["axis"]==axis)
            comparisons.append(dict(axis=axis,strong=strong,reference_center=center,reference_selected=center,
                selected_minus_center={k:0 for k in center},selected_minus_strong={k:center[k]-strong[k] for k in center}))
        write(out/"COMPARISONS.json",comparisons)
        write(out/"CALL_COUNTS.json",dict(cut_selection=1,new_sampling=0,new_seed_or_RNG=0,new_bundle_build=0,new_raw_correctness=0,
            new_P0_compile=0,new_compiled_postcheck=0,reused_raw_PASS=1,reused_compiled_wrappers_PASS=4,
            saved_event_roundtrips=4,saved_QPY_loads=8,saved_compiled_metric_counts=4,ground_or_DF=0,fit=0,N_or_13round=0,science_retry=0))
        write(out/"STOP.json",dict(status="COMPLETE_NO_ADDITIONAL_DEPTH_REDUCTION_MANDATORY_STOP",mandatory_stop=True,
            sample_id="q4_sample00",selected_cut=2,central_cut=2,delta_depth_X=0,delta_depth_Y=0,
            interpretation="Frozen descriptor policy selected the center on this one calibration path; no claim about other legal cuts or all-law optimum",new_compile=0,
            next_stage_authorized=False,retry=0,old_preflight_STOP_preserved=True))
        print("QUALIFIED REUSE: central cut2; delta depth0; no new build/compile/postcheck",flush=True)
    except BaseException as error:
        write(out/"STOP_REUSE_FAILURE.json",dict(status="FAIL_MANDATORY_STOP",mandatory_stop=True,error=str(error),traceback=traceback.format_exc(),retry=0))
        raise

if __name__=="__main__":main()
