#!/usr/bin/env python3
"""Exactly one saved q4_sample00 descriptor decision, before any price read."""
import argparse
from dataclasses import asdict
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

ROOT=Path(__file__).resolve().parents[1]
BASE="848ca0deecc08fd09bbb9027fb372905edd4414d"
C6A="6b6667026ec8b670f610e41141d195883a9eb513"
FEATURE="64072ba81bb70db1df1b34e94adb08e78926f24b"
FP="src/trottertracks/track_d_c7_4_models.py"
OLD="artifacts/track_d_c6a_paired_price/2026-11-06/run01"

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def write(path,value):
    with path.open("x") as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--archive",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    calls=0
    try:
        boot=load("cut_h4_namespace",ROOT/"scripts/native_df_qiskit_validation_support.py")
        boot.package("trotterlib",ROOT/"src/trotterlib");boot.package("trotterlib.df_trotter",ROOT/"src/trotterlib/df_trotter")
        sys.path.insert(0,str(ROOT/"src"))
        from trottertracks import track_d_cut_h4_saved as saved
        from trottertracks import track_d_cut_selection as selector
        feature_bytes=subprocess.check_output(["git","-C",str(ROOT),"show",FEATURE+":"+FP])
        if saved.sha(feature_bytes)!="6045e9a1b750494e8863928d119dc5b88448ad259936d0f4870bedec1dbdc33f":raise ValueError("FIXED_FEATURE_SOURCE")
        if (ROOT/"src/trottertracks/track_d_cut_selection.py").read_bytes()!=subprocess.check_output([
            "git","-C",str(ROOT),"show","a54d04059bfc61adef43aa2d40fb9c001b62378f:src/trottertracks/track_d_cut_selection.py"]):
            raise ValueError("SELECTOR_SOURCE_CHANGED")
        with tempfile.TemporaryDirectory(prefix="cut_h4_feature_") as temp:
            path=Path(temp)/"features.py";path.write_bytes(feature_bytes)
            features=load("trottertracks._h4_cut_features",path).event_features
        binding=next(r for r in json.loads((ROOT/OLD/"event_bindings.json").read_text()) if r["sample_id"]=="q4_sample00")
        manifest=json.loads((ROOT/OLD/"paths/q4_sample00/payload_archive.json").read_text())
        if (binding["q"],binding["sample"],manifest["sample_id"],manifest["execution_source"])!=(4,0,"q4_sample00",C6A):raise ValueError("FIXED_PATH")
        if manifest["sha256"]!="a631cd394073f7770d2723eb0cfa93fc6a8f703534b9cdce88b7c126381a007b":raise ValueError("FIXED_ARCHIVE")
        paths={"scripts/run_track_d_cut_h4_select.py","src/trottertracks/track_d_cut_h4_saved.py","tests/check_track_d_cut_h4_saved.py",OLD+"/event_bindings.json",OLD+"/paths/q4_sample00/payload_archive.json"}
        for module in list(sys.modules.values()):
            filename=getattr(module,"__file__",None)
            if filename:
                path=Path(filename).resolve()
                if path.is_relative_to(ROOT) and path.suffix==".py":paths.add(str(path.relative_to(ROOT)))
        hashes={p:saved.sha((ROOT/p).read_bytes()) for p in sorted(paths)}
        source=dict(base_commit=BASE,C6a_source=C6A,feature_source=FEATURE,feature_path=FP,feature_sha256=saved.sha(feature_bytes),
            source_sha256=hashes,archive_sha256=manifest["sha256"],sample_id="q4_sample00",binding_sha256=saved.sha(json.dumps(binding,sort_keys=True).encode()),
            python=sys.version,versions={p:importlib.metadata.version(p) for p in ("numpy","qiskit","scipy","mpmath","symengine")},
            rule=selector.RULE,phase="selection_before_individual_price_access",individual_prices_opened=False,
            source_fixed_before_event_decode=True,retry=0)
        write(out/"SOURCE.json",source)
        payload=saved.checked_payload(Path(args.archive),manifest)
        # Private lossless original bytes; no circuit or state is generated.
        local=out/"local";local.mkdir();(out/".gitignore").write_text("/local/\n")
        for name,data in payload.items():
            dest=local/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        from trottertracks.track_d_whole_path_price import LAW,INPUT_PACKET
        from trottertracks.track_d_h4_paired_reference import ENDPOINT
        record=saved.checked_record(payload["events.json"],binding,C6A,ENDPOINT,INPUT_PACKET,LAW)
        write(out/"INPUT_CHECK.json",dict(status="PASS",archive_sha256=manifest["sha256"],members_verified=len(payload),
            event_sha256=saved.sha(payload["events.json"]),sample_id="q4_sample00",steps=4,events_per_step=32,
            SOURCE_endpoint_packet_law_binding="PASS",private_payload_public=False))
        write(out/"CUT_CALL_START.json",dict(call=1,rule=selector.RULE,event_sha256=binding["record_sha256"],price_access_started=False))
        calls=1
        decision=selector.select_cut(record["ordered_events"],features)
        write(out/"DECISION.json",dict(**asdict(decision),source_manifest_sha256=saved.sha((out/"SOURCE.json").read_bytes()),
            event_sha256=binding["record_sha256"],decision_before_price_access=True))
        write(out/"SELECTION_STATUS.json",dict(status="DECISION_FROZEN_AWAIT_REUSE_CHECK",cut_selection_calls=calls,
            selected=decision.selected,new_bundle_build=0,new_compile=0,new_postcheck=0,individual_prices_opened=False))
        print(json.dumps(asdict(decision)),flush=True)
        if any(saved.sha((ROOT/p).read_bytes())!=h for p,h in hashes.items()):raise ValueError("SOURCE_CHANGED")
    except BaseException as error:
        write(out/"STOP_FAILURE.json",dict(status="FAIL_MANDATORY_STOP",mandatory_stop=True,error=str(error),
            traceback=traceback.format_exc(),cut_selection_calls=calls,retry=0))
        raise

if __name__=="__main__":main()
