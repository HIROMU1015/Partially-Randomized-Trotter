#!/usr/bin/env python3
"""AX1b entry point: default denial; metadata validation never reads science."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from trottertracks.resource_applicability.ax1b_contract import Stop,FLAGS
from trottertracks.resource_applicability.ax1b_execution import execute,validate_preparation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-preparation",action="store_true")
    parser.add_argument("--execute-saved-analysis",action="store_true")
    parser.add_argument("--authorization",type=Path)
    parser.add_argument("--launch-authorization-sha256")
    parser.add_argument("--preparation-manifest",type=Path,default=ROOT/"artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/identity_compatibility_manifest_v1.json")
    args=parser.parse_args(argv)
    try:
        if not args.validate_preparation and (not args.execute_saved_analysis or args.authorization is None):
            raise Stop("AUTHORIZATION","AX1b is not authorized; no science inputs read and no fit invoked")
        bundle=json.loads(args.preparation_manifest.read_text())
        if args.validate_preparation:
            if args.execute_saved_analysis:
                raise Stop("AUTHORIZATION","metadata validation and execution are separate modes")
            validate_preparation(ROOT,bundle)
            result=dict(status=bundle["status"],science_inputs_read=0,real_model_fits=0,**FLAGS)
        else:
            authorization=json.loads(args.authorization.read_text())
            result=execute(ROOT,bundle,authorization,args.execute_saved_analysis,args.launch_authorization_sha256)
        print(json.dumps(result,sort_keys=True))
        return 0
    except (Stop,OSError,ValueError,KeyError) as exc:
        print(json.dumps(dict(status=getattr(exc,"status","AX1B_STOP_IMPLEMENTATION"),reason=str(exc),mandatory_stop=True,next_stage_authorized=False)))
        return 2


if __name__=="__main__":
    sys.exit(main())
