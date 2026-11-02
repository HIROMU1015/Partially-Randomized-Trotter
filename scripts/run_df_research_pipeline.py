#!/usr/bin/env python3
"""P1 plan or artificial wiring only; never an old H4 science runner replacement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# This script lives in the same checkout as its src; no external worktree imports.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "synthetic", "scientific", "real", "resume"),
                        default="plan", help="plan or artificial synthetic wiring; scientific/real/resume are disabled")
    parser.add_argument("--output-dir", help="optional NEW output directory; requires legacy core dependencies; existing directories are refused")
    args = parser.parse_args(argv)
    if args.mode not in ("plan", "synthetic"):
        parser.error("P1 scientific execution is disabled; no input/sampling/compile was started")
    from trottertracks.resource_applicability.pipeline import compose, PipelineError
    from trottertracks.resource_applicability.h4_geometry_pipeline import H4GeometryAdapter
    from trottertracks.resource_applicability.pipeline_fixture import synthetic_context, synthetic_fixture
    try:
        context = synthetic_context()
        if args.mode == "synthetic":
            data, providers = synthetic_fixture(context)
            result = compose(context, H4GeometryAdapter(), mode=args.mode,
                             payload=data, providers=providers)
        else:
            result = compose(context, H4GeometryAdapter())
        if args.output_dir:
            from trottertracks.resource_applicability.pipeline import write_new_result_directory
            write_new_result_directory(result, args.output_dir)
    except (PipelineError, OSError) as exc:
        parser.error(f"pipeline refused: {exc}")
    except ModuleNotFoundError as exc:
        parser.error(f"synthetic fixture dependency unavailable: {exc.name}; no output created")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.qualification in ("PLAN_ONLY", "SYNTHETIC_END_TO_END_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
