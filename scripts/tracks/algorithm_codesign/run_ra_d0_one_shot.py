#!/usr/bin/env python3
"""Future authorized saved-table development one-shot. Review source rejects launch."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from trottertracks.algorithm_codesign.ra_d0.launch import verify_launch, consume_marker, SOURCE_DIR


def main():
    permit = verify_launch(ROOT)  # No registered data/solver/marker before gate.
    from trottertracks.algorithm_codesign.ra_d0.guard import BudgetGuard
    guard = BudgetGuard()  # Includes marker and saved-table loading in runtime.
    output = consume_marker(permit)
    guard.output_bytes = (output/"one_shot_consumed.json").stat().st_size
    try:
        with guard.enforce_OS_limits():
            from trottertracks.algorithm_codesign.ra_d0.engine import OneShotEngine
            manifest = json.loads((ROOT/SOURCE_DIR/"source_manifest_v3.json").read_text())
            table_path = "artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json"
            grid_path = "artifacts/track_b_ra_d0_preparation/2026-10-06/shot_grid_query_recipe_v1.json"
            inputs = {"source_commit": permit.source_commit, "authorization_commit": permit.authorization_commit,
                      "authorization_sha256": permit.authorization_sha256,
                      "candidate_table_sha256": manifest["critical_sha256"][table_path],
                      "shot_grid_sha256": manifest["critical_sha256"][grid_path],
                      "contract_sha256": manifest["critical_sha256"][SOURCE_DIR+"/execution_contract_v3.json"]}
            engine = OneShotEngine(json.loads((ROOT/table_path).read_text()), json.loads((ROOT/grid_path).read_text()), output, inputs, permit)
            engine.guard = guard
        result = engine.run()
    except Exception as error:
        result = {"classification": "D0_TECHNICAL_INCONCLUSIVE", "technical_reason": type(error).__name__+": "+str(error),
                  "source_commit": permit.source_commit, "authorization_commit": permit.authorization_commit,
                  "resource_usage": guard.usage(), "runs": 1, "retries": 0,
                  "mandatory_STOP": True, "next_stage_authorized": False}
        # Includes a failure during saved-input loading after marker consumption.
        if not (output/"result.json").exists():
            guard.write(output/"result.json", result, terminal=True)
    print(json.dumps({"classification": result["classification"], "mandatory_STOP": True}))


if __name__ == "__main__":
    try:
        main()
    except (PermissionError, FileNotFoundError) as error:
        raise SystemExit("LAUNCH_REJECTED_BEFORE_REGISTERED_OPTIMIZATION: "+str(error))
