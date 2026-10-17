#!/usr/bin/env python3
"""Export proposed scope and simulate its DAG. This CLI cannot execute science."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "trotterlib"))
import targeted_validation_plan as campaign


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, help="new metadata directory; never overwrite")
    args = p.parse_args(argv)
    plan = campaign.build_plan()
    report = dict(plan_digest=campaign.digest(plan), workload=campaign.planned_workload(plan),
                  current_frontier=campaign.simulate_frontier(plan))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        for name, value in (("campaign_plan_UNAPPROVED_v1.json", plan),
                            ("scheduling_validation_v1.json", report),
                            ("grants_INVALID_v1.json", dict(approved=False,
                             valid_for_science=False, plan_digest=campaign.digest(plan),
                             grants=[dict(scope=n["id"], authorized=False,
                                          grant_id=None, SOURCE=None, command=None)
                                     for n in plan["nodes"]]))):
            (args.output/name).write_text(json.dumps(value, ensure_ascii=False,
                                                    indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
