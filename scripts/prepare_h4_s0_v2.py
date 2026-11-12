"""Fixed S0 -> V2 metadata artifacts only. No new science, prices or G."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.h4_s0_v2_intake import prepare, A_PACKET


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory')
    result = prepare(ROOT)
    args.out.mkdir(parents=True, exist_ok=False)
    names = dict(configuration='CONFIGURATION_V2.json', identity_map='IDENTITY_MAP.json',
        evidence_plan='EVIDENCE_PLAN.json', price_requests='C_REQUESTS.json')
    for key, name in names.items():
        (args.out/name).write_text(json.dumps(result[key],indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    # Preserve the scientific handoff bytes, including its V1 IDs and status.
    raw = subprocess.check_output(['git','show',A_PACKET['commit']+':'+A_PACKET['path']],cwd=ROOT)
    (args.out/'ORIGINAL_S0_N_HANDOFF.json').write_bytes(raw)
    summary = {k:v for k,v in result.items() if k not in set(names)|{'original_S0_handoff'}}
    (args.out/'SOURCE_AND_SCOPE.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
