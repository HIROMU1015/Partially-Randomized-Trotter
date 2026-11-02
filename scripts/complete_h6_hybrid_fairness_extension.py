#!/usr/bin/env python3
"""Continue the authorized fixed sequence once the current exploration ends."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/h6_hybrid_fairness_extension/2026-10-27'
def run(script,*args):subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True)
def main():
    p=OUT/'stages/exploration/native.json'
    while not p.exists():time.sleep(10)
    assert json.loads(p.read_text())['all_exit_zero']
    run('analyze_h6_hybrid_fairness_extension.py','nominate')
    cp=OUT/'confirmation_plan.json';(OUT/'CONFIRMATION_SOURCE.json').write_text(json.dumps(dict(confirmation_plan_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),top_up_allowed=False),indent=2)+'\n')
    run('dispatch_h6_hybrid_confirmation.py')
    run('analyze_h6_hybrid_fairness_extension.py','finish')
    run('verify_h6_hybrid_fairness_extension.py')
    run('pack_h6_hybrid_fairness_extension.py')
    print('FIXED_PHASE_B_SCIENCE_AND_SAVE_COMPLETE_MANDATORY_STOP',flush=True)
if __name__=='__main__':main()
