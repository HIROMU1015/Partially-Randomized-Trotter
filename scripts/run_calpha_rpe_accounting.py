"""Consume fixed-law certificates; acquire only missing round bounds/shots."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
ART='artifacts/calpha_h4_step1/2026-11-10/'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operands',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['scripts/run_calpha_rpe_accounting.py','src/trotterlib/calpha_rpe_accounting.py',
           'src/trotterlib/calpha_interval_bounds.py','src/trotterlib/finite_rte_signal_validation.py']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    result=dict(started_UTC=datetime.now(timezone.utc).isoformat(),SOURCE_SHA256=hashes,status='STARTED')
    started=time.monotonic()
    try:
        import numpy as np
        from trotterlib.calpha_rpe_accounting import evaluate_all_rounds
        names=['run04_state_defect','run03_macrostep_gap','run06_reference_interval',
               'run07_reconstruction_bridge','run09_sampling_law']
        prior={name:json.loads((ROOT/(ART+name+'/result.json')).read_text()) for name in names}
        if hashlib.sha256(Path(args.operands).read_bytes()).hexdigest()!=prior['run04_state_defect']['operands_SHA256']:
            raise ValueError('PRESERVED_OPERANDS_IDENTITY_REQUIRED')
        with np.load(args.operands,allow_pickle=False) as raw:
            operands={key:raw[key].copy() for key in raw.files}
        operands['gap_lower_exact']=prior['run03_macrostep_gap']['gap_certificate']['gap_lower_exact']
        result['inputs']={name:hashlib.sha256((ROOT/(ART+name+'/result.json')).read_bytes()).hexdigest() for name in names}
        result.update(evaluate_all_rounds(operands,prior['run06_reference_interval'],
            prior['run07_reconstruction_bridge'],prior['run09_sampling_law']))
    except Exception as exc:
        result.update(status='TECHNICAL_EXCEPTION',reason=str(exc));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['SOURCE_unchanged']=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in hashes.items())
        (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(result['status'])
