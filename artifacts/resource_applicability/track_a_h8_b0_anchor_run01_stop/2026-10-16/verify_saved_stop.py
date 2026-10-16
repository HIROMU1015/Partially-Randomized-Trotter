"""Read-only check of saved STOP evidence; no resource gate or science launch."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def main():
    manifest=json.loads((HERE/'publication_files_v1.json').read_text())
    for row in manifest['files']:
        p=Path(row['path'])
        if p.is_absolute() or '..' in p.parts:raise ValueError('PUBLIC_PATH')
        raw=(ROOT/p).read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('PUBLIC_BYTES_CHANGED:'+str(p))
    d=json.loads((HERE/'attempt_STOP_v1.json').read_text())
    assert d['status']=='STOP_PRE_SCIENCE_FRESH_CPU_GATE' and d['reason']=='PHYSICAL_CORE_BUSY'
    assert d['driver_started'] is False and d['compile_workers_started']==0
    assert all(d[k]==0 for k in ('new_snapshot_decode','new_preparation_restore','new_native_evolution',
        'new_control_probe','new_compile','new_fit','new_residual','new_cost_reaccount','scientific_reservations'))
    assert all(d[k] is False for k in ('full_fresh_gate_passed','one_shot_science_lock_created',
        'control_directory_created','output_directory_created','automatic_retry','CPU_reassignment_after_STOP','next_stage_authorized'))
    assert d['launcher_native_exit_code'] is None and d['actual_failing_CPU_id'] is None
    assert d['grant_modified_after_launch'] is False and d['mandatory_STOP'] is True
    print(json.dumps(dict(status='PASS_SAVED_STOP_ONLY',new_compile=0,new_fit=0,
        public_files=len(manifest['files']),science_started=False,mandatory_STOP=True)))


if __name__=='__main__':main()
