"""Exclusive identity-only child; gate precedes numerical imports/input arrays."""
import json
from pathlib import Path
import resource
import sys
import time

from .launch import ART,RESULT,verify_child
from ..df_correction_comparison.guard import exclusive_json
from ..df_correction_comparison.storage import dump_json


def saved_frame_rows(root,contract):
    job=json.loads((root/contract['inputs']['saved_job']['path']).read_bytes())
    inputs=json.loads((root/contract['inputs']['saved_inputs']['path']).read_bytes())
    rows=job['frames']
    if (job['r']!=3 or job['stage']!='native' or rows!=inputs['3']
        or job['marker_sha256']!=contract['old_marker_sha256']):
        raise PermissionError('Saved E-r3 job/input metadata mismatch')
    if len(rows)!=3 or any(len(v)!=3 or any(len(row)!=2 or any(len(z)!=2 for z in row) for row in v) for v in rows):
        raise PermissionError('Only three 3x2 E-r3 frames are registered')
    return rows


def claim_child(root):
    provenance=verify_child(root)
    out=root/RESULT
    exclusive_json(out/'identity.started.json',dict(provenance,runs=1,retries=0))
    return provenance


def main():
    if len(sys.argv)!=1:raise PermissionError('Identity child takes no overrides')
    root=Path(__file__).resolve().parents[4]
    provenance=claim_child(root)
    out=root/RESULT
    contract=json.loads((root/ART/'contract_v1.json').read_bytes())
    from ..df_correction_comparison.worker import runtime_check
    runtime_check(contract,'identity')
    rows=saved_frame_rows(root,contract)
    # All registered numerical reconstruction starts AFTER the new capability,
    # exclusive child receipt, runtime checks and saved-input hash verification.
    from ..df_correction_comparison.native_evaluation import deserialize_frames
    from ..df_correction_comparison.intervals import configure
    from .evaluation import legacy_helpers,observe_path,compare_observations
    configure();frames=deserialize_frames(rows)
    start,cpu=time.monotonic(),time.process_time()
    helper=legacy_helpers(root/contract['legacy_helper']['path'])
    observations={};provenance.update(path_order=contract['path_order'],
        context='fresh diagnostic child; current runtime; current then legacy; no counter reset')
    for label in contract['path_order']:
        path=out/(label+'.json');saved=False
        def save(record):
            nonlocal saved
            dump_json(path,record,replace_existing=saved);saved=True
        observations[label]=observe_path(label,frames,contract['historical_expected_fingerprint'],helper,save,provenance)
    comparison=compare_observations(observations['current'],observations['legacy'])
    result={'status':'E_R3_IDENTITY_DIAGNOSTIC_COMPLETE','provenance':provenance,'path_order':contract['path_order'],
        'path_observations':{p:observations[p]['observation'] for p in contract['path_order']},
        'comparison':comparison,'definition_work':{p:observations[p]['definition_work'] for p in contract['path_order']},
        'native_transpile_calls':0,'RNG_calls':0,'registered_CD_E_r6_controlled_T_calls':0,
        '_process_receipt':{'evaluation_wall_seconds':time.monotonic()-start,'evaluation_CPU_seconds':time.process_time()-cpu,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024},
        'historical_cause_unique':False,'science_winner':None,'retries':0,'mandatory_STOP':True}
    dump_json(out/'identity_result.json',result)


if __name__=='__main__':main()
