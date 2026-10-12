"""Future separately authorized identity-only one-shot; no CLI scope override."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.e_r3_identity_diagnostic.launch import ART,RESULT,verify_launch,sha
from trottertracks.algorithm_codesign.df_correction_comparison.guard import Supervisor,Limits,exclusive_json
from trottertracks.algorithm_codesign.df_correction_comparison.storage import dump_json,load_json


def main():
    if len(sys.argv)!=1:raise PermissionError('Identity-only runner takes no overrides')
    provenance=verify_launch(ROOT)  # current preparation authorization fails here
    contract=json.loads((ROOT/ART/'contract_v1.json').read_bytes());caps=contract['caps']
    out=ROOT/RESULT
    supervisor=Supervisor(Limits(float('inf'),caps['CPU_seconds_total'],caps['aggregate_RSS_MiB'],caps['worker_RSS_MiB']),
        out,output_cap=caps['output_bytes']-262144,log_cap=caps['log_bytes'])
    from trottertracks.algorithm_codesign.df_correction_comparison.worker import runtime_check
    runtime_check(contract,'identity')  # before consuming marker, no registered array
    supervisor.check_total()
    out.mkdir(parents=True)  # existing namespace is a STOP, never replace it
    marker=out/'one_shot_marker.json'
    exclusive_json(marker,dict(provenance,runs=1,retries=0,mandatory_STOP=True,created_unix=time.time()))
    provenance['marker_sha256']=sha(marker)
    os.environ['PYTHONPATH']=str(ROOT/'src')
    status='E_R3_IDENTITY_DIAGNOSTIC_TECHNICAL_INCONCLUSIVE';reason=None;observations=None
    try:
        job={'label':'identity','output':str(out/'identity_result.json'),
             'command':[contract['runtimes']['numeric']['python'],'-B','-m',
                        'trottertracks.algorithm_codesign.e_r3_identity_diagnostic.worker']}
        supervisor.phase([job],Limits(float('inf'),caps['CPU_seconds_worker'],caps['aggregate_RSS_MiB'],caps['worker_RSS_MiB']),1)
        supervisor.check_total()
        saved=load_json(out/'identity_result.json')
        if saved['status']!='E_R3_IDENTITY_DIAGNOSTIC_COMPLETE' or saved['path_order']!=contract['path_order']:
            raise PermissionError('Identity child completion mismatch')
        if any(saved['provenance'].get(k)!=v for k,v in provenance.items()):
            raise PermissionError('Identity child provenance mismatch')
        status=saved['status'];observations=saved['path_observations']
    except BaseException as exc:
        reason=type(exc).__name__+': '+str(exc)
    receipt=supervisor.receipt()
    terminal={'status':status,'technical_reason':reason,'path_observations':observations,'provenance':provenance,
        'resource_receipt':receipt,'observation_mismatch_is_not_scientific_acceptance':True,
        'old_science_classification':'TECHNICAL_INCONCLUSIVE','historical_cause_unique':False,
        'retries':0,'mandatory_STOP':True}
    dump_json(out/'process_receipt.json',receipt)
    dump_json(out/'terminal_status.json',terminal)
    print(json.dumps({'status':status,'technical_reason':reason,'mandatory_STOP':True}))
    if reason:sys.exit(2)


if __name__=='__main__':main()
