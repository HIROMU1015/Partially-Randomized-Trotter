"""One authorized metadata recovery job under inherited external resource guard."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_et_saved_recovery.launch import ART, RESULT, verify_launch, verify_child
from trottertracks.algorithm_codesign.df_correction_et_saved_recovery.storage import dump_json, artifact_identity
from trottertracks.algorithm_codesign.df_correction_et_phase_v7.guard import Supervisor, Limits, exclusive_json


def check_original_files(contract):
    path = ROOT/contract['original_inventory']['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != contract['original_inventory']['sha256']:
        raise PermissionError('Original saved inventory changed')
    for row in json.loads(path.read_text())['files']:
        raw = (ROOT/row['path']).read_bytes()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise PermissionError('Original raw result changed: '+row['path'])


def main():
    contract = json.loads((ROOT/ART/'recovery_contract.json').read_text())
    original = json.loads((ROOT/contract['original_contract_path']).read_text())
    from trottertracks.algorithm_codesign.df_correction_et_phase_v7.worker import runtime_check
    if len(sys.argv)==2:
        job = json.loads(Path(sys.argv[1]).read_text())
        capability = verify_child(ROOT,job)
        if set(job) != {'marker_sha256'}:
            raise PermissionError('Recovery worker payload override')
        exclusive_json(ROOT/RESULT/'recovery.started.json',dict(capability,retries=0))
        runtime_check(original,'native');check_original_files(contract)
        from trottertracks.algorithm_codesign.df_correction_et_saved_recovery.accounting import recover
        value, receipt = recover(ROOT,contract)
        dump_json(ROOT/RESULT/'T_accounting.json',value)
        receipt['recovered_accounting_identity']=artifact_identity(ROOT/RESULT/'T_accounting.json')
        receipt['classification']='SAVED_E_T_ACCOUNTING_COMPLETE'
        receipt['mandatory_STOP']=True
        check_original_files(contract)
        dump_json(ROOT/RESULT/'recovery.json',receipt)
        return 0
    if len(sys.argv)!=1:
        raise PermissionError('No recovery overrides')
    provenance = verify_launch(ROOT)
    runtime_check(original,'native');check_original_files(contract)
    out = ROOT/RESULT;out.mkdir()
    exclusive_json(out/'one_shot_marker.json',dict(provenance,runs=1,retries=0,mandatory_STOP=True,created_unix=time.time()))
    marker_hash=hashlib.sha256((out/'one_shot_marker.json').read_bytes()).hexdigest()
    exclusive_json(out/'recovery.job.json',{'marker_sha256':marker_hash})
    caps=original['caps'];limits=Limits(caps['wall_seconds_total'],caps['CPU_seconds_total'],caps['aggregate_RSS_MiB'],caps['aggregate_RSS_MiB'])
    guard=Supervisor(limits,out,output_cap=caps['output_bytes']-262144,log_cap=caps['log_bytes'])
    classification='SAVED_RECOVERY_TECHNICAL_INCONCLUSIVE';reason=None
    try:
        guard.phase([{'label':'recovery','output':str(out/'recovery.json'),
                     'command':[original['runtimes']['numeric']['python'],'-B',str(Path(__file__).resolve()),str(out/'recovery.job.json')]}],limits,1)
        check_original_files(contract);guard.check_total()
        classification='SAVED_E_T_ACCOUNTING_COMPLETE'
    except BaseException as exc:
        reason=type(exc).__name__+': '+str(exc)
    dump_json(out/'terminal_status.json',{'classification':classification,'reason':reason,'provenance':provenance,
              'marker_sha256':marker_hash,'resource_receipt':guard.receipt(),'retries':0,
              'additional_science_calls':0,'original_terminal_reclassified':False,'mandatory_STOP':True})
    print(json.dumps({'classification':classification,'reason':reason,'mandatory_STOP':True}))
    return 0 if classification=='SAVED_E_T_ACCOUNTING_COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
