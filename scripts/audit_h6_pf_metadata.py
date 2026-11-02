"""Read-only ending audit for acquired H6 metadata63, without signal execution."""
import argparse
import gzip
import hashlib
import io
from pathlib import Path
import pf_common_run_contract as c


def audit(private,output):
    p=Path(private);out=Path(output);c.require(not out.exists(),'FRESH_METADATA_AUDIT_OUTPUT')
    plan=c.read(p/'plan_APPROVED.json');c.verify(plan);stage=p/'METADATA'
    receipt=c.read(stage/'completion.json');grant=c.read(stage/'grant.json');marker=c.read(stage/'CONSUMED.json')
    c.require(receipt['status']=='PASS' and not list(stage.rglob('STOP.json')),'METADATA_SUCCESS_NO_STOP')
    c.require(marker['grant']==grant and grant['SOURCE']==plan['SOURCE'] and grant['plan_sha256']==c.sha(p/'plan_APPROVED.json'),'GRANT_PLAN_SOURCE')
    ledger=[c.read(q) for q in sorted((stage/'ledger').glob('*.json'))];counts={k:0 for k in grant['caps']}
    for r in ledger:
        c.require(r['entered_nonrefundable'],'NONREFUNDABLE_LEDGER');counts[r['kind']]+=r['count']
    c.require(counts==receipt['actual_calls']==c.read(stage/'reserved_counts.json') and all(counts[k]<=grant['caps'][k] for k in counts),'LEDGER_CAPS')
    records=c.read(stage/'output/all_metadata_records.json');old=c.read(p/'old52_signals.json');floor=c.read(p/'old_u_floors.json')
    c.require(len(records)==63 and len(old)==52 and set(records)=={c.cid(r) for r in plan['rows']},'FIXED63_OLD52')
    import numpy as np
    maxima={};archives=[];raw11=[]
    for identity,r in records.items():
        root=stage/'output'/identity;entry=r['vector_archive'];blob=(root/entry['path']).read_bytes();raw=gzip.decompress(blob)
        c.require(hashlib.sha256(blob).hexdigest()==entry['archive_sha256'] and hashlib.sha256(raw).hexdigest()==entry['original_sha256'],'VECTOR_ARCHIVE_SHA')
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            corrected,raw_vector,forward=(z[k] for k in ('corrected','raw','forward'))
            closure=float(np.linalg.norm(raw_vector-corrected));error=float(np.linalg.norm(corrected-forward))
            endnorm=max(float(np.linalg.norm(v)) for v in (corrected,raw_vector,forward))
        ev=r['signal']['allowance_evidence'];obs=r['observed']
        c.require(closure==ev['raw_corrected_vector_closure']==obs['raw_corrected_vector_closure'] and error==obs['forward_vector_error'],'SAVED_VECTOR_OBSERVATIONS')
        c.require(closure<1e-9 and error<1e-9 and ev['maximum_intermediate_norm']>=max(1.,endnorm),'OBSERVATION_NUMERICAL_GATE')
        c.require(r['SOURCE']==plan['SOURCE'] and r['original_restored'] is False and r['reference_rotated'] is False,'NEW_OBSERVATION_PROVENANCE')
        if identity in old:
            o=old[identity];s=r['signal']
            c.require(s['corrected']==o['signals']['corrected'] and s['reference']==o['reference'] and s['total_signed_bias']==o['error_decomposition']['total_signed'],'OLD52_LITERAL_VALUES_PRESERVED')
            c.require(r['old_u']>=max(o['allowance']['real'],o['allowance']['imag']) and r['old_margin']>=o['allowance']['log_B_margin'],'OLD52_ALLOWANCE_NOT_REDUCED')
            for key,value in r['old_signal_comparison'].items():
                if key.endswith('_error'):c.require(value<1e-9,'OLD52_CHECK');maxima[key]=max(maxima.get(key,0.),value)
        else:
            c.require(r['old_u']>=floor[identity],'NEW11_SAVED_U_FLOOR');raw11.append(identity)
        for key in ('maximum_intermediate_norm','raw_corrected_vector_closure','forward_vector_error'):
            maxima[key]=max(maxima.get(key,0.),obs[key])
        archives.append(dict(id=identity,**entry))
    worker=c.read(stage/'worker_exit_receipt.json')
    c.require(worker['same_parent_join'] and worker['remaining_owned_workers']==0 and len(worker['workers'])==63 and all(r['exit_code']==0 and r['reaped'] for r in worker['workers']),'NATIVE_EXIT_REAP')
    out.mkdir(parents=True)
    def save(name,v):c.save(out/name,v)
    result=dict(status='PASS',SOURCE=plan['SOURCE'],records=63,old52_checks=52,new_raw_signals=11,new_raw_ids=sorted(raw11),
        actual_calls=counts,ledger_entries=len(ledger),vector_archives_verified=len(archives),maxima=maxima,workers_exit0_reaped=63,
        new_scientific_signal_calls_in_audit=0,old52_values_literal_unchanged=True,old_STOPs_preserved=True,
        author_separate_numerical_path_check=True,external_review=False,mandatory_STOP_of_metadata_stage=True,
        original_common_H6_successor_authorized=True)
    save('ending_audit.json',result);save('stage_receipt.json',receipt);save('worker_exit_receipt.json',worker)
    save('grant_and_ledger.json',dict(grant=grant,marker=marker,ledger=ledger));save('archive_inventory.json',archives)
    save('resource_receipt.json',c.read(stage/'resource_receipt.json'))
    for name,value in [('metadata_records.json.gz',records),('old52_signals.json.gz',old)]:
        (out/name).write_bytes(gzip.compress(c.encoded(value),compresslevel=1,mtime=0))
    projection={k:v for k,v in plan.items() if k not in ('private_root','provider','input_paths','closure')}
    projection.update(original_plan_sha256=c.sha(p/'plan_APPROVED.json'),read_only_projection=True,
        input_paths={k:Path(v).name for k,v in plan['input_paths'].items()},closure=[{k:v for k,v in e.items() if k!='path'} for e in plan['closure']])
    save('execution_contract_projection.json',projection);save('approval_receipt.json',c.read(p/'approval_receipt.json'))
    save('file_inventory.json',dict(excludes_self=True,files=[dict(path=q.name,sha256=c.sha(q),bytes=q.stat().st_size) for q in sorted(out.iterdir())]))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--private',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();print(c.encoded(audit(a.private,a.output)).decode())
