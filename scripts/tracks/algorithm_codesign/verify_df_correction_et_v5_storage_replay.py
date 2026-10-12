"""Only lossless saved/synthetic I-O. Never imports arrays/circuit evaluators."""
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_comparison import storage as old
from trottertracks.algorithm_codesign.df_correction_et_storage_v5 import storage as new


def usage():return sum((getattr(resource.getrusage(resource.RUSAGE_SELF),k) for k in ('ru_utime','ru_stime')))

def size(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())

def check_readback(path,expected):
    h=hashlib.sha256();n=0
    for b in new.iter_verified_bytes(path):h.update(b);n+=len(b)
    assert n==expected['bytes'] and h.hexdigest()==expected['sha256']


def main(mode):
    start,cpu=time.monotonic(),usage();tmp=Path(tempfile.mkdtemp(prefix='et-v5-io-'+mode+'-'));records=[];max_size=0
    if mode=='saved':
        source=ROOT/'artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v1'
        inventory=json.loads((ROOT/'artifacts/track_b_df_correction_et_execution_audit/2026-10-12/v1/original_run_inventory.json').read_text())
        for row in inventory['files']:
            raw=(ROOT/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        logical={source/'E-r3.json',source/'E-r3.json.prefix.json',source/'E-r6.json.prefix.json',source/'E-inputs.json',source/'CD_saved_substage_receipt.json'}
        logical|={p for p in source.glob('*.raw-provenance-*.json') if not p.name.endswith('.review.json')}
        logical|={Path(str(p)[:-len('.review.json')]) for p in source.glob('*.raw-provenance-*.review.json')}
        for path in sorted(logical):
            expected=old.artifact_identity(path);data=old.load_json(path);target=tmp/path.name
            new.dump_json(target,data);check_readback(target,expected)
            records.append({'original_path':str(path.relative_to(ROOT)),'original_bytes':expected['bytes'],'original_sha256':expected['sha256'],'exact_legacy_readback':True})
            max_size=max(max_size,size(tmp))
            # Only fabricated metadata state controls; no science status or field
            # is altered, no Q/T/U/payload computation is made.
            if '.raw-provenance-' in path.name:
                before=size(tmp)
                for i in range(3):
                    state=dict(data,_IO_CONTROL_STATE=i);new.dump_json(target,state,replace_existing=True)
                new.dump_json(target,data,replace_existing=True);check_readback(target,expected)
                records[-1]['four_IO_updates_physical_bytes_delta']=size(tmp)-before
                max_size=max(max_size,size(tmp))
        for row in inventory['files']:
            raw=(ROOT/row['path']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        details={'legacy_input_original_disk_bytes':inventory['bytes'],'saved_logical_records':len(logical),'old_284_file_hashes_before_after_unchanged':True}
    elif mode=='synthetic':
        # Literal data only. 33 MiB incompressible-under-JSON string exercises
        # the observed payload size, with no analytic or quantum calculation.
        payload={'mock_text':'x'*(33*1024*1024),'name':'circuit-MOCK','phase':'-1/37'}
        record={'recursive_payload':payload,'input_arrays':{'mock':{'raw_bytes_hex':'ab'*(512*1024)}},'phase':'mock-first'}
        target=tmp/'mock-observation.json';new.dump_json(target,record);before=size(tmp)
        for i in range(6):record['phase']='MOCK-'+str(i);new.dump_json(target,record,replace_existing=True)
        delta=size(tmp)-before
        value=new.load_json(target);assert value==record
        shared=[{'inventory':{'literal':'N'*400000,'mock_only':True},'label':'MOCK-'+str(i)} for i in range(4)]
        prefix={'prefix':{'Pauli_coefficients':[{'beta':'1/17','label':'MOCK'}]*10000,'native':shared,'compile_calls':4}}
        new.dump_json(tmp/'mock-native.prefix.json',prefix);before_final=size(tmp)
        new.dump_json(tmp/'mock-native.json',prefix['prefix']);assert new.load_json(tmp/'mock-native.json')==prefix['prefix']
        details={'literal_payload_MiB':33,'six_state_update_increment_bytes':delta,'prefix_final_increment_bytes':size(tmp)-before_final,'synthetic_only':True}
        assert delta<128*1024 and size(tmp)-before_final<64*1024
        max_size=size(tmp)
    else:raise ValueError('Only saved or synthetic I-O modes')
    assert max_size<new.OUTPUT_BYTES
    files=[p for p in tmp.rglob('*') if p.is_file()]
    assert all(p.stat().st_size<=524288 for p in files)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    assert peak<512
    return {'mode':mode,'scope':'PURE_IO_LOSSLESS_VERIFICATION_NOT_SCIENCE_EXECUTION','PASS':True,'scratch_directory':str(tmp),'physical_output_bytes':size(tmp),'max_observed_disk_bytes':max_size,'active_output_budget_bytes':new.OUTPUT_BYTES,'physical_file_count':len(files),'all_files_le_512KiB':True,'wall_seconds':time.monotonic()-start,'CPU_seconds':usage()-cpu,'peak_RSS_MiB':peak,'records':records,**details,'matrix_evaluation':0,'native_transpile':0,'synthesis':0,'science_runner':0,'GPU_query':0,'old_science_classification_unchanged':True,'production_completion_guarantee':False}


if __name__=='__main__':print(json.dumps(main(sys.argv[1]),indent=2))
