"""Metadata/off-domain regression only: no registered frame or synthesizer call."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_comparison import storage as st
from trottertracks.algorithm_codesign.df_correction_comparison import inventory_integrity as inv
from trottertracks.algorithm_codesign.df_correction_comparison.worker import claim_job, dump, runtime_check
from trottertracks.algorithm_codesign.df_correction_comparison.launch import RESULT
from trottertracks.algorithm_codesign.df_correction_comparison.guard import Supervisor, Limits, CapFailure


def encoded(data, ensure_ascii=True):
    return (json.dumps(data, default=str, indent=2, ensure_ascii=ensure_ascii)+'\n').encode()


def table(n=4097):
    # Fabricated metadata strings, no Pauli operator/coefficient computation.
    return {'Pauli_coefficients': [{'label':'MOCK-%05d'%i,
        'computational_beta':('1'*173)+'/7', 'zero_status':'MOCK_ONLY_NOT_SCIENCE',
        'ideal_beta_radius_upper':'0'} for i in range(n)],
        'partial_not_scientific_outcome':True}


def test_4097_metadata_rows_round_trip_exact_bytes_SHA_and_index(tmp_path):
    data=table();p=tmp_path/'metadata.json';expected=encoded(data)
    result=st.dump_json(p,data)
    assert result['storage']=='chunks' and not p.exists()
    idx=st.index_path(p);record=json.loads(idx.read_text())
    assert idx.stat().st_size<=st.REVIEW_BYTES
    assert record['top_level']['Pauli_coefficients']['items']==4097
    assert record['original_bytes']==len(expected)>st.REVIEW_BYTES
    assert record['original_sha256']==hashlib.sha256(expected).hexdigest()
    assert b''.join(st.iter_verified_bytes(p))==expected
    assert st.load_json(p)==data
    assert all((tmp_path/r['path']).stat().st_size<=st.CHUNK_BYTES for r in record['chunks'])


def test_unicode_long_rational_string_and_partial_last_chunk(tmp_path):
    data={'value':'1234567890'*90000+'/17','unicode':'軌道λ🧪'*60000}
    p=tmp_path/'unicode.json';st.dump_json(p,data,ensure_ascii=False)
    assert b''.join(st.iter_verified_bytes(p))==encoded(data,False)
    record=json.loads(st.index_path(p).read_text())
    for row in record['chunks']:(tmp_path/row['path']).read_bytes().decode('utf-8')
    assert record['chunks'][-1]['bytes']<st.CHUNK_BYTES
    assert st.load_json(p)==data


@pytest.mark.parametrize('data',[{'x':'small','r':'1/17'}, {'x':'x'*300000}])
def test_small_raw_reader_and_old_JSON_serialization_unchanged(tmp_path,data):
    p=tmp_path/'small.json';r=st.dump_json(p,data)
    assert r['storage']=='raw' and p.read_bytes()==encoded(data)
    assert not st.index_path(p).exists() and st.load_json(p)==data
    with pytest.raises(FileExistsError):st.dump_json(p,data)


@pytest.mark.parametrize('mutation',['bytes','order','whole_digest','escape','missing'])
def test_corrupt_chunk_or_index_is_not_returned_as_a_result(tmp_path,mutation):
    p=tmp_path/'bad.json';st.dump_json(p,table());idx=st.index_path(p)
    rec=json.loads(idx.read_text());first=tmp_path/rec['chunks'][0]['path']
    if mutation=='bytes':first.write_bytes(b'bad')
    elif mutation=='order':rec['chunks'].reverse()
    elif mutation=='whole_digest':rec['original_sha256']='0'*64
    elif mutation=='escape':rec['chunks'][0]['path']='../outside.txt'
    elif mutation=='missing':first.unlink()
    idx.write_text(json.dumps(rec))
    with pytest.raises((ValueError,FileNotFoundError)):st.load_json(p)


def test_interrupted_prefix_cleanup_raw_cannot_override_committed_index(tmp_path):
    p=tmp_path/'interrupted.json';data=table();st.dump_json(p,data);p.write_text('{}')
    assert st.load_json(p)==data
    record=json.loads(st.index_path(p).read_text())
    (tmp_path/record['chunks'][0]['path']).write_text('damaged')
    with pytest.raises(ValueError):st.load_json(p)


def test_prefix_updates_large_small_large_and_do_not_erase_old_evidence(tmp_path):
    p=tmp_path/'job.json.prefix.json';old=tmp_path/'old_marker.json';old.write_text('protected')
    dump(p,table());first=json.loads(st.index_path(p).read_text())
    dump(p,{'completed':1});assert st.load_json(p)=={'completed':1}
    assert all(not (tmp_path/r['path']).exists() for r in first['chunks'])
    dump(p,table(4100));assert len(st.load_json(p)['Pauli_coefficients'])==4100
    assert old.read_text()=='protected'


def test_interrupted_chunk_write_keeps_previous_prefix_readable(tmp_path,monkeypatch):
    p=tmp_path/'job.json.prefix.json';dump(p,{'completed':1})
    real=Path.open
    def fail(path,*args,**kwargs):
        if path.name=='chunk-00001.txt' and args and args[0]=='xb':
            raise OSError('synthetic disk failure')
        return real(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',fail)
    with pytest.raises(OSError):dump(p,table())
    assert st.load_json(p)=={'completed':1}
    assert list(tmp_path.rglob('chunk-00000.txt'))  # Failed-prefix evidence preserved.


def test_guard_reads_successful_chunked_child_without_changing_limits(tmp_path):
    p=tmp_path/'child.json'
    code=('from trottertracks.algorithm_codesign.df_correction_comparison.storage import dump_json;'
          'dump_json('+repr(str(p))+',{"text":"x"*600000,"_process_receipt":{"CPU_seconds":0.01}})')
    sup=Supervisor(Limits(10,10,10000,10000),tmp_path)
    job={'label':'metadata-child','output':str(p),'command':[sys.executable,'-B','-c',code]}
    assert len(sup.phase([job],Limits(5,5,10000,10000,5,5),1,per_key=True))==1
    assert st.artifact_exists(p) and not p.exists()


def test_chunks_still_count_towards_unchanged_aggregate_output_guard(tmp_path):
    st.dump_json(tmp_path/'metadata.json',table())
    sup=Supervisor(Limits(10,10,10000,10000),tmp_path,output_cap=1024)
    with pytest.raises(CapFailure):sup.check_total()


def frozen(tmp_path):
    out=tmp_path/RESULT;out.mkdir(parents=True)
    inventory={'unique_keys':[{'key':'mock-a','kind':'raw-rad','value':'1/7','scale':'1'},
                              {'key':'mock-b','kind':'raw-rad','value':'2/7','scale':'1'}]}
    binding=inv.freeze_inventory(out,inventory,{'source':'synthetic-only'},'mock-marker')
    return out,inventory,binding


def synth_job(out,inventory,binding):
    return {'stage':'synthesis','output':str(out/'T-0.json'),'marker_sha256':'mock-marker',
            'angle':inventory['unique_keys'][0],**binding}


def test_original_freeze_binding_reaches_job_start_and_result(tmp_path):
    out,inventory,binding=frozen(tmp_path)
    assert inv.verify_inventory(out,binding,marker_sha256='mock-marker',provenance={'source':'synthetic-only'})==inventory
    job=synth_job(out,inventory,binding);claim_job(tmp_path,job)
    start=json.loads((out/'T-0.started.json').read_text())
    assert all(start[k]==v for k,v in binding.items())
    row={'angle':job['angle'],**binding}
    inv.verify_synthesis_result(row,job['angle'],binding)
    with pytest.raises(FileExistsError):claim_job(tmp_path,job)


@pytest.mark.parametrize('mutation',['angle','key_delete','order','same_path_replacement','whitespace'])
def test_original_inventory_changes_rejected_before_worker_start(tmp_path,mutation):
    out,inventory,binding=frozen(tmp_path);p=out/inv.INVENTORY;p.chmod(0o644)
    if mutation=='angle':inventory['unique_keys'][0]['value']='3/7'
    elif mutation=='key_delete':inventory['unique_keys'].pop()
    elif mutation=='order':inventory['unique_keys'].reverse()
    if mutation=='same_path_replacement':
        p.unlink();p.write_text('{}')
    elif mutation=='whitespace':p.write_text(json.dumps(inventory,separators=(',',':')))
    else:p.write_text(json.dumps(inventory))
    with pytest.raises(PermissionError):claim_job(tmp_path,synth_job(out,inventory,binding))
    assert not (out/'T-0.started.json').exists()


def test_changed_inventory_and_matching_job_cannot_replace_original_receipt(tmp_path):
    out,inventory,binding=frozen(tmp_path);p=out/inv.INVENTORY;p.chmod(0o644)
    inventory['unique_keys'][0]['value']='3/7';p.write_text(json.dumps(inventory))
    changed=dict(binding,inventory_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    with pytest.raises(PermissionError):claim_job(tmp_path,synth_job(out,inventory,changed))
    assert not (out/'T-0.started.json').exists()


def test_receipt_change_marker_mismatch_and_bad_result_rejected(tmp_path):
    out,inventory,binding=frozen(tmp_path)
    with pytest.raises(PermissionError):inv.verify_inventory(out,binding,marker_sha256='other-marker')
    with pytest.raises(PermissionError):inv.verify_synthesis_result({'angle':inventory['unique_keys'][0]},inventory['unique_keys'][0],binding)
    r=out/inv.RECEIPT;r.chmod(0o644);r.write_text('{}')
    with pytest.raises(PermissionError):inv.verify_inventory(out,binding)


def test_final_aggregation_rejects_inventory_mutated_after_valid_child(tmp_path):
    out,inventory,binding=frozen(tmp_path);row={'angle':inventory['unique_keys'][0],**binding}
    inv.verify_synthesis_result(row,inventory['unique_keys'][0],binding)
    p=out/inv.INVENTORY;p.chmod(0o644);p.write_text('{}')
    with pytest.raises(PermissionError):inv.verify_inventory(out,binding)


def test_chunked_terminal_STOP_blocks_a_new_job(tmp_path):
    out=tmp_path/RESULT;out.mkdir(parents=True)
    st.dump_json(out/'terminal_status.json',{'status':'STOP','technical_reason':'x'*600000})
    with pytest.raises(PermissionError):claim_job(tmp_path,{'stage':'CD','output':str(out/'CD.json'),'marker_sha256':'mock'})


def test_parent_wrong_runtime_rejected_before_marker_or_child(tmp_path,monkeypatch):
    script=ROOT/'scripts/tracks/algorithm_codesign/run_df_correction_comparison_v2.py'
    spec=importlib.util.spec_from_file_location('prepared_coordinator',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path);monkeypatch.setattr(module,'verify_launch',lambda root:{'source':'mock'})
    art=tmp_path/module.ART;art.mkdir(parents=True);(art/'contract_v2.json').write_text('{}')
    def reject(*args):raise PermissionError('synthetic runtime mismatch')
    monkeypatch.setattr(sys,'argv',['runner'])
    monkeypatch.setattr('trottertracks.algorithm_codesign.df_correction_comparison.worker.runtime_check',reject)
    with pytest.raises(PermissionError,match='runtime mismatch'):module.main()
    assert not (tmp_path/module.RESULT/'one_shot_marker.json').exists()


def test_runtime_identity_hash_mismatch_is_not_a_fixture_evaluation():
    with pytest.raises(PermissionError,match='executable hash'):
        runtime_check({'runtimes':{'numeric':{'executable_sha256':'0'*64}}},'CD')
