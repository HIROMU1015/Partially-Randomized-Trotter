"""Synthetic metadata/I-O only. No registered data or computation is generated."""
from pathlib import Path
import hashlib
import json
import multiprocessing
import os
import sys
from fractions import Fraction as F

import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_et_storage_v5 import storage as st
from trottertracks.algorithm_codesign.df_correction_comparison import storage as old
from trottertracks.algorithm_codesign.df_correction_et_storage_v5.guard import Supervisor,Limits,CapFailure


def original_bytes(data,ensure_ascii=True):return (json.dumps(data,default=str,indent=2,ensure_ascii=ensure_ascii)+'\n').encode()


def disk_bytes(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


@pytest.mark.parametrize('data',[{'fraction':F(1,7),'tuple':(1,'x',True),'phase':'-1/17'},[1,2,'λ'],{'t':'b','ref':{'kind':st.MAGIC}},None])
def test_lossless_value_and_exact_original_bytes(tmp_path,data):
 p=tmp_path/'test.json';st.dump_json(p,data)
 assert b''.join(st.iter_verified_bytes(p))==original_bytes(data)
 assert st.load_json(p)==json.loads(original_bytes(data))
 assert st.artifact_identity(p)['sha256']==hashlib.sha256(original_bytes(data)).hexdigest()
 with pytest.raises(FileExistsError):st.dump_json(p,data)


@pytest.mark.parametrize('ensure_ascii',[True,False])
def test_unicode_and_large_shared_blob_chunks_individually_UTF8(tmp_path,ensure_ascii):
 data={'recursive_payload':{'text':'軌道λ🧪'*45000},'phase':'α'}
 p=tmp_path/'unicode.json';st.dump_json(p,data,ensure_ascii=ensure_ascii)
 assert b''.join(st.iter_verified_bytes(p))==original_bytes(data,ensure_ascii)
 for file in tmp_path.rglob('*'):
  if file.is_file():assert file.stat().st_size<=524288;file.read_bytes().decode('utf-8')


def test_large_payload_state_updates_and_native_prefix_final_share_immutable_bytes(tmp_path):
 payload={'data':'x'*2000000,'operation_name':'circuit-MOCK','global_phase':'-1/37'}
 row={'inventory':{'raw_angles':['1/17']*10000},'numeric_error':'1/100000000000'}
 p=tmp_path/'observation.json';data={'recursive_payload':payload,'input_arrays':{'mock':{'raw_bytes_hex':'ab'*300000}},'phase':'FIRST'}
 st.dump_json(p,data);before=disk_bytes(tmp_path)
 for i in range(4):data['phase']='STATE-'+str(i);st.dump_json(p,data,replace_existing=True)
 assert disk_bytes(tmp_path)-before<50000
 prefix={'prefix':{'Pauli_coefficients':[{'beta':'1/17'}]*30000,'native':[row],'compile_calls':2}}
 st.dump_json(tmp_path/'native.prefix.json',prefix)
 before=disk_bytes(tmp_path);st.dump_json(tmp_path/'native.json',prefix['prefix'])
 assert disk_bytes(tmp_path)-before<20000
 assert st.load_json(p)==data and st.load_json(tmp_path/'native.json')==prefix['prefix']


@pytest.mark.parametrize('mutation',['raw_blob','chunk','index','snapshot','missing','escape','logical_hash','old_snapshot'])
def test_corrupted_graph_is_rejected_before_returning_data(tmp_path,mutation):
 data={'recursive_payload':{'value':'x'*800000},'phase':'first'};p=tmp_path/'object.json'
 st.dump_json(p,data);data['phase']='second';st.dump_json(p,data,replace_existing=True)
 pointer=json.loads(p.read_text());sp=tmp_path/pointer['snapshot']['path'];snapshot=json.loads(sp.read_text())
 node=next(v for k,v in snapshot['tree']['v'] if k=='recursive_payload');ref=node['ref'];idx=tmp_path/ref['path'];index=json.loads(idx.read_text())
 if mutation=='raw_blob':
  small=tmp_path/'small.json';st.dump_json(small,{'recursive_payload':{'a':1}})
  sn=st._root(small);sr=next(v for k,v in sn['tree']['v'] if k=='recursive_payload')['ref'];(tmp_path/sr['path']).write_text('{}')
  with pytest.raises(ValueError):st.load_json(small)
  return
 if mutation=='chunk':(tmp_path/index['chunks'][0]['path']).write_text('bad')
 elif mutation=='index':idx.write_text('{}')
 elif mutation=='snapshot':sp.write_text('{}')
 elif mutation=='missing':(tmp_path/index['chunks'][0]['path']).unlink()
 elif mutation=='old_snapshot':(tmp_path/snapshot['previous_snapshot']['path']).write_text('{}')
 elif mutation=='escape':
  node['ref']['path']='../outside.json';raw=(json.dumps(snapshot,separators=(',',':'))+'\n').encode();sp.write_bytes(raw);pointer['snapshot']['sha256']=hashlib.sha256(raw).hexdigest();pointer['snapshot']['bytes']=len(raw);p.write_text(json.dumps(pointer))
 else:
  snapshot['logical_sha256']='0'*64;raw=(json.dumps(snapshot,separators=(',',':'))+'\n').encode();sp.write_bytes(raw);pointer['snapshot']['sha256']=hashlib.sha256(raw).hexdigest();pointer['snapshot']['bytes']=len(raw);p.write_text(json.dumps(pointer))
 with pytest.raises((ValueError,FileNotFoundError)):st.load_json(p)


def test_interruption_preserves_published_pointer_and_never_adopts_partial_blob(tmp_path,monkeypatch):
 p=tmp_path/'interrupt.json';a={'recursive_payload':{'data':'original'*1000},'phase':'old'};st.dump_json(p,a);before=p.read_bytes();real=st._write
 calls=0
 def fail(root,path,raw):
  nonlocal calls
  if path.name.startswith('chunk-'):
   calls+=1
   if calls==2:raise OSError('simulated interrupted write')
  return real(root,path,raw)
 monkeypatch.setattr(st,'_write',fail)
 b={'recursive_payload':{'data':'x'*900000},'phase':'new'}
 with pytest.raises(OSError):st.dump_json(p,b,replace_existing=True)
 assert p.read_bytes()==before and st.load_json(p)==a
 monkeypatch.setattr(st,'_write',real)
 with pytest.raises(ValueError,match='Incomplete prior blob'):st.dump_json(p,b,replace_existing=True)


def test_failure_at_latest_pointer_publish_preserves_old_snapshot(tmp_path,monkeypatch):
 p=tmp_path/'pointer.json';st.dump_json(p,{'phase':'first'});before=p.read_bytes()
 monkeypatch.setattr(st.os,'replace',lambda *a:(_ for _ in ()).throw(OSError('publish')))
 with pytest.raises(OSError):st.dump_json(p,{'phase':'new'},replace_existing=True)
 assert p.read_bytes()==before and st.load_json(p)=={'phase':'first'}


def test_legacy_reader_compatibility_and_legacy_replacement_prohibited(tmp_path):
 p=tmp_path/'legacy.json';old.dump_json(p,{'data':'x'*800000})
 assert st.load_json(p)==old.load_json(p)
 assert b''.join(st.iter_verified_bytes(p))==b''.join(old.iter_verified_bytes(p))
 with pytest.raises(FileExistsError):st.dump_json(p,{'data':'new'},replace_existing=True)
 assert not (tmp_path/'_records.lock').exists()
 q=tmp_path/'legacy_raw.json';q.write_text('[1,2]')
 assert st.load_json(q)==[1,2]
 with pytest.raises((ValueError,TypeError)):st.dump_json(q,{'new':1},replace_existing=True)


def test_writer_enforces_fixed_output_budget_without_destroying_checkpoint(tmp_path,monkeypatch):
 monkeypatch.setattr(st,'OUTPUT_BYTES',1024*1024)
 p=tmp_path/'cap.json';data={'recursive_payload':{'text':'x'*600000},'phase':'old'};st.dump_json(p,data);before=p.read_bytes()
 with pytest.raises(st.StorageBudgetExceeded):st.dump_json(p,{'recursive_payload':{'text':'y'*600000},'phase':'new'},replace_existing=True)
 assert p.read_bytes()==before and st.load_json(p)==data
 assert disk_bytes(tmp_path)<=st.OUTPUT_BYTES


def write_shared(root,label):
 from trottertracks.algorithm_codesign.df_correction_et_storage_v5 import storage
 storage.dump_json(Path(root)/(label+'.json'),{'recursive_payload':{'text':'shared'*150000},'phase':label})


def test_two_processes_publish_shared_blob_once_and_restore_both(tmp_path):
 ctx=multiprocessing.get_context('fork');ps=[ctx.Process(target=write_shared,args=(tmp_path,label)) for label in ('a','b')]
 for p in ps:p.start()
 for p in ps:p.join(10);assert p.exitcode==0
 assert disk_bytes(tmp_path)<1000000
 for label in ('a','b'):assert st.load_json(tmp_path/(label+'.json'))['phase']==label


def test_new_guard_reads_decoded_child_CPU_receipt(tmp_path):
 code=('from pathlib import Path;from trottertracks.algorithm_codesign.df_correction_et_storage_v5.storage import dump_json;'
       'dump_json(Path('+repr(str(tmp_path/'child.json'))+'),{"value":"x"*600000,"_process_receipt":{"CPU_seconds":0.01}})')
 old_path=os.environ.get('PYTHONPATH');os.environ['PYTHONPATH']=str(ROOT/'src')
 try:
  sup=Supervisor(Limits(10,10,10000,10000),tmp_path)
  jobs=[{'label':'synthetic-only','output':str(tmp_path/'child.json'),'command':[sys.executable,'-B','-c',code]}]
  assert len(sup.phase(jobs,Limits(5,5,10000,10000,5,5),1,per_key=True))==1
 finally:
  if old_path is None:os.environ.pop('PYTHONPATH',None)
  else:os.environ['PYTHONPATH']=old_path


def test_mutable_input_cannot_reuse_stale_blob_or_erase_prior_values(tmp_path):
 data={'recursive_payload':{'phase':'1/17','name':'circuit-MOCK'},'stage':'a'};p=tmp_path/'mutation.json'
 st.dump_json(p,data);first=st._root(p)
 data['recursive_payload']['phase']='-1/17';data['stage']='b';st.dump_json(p,data,replace_existing=True)
 assert st.load_json(p)==data
 previous=st._snapshot(tmp_path,st._root(p)['previous_snapshot'],p.name)
 assert st._decode(tmp_path,previous['tree'])['recursive_payload']['phase']=='1/17'
 assert previous['logical_sha256']==first['logical_sha256']


def test_logical_digest_and_physical_history_both_bound_to_native_inventory(tmp_path):
 from trottertracks.algorithm_codesign.df_correction_et_storage_v5 import acceptance as ac
 from trottertracks.algorithm_codesign.df_correction_comparison.inventory_integrity import freeze_inventory,verify_inventory
 from qiskit import QuantumCircuit
 q=QuantumCircuit(1);q.rz(.071,0)
 native={r:{'native':[{'arm':'A','controlled':False,'inventory':ac.native_inventory(q)}]} for r in (3,6)}
 for r in (3,6):st.dump_json(tmp_path/('E-r'+str(r)+'.json'),native[r])
 inventory={'unique_keys':[],'native_result_bindings':ac.bind_native_results(tmp_path,native)}
 binding=freeze_inventory(tmp_path,inventory,{'scope':'MOCK_ONLY'},'mock-marker')
 ac.verify_native_bindings(tmp_path,verify_inventory(tmp_path,binding))
 p=tmp_path/'E-r3.json';before=st.artifact_identity(p)
 st.dump_json(p,native[3],replace_existing=True)
 after=st.artifact_identity(p)
 assert before['sha256']==after['sha256'] and before['bytes']==after['bytes']
 assert before['physical_descriptor_sha256']!=after['physical_descriptor_sha256']
 with pytest.raises(PermissionError):ac.verify_native_bindings(tmp_path,inventory)
