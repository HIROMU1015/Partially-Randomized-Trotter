"""Passive real byte checks and fully invented end-to-end runner fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from trottertracks import c_compiled_raw_readiness as m

ROOT=Path(__file__).resolve().parents[2]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def fake_root(root,*,unmatched=False,broken=False):
    from trottertracks import c_pr_aware_df_fixed_certificate as fixed
    from trottertracks.c_per_word_population_range import _digest
    c=deepcopy(m.read(ROOT/m.PILOT));c['packages']={}
    # Every numeric operator below is invented; no saved molecular record is read.
    a,b=fixed.identity(9),fixed.identity(9)
    if unmatched:a['gates']=[dict(name='cx',qubits=[0,1],params_hex=[])]
    if broken:a['phase_hex']='nan'
    values=dict(compiled=a,raw=b,old_certificate=dict(left_id=_digest(a),right_id=_digest(b),
        lower_exact='0',upper_exact='2',wall_seconds=.1),old_perword=dict(compiled_id=_digest(a),
        raw_id=_digest(b),binding=dict(word=[[word,0.0.hex()] for word in c['word']])))
    c['roles']={k:'INVENTED/'+k+'.json' for k in values}
    for k,v in values.items():save(root/c['roles'][k],v)
    c['parent_contract']='INVENTED/parent.json';save(root/c['parent_contract'],dict(identity=c['identity'],word=c['word']))
    c['saved_inputs']={p:m.sha(root/p) for p in [*c['roles'].values(),c['parent_contract']]}
    fake_source='INVENTED/fake_source.py';(root/fake_source).write_text('# invented only\n')
    c['source_manifest']='INVENTED/SOURCE.json';save(root/c['source_manifest'],dict(files={fake_source:m.sha(root/fake_source)}))
    c['source_manifest_sha256']=m.sha(root/c['source_manifest']);save(root/m.PILOT,c)
    f=deepcopy(m.read(ROOT/m.FAILURE));f.update(source_manifest=c['source_manifest'],source_manifest_sha256=c['source_manifest_sha256'])
    save(root/m.FAILURE,f);return c


def test_real_cold_preflight_only_bytes_and_metadata():
    code="""
import sys
from pathlib import Path
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','scipy','flint','qiskit','ffsim','pyscf','openfermion'):
   raise AssertionError('NO_NUMERIC_IMPORT:'+name)
sys.meta_path.insert(0,Block())
from trottertracks import c_compiled_raw_readiness as m
r=m.preflight(Path.cwd())
assert r['scientific_calls']==r['arrays_decoded']==r['native_records_decoded']==0
assert r['actual_H4_new_upper'] is None and r['actual_H4_correspondence_checked'] is False
"""
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


@pytest.mark.parametrize('field,value',[('precision_bits',128),('maximum_logical_units',7),
    ('commute_CX',True),('retry',1),('word',['different']),('identity',{}),('new_compile',1)])
def test_contract_changes_rejected(tmp_path,field,value):
    c=fake_root(tmp_path);c[field]=value;save(tmp_path/m.PILOT,c)
    with pytest.raises(ValueError):m.preflight(tmp_path)


def test_no_unapproved_real_execution():
    with pytest.raises(PermissionError):m.run_pilot(ROOT)


def test_invented_runner_six_calls_and_no_old_fallback(tmp_path,monkeypatch):
    from trottertracks import c_arb_column_batch as old
    monkeypatch.setattr(old,'compare',lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO_DENSE_FALLBACK')))
    c=fake_root(tmp_path);r=m.run_pilot(tmp_path,execute_science=True)
    stop=m.read(tmp_path/c['fresh_output']/'STOP.json')
    assert r['status']=='LOCAL_SUPPORT_FIXED_PAIR_COMPLETE'
    assert stop['scientific_units']==6 and stop['failed'] is None
    assert r['certificate']['columns_evaluated']==0 and r['certificate']['dimension']==512
    assert r['Stage_C_ready'] is False
    with pytest.raises(ValueError,match='FRESH'):m.run_pilot(tmp_path,execute_science=True)


def test_invented_unmatched_stops_without_Arb_fallback(tmp_path):
    c=fake_root(tmp_path,unmatched=True);r=m.run_pilot(tmp_path,execute_science=True)
    assert r['upper_exact'] is None and r['local_Arb_evaluations']==0
    stop=m.read(tmp_path/c['fresh_output']/'STOP.json')
    assert stop['scientific_units']==4 and stop['call_counts'].get('local_certificate',0)==0


def test_invented_identity_FAIL_preserves_partial_and_STOP(tmp_path):
    c=fake_root(tmp_path,broken=True)
    with pytest.raises(ValueError):m.run_pilot(tmp_path,execute_science=True)
    out=tmp_path/c['fresh_output'];stop=m.read(out/'STOP.json')
    assert stop['scientific_units']==2 and stop['failed'] is not None
    assert (out/'partial_failure.json').exists() and stop['retry']==stop['resume']==0
