"""Source/metadata lineage and invented failures only; no real array decode."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from trottertracks import c_direct_m0_bound as m

ROOT=Path(__file__).resolve().parents[2]


def contract():return m.io.read(ROOT/m.CONTRACT)


def test_producer_lineage_binds_same_saved_numeric_oracle_input():
    c=contract();b=m.oracle_binding(ROOT,c)
    assert b==c['oracle_target_identity']
    assert b['H_file']==c['roles']['sector'] and b['ground_file']==c['roles']['ground']
    assert 'same existing numerical oracle' in b['qualification']
    assert b['exact_symmetry_evaluated'] is b['state_distance_evaluated'] is False
    assert 'runtime' in b['kind'].lower()


@pytest.mark.parametrize('field',['sector','ground','candidate'])
def test_different_input_or_ground_producer_path_rejected(field):
    c=contract();c['roles'][field]='wrong-file.npz'
    with pytest.raises(ValueError,match='PATHS'):m.oracle_binding(ROOT,c)


def test_different_candidate_oracle_target_rejected():
    c=contract();c['identity']['candidate']='0'*64
    with pytest.raises(ValueError,match='CANDIDATE'):m.oracle_binding(ROOT,c)


def test_source_producer_edits_do_not_transfer_old_lineage(monkeypatch):
    original=m.io.sha
    def sha(path):return '0'*64 if str(path).endswith(m.PRODUCERS[0]) else original(path)
    monkeypatch.setattr(m.io,'sha',sha)
    with pytest.raises(ValueError,match='PRODUCER_SOURCE'):m.oracle_binding(ROOT,contract())


@pytest.mark.parametrize('field,value',[
    ('m',1),('occurrences',2),('precision_bits',160),('fresh_sampling',1),
    ('new_build_compile',1),('new_eigenvalue_proposal',2),('Stage_C_executable',True),
    ('retry',1),('maximum_logical_units',16)])
def test_fixed_scope_does_not_expand(tmp_path,field,value):
    c=contract();c[field]=value;p=tmp_path/'bad.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='SCOPE'):m.preflight(ROOT,p)


def test_fixed_word_cannot_be_replaced_inside_same_scope(tmp_path):
    c=contract();c['word']=list(reversed(c['word']));p=tmp_path/'bad.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='FIXED_WORD'):m.preflight(ROOT,p)


def test_binding_cannot_declare_unevaluated_symmetry_as_pass(tmp_path):
    c=contract();c['oracle_target_identity']['exact_symmetry_evaluated']=True
    p=tmp_path/'bad.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='TARGET_BINDING'):m.preflight(ROOT,p)


@pytest.mark.parametrize('sha',['0'*40,m.BASE])
def test_scientific_source_commit_must_contain_the_frozen_manifest(tmp_path,sha):
    c=contract();c['scientific_SOURCE']=sha
    p=tmp_path/'bad-source.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='SCIENTIFIC_SOURCE_COMMIT'):m.preflight(ROOT,p)


def test_default_run_never_infers_execution_permission(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('NO_PREFLIGHT_OR_DECODE_BEFORE_PERMISSION')
    monkeypatch.setattr(m,'preflight',forbidden)
    with pytest.raises(PermissionError,match='15_CALL'):m.run(ROOT)


def test_default_cli_preflight_has_no_scientific_imports():
    code='''
import sys
from pathlib import Path
class Block:
 def find_spec(self,name,*a):
  if name.split('.')[0] in ('numpy','scipy','flint','qiskit','ffsim','pyscf'):
   raise AssertionError('NO_SCIENTIFIC_IMPORT:'+name)
sys.meta_path.insert(0,Block())
from trottertracks import c_direct_m0_bound as m
r=m.preflight(Path.cwd())
assert r['scientific_calls']==r['arrays_decoded']==r['QPY_decoded']==0
assert not r['numerical_oracle_qualified'] and r['mandatory_stop']
assert r['maximum_logical_units']==15
'''
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_current_preparation_or_frozen_contract_never_performs_math():
    c=contract();r=m.preflight(ROOT)
    assert r['scientific_calls']==0 and r['maximum_logical_units']==15
    assert not r['numerical_oracle_qualified'] and not r['Stage_C_ready']
    assert r['execution_ready']==(c['executable'] and c['scientific_SOURCE'] is not None)


@pytest.mark.parametrize('error',[RuntimeError('INVENTED failure'),KeyboardInterrupt('INVENTED interruption')])
def test_partial_exception_and_stop_are_retained(tmp_path,monkeypatch,error):
    c=dict(source_manifest='source.json',fresh_output='fresh')
    (tmp_path/'contract.json').write_text(json.dumps(c));(tmp_path/'source.json').write_text('{}')
    monkeypatch.setattr(m,'preflight',lambda *a:dict(execution_ready=True,oracle_binding=dict(kind='INVENTED')))
    monkeypatch.setattr(m.prior.z,'runtime_model',lambda:None)
    def fail(root,c,out,call):
        call('metadata_load','INVENTED',lambda:m.io.save(out/'completed.json',dict(invented=True)))
        raise error
    monkeypatch.setattr(m.prior,'execute',fail)
    with pytest.raises(type(error)):m.run(tmp_path,tmp_path/'contract.json',execute_science=True)
    out=tmp_path/'fresh';s=m.io.read(out/'STOP.json')
    assert s['scientific_units']==1 and s['failed'] is not None
    assert (out/'completed.json').exists() and (out/'partial_failure.json').exists()
    assert s['retry']==s['resume']==0 and s['mandatory_stop']


def test_previous_draft_and_stop_remain_distinct():
    r=m.prior.preflight(ROOT)
    assert not r['execution_ready']
    assert 'saved_H_vs_existing_own_DF_oracle_target_identity' in r['missing']
    assert m.CONTRACT!=m.prior.CONTRACT
