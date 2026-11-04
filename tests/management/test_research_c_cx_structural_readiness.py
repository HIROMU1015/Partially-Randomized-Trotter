"""Permission/lifecycle tests use invented descriptors under tmp_path only."""
import json
from pathlib import Path
import pytest
from trottertracks import c_cx_structural_readiness as m,c_cx_structural_blocks as b
from test_research_c_cx_structural_blocks import g,r


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def mock_root(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(m,'preflight',lambda root:{'mock':True})
    a,c=r(9,[g('cx',[0,1]),g('rz',[0],.25),g('cx',[0,1])]),r(9,[g('rz',[0],.25)])
    if kind=='support':a=r(9,[g('cx',[q,q+1]) for q in range(4)]);c=r(9)
    saved,p=b.make_plan(a,c)
    word=['INVENTED']
    values=dict(compiled=a,raw=c,saved_plan=dict(plan=saved),
        old_certificate=dict(left_id=b._digest(a),right_id=b._digest(c),lower_exact='0',upper_exact='2',wall_seconds=1.),
        old_perword=dict(compiled_id=b._digest(a),raw_id=b._digest(c),binding=dict(word=[['INVENTED',0]])))
    if kind=='identity_FAIL':values['old_perword']['raw_id']='bad'
    roles={key:'invented/'+key+'.json' for key in values}
    for key,value in values.items():save(tmp_path/roles[key],value)
    save(tmp_path/'source.json',{'files':{}})
    save(tmp_path/m.CONTRACT,dict(roles=roles,fresh_output=m.FAMILY+'/block-pilot01',source_manifest='source.json',word=word))


def test_permission_is_separate_and_default_denies(tmp_path):
    with pytest.raises(PermissionError):m.run_pilot(tmp_path)


@pytest.mark.parametrize('kind,units',[('complete',6),('support',4),('identity_FAIL',2),('numeric_FAIL',4)])
def test_mock_calls_partial_STOP_and_no_resume(tmp_path,monkeypatch,kind,units):
    mock_root(tmp_path,monkeypatch,kind)
    if kind=='numeric_FAIL':
        def fail(*a,**k):raise ValueError('INVENTED_ENCLOSURE_FAIL')
        monkeypatch.setattr(b,'evaluate',fail)
    if kind.endswith('FAIL'):
        with pytest.raises(ValueError):m.run_pilot(tmp_path,execute_science=True)
    else:m.run_pilot(tmp_path,execute_science=True)
    out=tmp_path/m.FAMILY/'block-pilot01';stop=m.read(out/'STOP.json')
    assert stop['scientific_units']==units and stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    assert (stop['failed'] is not None)==kind.endswith('FAIL')
    if kind.endswith('FAIL'):assert (out/'partial_failure.json').exists()
    previous=(out/'science_calls.jsonl').read_bytes()
    with pytest.raises(FileExistsError):m.run_pilot(tmp_path,execute_science=True)
    assert previous==(out/'science_calls.jsonl').read_bytes()


def test_real_preflight_cold_bytes_and_metadata_only():
    import subprocess,sys
    root=Path(__file__).resolve().parents[2]
    script="""
import builtins
original=builtins.__import__
def restricted(name,*a,**k):
 if name.split('.')[0] in ('numpy','scipy','qiskit','flint'):raise AssertionError('NUMERICAL_IMPORT:'+name)
 return original(name,*a,**k)
builtins.__import__=restricted
from trottertracks.c_cx_structural_readiness import preflight
assert preflight('.') ['scientific_calls']==0
"""
    subprocess.run([sys.executable,'-B','-c',script],cwd=root,check=True)


@pytest.mark.parametrize('field,value',[('maximum_logical_units',8),('precision_bits',160),('retry',1),('block_policy',dict(max_support=4,guard=0,exact_clifford=True))])
def test_fixed_schema_mutation_rejected_before_science(tmp_path,field,value):
    root=Path(__file__).resolve().parents[2];c=m.read(root/m.CONTRACT);c[field]=value
    save(tmp_path/m.CONTRACT,c)
    with pytest.raises(ValueError,match='FIXED_BLOCK_PILOT_SCHEMA'):m.preflight(tmp_path)
