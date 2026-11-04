import json,subprocess,sys
from pathlib import Path
import pytest
from trottertracks import c_pauli_frame_readiness as m,c_cx_structural_blocks as b
from test_research_c_relative_pauli import g,r


def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_default_no_permission_refuses_before_any_input(tmp_path):
    with pytest.raises(PermissionError):m.run_pilot(tmp_path)


@pytest.mark.parametrize('kind,units',[('complete',6),('support',4),('identity_FAIL',2),('numerical_FAIL',4)])
def test_mock_lifecycle_partial_STOP_and_no_retry(tmp_path,monkeypatch,kind,units):
    from trottertracks import c_pauli_frame_enclosure as frame
    monkeypatch.setattr(m,'preflight',lambda root:{'mock':True})
    a,c=r(9,[g('z',[0]),g('cx',[0,1]),g('z',[0])]),r(9,[g('cx',[0,1])])
    if kind=='support':a,c=r(9,[g('cx',[q,q+1]) for q in range(4)]),r(9)
    saved,p=b.make_plan(a,c);keya,keyc=b._digest(a),b._digest(c)
    word=['INVENTED'];old=dict(left_id=keya,right_id=keyc,lower_exact='0',upper_exact='2',wall_seconds=1.)
    values=dict(compiled=a,raw=c,perword=dict(compiled_id=keya,raw_id=keyc,binding=dict(word=[['INVENTED',0]])),sharp=old,
        loose=dict(old),primitive_plan=dict(plan=saved),old_structural_plan=dict(plan=p))
    if kind=='identity_FAIL':values['sharp']=dict(old,left_id='bad')
    roles={k:'invented/'+k+'.json' for k in values}
    for k,v in values.items():save(tmp_path/roles[k],v)
    save(tmp_path/'source.json',dict(files={}))
    save(tmp_path/m.CONTRACT,dict(roles=roles,word=word,fresh_output=m.FAMILY+'/frame-pilot01',source_manifest='source.json'))
    if kind=='numerical_FAIL':
        def fail(*a,**k):raise ValueError('INVENTED_ARITHMETIC_FAIL')
        monkeypatch.setattr(frame,'evaluate',fail)
    if kind.endswith('FAIL'):
        with pytest.raises(ValueError):m.run_pilot(tmp_path,execute_science=True)
    else:m.run_pilot(tmp_path,execute_science=True)
    out=tmp_path/m.FAMILY/'frame-pilot01';s=m.read(out/'STOP.json');previous=(out/'science_calls.jsonl').read_bytes()
    assert s['scientific_units']==units and s['retry']==s['resume']==0 and s['mandatory_stop']
    assert (s['failed'] is not None)==kind.endswith('FAIL')
    with pytest.raises(FileExistsError):m.run_pilot(tmp_path,execute_science=True)
    assert previous==(out/'science_calls.jsonl').read_bytes()


def test_new_real_preflight_is_cold_bytes_only():
    root=Path(__file__).resolve().parents[2]
    code="""
import builtins
old=builtins.__import__
def deny(name,*a,**k):
 if name.split('.')[0] in ('numpy','qiskit','flint','scipy'):raise AssertionError(name)
 return old(name,*a,**k)
builtins.__import__=deny
from trottertracks.c_pauli_frame_readiness import preflight
assert preflight('.') ['scientific_calls']==0
"""
    subprocess.run([sys.executable,'-B','-c',code],cwd=root,check=True)


@pytest.mark.parametrize('field,value',[('maximum_logical_units',7),('method','other'),('precision_bits',160),('retry',1)])
def test_contract_mutation_FAIL(tmp_path,field,value):
    root=Path(__file__).resolve().parents[2];c=m.read(root/m.CONTRACT);c[field]=value;save(tmp_path/m.CONTRACT,c)
    with pytest.raises(ValueError,match='FIXED_PAULI_FRAME_CONTRACT'):m.preflight(tmp_path)
