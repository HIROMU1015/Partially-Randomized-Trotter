"""Scalar-only NPZ fixtures + metadata/mock runtime. No scientific run/circuit."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest

from trottertracks import candidate_price_adapter as D, ld12_price_requests as L
from trottertracks import compiled_price_fast as F

ROOT=Path(__file__).resolve().parents[1]
BASE='240a6be56c5096e7ede27a27fa8f7d8dbfee290b'
C_SOURCE='3802bcc0212325091515325a889fcab20eabd909'
D_SOURCE='748be6875b5007707d864e16aa7a4fc4250f3714'
CONTEXT=D.NativeRuntime.context
TECH=dict(fixture='TECHNICAL_ONLY_NO_EXECUTION')


@pytest.fixture(autouse=True)
def no_science(monkeypatch):
    def fail(*a,**kw):raise AssertionError('No physical H/preparation/sampling/build/compile')
    for cls,name in ((D.NativeRuntime,'__init__'),(D.NativeRuntime,'context'),
                     (D.NativeRuntime,'sample'),(D.Adapter,'__init__'),(D.Adapter,'acquire')):
        monkeypatch.setattr(cls,name,fail)
    monkeypatch.setattr(L,'_saved_constant',lambda root,binding:constant(binding))


def constant(binding,value=.125):
    return dict(quantity='HAMILTONIAN_CONSTANT',value=value,source=dict(
        fixture='ARTIFICIAL_METADATA',hamiltonian_hash=binding['hamiltonian'],
        snapshot=dict(sha256=binding['snapshot'])))


@pytest.fixture(scope='module')
def assets():return L._read(ROOT)


def req(assets,value=.125,axes=('X','Y')):
    menu,configs=assets;c=configs['PR-LD2']
    return L._request(menu,c,'PR-LD2',12,axes,TECH,constant(c['spec']['binding'],value))


def fixture_inputs(tmp_path,value=1.25,scalar_shape=False):
    snapshot=tmp_path/'snapshot.npz'
    raw=np.array(value if scalar_shape else [value],dtype='<f8')
    # The object-valued OTHER member must never be loaded.
    np.savez(snapshot,constant=raw,OTHER=np.array([object()],dtype=object))
    a=np.ascontiguousarray(raw);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    b=dict(snapshot=sha(snapshot),hamiltonian='INVENTED_H',state='INVENTED_STATE',rank=10)
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(status='COMPLETE_MANDATORY_STOP',sha256=sha(snapshot),
        bytes=snapshot.stat().st_size,hamiltonian_hash=b['hamiltonian'],state_hash=b['state'],rank=10,
        arrays=dict(constant=dict(shape=[1],dtype='<f8',sha256=hashlib.sha256(a.tobytes()).hexdigest())))))
    files={key:dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
        for key,path in [('H4_tol_snapshot.npz',snapshot),('H4_tol_receipt.json',receipt)]}
    return files,b


@pytest.mark.parametrize('value',[0.,1.25])
@pytest.mark.parametrize('scalar_shape',[False,True])
def test_source_bound_constant_reads_only_constant_member(tmp_path,monkeypatch,value,scalar_shape):
    files,binding=fixture_inputs(tmp_path,value,scalar_shape)
    original=np.lib.npyio.NpzFile.__getitem__;seen=[]
    def get(archive,name):
        assert name=='constant';seen.append(name);return original(archive,name)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',get)
    result=L._constant_from_inputs(files,binding,{'fixture':'PINNED_INPUT_LOCATOR'})
    assert result['quantity']=='HAMILTONIAN_CONSTANT' and result['value'].hex()==value.hex()
    assert seen==['constant'] and result['source']['snapshot']==files['H4_tol_snapshot.npz']


@pytest.mark.parametrize('problem',['missing','bytes','snapshot','H','array'])
def test_missing_or_mismatched_constant_source_has_no_fallback(tmp_path,problem):
    files,b=fixture_inputs(tmp_path)
    if problem=='missing':Path(files['H4_tol_snapshot.npz']['path']).unlink()
    if problem=='bytes':files['H4_tol_receipt.json']['sha256']='FOREIGN'
    if problem=='snapshot':b['snapshot']='FOREIGN'
    if problem=='H':b['hamiltonian']='FOREIGN'
    if problem=='array':
        path=Path(files['H4_tol_receipt.json']['path']);data=json.loads(path.read_text())
        data['arrays']['constant']['sha256']='FOREIGN';path.write_text(json.dumps(data))
        files['H4_tol_receipt.json'].update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises((ValueError,FileNotFoundError)):L._constant_from_inputs(files,b,{'fixture':'SOURCE'})


@pytest.mark.parametrize('value',[0.,.125])
def test_tail_identity_and_total_are_distinct_typed_quantities(assets,value):
    r=req(assets,value);pc=r['physical_context'];s=pc['input_source']['scalar_binding']
    tail=next(i for i in assets[0]['candidates'] if i['label']=='PR-LD2')['tail_descriptor']['scalar']
    assert s['hamiltonian_constant']['quantity']=='HAMILTONIAN_CONSTANT'
    assert s['tail_identity']==dict(quantity='TAIL_EXTRACTED_IDENTITY_COEFFICIENT',value=tail,
        source=dict(menu=L._ref('MENU.json',label='PR-LD2'),tail=pc['input_source']['tail']))
    assert pc['expected_scalar'].hex()==(value+tail).hex()
    assert s['expected_total']['value']==pc['expected_scalar']
    assert r['context_key']==D._key(pc)
    assert r['request_id']==D._key({k:v for k,v in r.items() if k!='request_id'})


def test_tail_record_cannot_supply_hamiltonian_constant(assets):
    menu,cs=assets;c=cs['PR-LD2'];bad=constant(c['spec']['binding'])
    bad['quantity']='TAIL_EXTRACTED_IDENTITY_COEFFICIENT'
    with pytest.raises(ValueError,match='distinct'):L._request(menu,c,'PR-LD2',12,('X','Y'),TECH,bad)


def test_original_runtime_binary64_guard_accepts_total_rejects_tail_and_one_ulp(assets,monkeypatch):
    r=req(assets);pc=r['physical_context'];c=pc['physical_candidate']
    runtime=D.NativeRuntime.__new__(D.NativeRuntime)
    runtime.binding=deepcopy(pc['binding']);runtime.input_source=deepcopy(pc['input_source'])
    runtime.execution_source=TECH;runtime.contexts={};runtime.preparations={}
    runtime.ham=NS(constant=.125,n_blocks=10)
    tail=pc['input_source']['scalar_binding']['tail_identity']['value']
    prep=NS(ld=2,num_system_qubits=8,hamiltonian_hash=pc['binding']['hamiltonian'],
        deterministic_fragment_indices=(0,1),randomized_block_indices=tuple(range(2,10)),
        preparation_hash='MOCK_ONLY',extracted_identity_coefficient=tail,deterministic_blocks=(),
        rte_preparation=NS(symbolic_tail=NS(lambda_r=float(D.p.accounting.exact_scalar(pc['split']['lambda_R'])))))
    tools=NS(h={'prepare':lambda *a:prep},Accounting=None,native=None,compile=None,
        CompilerSettings=None,fingerprint_context=None)
    monkeypatch.setattr(runtime,'_tools',lambda:tools)
    assert CONTEXT(runtime,r).scalar.hex()==pc['expected_scalar'].hex()
    for wrong in (tail,math.nextafter(pc['expected_scalar'],math.inf)):
        other=D.from_b(assets[1]['PR-LD2'],12,c,input_source=pc['input_source'],execution_source=TECH,expected_scalar=wrong)
        with pytest.raises(ValueError,match='scalar'):CONTEXT(runtime,other)


def test_xy_source_and_provenance_keys_are_preserved(assets):
    x=req(assets,axes=('X',));y=req(assets,axes=('Y',));both=req(assets)
    assert x['context_key']==y['context_key']==both['context_key']
    assert len({r['request_id'] for r in (x,y,both)})==3
    pending=D.pending_price_records(both)
    assert [r['axis'] for r in pending]==['X','Y']
    assert all(r['mean']['rz_count'] is None for r in pending)
    menu,cs=assets;changed=constant(cs['PR-LD2']['spec']['binding'])
    changed['source']['fixture']='DIFFERENT_SCIENTIFIC_SOURCE'
    other=L._request(menu,cs['PR-LD2'],'PR-LD2',12,('X','Y'),TECH,changed)
    assert other['context_key']!=both['context_key']
    assert other['identity']==both['identity'] and other['law_binding']==both['law_binding']


def test_actual_structural_loop_applies_scalar_once_per_xy_word():
    class Accounting:
        def __init__(self):self.native=self.folded=NS(num_qubits=9,num_clbits=1);self.word=[]
        def gate(self,*args,**kwargs):self.word.append(args)
        def tail(self,event):self.word.append(('MOCK_TAIL',))
        def block(self,*args):self.word.append(('MOCK_BLOCK',))
        def finish(self):return self.word
    words=F.structural_pair(Accounting,[[]],dict(system='H4',q=3,T=.75),.375,['MOCK']*3)
    for word in words:
        scalar=[x for x in word if x[0]=='p']
        assert scalar==[('p',(8,),(-.375*.75,),'scalar_once')]
        assert sum(x[0]=='MOCK_TAIL' for x in word)==3
        assert word[-1][0]=='measure'
    assert not any(x[0]=='sdg' for x in words[0]) and any(x[0]=='sdg' for x in words[1])


def c_api():
    # Read-only C source and its pinned D constructor. No C branch edits/import graph.
    scope=dict(__package__='trottertracks',__name__='trottertracks._D748_scalar_metadata')
    exec(compile(D.pinned._blob(ROOT,D_SOURCE,'src/trottertracks/candidate_price_adapter.py'),
                 D_SOURCE+':candidate_price_adapter.py','exec'),scope)
    api=NS(**{k:scope[k] for k in ('from_a_job','check_request')})
    tree=ast.parse(D.pinned._blob(ROOT,C_SOURCE,'src/trottertracks/rpe_d_price_connection.py'))
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_requests','_paths')]
    I=NS(providers=lambda root:NS(Price=api),C=NS(p=D.p),price_context=lambda r,m:dict(fixture_m=m))
    env=dict(I=I,PATH='PR_CANDIDATE_PAIRED_PATH_PRICE')
    exec(compile(tree,C_SOURCE+':metadata_subset','exec'),env)
    return NS(**{k:env[k] for k in ('_requests','_paths')})


def test_C380_metadata_reconstructs_new_request_and_rejects_wrong_resolved_scalar(assets):
    r=req(assets);config=assets[1]['PR-LD2'];jobs=[]
    for row in config['rounds']:
        pc={k:deepcopy(config['spec'][k]) for k in ('binding','profile','split','law')}
        pc.update(physical_candidate=row['physical_candidate'],structure=row['structure'])
        jobs.append(dict(m=row['m'],B_configuration_id=config['id'],B_candidate_id=row['candidate']['id'],
            B_identity=row['identity'],B_law_binding=row['law_binding'],physical_context=pc))
    plan=dict(configuration=config,pending=dict(jobs=jobs));rows=D.pending_price_records(r)
    for row in rows:row.update(d_request=r,physical_context=dict(fixture_m=12))
    api=c_api();matched=api._requests(ROOT,plan,rows)
    assert matched[r['request_id']]==r
    pc=r['physical_context']
    path=dict(quantity='PR_CANDIDATE_PAIRED_PATH_PRICE',request=r,qualified_expectation=False,
        cohort='ARTIFICIAL_NO_EVENTS',index=0,conditional=False,records=[],
        source=dict(execution=pc['execution_source'],input=pc['input_source']),
        resolved_context=dict(preparation_hash='MOCK',tail_hash='MOCK',lambda_R=pc['split']['lambda_R'],
            scalar=pc['expected_scalar'],deterministic_fragments=[0,1],randomized_fragments=list(range(2,10))))
    assert api._paths(matched,[path],D.p.accounting.exact_scalar)
    path['resolved_context']['scalar']=pc['input_source']['scalar_binding']['tail_identity']['value']
    with pytest.raises(ValueError,match='scalar'):api._paths(matched,[path],D.p.accounting.exact_scalar)


def test_historical_request_remains_separate_and_evidence_files_unmodified(assets):
    path='docs/research/d_ld2_m12_t2_20261113/run01/REQUEST.json'
    old=json.loads(D.pinned._blob(ROOT,BASE,path));before=deepcopy(old);new=req(assets)
    assert 'scalar_binding' not in old['physical_context']['input_source']
    assert new['request_id']!=old['request_id'] and new['context_key']!=old['context_key']
    assert new['identity']==old['identity'] and new['law_binding']==old['law_binding']
    assert old==before
    assert not subprocess.check_output(['git','diff',BASE,'--',
        'docs/research/d_ld2_m12_t2_20261113','docs/research/d_t2_scalar_diagnostic_20261113'])
