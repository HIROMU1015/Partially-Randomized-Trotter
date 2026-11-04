"""Synthetic NPZ/H/basis only for numerical execution; real files bytes-only."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from trottertracks import c_saved_eigh_difference as m
from trottertracks import c_hermitian_oracle_bridge as bridge

ROOT = Path(__file__).resolve().parents[2]


def fixture(tmp_path, a, *, dtype='<c16', basis=None):
    root = tmp_path/'invented'; root.mkdir()
    c = deepcopy(m.fixed_spec())
    c['source_manifest'] = 'invented_SOURCE.json'
    c['shape'] = list(np.asarray(a).shape)
    c['expected_basis_indices'] = list(range(len(a)))
    c['fresh_output'] = 'fresh'
    for role, data in [('H', {'H':np.asarray(a, dtype=dtype)}),
                       ('basis', {'sector_basis_indices':np.asarray(
                           list(range(len(a))) if basis is None else basis, dtype='<i8')})]:
        p = root/c['roles'][role]; p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **data)
    c['saved_inputs'] = {p:m.io.sha(root/p) for p in c['roles'].values()}
    m.io.save(root/c['source_manifest'], {'kind':'ARTIFICIAL_ONLY'})
    c['source_manifest_sha256'] = m.io.sha(root/c['source_manifest'])
    metadata = {
        m.ORIGIN:{'endpoints':{'PR':{'reference':c['roles']['H']}},
                  'candidate_sha256':{'PR':m.CANDIDATE},
                  'physical_reference':c['roles']['basis'],
                  'policy':{'oracle':'own_DF_ground_state_oracle'},
                  'saved_inputs':c['saved_inputs']},
        m.LINEAGE:{'H':{'path':c['roles']['H'],'sha256':c['saved_inputs'][c['roles']['H']]},
                   'numpy':{'version':'1.26.4','UPLO':'default L'}},
        m.OLD_STOP:{'scientific_units':3,'retry':0,'resume':0,'mandatory_stop':True,
                    'failed':{'error':'EXACT_SYMMETRY_REQUIRED_NO_HERMITIANIZATION'}},
    }
    for name, value in metadata.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);m.io.save(p,value)
    out = root/c['fresh_output']; out.mkdir()
    return root, c, out


def attempt(tmp_path, a, **kw):
    root,c,out = fixture(tmp_path, a, **kw)
    result = m._attempt(root,c,out)
    return result,out


@pytest.mark.parametrize('a,expected_square', [
    ([[0,1-2j],[1+2j,3]], F(0)),
    ([[0,3],[1,2]], F(4)),  # upper-only mismatch
    ([[0,1],[3,2]], F(4)),  # lower determines the replacement upper
    ([[2j,0],[0,-3j]], F(13)),
    ([[0,1j],[1j,2]], F(4)),
    ([[0,float.fromhex('0x1p-100')],[0,2]], F(1,2**200)),
    ([[0,2**40],[0,2]], F(2**80)),
])
def test_full_four_call_synthetic_npz_pipeline(tmp_path,a,expected_square):
    r,out = attempt(tmp_path,a)
    assert F(r['frobenius_squared_exact']) == expected_square
    lo,hi = map(F,r['frobenius_norm_interval_exact'])
    assert 0 <= lo <= hi and lo*lo <= expected_square <= hi*hi
    assert r['precision_bits'] == 192 and r['rounding_enclosed']
    if expected_square == 13:
        assert lo < hi  # irrational square root must have outward width
    assert r['difference_exactly_zero'] == (expected_square == 0)
    assert not r['operator_norm_is_exact_value'] and r['magnitude_threshold'] is None
    assert r['research_target_adopted'] is False and r['XY_signal'] is None
    stop=m.io.read(out/'STOP.json')
    assert stop['scientific_units'] == stop['maximum'] == 4
    assert stop['call_counts'] == m.LIMITS and stop['failed'] is None
    assert stop['retry'] == stop['resume'] == 0 and stop['mandatory_stop']
    rows=[json.loads(s) for s in (out/'science_calls.jsonl').read_text().splitlines()]
    assert [s['kind'] for s in rows if s['event']=='attempt'] == list(m.LIMITS)
    assert [s['event'] for s in rows].count('COMPLETE') == 4
    saved=m.io.read(out/'H_saved_exact.json')['components']
    eigh=m.io.read(out/'H_eigh_exact.json')['components']
    delta=m.io.read(out/'difference_exact.json')['components']
    for i in range(len(a)):
        for j in range(len(a)):
            for k in range(2):
                assert F(delta[i][j][k]) == F(saved[i][j][k])-F(eigh[i][j][k])
    b=np.array([[complex(float(F(x)),float(F(y))) for x,y in row] for row in eigh])
    # Observational check on invented data supplements the exact square proof.
    assert np.linalg.norm(np.asarray(a)-b,2) <= float(hi)*(1+1e-14)+1e-300


def test_binary64_bits_and_subnormal_are_not_decimal_rounded(tmp_path):
    tiny=float.fromhex('0x0.0000000000001p-1022')
    root,c,out=fixture(tmp_path,[[0,.1+tiny*1j],[0,2]])
    a,_=m.load_exact(root,c)
    assert a[0][1] == (F(.1),F(tiny))
    assert a[0][1][0] != F(1,10)


def test_lower_completion_and_no_input_overwrite(tmp_path):
    root,c,out=fixture(tmp_path,[[1+2j,9-5j],[1+3j,4-7j]])
    before={p:(root/p).read_bytes() for p in c['roles'].values()}
    m._attempt(root,c,out)
    b=m.io.read(out/'H_eigh_exact.json')['components']
    assert b == [[['1','0'],['1','-3']],[['1','3'],['4','0']]]
    assert all((root/p).read_bytes()==value for p,value in before.items())


@pytest.mark.parametrize('fault,expected',[
    ('hash','DECODED_INPUT'),('dtype','H_SHAPE_DTYPE'),('shape','H_SHAPE_DTYPE'),
    ('nan','NONFINITE'),('basis','SECTOR_BASIS')])
def test_partial_failure_retains_journal_and_stop(tmp_path,fault,expected):
    a=[[0,0],[0,2]]
    if fault=='nan':a[0][1]=complex(float('nan'),0)
    root,c,out=fixture(tmp_path,a,dtype='<f8' if fault=='dtype' else '<c16',
                       basis=[1,0] if fault=='basis' else None)
    if fault=='hash':(root/c['roles']['H']).write_bytes(b'wrong bytes')
    if fault=='shape':c['shape']=[3,3]
    with pytest.raises(ValueError,match=expected):m._attempt(root,c,out)
    stop=m.io.read(out/'STOP.json')
    assert stop['scientific_units']==2 and stop['failed']['kind']=='arrays_load_exact'
    assert stop['mandatory_stop'] and stop['retry']==stop['resume']==0
    assert (out/'producer_binding.json').exists() and (out/'partial_failure.json').exists()
    assert not (out/'result.json').exists()


def test_invalid_enclosure_keeps_earlier_exact_evidence(tmp_path,monkeypatch):
    root,c,out=fixture(tmp_path,[[0,1],[0,2]])
    original=bridge.difference_bound
    def invalid(a,b):return dict(original(a,b),operator_norm_upper_exact='0')
    monkeypatch.setattr(bridge,'difference_bound',invalid)
    with pytest.raises(ValueError,match='ENCLOSURE'):m._attempt(root,c,out)
    assert (out/'H_eigh_exact.json').exists()
    stop=m.io.read(out/'STOP.json')
    assert stop['scientific_units']==4 and stop['failed']['kind']=='difference_norm'


def test_permission_precedes_preflight_and_every_array_access(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('NO_WORK_BEFORE_PERMISSION')
    monkeypatch.setattr(m,'preflight',forbidden)
    with pytest.raises(PermissionError,match='FOUR_CALL'):m.run(ROOT)


@pytest.mark.parametrize('field,value', [
    ('UPLO','U'),('shape',[35,35]),('dtype','<f8'),('retry',1),('resume',1),
    ('maximum_logical_units',5),('ground_eigensolve',1),('target_adopted',True)])
def test_contract_scope_changes_are_rejected_without_arrays(tmp_path,field,value):
    c=m.io.read(ROOT/m.CONTRACT);c[field]=value
    p=tmp_path/'wrong.json';m.io.save(p,c)
    with pytest.raises(ValueError,match='FIXED_DIFFERENCE_ONLY_SCOPE'):m.preflight(ROOT,p)


@pytest.mark.parametrize('fault', ['source','input','source_commit'])
def test_wrong_source_and_input_identity_without_decoding(tmp_path,fault):
    c=m.io.read(ROOT/m.CONTRACT)
    if fault=='source':c['source_manifest_sha256']='0'*64
    elif fault=='input':c['saved_inputs'][m.H_FILE]='0'*64
    else:c['scientific_SOURCE']='0'*40
    p=tmp_path/'wrong.json';m.io.save(p,c)
    with pytest.raises(ValueError,match='SOURCE|INPUT'):m.preflight(ROOT,p)


def test_consumed_fresh_output_rejects_resume(tmp_path,monkeypatch):
    consumed=tmp_path/'consumed';consumed.mkdir();m.io.save(consumed/'STOP.json',{'mandatory_stop':True})
    original=m._path
    def path(root,name):
        return consumed if name==m.fixed_spec()['fresh_output'] else original(root,name)
    monkeypatch.setattr(m,'_path',path)
    with pytest.raises(ValueError,match='FRESH_NO_RETRY_RESUME'):m.preflight(ROOT)


def test_passive_preflight_has_no_scientific_imports():
    code='''
import sys,json
from pathlib import Path
class Block:
 def find_spec(self,name,*a):
  if name.split('.')[0] in ('numpy','scipy','flint','qiskit','ffsim','pyscf'):
   raise AssertionError('NO_NUMERICAL_IMPORT:'+name)
sys.meta_path.insert(0,Block())
from trottertracks import c_saved_eigh_difference as m
r=m.preflight(Path.cwd())
assert r['scientific_calls']==r['arrays_decoded']==0
assert not r['scientific_execution_permitted'] and r['mandatory_stop']
assert r['maximum_logical_units']==4
print(json.dumps(r))
'''
    r=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,text=True,capture_output=True)
    assert r.returncode==0,r.stderr


@pytest.mark.parametrize('flag',['--resume','--retry'])
def test_no_retry_resume_cli(flag):
    r=subprocess.run([sys.executable,'-B',m.ENTRY,flag],cwd=ROOT,text=True,capture_output=True)
    assert r.returncode==2 and 'unrecognized arguments' in r.stderr
