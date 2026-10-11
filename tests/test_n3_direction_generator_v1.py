"""New synthetic fixtures only; never evaluate the 12 frozen transfer cases."""
from fractions import Fraction as F
import importlib.util
import itertools
from pathlib import Path
import numpy as np
import pytest
from trottertracks.representation_exploration import n3_direction_generator_v1 as n
spec=importlib.util.spec_from_file_location('independent_n3_checker',Path(__file__).resolve().parents[1]/'scripts/verify_n3_direction_generator_v1.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

@pytest.mark.parametrize('M,ok,status',[
 ([[2,1],[1,2]],True,'PSD'),([[0,0],[0,1]],True,'PSD'),([[1,1],[1,1]],True,'PSD'),([[0,1],[1,0]],False,'ZERO_PIVOT_NONZERO_ROW'),([[-1,0],[0,2]],False,'NEGATIVE_PIVOT'),([[1,2],[2,1]],False,'NEGATIVE_PIVOT')])
def test_exact_psd_positive_negative_zero(M,ok,status):
    M=[[F(x) for x in row] for row in M];proof=n.psd(M,n.Arithmetic());assert proof['accepted']==ok and proof['status']==status;assert audit.Audit().proof(M,proof)==ok

def fixture(ext=3,v=F(1,20)):
    h=[[F(0),F(1,100),v],[F(1,100),F(2),F(0)],[v,F(0),F(ext)]]
    return {'id':'synthetic_generated','base':'SYNTHETIC','modes':3,'particles':1,'active':[0,1],'external':2,'scale':[1,1],'amplitude':n.enc(v),'direction':'synthetic','h':[[n.enc(x) for x in row] for row in h],'density':[{'i':2,'j':0,'coefficient':[1,10]}]}

def test_new_synthetic_generator_and_checker():
    c=fixture();r=n.generate_case(c);accepted=audit.Audit().case(c,r);assert accepted['direct_one_variable'];assert n.diagnose_case(c,r)['width']<.004

def test_zero_coupling_endpoint_and_low_external():
    for ext,ok in [(3,True),(-3,False)]:
        c=fixture(ext,F(0));r=n.generate_case(c);a=audit.Audit().case(c,r);assert a['direct_one_variable']==ok

def test_permuted_modes_and_unordered_density():
    c=fixture();p=[2,0,1];cp=dict(c);cp['h']=[[c['h'][p[i]][p[j]] for j in range(3)] for i in range(3)];cp['active']=[1,2];cp['external']=0;cp['density']=[{'i':0,'j':1,'coefficient':[1,10]}]
    r=n.generate_case(c);rp=n.generate_case(cp);assert [m['accepted'] for m in r['methods']]==[m['accepted'] for m in rp['methods']];audit.Audit().case(cp,rp)

def test_scalar_boundary_and_original_witness():
    A=[[F(0)]];v=[F(1)];ar=n.Arithmetic();w=n.original_witness(A,v,1,F(0),F(62499,250),F(0),F(-1),ar)
    assert w['accepted'] and n.dec(w['affine_margin'])==0 and n.dec(w['PSD']['pivots'][0])==0;assert audit.Audit().witness(A,v,1,F(0),F(62499,250),w)

def test_tampered_evidence_refused():
    M=[[F(1),F(0)],[F(0),F(1)]];p=n.psd(M,n.Arithmetic());p['pivots'][0]=[2,1]
    with pytest.raises(ValueError):audit.Audit().proof(M,p)

def test_negative_density_duplicate_and_oracle_refused():
    c=fixture();c['density'][0]['coefficient']=[-1,10]
    with pytest.raises(ValueError):n.decode_case(c)
    c=fixture();c['density'].append({'i':0,'j':2,'coefficient':[1,10]})
    with pytest.raises(ValueError):n.decode_case(c)
    c=fixture();c['true_gap']=1
    with pytest.raises(ValueError):n.decode_case(c)

def test_arithmetic_cap():
    with pytest.raises(ArithmeticError):n.Arithmetic().check(F(1,1<<4096))

def test_no_many_body_selector_access(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('many-body truth access')
    monkeypatch.setattr(n,'sector_operators',forbidden);monkeypatch.setattr(n,'diagnose_case',forbidden);assert n.generate_case(fixture())['methods'][4]['accepted']

def test_cached_sector_matches_direct_occupation_reference():
    # Synthetic sector construction checked against an independent occupation-list implementation.
    nmode,nu=4,2;active=(0,1,3);states,ops,aids=n.sector_operators(nmode,nu,active);h=np.arange(16,dtype=float).reshape(4,4)/100;h=h+h.T;ref=np.zeros((6,6));occs=list(itertools.combinations(range(nmode),nu))
    for col,occ in enumerate(occs):
        for q in occ:
            remaining=list(occ);removed_sign=(-1)**remaining.index(q);remaining.remove(q)
            for p in range(nmode):
                if p in remaining:continue
                inserted_sign=(-1)**sum(x<p for x in remaining);target=tuple(sorted(remaining+[p]));ref[occs.index(target),col]+=h[p,q]*removed_sign*inserted_sign
    assert np.allclose(np.tensordot(h,ops,axes=2),ref,atol=1e-16);assert n.sector_operators(nmode,nu,active)[1] is ops
