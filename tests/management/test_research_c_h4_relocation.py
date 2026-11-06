"""Invented J and temporary archives only; real molecular arrays never opened."""
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_h4_z_relocation as m
from trottertracks import c_near_objective as near
from trottertracks.c1_general_k_adapter import general_k_coefficients


@pytest.fixture(autouse=True)
def forbid_real_archive_reads(monkeypatch, tmp_path):
    original=np.load
    def guarded(path,*args,**kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path), 'REAL_SAVED_ARRAY_READ_FORBIDDEN'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)


def invented(n=4, rank=8, ld=2):
    a=np.random.default_rng(913).normal(size=(rank,n,n))
    return (a+a.transpose(0,2,1))/2,ld


@pytest.mark.parametrize('n,rank,ld',[(1,2,1),(2,3,1),(4,8,2)])
def test_signed_mass_identities_and_existing_diagnostics(n,rank,ld):
    J,ld=invented(n,rank,ld);before=J.tobytes();row=m.diagnose(J,ld=ld)
    old=near.mass_diagnostics(J,near.Objective(rank,n,ld,1.,1.,0.))
    assert row['M']==old['exact_tail_mass'] and row['E']==old['exact_entry_mass']
    assert row['E_minus_M']==pytest.approx(old['row_cancellation_gap'])
    assert row['M']==pytest.approx(row['Z']+row['M_ZZ'])
    assert row['E']/5-1e-12<=row['M_ZZ']<=row['E']/3+1e-12
    assert all(row['checks'].values()) and J.tobytes()==before


@pytest.mark.parametrize('sign',[1.,-1.])
def test_no_cancellation_and_original_frame_order(sign):
    J,ld=invented();J=sign*abs(J)
    row=m.diagnose(J,ld=ld);assert row['M']==pytest.approx(row['E'])
    J[:ld]+=np.eye(4)*1e12
    assert m.diagnose(J,ld=ld)==row


@pytest.mark.parametrize('kind',['diag','offdiag','cancel'])
def test_exact_diagonal_offdiagonal_weights_and_zero_rows(kind):
    J=np.zeros((2,2,2))
    J[1]={'diag':[[4.,0.],[0.,-8.]],'offdiag':[[0.,2.],[2.,0.]],
          'cancel':[[1.,-1.],[-1.,1.]]}[kind]
    r=m.diagnose(J,ld=1)
    if kind=='diag':assert r['D_diag']==12 and r['O']==0 and r['M_ZZ']==3 and r['M_ZZ']==r['E']/5
    if kind=='offdiag':assert r['D_diag']==0 and r['O']==2 and r['M_ZZ']==2 and r['M_ZZ']==r['E']/3
    if kind=='cancel':assert r['Z']==0 and r['E_minus_M']==4


def test_general_K_I_Z_ZZ_map_and_relocation_pointwise_all_occupancies():
    J,_=invented(n=2,rank=3,ld=1)
    for a in J:
        K=np.repeat(np.repeat(a/2,2,axis=0),2,axis=1)
        terms=general_k_coefficients(K)
        np.testing.assert_allclose([v for s,v in terms if s],near.pr.coefficients(a[None],np)[0],atol=1e-14)
        r=m.diagnose(a[None],ld=0)
        assert r['M_ZZ']==pytest.approx(sum(abs(v) for s,v in terms if len(s)==2))
        assert r['Z']==pytest.approx(sum(abs(v) for s,v in terms if len(s)==1))
        # Artificial diagonal number polynomials, not a molecular Hamiltonian.
        for occupation in itertools.product((0,1),repeat=4):
            n=np.array(occupation);z=1-2*n
            number=float(n@K@n)
            s=a.sum(axis=1);one_body=float(np.repeat(s,2)@n);shift=-float(s.sum())
            identity=terms[0][1]
            zz=sum(v*z[supp[0]]*z[supp[1]] for supp,v in terms if len(supp)==2)
            assert number==pytest.approx(identity+one_body+shift+zz,abs=1e-13)


@pytest.mark.parametrize('fault',['asymmetric','nan','complex','float32','ld','shape'])
def test_invalid_coefficients_fail_without_modifying_input(fault):
    a,ld=invented()
    if fault=='asymmetric':a[2,0,1]+=1
    if fault=='nan':a[2,0,1]=np.nan
    if fault=='complex':a=a.astype(complex)
    if fault=='float32':a=a.astype(np.float32)
    if fault=='ld':ld=8
    if fault=='shape':a=a[:,:,:3]
    with pytest.raises(ValueError):m.diagnose(a,ld=ld)


def tagged(a,arm):
    return dict(m.diagnose(a,ld=2),candidate_id=arm,start='C0',budget=64,strength=.1,arm=arm)


def test_zero_mass_and_ratio_rules():
    zero=np.zeros((8,4,4));p,e=tagged(zero,'PR'),tagged(zero,'ENTRY')
    assert p['Z_over_M'] is None and p['zero_mass']
    r=m.compare_pair(p,e,pairing='matched')
    assert r['ratio'] is None and r['classification']=='NUMERICALLY_SAME'
    J,_=invented();e=tagged(J,'ENTRY');p=tagged(J*.5,'PR')
    r=m.compare_pair(p,e,pairing='matched')
    assert r['ratio']==pytest.approx(.5) and r['classification']=='PR_LOWER'
    assert m.compare_pair(e|dict(arm='PR'),p|dict(arm='ENTRY'),pairing='matched')['classification']=='ENTRY_LOWER'
    with pytest.raises(ValueError):m.compare_pair(p,e|dict(budget=16),pairing='matched')
    with pytest.raises(ValueError):m.compare_pair(p,e|dict(strength=.03),pairing='matched')
    assert m.compare_pair(p,e|dict(strength=.03),pairing='frozen_fair_policy_different_strength_explicit')['ratio']==pytest.approx(.5)


@pytest.fixture
def archive(tmp_path):
    root=tmp_path;rows=[];reports={}
    for start,strength,budget,arm in itertools.product(('C0','P1','P2'),(.03,.1,.3),(16,64),('PR','ENTRY')):
        key=f'{start}_{strength}_{arm}';candidate=f'{key}_{budget}';path=root/(candidate+'.npz')
        J,ld=invented();J=J*(1+len(rows)/100)
        # Reading U would require pickle; the existing whitelist only reads J.
        np.savez(path,J=J,U=np.array([object()],dtype=object))
        d=m.diagnose(J,ld=ld);ident=m.io.array_record(J)
        r=reports.setdefault(key,dict(start=start,strength=strength,arm=arm,source_sha256='a'*64,checkpoints={}))
        r['checkpoints'][str(budget)]=dict(array_file_sha256=m.io.sha(path),array_identity=dict(J=ident))
        rows.append(dict(candidate_id=candidate,start=start,strength=strength,budget=budget,arm=arm,
            path=str(path),file_sha256=m.io.sha(path),J_identity=ident,
            original_SOURCE_sha256='a'*64,report_path=key+'.json',
            series='picklefix' if (start,arm,strength)==('C0','ENTRY',.3) else 'original',
            saved_masses=dict(M=d['M'],E=d['E'],E_minus_M=d['E_minus_M'])))
    for key,r in reports.items():m.io.save(root/(key+'.json'),r)
    for row in rows:row['report_sha256']=m.io.sha(root/row['report_path'])
    inventory=dict(system=m.SYSTEM,checkpoints=rows,references=[],protected=[],
        inherited_scientific_SOURCE_files={},scientific_archive_root=str(root))
    m.io.save(root/'inputs.json',inventory);m.io.save(root/'source.json',dict(files={}))
    contract=dict(scope=m.VERSION,system=m.SYSTEM,retry=0,resume=0,mandatory_stop=True,
        numerics=dict(atol_Ha=m.ATOL,rtol=m.RTOL),maximum=dict(checkpoints=36,neutral_references=1,J_reads=37),
        comparisons=dict(matched_pairs=18,frozen_common='C0/.1/64',frozen_fair='C0/PR.1/ENTRY.03/64'),
        inventory='inputs.json',inventory_sha256=m.io.sha(root/'inputs.json'),SOURCE='source.json',
        SOURCE_sha256=m.io.sha(root/'source.json'),fresh_output='fresh')
    m.io.save(root/'contract.json',contract)
    return root,inventory,contract


def test_passive_preflight_and_synthetic_end_to_end(archive):
    root,inventory,c=archive
    before={p:m.io.sha(p) for p in root.iterdir() if p.is_file()}
    p=m.preflight(root,'contract.json');assert p['J_arrays_decoded']==0
    out=m.execute(root,'contract.json')
    result=m.io.read(out/'result.json')
    assert len(result['rows'])==36 and len(result['matched_pairs'])==18
    assert result['total_RZ'] is None and m.io.read(out/'STOP.json')['mandatory_stop']
    assert all(m.io.sha(p)==h for p,h in before.items())
    with pytest.raises(FileExistsError):m.execute(root,'contract.json')


@pytest.mark.parametrize('fault',['missing','changed_bytes','missing_row','wrong_metadata'])
def test_missing_or_corrupt_saved_data_fail_in_byte_preflight(archive,fault):
    root,inv,c=archive
    if fault=='missing':Path(inv['checkpoints'][0]['path']).unlink()
    if fault=='changed_bytes':Path(inv['checkpoints'][0]['path']).write_bytes(b'corrupted')
    if fault=='missing_row':inv['checkpoints'].pop()
    if fault=='wrong_metadata':inv['checkpoints'][0]['arm']='ENTRY'
    if fault in ('missing_row','wrong_metadata'):
        (root/'inputs.json').write_text(json.dumps(inv))
        c['inventory_sha256']=m.io.sha(root/'inputs.json')
        (root/'contract.json').write_text(json.dumps(c))
    with pytest.raises((ValueError,FileNotFoundError)):m.preflight(root,'contract.json')
    assert not (root/'fresh').exists()


def test_saved_mass_disagreement_stops_synthetic_analysis(archive):
    root,inv,c=archive
    inv['checkpoints'][0]['saved_masses']['M']+=1
    (root/'inputs.json').write_text(json.dumps(inv))
    c['inventory_sha256']=m.io.sha(root/'inputs.json')
    (root/'contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='SAVED_MASS_MISMATCH'):m.execute(root,'contract.json')
    stop=m.io.read(root/'fresh/STOP.json')
    assert stop['failed'] and stop['checkpoint_rows']==0


def test_import_is_passive():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_h4_z_relocation;assert 'numpy' not in sys.modules;assert 'ffsim' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_byte_preflight_cannot_decode_arrays(archive,monkeypatch):
    root,inv,c=archive
    def forbidden(*a,**k):raise AssertionError('NUMERICAL_DECODE_IN_PREFLIGHT')
    monkeypatch.setattr(np,'load',forbidden)
    assert m.preflight(root,'contract.json')['J_arrays_decoded']==0


@pytest.mark.parametrize('kind',['source','protected'])
def test_source_or_protected_evidence_changes_are_rejected(archive,kind):
    root,inv,c=archive
    path=root/'protected.txt';path.write_text('original')
    if kind=='source':
        (root/'source.json').write_text(json.dumps(dict(files={'protected.txt':m.io.sha(path)})))
        c['SOURCE_sha256']=m.io.sha(root/'source.json')
    else:
        inv['protected']=[dict(path=str(path),sha256=m.io.sha(path))]
        (root/'inputs.json').write_text(json.dumps(inv));c['inventory_sha256']=m.io.sha(root/'inputs.json')
    (root/'contract.json').write_text(json.dumps(c));path.write_text('changed')
    with pytest.raises(ValueError):m.preflight(root,'contract.json')


def test_neutral_reference_is_separate_from_36_checkpoints(archive):
    root,inv,c=archive
    row=dict(inv['checkpoints'][0],candidate_id='NEUTRAL',arm='REFERENCE',strength=None,budget=0,
        saved_masses={'M':inv['checkpoints'][0]['saved_masses']['M']})
    m.io.save(root/'neutral.json',dict(starts=dict(C0=dict(identity=dict(
        cold_sha256=row['file_sha256'],array_identity=dict(J=row['J_identity']))))))
    row.update(report_path='neutral.json',report_sha256=m.io.sha(root/'neutral.json'))
    inv['references']=[row];(root/'inputs.json').write_text(json.dumps(inv))
    c['inventory_sha256']=m.io.sha(root/'inputs.json');(root/'contract.json').write_text(json.dumps(c))
    result=m.io.read(m.execute(root,'contract.json')/'result.json')
    assert len(result['rows'])==36 and len(result['references'])==1


def test_unrelated_historical_document_is_not_a_live_dependency(archive):
    root,inv,c=archive
    inv['inherited_scientific_SOURCE_files']={'unrelated_historical_note.md':'a'*64}
    (root/'inputs.json').write_text(json.dumps(inv))
    c['inventory_sha256']=m.io.sha(root/'inputs.json');(root/'contract.json').write_text(json.dumps(c))
    assert m.preflight(root,'contract.json')['J_arrays_decoded']==0
