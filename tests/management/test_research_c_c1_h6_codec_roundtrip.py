"""Eval-env artificial Hamiltonian/dispatch/STOP tests; no molecular values."""
from pathlib import Path
import json,sys
from unittest.mock import Mock,patch
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h6_codec_roundtrip as m


@pytest.mark.parametrize('corrupt_after',[False,True])
def test_artificial_parent_end_to_end_equivalence_budget_and_failure_STOP(tmp_path,corrupt_after):
    root=tmp_path/'root';root.mkdir();out=root/'out'
    J=np.array([[[.2,.03],[.03,.1]]]);U=np.array([[[.8,-.6],[.6,.8]]])
    packet=dict(h=np.diag([-.2,.1]),g=m.old.reconstruct(J,U),constant=np.asarray(.125))
    np.savez(root/'input.npz',**packet)
    ctx=m.old.molecule.SectorCAR(2);physical=ctx.physical(packet['h'],packet['g'],packet['constant'])
    reference=m.old.molecule.rank_assessment(ctx,packet,J,U,physical,14)[0]
    rows={}
    for r in (14,18):m.io.save(root/f'R{r}.json',reference);rows[str(r)]=f'R{r}.json'
    contract=dict(entry='UNUSED',original_input='input.npz',DF_reserve_Ha=.00159/8,saved_sector_rows=rows)
    def worker(*a,**kw):
        p=out/'codec';p.mkdir()
        for r in (14,18):
            for state in ('before','after'):
                np.savez(p/f'R{r}_{state}.npz',J=J*(1.2 if corrupt_after and state=='after' else 1),U=U)
        m.io.save(p/'STOP.json',dict(call_counts=m.CODEC_LIMITS))
    forbidden=Mock(side_effect=AssertionError('No fit/PF/native'))
    with patch.object(m,'preflight',return_value=contract),patch.object(m.subprocess,'run',side_effect=worker),patch.object(m.old.molecule,'factor',forbidden),patch.object(m.old.molecule,'PF_rounds',forbidden),patch.object(m.old.molecule,'native_H2',forbidden):
        if corrupt_after:
            with pytest.raises(ValueError):m.run(root,root/'contract.json',out,Path('/venv/python'))
        else:m.run(root,root/'contract.json',out,Path('/venv/python'))
    stop=m.io.read(out/'STOP.json');assert stop['failed']==corrupt_after
    assert stop['retry']==stop['resume']==0 and stop['scientific_units']<=20
    assert (out/'R14_after_sector.json').exists();forbidden.assert_not_called()
    if not corrupt_after:
        assert stop['scientific_units']==20
        assert all(x['status']=='PASS' for x in m.io.read(out/'result.json')['ranks'].values())
    else:assert (out/'failure.json').exists() and not (out/'result.json').exists()
    original=(out/'STOP.json').read_bytes()
    with patch.object(m,'preflight',return_value=contract),pytest.raises(FileExistsError):m.run(root,root/'contract.json',out,'UNUSED')
    assert (out/'STOP.json').read_bytes()==original


def test_byte_preflight_is_passive_and_detects_changed_scientific_conditions(tmp_path):
    source=tmp_path/'source';source.write_text('artificial')
    manifest=tmp_path/'SOURCE.json';m.io.save(manifest,dict(files={'source':m.io.sha(source)}))
    c=dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT',ranks=[14,18],codec_calls=m.CODEC_LIMITS,
        sector_calls=m.SECTOR_LIMITS,maximum_logical_units=20,new_fit=0,retry=0,resume=0,
        DF_reserve_Ha=.00159/8,guard_policy=m.old.molecule.GUARD_POLICY,allowances=m.ALLOWANCES,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(manifest),saved_inputs={},evaluation_packages={})
    path=tmp_path/'contract.json';m.io.save(path,c)
    with patch.object(np,'load',side_effect=AssertionError('No molecular decode')):assert m.preflight(tmp_path,path)==c
    c['DF_reserve_Ha']*=2;path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='SCOPE'):m.preflight(tmp_path,path)


def test_call_scope_excludes_initialization_search_and_optimizer():
    assert sum(m.CODEC_LIMITS.values())==9 and sum(m.SECTOR_LIMITS.values())==11
    assert not set(m.CODEC_LIMITS|m.SECTOR_LIMITS)&{'fit','initializer','gradient','PF','compile','shots'}
