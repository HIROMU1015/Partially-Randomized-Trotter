"""Finite-family plumbing on invented data; never load the saved H4 arrays."""
from pathlib import Path
from fractions import Fraction
import json,subprocess,sys
import numpy as np
import pytest
from trottertracks import c_fixed_word_pilot as m

ROOT=Path(__file__).resolve().parents[2]


def fixture(root,monkeypatch):
    from test_research_c_pr_integrated_pipeline import synthetic_packet
    from trottertracks.c1_molecule_extension import SectorCAR
    from qiskit import qpy
    p,J,U=synthetic_packet();ctx=SectorCAR(4);G=np.asarray(ctx.generators(**p,J=J,U=U));H=sum(G)
    load=m.perf.loader
    for leaf,data in [('input',p),('candidate',dict(J=J,U=U)),('sector',dict(J=J,U=U,generators=G,H=H)),('basis',dict(sector_basis_indices=ctx.sector.basis_indices))]:load.save_arrays(root/(leaf+'.npz'),**data)
    ev,vec=np.linalg.eigh(H);load.save_arrays(root/'ground.npz',energy=ev[0],sector_state=vec[:,0])
    digest=m.io.sha(root/'candidate.npz');old_identity=m.integration.previous.identity
    monkeypatch.setattr(m.integration.previous,'identity',lambda:dict(old_identity(),candidate=digest))
    arm=m.perf.b.prepare_arm(**p,J=J,U=U,role='proposed',source_identity=digest,input_identity=m.io.sha(root/'input.npz'),ld=2)
    atoms=[dict(atom_id=a.atom_id,frame_id=a.frame_id,support=list(a.support),coefficient=str(Fraction(a.coefficient))) for a in arm.plan.atoms]
    def save(n,x):(root/n).write_text(json.dumps(x)+'\n')
    save('atoms.json',dict(atoms=atoms));save('source_old.json',dict(invented=True));oldsha=m.io.sha(root/'source_old.json')
    save('origin.json',dict(candidate_sha256=dict(PR=digest),source_manifest_sha256=oldsha))
    save('result_old.json',dict(arms=dict(PR=dict(candidate=digest,PR=dict(native_admitted=True),D=dict(saved_DF=dict(budget_PASS=True,candidate_sha256=digest))))))
    save('stop_old.json',dict(failed=False,scientific_units=1231,retry=0,resume=0,mandatory_stop=True))
    shells={}
    for label in ('head','X','Y'):
        q=m.perf.b2.shell_build(8,label);path=label+'.qpy'
        with (root/path).open('xb') as f:qpy.dump(q,f)
        shells[label]=dict(raw=path,compiled=path)
    entry=root/m.ENTRY;entry.parent.mkdir();entry.write_text('# synthetic entry\n')
    save('SOURCE.json',dict(files={m.ENTRY:m.io.sha(entry)}))
    from trottertracks.c1_h4_native_fair import GAUSSIAN,PHASE
    c=dict(m.fixed(),input='input.npz',candidate='candidate.npz',sector='sector.npz',basis='basis.npz',ground='ground.npz',atoms='atoms.json',
        origin_contract='origin.json',origin_result='result_old.json',origin_STOP='stop_old.json',origin_SOURCE='source_old.json',origin_SOURCE_sha256=oldsha,
        shells=shells,words=m.words(atoms),packages={},gaussian=GAUSSIAN,compiler_phase=PHASE,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),saved_inputs={str(p.relative_to(root)):m.io.sha(p) for p in root.iterdir() if p.is_file()})
    save('contract.json',c);return c


@pytest.mark.parametrize('fault',['none','candidate','words','source','fresh','resume'])
def test_preflight_passive_fixed_family(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive array decode'))
    if fault=='words':c['words'][0]['atom_ids'].reverse();c['words'][0]['atom_ids'][0]='WRONG'
    if fault=='candidate':(tmp_path/'candidate.npz').write_bytes(b'wrong bytes')
    if fault=='source':(tmp_path/m.ENTRY).write_text('changed')
    if fault=='fresh':(tmp_path/c['fresh_output']).mkdir(parents=True)
    if fault=='resume':c['resume']=1
    (tmp_path/'contract.json').write_text(json.dumps(c))
    if fault=='none':assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    else:
        with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'contract.json')


def test_word_rules_no_coefficient_optimization_or_random_draws():
    rows=[dict(frame_id='CDF_proposed:F'+str(f),atom_id='CDF_proposed:F'+str(f)+':'+support,support=[0] if support=='0' else [0,1]) for f in range(2,8) for support in ('0','0,1')]
    words=m.words(rows)
    assert len(words)==4 and all(len(w['atom_ids'])==32 for w in words)
    assert words==m.words(list(reversed(rows)))
    assert len({tuple(w['atom_ids']) for w in words})==4
    assert m.MAX_CALLS==86 and sum(m.COMMON.values())==35
    with pytest.raises(ValueError):m.words([])


@pytest.mark.parametrize('fail',[False,True])
def test_artificial_preparation_and_partial_STOP(tmp_path,monkeypatch,fail):
    c=fixture(tmp_path,monkeypatch);out=tmp_path/'out';out.mkdir();call=m.perf.Calls(out,m.LIMITS,m.MAX_CALLS)
    def native(root,c,out,call,a,**kw):
        assert len(kw['fixed_words'])==4 and all(len(w)==32 for w in kw['fixed_words'])
        assert a['config'].rte_steps==32 and kw['sampling']==m.POLICY
        if fail:return call('raw_operator_gate','artificial wrong phase',lambda:(_ for _ in ()).throw(ValueError('PHASE_FAIL')))
        return [dict(rz_count=1)]*4,dict(status='ARTIFICIAL_FINITE')
    if fail:
        with pytest.raises(ValueError,match='PHASE_FAIL'):m.execute(tmp_path,c,out,call,native=native)
        assert call.stop.is_set()
    else:
        r=m.execute(tmp_path,c,out,call,native=native)
        assert r['whole_law_synthesis_bound'] is None and not r['Stage_C_ready']
        assert call.counts['event_catalog']==1 and call.counts['word_gate']==1
    assert sum(call.counts.values())<=86


def test_passive_import_and_no_permission_no_runner():
    with pytest.raises(PermissionError):m.run(Path('.'),Path('absent'))
    code="import sys;from trottertracks import c_fixed_word_pilot;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','flint','ffsim'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_invented_four_word_end_to_end_real_native_CAR_P0_and_STOP(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch)
    r=m.run(tmp_path,tmp_path/'contract.json',execute_science=True)
    out=tmp_path/c['fresh_output'];stop=m.io.read(out/'STOP.json')
    assert r['status']=='FIXED_FOUR_WORD_NUMERICAL_PILOT_COMPLETE'
    assert stop['scientific_units']<=86 and not stop['failed']
    assert stop['call_counts']['relative_gaussian']<=4
    assert stop['call_counts']['compile']==stop['call_counts']['fixed_word_events']==4
    assert len(list((out/'circuits').glob('*.qpy')))==16
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    with pytest.raises(ValueError,match='FRESH_NO_RETRY_RESUME'):m.preflight(tmp_path,tmp_path/'contract.json')
