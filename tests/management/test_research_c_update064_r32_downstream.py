"""Invented arrays/circuits only; fixed H4 candidate files remain opaque."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_update064_r32_downstream as m


def test_passive_import_scope_and_budget():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_update064_r32_downstream as m;assert m.MAX_CALLS==617;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)
    assert [sum(x.values()) for x in (m.A_LIMITS,m.B_LIMITS,m.C_LIMITS,m.D_LIMITS)] == [23,90,357,141]
    assert m.SAMPLING['events_per_sample'] == m.POLICY['r'] == 32
    assert not m.SAMPLING['paired'] and not m.SAMPLING['adaptive']
    assert not {'fit','optimizer','DF_norm'} & m.LIMITS.keys()


def metadata():
    df = dict(candidate_sha256=m.CANDIDATE, actual_rank=8, budget_PASS=True, budget_Ha=.00019875,
        sector_error=1e-9, numerical_guard=1e-9, kappa='original_g_corrected', scalar='nuclear_once')
    last = dict(id='update064', physical=df, qualification=dict(status='NUMERICAL_PILOT_ADMITTED'))
    return dict(physical=df, result=dict(trajectory=dict(history=[last], physically_qualified_updates=64)),
        STOP=dict(failed=False, mandatory_stop=True, scientific_units=1042, stop_reason='MAXIMUM_ACCEPTED_UPDATES'),
        contract=dict(scope='C_PR_REDUCED_U_BOUNDED_ARMIJO_V1', objective=dict(rank=8,ld=2),
            initial_candidate=dict(sha256=m.retuning.CANDIDATE), input_paths=dict(input='input.npz'),
            saved_inputs={'input.npz':m.INPUT}, source_manifest_sha256='SOURCE'), SOURCE={})


@pytest.mark.parametrize('fault', ['candidate','DF','nan','rank','oldinitial','STOP','SOURCE','input','endpoint'])
def test_metadata_rejects_contamination_before_decode(tmp_path, monkeypatch, fault):
    s = metadata()
    if fault == 'candidate': s['physical']['candidate_sha256'] = m.retuning.CANDIDATE
    if fault == 'DF': s['physical']['budget_PASS'] = False
    if fault == 'nan': s['physical']['sector_error'] = float('nan')
    if fault == 'rank': s['contract']['objective']['rank'] = 7
    if fault == 'oldinitial': s['contract']['initial_candidate']['sha256'] = 'OTHER'
    if fault == 'STOP': s['STOP']['failed'] = True
    if fault == 'SOURCE': s['contract']['source_manifest_sha256'] = 'WRONG'
    if fault == 'input': s['contract']['saved_inputs']['input.npz'] = 'OTHER'
    if fault == 'endpoint': s['result']['trajectory']['history'][-1]['id'] = 'update005'
    monkeypatch.setattr(m.io, 'read', lambda p:s[Path(p).name])
    monkeypatch.setattr(m.io, 'sha', lambda p:'SOURCE')
    monkeypatch.setattr(np, 'load', lambda *a, **kw:pytest.fail('numeric decode'))
    c = dict(input='input.npz', peer_paths={k:k for k in s})
    with pytest.raises(ValueError, match='UPDATE064'): m.evidence(tmp_path, tmp_path, c)


def passive(root):
    from trottertracks.c1_h4_native_fair import GAUSSIAN, PHASE
    (root/'opaque.npz').write_bytes(b'invented opaque data,not NPZ')
    entry = root/m.ENTRY; entry.parent.mkdir(); entry.write_text('# invented entry\n')
    m.io.save(root/'SOURCE.json', dict(files={m.ENTRY:m.io.sha(entry)}))
    digest = m.io.sha(root/'opaque.npz')
    c = dict(base_commit=m.BASE, candidate_commit=m.PEER, scope=m.SCOPE, implementation=m.VERSION,
        entry=m.ENTRY, policy=m.POLICY, sampling=m.SAMPLING, target=m.m4.TARGET,
        compiler=m.retuning.COMPILER, allowances=m.m3.ALLOWANCES, scenarios=list(m.perf.accounting.SCENARIOS),
        call_limits=m.LIMITS, maximum_logical_units=m.MAX_CALLS, fresh_output=m.FRESH,
        retry=0, resume=0, new_fit=0, optimizer=0, new_DF_error=0, mandatory_stop=True,
        source_manifest='SOURCE.json', source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        peer_paths={k:'opaque.npz' for k in m.PEER_ROLES}, peer_inputs={'opaque.npz':digest},
        input='opaque.npz', records={}, dependency_lock='opaque.npz', shells={},
        saved_inputs={'opaque.npz':digest}, packages={}, gaussian=GAUSSIAN, compiler_phase=PHASE)
    return c


@pytest.mark.parametrize('fault', ['none','r','SOURCE','peer','input','fresh','resume'])
def test_passive_fixed_source_peer_fresh_no_decode(tmp_path, monkeypatch, fault):
    c = passive(tmp_path); digest = c['saved_inputs']['opaque.npz']
    monkeypatch.setattr(m, 'CANDIDATE', digest); monkeypatch.setattr(m, 'INPUT', digest)
    monkeypatch.setattr(m, 'evidence', lambda *a:{})
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw:('WRONG' if fault == 'peer' else m.PEER)+'\n')
    monkeypatch.setattr(np, 'load', lambda *a, **kw:pytest.fail('numeric decode'))
    if fault == 'r': c = deepcopy(c); c['policy']['r'] = 64
    if fault == 'SOURCE': (tmp_path/m.ENTRY).write_text('changed')
    if fault == 'input': (tmp_path/'opaque.npz').write_bytes(b'changed')
    if fault == 'fresh': (tmp_path/m.FRESH).mkdir(parents=True)
    if fault == 'resume': c = deepcopy(c); c['resume'] = 1
    m.io.save(tmp_path/'c.json', c)
    if fault == 'none': assert m.preflight(tmp_path,tmp_path/'c.json',tmp_path) == c
    else:
        with pytest.raises(ValueError): m.preflight(tmp_path,tmp_path/'c.json',tmp_path)


def synthetic(root, monkeypatch):
    from test_research_c_pr_integrated_pipeline import synthetic_packet
    from trottertracks.c1_molecule_extension import SectorCAR
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    from qiskit import qpy
    p, J, U = synthetic_packet(); ctx = SectorCAR(4)
    G = np.asarray(ctx.generators(**p,J=J,U=U)); H = sum(G)
    load = m.perf.loader
    load.save_arrays(root/'input.npz', **p)
    load.save_arrays(root/'new.npz', J=J,U=U)
    load.save_arrays(root/'physical.npz', J=J,U=U,generators=G,H=H)
    load.save_arrays(root/'basis.npz', sector_basis_indices=ctx.sector.basis_indices)
    arm = m.perf.b.prepare_arm(**p,J=J,U=U,role='proposed',source_identity=m.CANDIDATE,input_identity=m.INPUT,ld=2)
    df = dict(metadata()['physical'], exact_tail_lambda_Ha=sum(abs(float(x.coefficient)) for x in arm.plan.atoms))
    saved = dict(physical=df, old_PR=dict(lanes={}), old_D=dict(ideal_exact_native={}), shells={})
    scope = m.perf.accounting.COST_SCOPE
    cost = dict(totals={k:1e12 for k in m.m4.METRICS}, cost_scope=scope)
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        saved['old_PR']['lanes'][lane] = dict(conditional_totals=cost)
    saved['old_D']['ideal_exact_native']['cost'] = cost
    c = dict(input='input.npz', peer_paths=dict(candidate='new.npz',reference='physical.npz',basis='basis.npz'),
        source_manifest_sha256='INVENTED_SOURCE', shells={})
    for label in ('head','X','Y'):
        q = m.perf.b2.shell_build(8,label); name = label+'.qpy'
        with (root/name).open('xb') as f: qpy.dump(q,f)
        c['shells'][label] = {k:name for k in ('raw','compiled')}
        saved['shells'][label] = dict(metrics=circuit_metrics(q))
    monkeypatch.setattr(m, 'evidence', lambda *a:saved)
    return c, saved


def mock_native(monkeypatch):
    def native(root,c,out,call,a,**kw):
        assert kw['sampling'] == m.SAMPLING and a['config'].rte_steps == 32
        assert a['arm'].model.source_identity == m.CANDIDATE
        rows = [{k:100 for k in m.m4.METRICS} for _ in range(32)]
        return rows, m.perf.b2.calibration(rows)
    monkeypatch.setattr(m.perf, 'native', native)


def test_artificial_new_ground_PF_RTE_lanes_cost_and_actual_sameJU_D(tmp_path, monkeypatch):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        c, saved = synthetic(tmp_path,monkeypatch); mock_native(monkeypatch)
        out = tmp_path/'out'; out.mkdir(); call = m.m4.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
        result = m.execute(tmp_path,tmp_path,c,out,call)
    assert result['status'] == 'UPDATE064_PR_AND_SAME_JU_D_COMPLETE'
    assert dict((k[2:],v) for k,v in call.counts.items() if k.startswith('A.')) == m.A_LIMITS
    assert call.counts['A.ground_reference'] == 2  # New candidate and physical,never old ground.
    assert call.counts['B.exact_evolution'] == call.counts['B.round_reference'] == 13
    assert call.counts['D.compile'] == 1 and call.counts['D.PF_round'] == 13
    assert call.counts['D.ground_load'] == 0 and sum(call.counts.values()) <= m.MAX_CALLS
    assert len(list((out/'D/circuits').glob('*.qpy'))) == 4
    assert m.io.read(out/'D/XY.json')['status'] == 'PASS'
    assert result['D']['candidate'] == result['PR']['candidate'] == m.CANDIDATE
    assert result['PR']['lanes']['evidence']['total_shots'] is None
    assert result['PR']['lanes']['evidence']['conditional_totals']['totals'] is None
    lane = result['PR']['lanes']['ideal_exact_native']; plans = lane['rounds']
    W = sum(2*r['shots']*(1<<r['m']) for r in plans)
    shell = sum(2*r['shots']*(saved['shells']['head']['metrics']['rz_count']+
        saved['shells']['X']['metrics']['rz_count']) for r in plans)
    assert lane['weighted_SE']['kernel_reuse_weight'] == W
    assert lane['conditional_totals']['totals']['rz_count'] == W*100+shell
    assert result['actual_CA_achieved'] is result['causal_objective_effect'] is result['update064_r32_optimality'] is None
    assert len(result['comparisons']) == 5
    # Ground is bound to the new Hamiltonian; a deliberately displaced old energy fails.
    aH = np.load(tmp_path/'physical.npz')['H']; ground = np.load(out/'A/ground_candidate.npz')
    psi, energy = ground['sector_state'], ground['energy']
    assert np.linalg.norm(aH@psi-energy*psi) < 1e-10
    assert np.linalg.norm(aH@psi-(energy+.1)*psi) > .09


def test_artificial_actual_native32_complete_pipeline_and_budget(tmp_path, monkeypatch):
    """Real invented full512 PR32 and D circuit paths; no native mocks."""
    with threadpoolctl_context():
        c, _ = synthetic(tmp_path,monkeypatch)
        out=tmp_path/'out';out.mkdir();call=m.m4.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
        result=m.execute(tmp_path,tmp_path,c,out,call)
    assert result['status']=='UPDATE064_PR_AND_SAME_JU_D_COMPLETE'
    for key in ('trajectory_sample','kernel_build','compile','raw_operator_gate',
            'compiled_operator_gate','adjoint_pair_gate','XY_gate','kernel_metrics'):
        assert call.counts['C.'+key]==32
    assert call.counts['D.compile']==1 and not call.stop.is_set()
    assert len(list((out/'C/circuits').glob('*.qpy')))==128
    assert sum(call.counts.values())<=617
    assert all(v<=m.LIMITS[k] for k,v in call.counts.items())
    assert m.io.read(out/'C/samples/sample00_XY.json')['status']=='PASS'
    events=m.io.read(out/'C/samples/sample00_events.json')
    assert events['JU']==m.CANDIDATE and len(events['events'])==32
    assert not events['old_cohort_pooled'] and events['quantum_shots']==0
    assert result['PR']['calibration']['cost_support'] is None


@pytest.mark.parametrize('mode', ['infeasible','unresolved','mixed_JU','compile_FAIL'])
def test_dependent_STAGE_STOP_no_fallback(tmp_path, monkeypatch, mode):
    from threadpoolctl import threadpool_limits
    c, saved = synthetic(tmp_path,monkeypatch); mock_native(monkeypatch)
    if mode == 'mixed_JU':
        with np.load(tmp_path/'new.npz') as x: J,U=x['J'],x['U']
        J[0,0,0] += .01
        (tmp_path/'new.npz').unlink(); m.perf.loader.save_arrays(tmp_path/'new.npz',J=J,U=U)
    if mode in ('infeasible','unresolved'):
        from trottertracks import c1_stage3_feasibility as f
        monkeypatch.setattr(f,'summarize',lambda *a:dict(status='UNRESOLVED' if mode=='unresolved' else 'INFEASIBLE'))
    if mode == 'compile_FAIL':
        def fail(root,c,out,call,a,**kw):
            return call('compile','injected',lambda:(_ for _ in ()).throw(ValueError('ARTIFICIAL_COMPILE_FAIL')))
        monkeypatch.setattr(m.perf,'native',fail)
    out=tmp_path/'out';out.mkdir();call=m.m4.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpoolctl_context():
        if mode in ('mixed_JU','compile_FAIL'):
            with pytest.raises(ValueError): m.execute(tmp_path,tmp_path,c,out,call)
            assert call.stop.is_set() and call.counts['D.compile'] == 0
            if mode == 'mixed_JU': assert call.counts['B.kernel_reference'] == 0
        else:
            result=m.execute(tmp_path,tmp_path,c,out,call)
            assert result['status']=='STOP_PR_UNQUALIFIED_NO_C_D' and result['D'] is None
            assert call.counts['C.compile'] == call.counts['D.compile'] == 0
            assert all(v['ratio'] is None for v in result['comparisons'].values())


def threadpoolctl_context():
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)


def test_D_unqualified_preserves_PR_but_skips_D_native(tmp_path, monkeypatch):
    from trottertracks import c1_molecule_extension as molecule
    c, _ = synthetic(tmp_path,monkeypatch); mock_native(monkeypatch)
    def rounds(G,s,call,label,**kw):
        return [call('PF_round_assessment',str(i),lambda i=i:dict(m=i,pf_operator_error=1.5)) for i in range(13)]
    monkeypatch.setattr(molecule,'PF_rounds',rounds)
    out=tmp_path/'out';out.mkdir();call=m.m4.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpoolctl_context(): result=m.execute(tmp_path,tmp_path,c,out,call)
    assert result['status']=='STOP_D_PF_UNQUALIFIED'
    assert result['PR']['calibration'] is not None and result['D']['ideal_exact_native']['cost']['totals'] is None
    assert call.counts['D.compile']==call.counts['D.kernel_build']==0
    assert result['comparisons']['update064_PR/D:ideal_exact_native']['ratio'] is None


def test_partial_failure_STOP_and_no_resume(tmp_path,monkeypatch):
    m.io.save(tmp_path/'SOURCE.json',dict(files={}))
    def preflight(root,path,peer):
        if (root/m.FRESH).exists(): raise ValueError('FRESH_NO_RETRY_RESUME')
        return dict(source_manifest='SOURCE.json')
    monkeypatch.setattr(m,'preflight',preflight)
    def execute(root,peer,c,out,call):
        return call('A.evidence_load','injected',lambda:(_ for _ in ()).throw(ValueError('INVENTED_IDENTITY_FAIL')))
    monkeypatch.setattr(m,'execute',execute)
    with pytest.raises(ValueError,match='INVENTED_IDENTITY_FAIL'): m.run(tmp_path,tmp_path/'c',tmp_path)
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json')
    assert stop['failed'] and stop['scientific_units']==1 and stop['retry']==stop['resume']==0
    with pytest.raises(ValueError,match='FRESH_NO_RETRY_RESUME'): m.run(tmp_path,tmp_path/'c',tmp_path)
