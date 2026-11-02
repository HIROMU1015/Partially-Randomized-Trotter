"""Only invented arrays, circuits and provenance. Never decodes saved H4 J/U."""
from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_stage_b2 as m
b = m.b


def artificial(n=2, rank=8, role='baseline'):
    rng = np.random.default_rng(103+n)
    a = rng.normal(size=(rank,n,n))*.003
    J = a+a.transpose(0,2,1)
    # Known sparse nontrivial rotations keep full512 artificial tests small.
    U = []
    for i in range(rank):
        v = np.eye(n); theta = .13*(i+1)
        v[:2,:2] = [[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]
        U.append(v)
    h = np.diag(np.linspace(-.2,.3,n)); h[0,1] = h[1,0] = .07
    g = np.zeros((n,)*4)
    if role == 'proposed': J = J*.7
    arm = b.prepare_arm(h,g,.31,J,np.array(U),role=role,source_identity='invented:'+role,input_identity='I',ld=2)
    return arm,dict(h=h,g=g,constant=np.array(.31))


def full_context(arm,p):
    car = b.stage1.car_bilinears(arm.model.modes)
    G = [b.stage1.raw_base(b.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)]
    G += [b.stage1.raw_frame(J,U,car)[0] for J,U in zip(arm.J,arm.U)]
    return G,m.reference_context(arm,G,car,Fraction(1,40))


def test_passive_import_and_fixed_counts():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_stage_b2 as m;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'));print(m.MAX_CALLS)"
    done = subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
    assert int(done.stdout) == sum(m.LIMITS.values())
    assert m.LIMITS['compile'] == 67 and m.LIMITS['trajectory_sample'] == 64


def test_seed_namespaces_reproducibility_and_K0_r16():
    seeds = set()
    for role in b.ARMS:
        arm,_ = artificial(role=role); tail,c,dist = b.configure(arm,Fraction(1,40),16,0)
        for source in ('sourceA','sourceB'):
            for i in range(m.SAMPLES):
                events,record = m.sample_trajectory(arm,tail,dist,c,i,source)
                assert len(events) == 16 and all(e.taylor_order == 0 for e in events)
                assert not record['reused_as_coherent_repetitions'] and record['quantum_shots'] == 0
                assert record['seed'] not in seeds; seeds.add(record['seed'])
        first,rec = m.sample_trajectory(arm,tail,dist,c,0,'sourceA')
        repeat,rec2 = m.sample_trajectory(arm,tail,dist,c,0,'sourceA')
        assert first == repeat and rec == rec2
    assert len(seeds) == 128


@pytest.mark.parametrize('n',[2,4])
def test_full_controlled_reference_precision_P0_adjoint_and_XY(n):
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    arm,p = artificial(n,rank=3); G,ctx = full_context(arm,p)
    tail,c,dist = b.configure(arm,Fraction(1,40),16,0)
    events,_ = m.sample_trajectory(arm,tail,dist,c,0,'artificial-source')
    assert any(e.application_sequence[-1].coefficient_sign < 0 for e in events)
    ref = m.trajectory_reference(arm,ctx,Fraction(1,40),events)
    # Existing uncached helper remains the numerical reference for the cache adapter.
    expected = b.trajectory_reference(arm,G,ctx['atoms'],Fraction(1,40),events,c)
    np.testing.assert_allclose(ref,expected,atol=1e-13)
    raw = b.build_kernel(arm,Fraction(1,40),events,c)
    compiled = compile_checked(raw,CompilerProfile())
    R,target,reports = m.raw_gate(raw,ref)
    C,reports2 = m.compiled_gate(compiled,R,target)
    assert len(R) == 1<<(2*n+1)
    adjoints,reports3 = m.adjoint_pair_gate(raw,compiled,R,C,target)
    assert set(adjoints) == {'raw','compiled'}
    shells = {}
    for label in m.SHELLS:
        s = m.shell_build(2*n,label); cs = compile_checked(s,CompilerProfile())
        shells[label],sr = m.shell_assessment(s,cs,label)
    rng = np.random.default_rng(7); psi = rng.normal(size=len(ref))+1j*rng.normal(size=len(ref)); psi /= np.linalg.norm(psi)
    assert m.XY_gate(R,C,ref,psi,shells)['status'] == 'PASS'
    # This is the shared helper's full-control check too; production avoids
    # decoding the same matrices twice and charging metrics before XY gates.
    assert b.assess_native_probe(raw,compiled,ref)['cost_support'] is None


@pytest.mark.parametrize('fault',['relative_phase','gaussian_direction','frame_order','compile'])
def test_controlled_and_reference_mismatches_are_not_phase_fitted(fault):
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    arm,p = artificial(rank=3); G,ctx = full_context(arm,p)
    tail,c,dist = b.configure(arm,Fraction(1,40),16,0)
    events,_ = m.sample_trajectory(arm,tail,dist,c,0,'S')
    raw = b.build_kernel(arm,Fraction(1,40),events,c)
    ref = m.trajectory_reference(arm,ctx,Fraction(1,40),events)
    if fault == 'relative_phase': raw.p(.02,arm.model.modes)
    elif fault == 'gaussian_direction':
        wrong = b.prepare_arm(**p,J=arm.J,U=arm.U.transpose(0,2,1),role='baseline',
            source_identity=arm.model.source_identity,input_identity='I',ld=2)
        wt,wc,wd = b.configure(wrong,Fraction(1,40),16,0)
        we,_ = m.sample_trajectory(wrong,wt,wd,wc,0,'S')
        raw = b.build_kernel(wrong,Fraction(1,40),we,wc)
    elif fault == 'frame_order':
        wrong = dict(ctx,left=ctx['right'],right=ctx['left'])
        ref = m.trajectory_reference(arm,wrong,Fraction(1,40),events)
    if fault != 'compile':
        with pytest.raises(ValueError): m.raw_gate(raw,ref)
    else:
        R,target,_ = m.raw_gate(raw,ref)
        compiled = compile_checked(raw,CompilerProfile()); compiled.rz(.03,0)
        with pytest.raises(ValueError): m.compiled_gate(compiled,R,target)


def test_cost_descriptive_statistics_and_unknown_propagation():
    from trottertracks.c1_stage3_accuracy import METRICS
    rows = [{k:x for k in METRICS} for x in (2,4,6)]
    summary = m.calibration(rows)
    assert all(x == 4 for x in summary['sample_mean'].values())
    assert all(x == 4 for x in summary['sample_variance'].values())
    assert summary['sample_mean_standard_error']['rz_count'] == pytest.approx(math.sqrt(4/3))
    assert summary['observed_min']['rz_count'] == 2 and summary['observed_max']['rz_count'] == 6
    assert summary['cost_support'] is summary['ideal_law_expectation'] is summary['lower'] is None
    comp = m.comparison(dict(baseline=summary,proposed=summary))
    assert comp['guaranteed_advantage'] is comp['shots'] is comp['CA_achieved'] is None
    assert m.calibration(rows[:1])['sample_variance'] is None
    assert m.calibration(rows[:1])['sample_mean_standard_error'] is None


def invented_evidence(root,n=2,rank=8):
    root = Path(root); result = dict(status='STAGE_A_PROXY_AND_DF_PASS',sector={},fit_diagnostics={})
    records = dict(result='A_result.json',STOP='A_STOP.json',SOURCE='A_SOURCE.json')
    B1 = dict(status='B1_MECHANISM_EVALUATION_COMPLETE',arms={})
    b1records = dict(result='B1_result.json',STOP='B1_STOP.json',SOURCE='B1_SOURCE.json')
    exports = {}; refs = {}; grounds = {}
    for role in b.ARMS:
        arm,p = artificial(n,rank,role)
        exports[role] = role+'.npz'; np.savez(root/exports[role],J=arm.J,U=arm.U,x=np.zeros(128),jac=np.zeros(128))
        result['sector'][role] = dict(actual_rank=8,sector_error=1e-8,numerical_guard=1e-9,guard_policy=m.b1.molecule.GUARD_POLICY)
        result['fit_diagnostics'][role] = dict(optimizer=dict(success=True),diagnostic=dict(exact_tail_mass_Ha=.1))
        ctx = m.b1.molecule.SectorCAR(n); G = ctx.generators(p['h'],p['g'],p['constant'],arm.J,arm.U)
        refs[role] = role+'_ref.npz'; grounds[role] = role+'_ground.npz'
        indices = b.stage1.bit_reverse_indices(2*n)[ctx.sector.basis_indices]
        np.savez(root/refs[role],generators=np.array(G),full_input_indices=indices)
        psi = np.ones(ctx.d)/math.sqrt(ctx.d); np.savez(root/grounds[role],sector_state=psi)
        tail,c,dist = b.configure(arm,Fraction(1,40),16,0)
        for suffix,value in [('ground',dict(status='PASS')),('tail',dict(lambda_R=tail.lambda_r,
                identity=arm.plan.extracted_identity,atoms=len(arm.plan.atoms),components=[dict(probability=x.probability) for x in tail.components]))]:
            path=role+'_'+suffix+'.json'; m.io.save(root/path,value); b1records[role+'_'+suffix] = path
        B1['arms'][role] = dict(classification=dict(status='POTENTIALLY_FEASIBLE'),saved_DF_bound_Ha=1e-8+1e-9,
            rounds=[dict(m=i,feasibility=dict(status='POTENTIALLY_FEASIBLE')) for i in range(13)])
    np.savez(root/'input.npz',**p)
    for path,value in [(records['result'],result),(records['STOP'],dict(failed=False,scientific_units=30,retry=0,resume=0,mandatory_stop=True)),
            (records['SOURCE'],dict(execution_commit='e1085c95f5c01b68566e17d543ace5832cb94061')),
            (b1records['result'],B1),(b1records['STOP'],dict(failed=False,scientific_units=141,call_counts=m.b1.LIMITS,retry=0,resume=0,mandatory_stop=True)),
            (b1records['SOURCE'],dict(execution_commit='7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9'))]: m.io.save(root/path,value)
    m.io.save(root/'source.json',dict(files={}))
    # n=4 allows genuine passive export-header preflight without molecular data.
    required = {'input.npz',*records.values(),*b1records.values(),*refs.values(),*grounds.values(),*exports.values()}
    c = dict(stage_A_input='input.npz',stage_A_records=records,B1_records=b1records,
        B1_references=refs,B1_grounds=grounds,final_exports=exports,
        saved_inputs={p:m.io.sha(root/p) for p in required},source_manifest='source.json',
        source_manifest_sha256=m.io.sha(root/'source.json'),original_input=dict(sha256='I'))
    return c


def test_passive_preflight_hash_header_provenance_and_policy(tmp_path,monkeypatch):
    c = invented_evidence(tmp_path,n=4)
    from trottertracks.c1_h6_native_precision import VERSION as gaussian
    from trottertracks.c1_h6_compiler_precision import VERSION as phase
    original = tmp_path/'original'; original.write_bytes(b'invented byte packet')
    c.update(scope=m.SCOPE,status='UNAPPROVED_PROPOSAL',implementation=m.VERSION,policy=m.POLICY,
        allowances=m.ALLOWANCES,compiler=m.profile_dict(),gaussian=gaussian,compiler_phase=phase,
        call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,
        sampling=m.sampling_policy(),original_input=dict(sha256=m.io.sha(original)),evaluation_packages={})
    path=tmp_path/'c.json'; m.io.save(path,c)
    monkeypatch.setattr(np,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO_SCIENTIFIC_DECODE')))
    assert m.preflight(tmp_path,path,original)==c
    c['policy']=dict(m.POLICY,r=4); path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='SCOPE_OR_POLICY'): m.preflight(tmp_path,path,original)
    c['policy']=m.POLICY; path.write_text(json.dumps(c))
    (tmp_path/c['B1_records']['STOP']).write_text('{}')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'): m.preflight(tmp_path,path,original)


@pytest.mark.parametrize('fault',[None,'raw','compile','law','input'])
def test_artificial_runner_end_to_end_partial_STOP_and_fresh(tmp_path,monkeypatch,fault):
    c = invented_evidence(tmp_path)
    monkeypatch.setattr(m,'preflight',lambda *a:c)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'SYNTHETIC_SOURCE\n')
    actual_execute=m.execute
    # Four real synthetic complete kernels through the production pipeline.
    monkeypatch.setattr(m,'execute',lambda c,r,o,call:actual_execute(c,r,o,call,samples_per_arm=2))
    if fault=='raw': monkeypatch.setattr(m,'raw_gate',lambda *a:(_ for _ in ()).throw(ValueError('INVENTED_RAW_FAIL')))
    if fault=='compile':
        from trottertracks import c1_h4_native_fair
        original_compile=c1_h4_native_fair.compile_checked
        def bad(q,p):
            value=original_compile(q,p)
            if not q.num_clbits and q.metadata and 'native_policy' in q.metadata: value.rz(.03,0)
            return value
        monkeypatch.setattr(c1_h4_native_fair,'compile_checked',bad)
    if fault=='law': monkeypatch.setattr(m,'law_gate',lambda *a:(_ for _ in ()).throw(ValueError('LAW_FAIL')))
    if fault=='input': (tmp_path/c['stage_A_input']).write_bytes(b'bad input')
    out = tmp_path/m.FRESH
    if fault:
        with pytest.raises(Exception): m.run(tmp_path,'contract','input',out)
        stop=m.io.read(out/'STOP.json'); assert stop['failed'] and not (out/'result.json').exists()
        assert stop['call_counts'].get('kernel_metrics',0)==0
    else:
        m.run(tmp_path,'contract','input',out)
        stop=m.io.read(out/'STOP.json'); assert not stop['failed'] and stop['retry']==stop['resume']==0
        expected=dict(m.LIMITS)
        for k in ('trajectory_sample','kernel_build','trajectory_reference','raw_operator_gate',
                'compiled_operator_gate','adjoint_pair_gate','XY_gate','kernel_metrics'): expected[k]=4
        expected['compile']=7
        assert stop['call_counts']==expected
        assert stop['scientific_units']==sum(expected.values())
        result=m.io.read(out/'result.json')
        assert all(a['samples']==2 for a in result['arms'].values())
        assert result['cost_support'] is result['total_resources'] is result['CA_achieved'] is None
        assert len(list((out/'circuits').glob('*.qpy')))==22
        assert len(list((out/'references').glob('*.npz')))==4
    with pytest.raises(ValueError,match='FRESH'): m.run(tmp_path,'contract','input',out)


def test_full_cohort_call_accounting_mock_end_to_end(tmp_path,monkeypatch):
    """64 logical samples without 64 expensive duplicate artificial compiles."""
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks import c1_h4_native_fair as fair
    c=invented_evidence(tmp_path); out=tmp_path/'full-cohort'; out.mkdir()
    # Reuse an invented identity kernel and reference only inside this mock test.
    from qiskit import QuantumCircuit
    monkeypatch.setattr(b,'build_kernel',lambda arm,*a,**k:QuantumCircuit(arm.model.modes+1))
    monkeypatch.setattr(m,'trajectory_reference',lambda arm,*a:np.eye(1<<arm.model.modes,dtype=complex))
    monkeypatch.setattr(fair,'compile_checked',lambda qc,p:qc.copy())
    call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    result=m.execute(c,tmp_path,out,call)
    assert dict(call.counts)==m.LIMITS and sum(call.counts.values())==m.MAX_CALLS
    assert all(a['samples']==32 for a in result['arms'].values())
    assert len(list((out/'circuits').glob('*.qpy')))==262
    assert m.MAX_CALLS > 141
