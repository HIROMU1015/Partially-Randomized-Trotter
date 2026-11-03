"""Invented matrices and mock native dispatch only; saved H4 arrays stay closed."""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c_rte_r_retuning as m


def test_passive_import_and_registered_scope():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_rte_r_retuning as m;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'));assert m.R_VALUES==(16,32,64,128);assert m.MAX_CALLS==1265"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)
    assert not set(m.LIMITS) & {'fit', 'optimizer', 'PF_kernel', 'exact_evolution', 'DF_error'}
    with pytest.raises(ValueError): m.sampling(128)


@pytest.mark.parametrize('r', m.R_VALUES)
def test_invented_finite_RTE_normalization_raw_signal_identity_and_events(r):
    from test_research_c_pr_aware_df_stage_b2 import artificial
    from threadpoolctl import threadpool_limits
    b = m.perf.b
    with threadpool_limits(limits=1):
        arm, p = artificial(n=2, role='proposed')
        car = b.stage1.car_bilinears(4)
        G = [b.stage1.raw_base(b.stage1.corrected_kappa(p['h'], p['g']), p['constant'], car)]
        G += [b.stage1.raw_frame(J, U, car)[0] for J, U in zip(arm.J, arm.U)]
        tail, config, dist = b.configure(arm, Fraction(1, 10), r, 0)
        reference = b.kernels(arm, G, Fraction(1, 10), config)
        pf128 = b.kernels(arm, G, Fraction(1, 10), b.configure(arm, Fraction(1, 10), 128, 0)[1])['pf']
        np.testing.assert_array_equal(pf128, reference['pf'])
        actual = m.m4.corrected_kernel(G, config, ld=2, identity_tail=arm.plan.extracted_identity,
            saved_pf=pf128, expected_steps=r)
        np.testing.assert_allclose(actual['corrected'], reference['corrected'], atol=1e-14, rtol=0)
        psi = np.linalg.eigh(sum(G))[1][:, 0]
        assert m.normalization(tail, config, r)['status'] == 'PASS'
        for index in range(13):
            ref = b.reference.reference_round(reference, (sum(G), Fraction(1, 10)*(1 << index)),
                repetitions=1 << index, state=psi)
            row = m.m4.rte_round(actual, index, ref['pf_operator_error'], state=psi)
            assert row['PF_reused'] and row['pf_operator_error'] == ref['pf_operator_error']
            assert row['rte_corrected_operator_error'] == ref['rte_corrected_operator_error']
            assert row['attenuation'] == ref['attenuation']
            np.testing.assert_allclose(complex(*row['raw_signal_parts']), ref['raw_signal'], atol=1e-14, rtol=0)
        events = b.independent_occurrence(arm, tail, dist, config, master='invented',
            round_index=0, axis='X', shot=0, occurrence=0)
        assert len(events) == r and all(x.taylor_order == 0 for x in events)
        seeds = [b.connection.trajectory_seed('invented', arm.role, arm.model.source_identity, 0, 'X', 0, i)
            for i in (0, 1)]
        assert seeds[0] != seeds[1]
        with pytest.raises(ValueError, match='H_CONFIG'):
            m.m4.corrected_kernel(G, config, ld=2, identity_tail=0, saved_pf=pf128, expected_steps=r+1)


def scalar_saved():
    rounds = [dict(m=i, reference=dict(pf_operator_error=0., rte_corrected_operator_error=0., attenuation=1.))
        for i in range(13)]
    df = dict(candidate_sha256=m.CANDIDATE, actual_rank=8, budget_PASS=True, budget_Ha=.00019875,
        sector_error=1e-9, numerical_guard=1e-9, kappa='original_g_corrected', scalar='nuclear_once',
        exact_tail_lambda_Ha=1.)
    lanes = {lane: dict(rounds=[dict(m=i, target_id='integrated:'+m.CANDIDATE) for i in range(13)],
        conditional_totals=dict(totals={k:1e12 for k in m.m4.METRICS})) for lane in m.accounting.LANES}
    return dict(parent=dict(selected=dict(id='update005', physical=df)), physical=df,
        performance=dict(status='PERFORMANCE_NATIVE_COMPLETE', rounds=rounds, lanes=lanes),
        parent_STOP=dict(failed=False, scientific_units=543),
        performance_STOP=dict(failed=False, call_counts={k:32 for k in
            ('trajectory_sample','compile','raw_operator_gate','compiled_operator_gate','XY_gate')}),
        parent_contract=dict(sampling=m.perf.SAMPLING, target=m.m4.TARGET, compiler=m.COMPILER,
            input_paths={'input':'input.npz'}, saved_inputs={'input.npz':m.INPUT}, source_manifest_sha256='S'),
        parent_SOURCE={}, ground=dict(status='PASS', alias_margin=1., energy_Ha=0.),
        physical_ground=dict(status='PASS', alias_margin=1.), law=dict(status='PASS', lambda_R=1., identity=0., atoms=1),
        D=dict(status='SAME_JU_STRONG_D_COMPLETE', selected_export=m.CANDIDATE, saved_DF=df,
            ideal_exact_native=dict(cost=dict(totals={k:1e10 for k in m.m4.METRICS})),
            numerical=dict(cost=dict(totals={k:1e10 for k in m.m4.METRICS}))),
        D_STOP=dict(failed=False, scientific_units=156), D_SOURCE={},
        D_contract=dict(target=m.m4.TARGET, compiler=m.COMPILER, source_manifest_sha256='S'),
        shell_reports={key:dict(metrics={k:1 for k in m.m4.METRICS}) for key in ('head','X','Y')})


@pytest.mark.parametrize('mismatch', ['candidate', 'DF', 'source', 'ground', 'cohort'])
def test_metadata_identity_FAILURE_before_decode(tmp_path, monkeypatch, mismatch):
    saved = scalar_saved()
    if mismatch == 'candidate': saved['D']['selected_export'] = 'other'
    if mismatch == 'DF': saved['physical']['budget_PASS'] = False
    if mismatch == 'source': saved['parent_contract']['source_manifest_sha256'] = 'wrong'
    if mismatch == 'ground': saved['ground']['alias_margin'] = -1
    if mismatch == 'cohort': saved['performance_STOP']['call_counts']['compile'] = 31
    c = dict(input='input.npz', records={k:k for k in saved})
    monkeypatch.setattr(m.io, 'read', lambda path:saved[Path(path).name])
    monkeypatch.setattr(m.io, 'sha', lambda path:'S')
    monkeypatch.setattr(np, 'load', lambda *a, **k:pytest.fail('molecular arrays must stay closed'))
    with pytest.raises(ValueError): m.evidence(tmp_path, c)


def passive_fixture(root):
    opaque = root/'opaque.npz'; opaque.write_bytes(b'closed artificial binary')
    m.io.save(root/'source.json', dict(files={}))
    c = dict(base_commit=m.BASE, scope=m.SCOPE, implementation=m.VERSION, policy=m.POLICY,
        r_values=list(m.R_VALUES), new_r_values=list(m.NEW_R), saved_r=128, target=m.m4.TARGET,
        compiler=m.COMPILER, allowances=m.m3.ALLOWANCES, scenarios=list(m.accounting.SCENARIOS),
        call_limits=m.LIMITS, maximum_logical_units=m.MAX_CALLS, fresh_output=m.FRESH,
        retry=0, resume=0, new_fit=0, optimizer=0, new_PF_error=0, new_DF_error=0,
        new_ground_eigensolve=0, sampling={str(r):m.sampling(r) for r in m.NEW_R}, packages={},
        input='opaque.npz', selected='opaque.npz', physical_reference='opaque.npz',
        sector_reference='opaque.npz', basis='opaque.npz', ground='opaque.npz',
        records={k:'opaque.npz' for k in m.RECORDS}, source_manifest='source.json',
        source_manifest_sha256=m.io.sha(root/'source.json'), saved_inputs={'opaque.npz':m.io.sha(opaque)},
        shells={k:{x:'opaque.npz' for x in ('raw','compiled')} for k in ('head','X','Y')})
    from trottertracks.c1_h4_native_fair import GAUSSIAN, PHASE
    c.update(gaussian=GAUSSIAN, compiler_phase=PHASE)
    return c


def test_passive_preflight_source_drift_freshness_and_fixed_grid(tmp_path, monkeypatch):
    c = passive_fixture(tmp_path); digest = c['saved_inputs']['opaque.npz']
    monkeypatch.setattr(m, 'CANDIDATE', digest); monkeypatch.setattr(m, 'INPUT', digest)
    monkeypatch.setattr(m, 'evidence', lambda *a:{})
    monkeypatch.setattr(np, 'load', lambda *a, **k:pytest.fail('passive decode'))
    m.io.save(tmp_path/'c.json', c); m.preflight(tmp_path, tmp_path/'c.json')
    altered = deepcopy(c); altered['r_values'].append(256)
    m.io.save(tmp_path/'wrong.json', altered)
    with pytest.raises(ValueError, match='FIXED_RETUNING'): m.preflight(tmp_path, tmp_path/'wrong.json')
    (tmp_path/'opaque.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SOURCE_INPUT'): m.preflight(tmp_path, tmp_path/'c.json')


@pytest.mark.parametrize('mode', ['pass','infeasible','unresolved','correctness'])
def test_artificial_RTE_RPE_accounting_pipeline_and_STOP(tmp_path, monkeypatch, mode):
    from test_research_c_pr_integrated_pipeline import synthetic_packet
    from trottertracks.c1_molecule_extension import SectorCAR
    from threadpoolctl import threadpool_limits
    b = m.perf.b
    with threadpool_limits(limits=1):
        p, J, U = synthetic_packet(); ctx = SectorCAR(4)
        G = ctx.generators(**p, J=J, U=U)
        arm = b.prepare_arm(**p, J=J, U=U, role='proposed', source_identity='invented', input_identity='invented', ld=2)
        tail, config, _ = b.configure(arm, Fraction(1,10), 128, 0)
        pf = b.kernels(arm, G, Fraction(1,10), config)['pf']
        psi = np.linalg.eigh(sum(G))[1][:,0]
    saved = scalar_saved(); saved['law'].update(lambda_R=tail.lambda_r, identity=arm.plan.extracted_identity, atoms=len(arm.plan.atoms))
    a = dict(saved=saved, p=p, arm=arm, ctx=ctx, sector=dict(generators=np.array(G), H=sum(G), pf=pf),
        ground=psi, target=m.m4.accuracy.AccuracyTarget(**m.m4.TARGET))
    def prepare(*args):
        m.io.save(args[2]/'saved_r128.json', dict(r=128, saved=True)); return a
    def native(root,c,out,call,packet,**kw):
        if mode == 'correctness':
            return call('compile','injected failure',lambda:(_ for _ in ()).throw(ValueError('ARTIFICIAL_COMPILE_FAIL')))
        assert packet['config'].rte_steps==kw['sampling']['events_per_sample']
        rows=[{k:100 for k in m.m4.METRICS} for _ in range(32)]
        kw['prepared']['origin']=str(out)
        return rows,m.perf.b2.calibration(rows)
    monkeypatch.setattr(m.perf, 'native', native)
    if mode in ('infeasible','unresolved'):
        original=m.assess_lanes
        def lanes(*args):
            values=original(*args)
            for x in values.values():x['energy']['same_accuracy']=False
            return values
        monkeypatch.setattr(m,'assess_lanes',lanes)
        if mode=='unresolved':
            from trottertracks import c1_stage3_feasibility as f
            monkeypatch.setattr(f,'summarize',lambda *a:dict(status='UNRESOLVED',reason='artificial missing evidence'))
    # Stop legacy PF/full-reference engines from being invoked by the new path.
    monkeypatch.setattr(b.reference,'reference_kernels',lambda *a,**kw:pytest.fail('new PF engine'))
    out=tmp_path/'out';out.mkdir();call=m.m4.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    if mode=='correctness':
        with pytest.raises(ValueError,match='ARTIFICIAL_COMPILE_FAIL'):
            m.execute(tmp_path,{},out,call,prepare_function=prepare)
        assert call.stop.is_set() and (out/'r16/B.json').exists() and not (out/'r32').exists()
    else:
        with threadpool_limits(limits=1): result=m.execute(tmp_path,{},out,call,prepare_function=prepare)
        assert list(result['new'])==['16','32','64']
        for value in result['new'].values():
            assert value['lanes']['evidence']['total_shots'] is None
            assert value['actual_CA_achieved'] is None and value['cost_support'] is None
            assert value['native_admitted']==(mode=='pass')
            if mode=='pass':
                lane=value['lanes']['ideal_exact_native']; plans=lane['rounds']
                W=sum(2*x['shots']*(1<<x['m']) for x in plans)
                assert lane['weighted_SE']['kernel_reuse_weight']==W
                shell=sum(4*x['shots'] for x in plans)
                assert lane['conditional_totals']['totals']['rz_count']==W*100+shell
                assert value['comparisons']['ideal_exact_native']['ratios'] is not None
            else:assert value['comparisons']['ideal_exact_native']['ratios'] is None
        assert not call.stop.is_set()


def test_run_failure_preserves_STOP_and_forbids_resume(tmp_path, monkeypatch):
    c=dict(source_manifest='source.json');m.io.save(tmp_path/'source.json',dict(files={}))
    def preflight(root,path):
        if (root/m.FRESH).exists():raise ValueError('FRESH_NO_RETRY_RESUME')
        return c
    monkeypatch.setattr(m,'preflight',preflight)
    def execute(root,c,out,call):
        return call('evidence_load','synthetic',lambda:(_ for _ in ()).throw(ValueError('INPUT_IDENTITY_FAIL')))
    monkeypatch.setattr(m,'execute',execute)
    with pytest.raises(ValueError,match='INPUT_IDENTITY_FAIL'):m.run(tmp_path,tmp_path/'contract')
    stop=m.io.read(tmp_path/m.FRESH/'STOP.json')
    assert stop['failed'] and stop['scientific_units']==1 and stop['retry']==stop['resume']==0
    with pytest.raises(ValueError,match='FRESH_NO_RETRY_RESUME'):m.run(tmp_path,tmp_path/'contract')


def test_actual_artificial_Stage_A_reuses_sector_PF_and_ground(tmp_path, monkeypatch):
    from test_research_c_pr_integrated_pipeline import synthetic_packet
    from trottertracks.c1_molecule_extension import SectorCAR
    from threadpoolctl import threadpool_limits
    p,J,U=synthetic_packet();ctx=SectorCAR(4)
    with threadpool_limits(limits=1):
        G=np.asarray(ctx.generators(**p,J=J,U=U));H=sum(G)
        values,vectors=np.linalg.eigh(H)
    loader=m.perf.loader
    loader.save_arrays(tmp_path/'input.npz',**p)
    loader.save_arrays(tmp_path/'selected.npz',J=J,U=U)
    loader.save_arrays(tmp_path/'physical.npz',J=J,U=U,generators=G,H=H)
    loader.save_arrays(tmp_path/'reference.npz',generators=G,H=H,pf=np.eye(36,dtype=complex))
    loader.save_arrays(tmp_path/'basis.npz',sector_basis_indices=ctx.sector.basis_indices)
    loader.save_arrays(tmp_path/'ground.npz',energy=values[0],sector_state=vectors[:,0])
    saved=scalar_saved();saved['ground']['energy_Ha']=float(values[0])
    monkeypatch.setattr(m,'evidence',lambda *a:saved)
    monkeypatch.setattr(m.perf.b,'kernels',lambda *a,**kw:pytest.fail('Stage A PF recomputation'))
    c=dict(input='input.npz',selected='selected.npz',physical_reference='physical.npz',
        sector_reference='reference.npz',basis='basis.npz',ground='ground.npz')
    out=tmp_path/'out';out.mkdir();call=m.m4.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpool_limits(limits=1):a=m.prepare(tmp_path,c,out,call)
    assert dict(call.counts)==m.A_LIMITS
    assert a['ground'].shape==(36,) and a['arm'].model.source_identity==m.CANDIDATE
    assert m.io.read(out/'identity.json')['new_PF_error']==0
    assert m.io.read(out/'ground.json')['new_eigensolve']==0


def test_custom_native32_dispatch_and_shared_context(tmp_path, monkeypatch):
    from test_research_c_pr_integrated_native import test_fixed32_native_dispatch_and_failure as legacy
    original=m.perf.native
    def wrapped(root,c,out,call,a):
        # Reuse the existing fully mocked QPY/native test setup; all arrays are invented.
        occurrence=m.perf.b.independent_occurrence
        r=16; a['config']=SimpleNamespace(rte_steps=r)
        monkeypatch.setattr(m.perf.b,'independent_occurrence',lambda *x,**kw:occurrence(*x,**kw)[:r])
        prepared={}
        first=original(root,c,out,call,a,sampling=m.sampling(r),prepared=prepared)
        assert prepared and call.counts['raw_factor']==9
        a['config']=SimpleNamespace(rte_steps=32)
        nextout=root/'next';nextout.mkdir();nextcall=m.perf.Calls(nextout,m.perf.LIMITS,m.perf.MAX_CALLS)
        monkeypatch.setattr(m.perf.b,'independent_occurrence',lambda *x,**kw:tuple(
            SimpleNamespace(taylor_order=0,to_dict=lambda:{'artificial':True}) for _ in range(32)))
        monkeypatch.setattr(m.perf.b.stage1,'car_bilinears',lambda *x:pytest.fail('shared CAR recomputation'))
        second=original(root,c,nextout,nextcall,a,sampling=m.sampling(32),prepared=prepared)
        assert len(second[0])==32 and nextcall.counts['compile']==32
        assert nextcall.counts['raw_factor']==nextcall.counts['gaussian_build']==0
        assert len(list((nextout/'circuits').glob('*.qpy')))==128
        event=m.io.read(nextout/'samples/sample00_events.json')
        assert len(event['events'])==32 and event['coherent_occurrences_independent']
        return first
    monkeypatch.setattr(m.perf,'native',wrapped)
    legacy(tmp_path,monkeypatch,False)
