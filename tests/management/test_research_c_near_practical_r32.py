"""Artificial data only: paired identities, gating, stats, counts and STOP."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import collections,json,subprocess,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_near_practical_r32 as m


def fixture(root,monkeypatch):
    from test_research_c_pr_integrated_pipeline import synthetic_packet
    from trottertracks.c1_molecule_extension import SectorCAR
    from qiskit import qpy
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    p,J,U=synthetic_packet();loader=m.perf.loader;ctx=SectorCAR(4)
    G=np.asarray(ctx.generators(**p,J=J,U=U));H=sum(G)
    loader.save_arrays(root/'input.npz',**p)
    loader.save_arrays(root/'basis.npz',H=H,sector_basis_indices=ctx.sector.basis_indices)
    endpoints={};arms={};digests={}
    for name in m.CANDIDATES:
        loader.save_arrays(root/(name+'.npz'),J=J,U=U)
        loader.save_arrays(root/(name+'_ref.npz'),J=J,U=U,generators=G,H=H)
        digest=m.io.sha(root/(name+'.npz'));digests[name]=digest
        # Distinct identity keys are tested separately; artificial J/U may coincide.
        df=dict(candidate_sha256=digest,actual_rank=8,budget_PASS=True,budget_Ha=.00019875,
            sector_error=1e-9,numerical_guard=1e-9,guard_policy='sector_operation_scale_guard8_proposal_v1',
            kappa='original_g_corrected',scalar='nuclear_once',exact_tail_lambda_Ha=0.)
        model=m.perf.b.prepare_arm(**p,J=J,U=U,role='proposed',source_identity=digest,input_identity=m.io.sha(root/'input.npz'),ld=2)
        df['exact_tail_lambda_Ha']=sum(abs(float(x.coefficient)) for x in model.plan.atoms)
        m.io.save(root/(name+'_df.json'),df)
        endpoints[name]=dict(candidate=name+'.npz',reference=name+'_ref.npz',physical=name+'_df.json')
        arms[name]=dict(endpoint_sha256=digest,physical=df,endpoint_eligible=True,trajectory=dict(accepted_U_updates=16),status='MAXIMUM_ACCEPTED_UPDATES',candidate_identity=dict(source_manifest_sha256='SOURCE'))
    monkeypatch.setattr(m,'CANDIDATES',digests);monkeypatch.setattr(m,'INPUT',m.io.sha(root/'input.npz'))
    origin={k:'origin_'+k+'.json' for k in ('result','STOP','contract','SOURCE')}
    m.io.save(root/origin['SOURCE'],dict(synthetic=True));source_digest=m.io.sha(root/origin['SOURCE'])
    for v in arms.values():v['candidate_identity']['source_manifest_sha256']=source_digest
    m.io.save(root/origin['result'],dict(status='PAIRED_FINITE_RECIPE_ENDPOINTS_RECORDED',arms=arms))
    m.io.save(root/origin['STOP'],dict(failed=False,scientific_units=645,retry=0,resume=0,mandatory_stop=True))
    m.io.save(root/origin['contract'],dict(scope='C_NEAR_OBJECTIVE_PRACTICAL_COMPARISON_II_O_V1',saved_inputs={'input.npz':m.INPUT},source_manifest_sha256=source_digest))
    shells={};reports={}
    for label in ('head','X','Y'):
        q=m.perf.b2.shell_build(8,label);path=label+'.qpy'
        with (root/path).open('xb') as f:qpy.dump(q,f)
        shells[label]={'raw':path,'compiled':path};reports[label]=dict(metrics=circuit_metrics(q))
    m.io.save(root/'shells.json',reports)
    entry=root/m.ENTRY;entry.parent.mkdir();entry.write_text('# artificial entry\n')
    m.io.save(root/'SOURCE.json',dict(files={m.ENTRY:m.io.sha(entry)}))
    from trottertracks.c1_h4_native_fair import GAUSSIAN,PHASE
    c=dict(**m.fixed(),input='input.npz',physical_reference='basis.npz',origin_records=origin,endpoints=endpoints,
        shell_reports='shells.json',shells=shells,packages={},gaussian=GAUSSIAN,compiler_phase=PHASE,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'))
    c['saved_inputs']={str(p.relative_to(root)):m.io.sha(p) for p in root.iterdir() if p.is_file() and p.name!='SOURCE.json'}
    m.io.save(root/'contract.json',c);return c


def fake_strong(root,c,out,call,a):
    cost=dict(totals={k:1e12 for k in m.downstream.m4.METRICS},cost_scope=m.perf.accounting.COST_SCOPE)
    return dict(candidate=a['candidate_identity'],native_admitted=True,ideal_exact_native=dict(cost=cost),numerical=dict(cost=cost))


def fake_native(root,c,out,call,a,result,packet,*,sampling):
    assert sampling['events_per_sample']==a['config'].rte_steps==32
    name=a['candidate_label'];assert sampling==m.sampling(name)
    rows=[{k:100+(i%5) for k in m.downstream.m4.METRICS} for i in range(32)]
    cal=m.perf.b2.calibration(rows);packet.update(kernel_samples=rows,saved_calibration=cal)
    kernel=call('cost_calibration','artificial',lambda:m.perf.accounting.cost_calibration(packet))
    costs=[call('serial_cost',str(mo)+axis,lambda mo=mo,axis=axis:m.downstream.m3.serial_cost(packet,kernel,mo,axis)) for mo in range(13) for axis in ('X','Y')]
    for lane,v in result['lanes'].items():
        v['conditional_totals']=call('total_aggregation',lane,lambda:m.downstream.m4.accuracy.aggregate_totals(v['energy'],v['rounds'],costs,cost_scope=m.perf.accounting.COST_SCOPE))
        v['weighted_SE']=call('calibration_uncertainty',lane,lambda:m.downstream.m3.calibration_uncertainty(packet,v['rounds'],v['conditional_totals']))
    result['calibration']=cal;return result


@pytest.mark.parametrize('mode',['pass','PR_infeasible','ENTRY_alias','correctness'])
def test_artificial_e2e_gate_other_arm_independent_D_and_fresh_STOP(tmp_path,monkeypatch,mode):
    c=fixture(tmp_path,monkeypatch);out=tmp_path/'out';out.mkdir();call=m.downstream.m4.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    def prepare(*a,**k):
        if mode=='correctness':raise ValueError('ARTIFICIAL_CORRECTNESS')
        if mode=='ENTRY_alias' and a[3]=='ENTRY':return None,dict(status='UNRESOLVED_GROUND_ALIAS')
        return m.prepare_arm(*a,**k)
    def reference(a,out,call):
        result,packet=m.downstream.references(a,out,call)
        if mode=='PR_infeasible' and a['candidate_label']=='PR':result['native_admitted']=False
        return result,packet
    if mode=='correctness':
        with pytest.raises(ValueError,match='ARTIFICIAL_CORRECTNESS'):
            m.execute(tmp_path,c,out,call,preparer=prepare,reference=reference,native=fake_native,strong=fake_strong)
    else:
        r=m.execute(tmp_path,c,out,call,preparer=prepare,reference=reference,native=fake_native,strong=fake_strong)
        if mode=='pass':
            assert r['comparison']['primary']['difference_interval'] is not None
            assert r['arms']['PR']['PR']['candidate']==m.CANDIDATES['PR']
            assert r['arms']['ENTRY']['PR']['candidate']==m.CANDIDATES['ENTRY']
            assert r['arms']['PR']['PR']['lanes']['evidence']['conditional_totals']['totals'] is None
        else:
            assert r['comparison']['status']=='UNRESOLVED_ONE_OR_BOTH_PR_INELIGIBLE'
            assert r['arms']['ENTRY']['statistics'] is not None if mode=='PR_infeasible' else r['arms']['PR']['statistics'] is not None
            if mode=='PR_infeasible':assert r['arms']['PR']['D'] is not None and r['arms']['PR']['statistics'] is None
        assert sum(call.counts.values())<=m.MAX_CALLS
        assert r['actual_CA_achieved'] is None and r['guaranteed_resource_superiority'] is None


@pytest.mark.parametrize('fault',['source','input','r','candidate','fresh','none'])
def test_passive_drift_or_wrong_candidate_before_array_decode(tmp_path,monkeypatch,fault):
    fixture(tmp_path,monkeypatch);monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decode'))
    if fault=='source':(tmp_path/m.ENTRY).write_text('wrong source')
    if fault=='input':(tmp_path/'input.npz').write_bytes(b'wrong input')
    if fault in ('r','candidate'):
        c=m.io.read(tmp_path/'contract.json')
        if fault=='r':c['policy']['r']=16
        else:c['candidate_sha256']['ENTRY']='WRONG'
        (tmp_path/'contract.json').write_text(json.dumps(c))
    if fault=='fresh':(tmp_path/m.FRESH).mkdir(parents=True)
    if fault=='none':assert m.preflight(tmp_path,tmp_path/'contract.json')
    else:
        with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'contract.json')


def test_budget_seed_identity_and_passive_import():
    assert m.MAX_CALLS==1231
    assert [sum(x.values()) for x in (m.A,m.B,m.C,m.D,m.E)]==[17,90,357,141,6]
    x=m.seed_manifest('INVENTED');assert len(x['rows'])==len({r['seed'] for r in x['rows']})==64
    assert m.sampling('PR')!=m.sampling('ENTRY')
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_near_practical_r32;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    with pytest.raises(PermissionError):m.run(Path('.'),Path('absent.json'))


def samples(offset):
    rng=np.random.default_rng(912)
    return [{k:int(offset+x) for k in m.downstream.m4.METRICS} for x in rng.integers(-5,6,32)]


def test_affine_Welch_interval_and_zero_variance_not_support():
    from scipy.stats import ttest_ind
    stats=m.statistics
    p,e=samples(100),samples(200);a=dict(W=100,shell=dict(rz_count=3));b=dict(W=200,shell=dict(rz_count=7))
    r=stats.primary(p,e,a,b)
    expected=ttest_ind([100*x['rz_count']+3 for x in p],[200*x['rz_count']+7 for x in e],equal_var=False).confidence_interval(.95)
    np.testing.assert_allclose(r['difference_interval'],[expected.low,expected.high])
    assert r['point_ratio_PR_ENTRY']<1 and r['ratio_interval'] is None
    assert r['actual_CA_achieved'] is None and not r['ideal_law_price_transfer_qualified']
    constant=[{k:1 for k in m.downstream.m4.METRICS} for _ in range(32)]
    assert stats.primary(constant,e,a,b)['decision']=='UNRESOLVED_ZERO_COHORT_VARIANCE'
