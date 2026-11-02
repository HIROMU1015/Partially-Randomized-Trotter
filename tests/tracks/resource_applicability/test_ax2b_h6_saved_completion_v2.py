"""Synthetic arrays/solver/dummy children only. Real H6 NPZ decode is forbidden."""
from copy import deepcopy
import builtins
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from trottertracks.resource_applicability import ax2b_h6_saved_completion_contract_v2 as contract
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h6_weighted_projection_v1 import accept,reconstruct,canonical_antisym
from trottertracks.resource_applicability.ax2b_h6_saved_completion_port_v2 import SavedCompletionPort,load_saved_payload,atomic_npz
from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
from trottertracks.resource_applicability.ax2b_h6_saved_completion_watchdog_v2 import CompletionProgress,supervise
from trottertracks.resource_applicability.ax2b_h6_saved_completion_audit_v2 import audit_saved
from trottertracks.resource_applicability.ax2b_h6_df_diagnostic_port_v1 import describe
from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
from trottertracks.resource_applicability.ax2b_limits import CallBudget
ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def no_real_science(monkeypatch,tmp_path):
    def forbidden(*a,**k):raise AssertionError('REAL_SCIENCE_FORBIDDEN')
    import openfermion,openfermionpyscf,pyscf.gto,scipy.sparse.linalg,qiskit
    import trotterlib.df_hamiltonian as df
    import trotterlib.rte as rte
    for obj,name in ((openfermion,'low_rank_two_body_decomposition'),(openfermion.MolecularData,'__init__'),
        (openfermionpyscf,'run_pyscf'),(pyscf.gto.Mole,'build'),(scipy.sparse.linalg,'eigsh'),
        (df,'build_df_h_d_from_molecule'),(qiskit.QuantumCircuit,'__init__'),(qiskit,'transpile')):
        monkeypatch.setattr(obj,name,forbidden)
    for name in ('sample_rte_events','iter_sample_rte_events','sample_event_mean_operator'):
        monkeypatch.setattr(rte,name,forbidden)
    original=np.load
    def temp_only(path,*a,**k):
        assert Path(path).resolve().is_relative_to(tmp_path.resolve()),'REAL_NPZ_DECODE_FORBIDDEN'
        return original(path,*a,**k)
    monkeypatch.setattr(np,'load',temp_only)
    real_open,real_path=builtins.open,Path.open
    def check(path):
        if isinstance(path,(str,Path)) and '/artifacts/' in str(path) and not Path(path).resolve().is_relative_to(tmp_path.resolve()):
            raise AssertionError('REAL_ARTIFACT_IO_FORBIDDEN')
    def safe_open(path,*a,**k):check(path);return real_open(path,*a,**k)
    def safe_path(path,*a,**k):check(path);return real_path(path,*a,**k)
    monkeypatch.setattr(builtins,'open',safe_open);monkeypatch.setattr(Path,'open',safe_path)


def small(weight=1e-8,defect=1e-6,count=1):
    g=np.eye(4,dtype='<c16');g[0,2]=g[1,3]=defect
    return dict(constant=.1,one_body=np.eye(4,dtype='<c16'),two_body=np.zeros((4,)*4,dtype='<c16'),
        lambdas=np.full(count,weight,dtype='<f8'),g_matrices=np.stack([g]*count),
        correction=np.zeros((4,4),dtype='<c16'),truncation=np.array(0.))


def test_weak_lambda_retained_and_strong_lambda_rejected():
    a,r,_=accept(**small())
    assert r['old_failed_fragment_indices']==[0] and r['status']=='PASS_ENGINEERING'
    assert not r['representation_error_certified'] and np.array_equal(a['lambdas'],[1e-8])
    assert float(r['eta_N_hartree_decimal']['12'])<1e-10
    with pytest.raises(ValueError,match='WEIGHTED_BUDGET'):accept(**small(weight=1.))


def test_fragment_sum_has_no_cancellation_and_zero_lambda_not_deleted():
    _,r,_=accept(**small())
    term=float(r['eta_N_hartree_decimal']['12'])
    weight=1e-8*6e-11/term
    accept(**small(weight=weight))
    p=small(weight=weight,count=2);p['lambdas'][1]*=-1
    with pytest.raises(ValueError,match='WEIGHTED_BUDGET'):accept(**p)
    p=small(count=2);p['lambdas'][:]=[0.,-1e-8]
    a,r,_=accept(**p);assert len(a['lambdas'])==2 and a['lambdas'][1]<0


@pytest.mark.parametrize('failure',['complex_lambda','nonfinite','cross_spin','alpha_beta','truncation','layout'])
def test_structure_is_independent_of_small_weight(failure):
    p=small()
    if failure=='complex_lambda':p['lambdas']=np.array([1e-8+0j])
    if failure=='nonfinite':p['g_matrices'][0,0,0]=np.nan
    if failure=='cross_spin':p['g_matrices'][0,0,1]=1e-30
    if failure=='alpha_beta':p['g_matrices'][0,0,0]+=1e-12
    if failure=='truncation':p['truncation']=2e-8
    if failure=='layout':p['g_matrices']=p['g_matrices'][:,:,:3]
    with pytest.raises(ValueError):accept(**p)


def test_one_body_projection_counted_once_and_correction_not_replaced():
    p=small(weight=0.,defect=0.);p['correction'][0,2]=p['correction'][1,3]=1e-12
    a,r,c=accept(**p)
    dh=r['corrected_one_body_change_frobenius']
    assert float(r['eta_N_hartree_decimal']['12'])==pytest.approx(12*dh)
    assert np.array_equal(a['one_body'],(p['one_body']+p['correction']+(p['one_body']+p['correction']).conj().T)/2)
    assert np.linalg.norm(c['raw_one_body_residual']-p['correction'])<1e-15
    assert r['error_ledger']['PF_RTE'] is None and r['error_ledger']['measurement'] is None


def fermion_matrix(n,operations):
    """Independent CAR bit transitions; rightmost operator acts first."""
    out=np.zeros((1<<n,1<<n),dtype=complex)
    for column in range(1<<n):
        state=column;amplitude=1
        for mode,creation in reversed(operations):
            occupied=(state>>mode)&1
            if occupied==creation:amplitude=0;break
            amplitude*=(-1)**((state&((1<<mode)-1)).bit_count())
            state^=1<<mode
        out[state,column]+=amplitude
    return out


def test_normal_order_matches_independent_fermion_operator_without_conjugation():
    n=4;g=np.array([[.2,1j,.1,0],[.3,.4,0,0],[0,.2,.5,.1j],[0,0,.6,.7]])
    h=np.eye(n)*.17;weights=np.array([-.7,.2]);blocks=np.stack([g,g.T])
    one,two=reconstruct(h,weights,blocks)
    def second(a):return sum(a[p,q]*fermion_matrix(n,[(p,True),(q,False)]) for p in range(n) for q in range(n))
    target=second(h)+sum(w*second(b)@second(b) for w,b in zip(weights,blocks))
    actual=second(one)+sum(two[p,q,r,s]*fermion_matrix(n,[(p,True),(q,True),(r,False),(s,False)])
        for p in range(n) for q in range(n) for r in range(n) for s in range(n))
    assert np.linalg.norm(actual-target)<1e-13
    wrong=second(h)+sum(w*second(b).conj().T@second(b) for w,b in zip(weights,blocks))
    assert np.linalg.norm(actual-wrong)>.1


def fixture_payload():
    weights=np.zeros(19,dtype='<f8');weights[:2]=[.2,-.1]
    gs=np.zeros((19,12,12),dtype='<c16');gs[0]=np.eye(12);gs[1]=np.eye(12)*.3
    one=np.eye(12,dtype='<c16')*.2
    correction=-sum(w*g@g for w,g in zip(weights,gs))
    two=-sum(w*np.einsum('pr,qs->pqrs',g,g) for w,g in zip(weights,gs))
    summary,hyp=describe(one,two,weights,gs,correction,0.,budget=CallBudget(fragment_diagnostics=36))
    return {'constant':.1,'one_body':one,'two_body':two,'lambdas':weights,'g_matrices':gs,
        'correction':correction,'truncation':0.,'summary':summary,
        'diagnostic_coefficients':(hyp['normal_order_one_body_raw_residual'],canonical_antisym(hyp['normal_order_two_body_raw']),hyp['normal_order_two_body_target_antisym']),
        'hypothetical_projection':(hyp['corrected_one_body_hypothetical_hermitian'],hyp['g_hypothetical_hermitian'])}


def sealed(tmp_path,monkeypatch):
    m=contract.preparation();cpus=sorted(os.sched_getaffinity(0))[:4];out=tmp_path/contract.NAMESPACE/'one_shot'
    m.update(execution_plan_sealed=True,source_commit='a'*40,source_hashes={'fixture':'b'*64},
        input_identity={'synthetic_only':True},environment={'fixture':True},
        assigned_resources=contract.resources(cpus),
        exclusive_output={'repository_path':contract.NAMESPACE+'one_shot','absolute_path':str(out.resolve())})
    grant={'schema':'track_a_h6_saved_df_completion_authorization_v2','approved_by_user':True,
        'kind':contract.KIND,'manifest_digest':digest(m),'assigned_cpus':cpus,'exclusive_output':str(out.resolve()),'retry':False,'resume':False}
    monkeypatch.setattr(contract,'verify_sources',lambda *a:None)
    monkeypatch.setattr(contract,'verify_parents',lambda *a:{'synthetic_only':True})
    monkeypatch.setattr(contract,'environment',lambda:{'fixture':True})
    return m,grant,out


def fake_solver(op,**kwargs):
    v=kwargs['v0'];assert np.count_nonzero(v)==1
    assert np.array_equal(op@v,2*v) and np.array_equal(op.rmatvec(v),2*v)
    return np.array([2.]),(3j*v).reshape(-1,1)


def run_fixture(tmp_path,monkeypatch,*,cap=10000,failure=None):
    m,g,_=sealed(tmp_path,monkeypatch);m['plan']['caps']['solver_matvec']=cap
    g['manifest_digest']=digest(m);out=tmp_path/'synthetic';out.mkdir()
    writer=AtomicWriter(out,byte_cap=32*2**20);progress=CompletionProgress(writer,cap=256)
    raw=json.dumps(g,indent=1).encode()+b'\n';sha=hashlib.sha256(raw).hexdigest()
    writer.write('frozen_completion.json',m);writer.write('authorization.json',g);(out/'authorization_source.json').write_bytes(raw)
    binding={'manifest_digest':digest(m),'authorization_digest':digest(g)}
    writer.write('launch_binding.json',binding)
    writer.write('worker_claim.json',dict(binding,assigned_resources=contract.resources(g['assigned_cpus']),retry=False,resume=False,authorization_sha256=sha))
    writer.write('parallel_resource_receipt.json',{'assigned_cpus':g['assigned_cpus'],
        'numba_threads':4,'operator_backend':'numba','block_chunk_size':1,'science_workers':1,
        'thread_environment':{'NUMBA_NUM_THREADS':'4','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}})
    payload=fixture_payload()
    if failure=='summary':payload['summary']['fragments'][0]['lambda']+=.1
    if failure=='coefficient':payload['diagnostic_coefficients'][0][0,0]=1.
    if failure=='hypothetical':payload['hypothetical_projection'][1][0,0,0]+=1.
    def solver(op,**kwargs):
        if failure=='solver':raise RuntimeError('SYNTHETIC_NO_CONVERGENCE')
        e,v=fake_solver(op,**kwargs)
        return (np.array([3.]),v) if failure=='residual' else (e,v)
    port=SavedCompletionPort(m,writer,progress,root=tmp_path,grant_sha256=sha,payload=payload,
        solver=solver,operator_factory=lambda h,s:np.eye(400)*2)
    port.execute()
    terminal={'status':'H6_SAVED_DF_COMPLETION_COMPLETE','reason':None,'df_receipt_saved':True,
        'snapshot_receipt_saved':True,'calls_attempted':port.calls.used,'calls_completed':port.completed,
        'H6_input_accepted':True,'N':None,'G':None,'numerical_allowance_certified':False,
        'accuracy_eligibility':'UNDETERMINED','H6_status':'H6_NOT_AUTHORIZED',
        'contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}
    writer.write('worker_terminal.json',terminal,terminal=True)
    writer.write('terminal_status.json',{**terminal,'worker_terminal':terminal,'worker_exit_code':0},terminal=True)
    return port,out


def test_snapshot_roundtrip_and_stdlib_audit(tmp_path,monkeypatch):
    port,out=run_fixture(tmp_path,monkeypatch)
    assert port.calls.used['solver_matvec']==3 and port.calls.used['df_decomposition']==0
    assert audit_saved(out)['status']=='SAVED_COMPLETION_BYTES_PASS'
    r=subprocess.run([sys.executable,'-S',str(ROOT/contract.AUDITOR),'--saved-run',str(out),
        '--output',str(tmp_path/'audit.json')],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


@pytest.mark.parametrize('failure',['summary','coefficient','hypothetical'])
def test_independent_mismatch_stops_before_solver(tmp_path,monkeypatch,failure):
    with pytest.raises(ValueError):run_fixture(tmp_path,monkeypatch,failure=failure)
    assert not (tmp_path/'synthetic/h6_input_snapshot.npz').exists()


def test_shared_residual_budget_stops_before_snapshot(tmp_path,monkeypatch):
    with pytest.raises(RuntimeError,match='CALL_BUDGET:solver_matvec'):run_fixture(tmp_path,monkeypatch,cap=2)
    assert not (tmp_path/'synthetic/h6_input_snapshot.npz').exists()


@pytest.mark.parametrize('failure',['solver','residual'])
def test_solver_failure_preserves_df_without_snapshot(tmp_path,monkeypatch,failure):
    with pytest.raises((ValueError,RuntimeError)):run_fixture(tmp_path,monkeypatch,failure=failure)
    assert (tmp_path/'synthetic/df_receipt.json').exists()
    assert not (tmp_path/'synthetic/h6_input_snapshot.npz').exists()


def test_parent_stop_keeps_completed_bytes_without_promotion(tmp_path,monkeypatch):
    _,out=run_fixture(tmp_path,monkeypatch)
    p=out/'terminal_status.json';row=json.loads(p.read_text())
    row.update(status='H6_SAVED_DF_COMPLETION_STOP',reason='SYNTHETIC_PARENT_CAP');p.write_text(json.dumps(row))
    result=audit_saved(out)
    assert result['status']=='SAVED_COMPLETION_STOP_RECORDED'
    assert 'h6_input_snapshot.npz' in result['file_hashes']


def test_missing_snapshot_cannot_be_complete(tmp_path,monkeypatch):
    _,out=run_fixture(tmp_path,monkeypatch);(out/'h6_input_snapshot.npz').unlink()
    with pytest.raises(ValueError,match='MISSING_RECORDS'):audit_saved(out)


def test_tamper_and_old_policy_rejected(tmp_path,monkeypatch):
    _,out=run_fixture(tmp_path,monkeypatch)
    p=out/'h6_input_snapshot.npz';original=p.read_bytes();p.write_bytes(original+b'bad')
    with pytest.raises(ValueError,match='NPZ_BYTES'):audit_saved(out)
    p.write_bytes(original)
    snapshot=json.loads((out/'snapshot_receipt.json').read_text());df=json.loads((out/'df_receipt.json').read_text())
    df['projection_receipt']['policy']['name']='TOL_ONLY_NO_CONFIG_FALLBACK'
    with pytest.raises(ValueError,match='POLICY_BINDING'):load_h6_snapshot(p,snapshot,df)


def test_readonly_importer_rejects_bytes_before_decode(tmp_path):
    path='integrals.npz';(tmp_path/path).write_bytes(b'changed')
    (tmp_path/'integral_receipt.json').write_text('{}')
    with pytest.raises(ValueError,match='BEFORE_DECODE'):
        load_saved_payload(tmp_path,{'integrals':{'input_path':path,'input_sha256':'b'*64}})


def test_readonly_importer_complete_synthetic_parents(tmp_path):
    p=fixture_payload();hyp=p.pop('hypothetical_projection');coeff=p.pop('diagnostic_coefficients')
    raw_dir=tmp_path/contract.RAW;raw_dir.mkdir(parents=True)
    writer=AtomicWriter(raw_dir,byte_cap=32*2**20)
    integral=atomic_npz(writer,'integrals.npz',{'constant':np.array(p['constant']),
        'one_body':p['one_body'],'two_body':p['two_body']},expanded_cap=16*2**20)
    writer.write('integral_receipt.json',integral)
    raw=atomic_npz(writer,'raw_decomposition.npz',{'lambdas_raw':p['lambdas'],'g_matrices_raw':p['g_matrices'],
        'one_body_correction_raw':p['correction'],'truncation_value_raw':np.array(p['truncation'])},expanded_cap=4*2**20)
    writer.write('raw_decomposition_receipt.json',raw)
    # Diagnostic stores un-antisymmetrized quartic coefficients.
    t=-sum(w*np.einsum('pr,qs->pqrs',g,g) for w,g in zip(p['lambdas'],p['g_matrices']))
    projection=atomic_npz(writer,'hypothetical_hermitization.npz',{
        'normal_order_one_body_raw_residual':coeff[0],'normal_order_two_body_raw':t,
        'normal_order_two_body_target_antisym':coeff[2],
        'corrected_one_body_hypothetical_hermitian':hyp[0],'g_hypothetical_hermitian':hyp[1]},expanded_cap=4*2**20)
    writer.write('hypothetical_receipt.json',projection);writer.write('diagnostic_summary.json',p['summary'])
    identity={'integrals':{'input_path':contract.RAW+'integrals.npz','input_sha256':integral['sha256']},
        'raw_path':contract.RAW+'raw_decomposition.npz','raw_sha256':raw['sha256'],
        'hypothetical_sha256':projection['sha256'],'summary_sha256':contract.file_hash(raw_dir/'diagnostic_summary.json')}
    before={f.name:f.read_bytes() for f in raw_dir.iterdir()}
    loaded=load_saved_payload(tmp_path,identity)
    assert not loaded['g_matrices'].flags.writeable and not loaded['one_body'].flags.writeable
    accept(**{k:v for k,v in loaded.items() if k!='hypothetical_projection'})
    assert before=={f.name:f.read_bytes() for f in raw_dir.iterdir()}


def test_decision_margin_rejects_near_budget():
    _,r,_=accept(**small());eta=float(r['eta_N_hartree_decimal']['12'])
    with pytest.raises(ValueError,match='WEIGHTED_BUDGET'):
        accept(**small(weight=1e-8*9.95e-11/eta))


@pytest.mark.parametrize('mode',['metadata','no_grant'])
def test_default_and_rejected_cli_without_scientific_site_packages(tmp_path,mode):
    output=tmp_path/'never';args=['--execute'] if mode=='no_grant' else []
    r=subprocess.run([sys.executable,'-S',str(ROOT/contract.RUNNER),'--output',str(output),*args],capture_output=True,text=True)
    if mode=='metadata':assert r.returncode==0 and json.loads(output.read_text())==contract.preparation()
    else:assert r.returncode!=0 and not output.exists() and 'numpy' not in r.stderr


def test_valid_metadata_seal_and_worker_replay_rejection(tmp_path,monkeypatch):
    m,g,out=sealed(tmp_path,monkeypatch);assert contract.validate_launch(tmp_path,m,g,out,requested=True)==g['assigned_cpus']
    out.mkdir(parents=True);(out/'launch_binding.json').write_text(json.dumps({'manifest_digest':digest(m),'authorization_digest':digest(g)}))
    (out/'worker_claim.json').write_text('{}')
    with pytest.raises(ValueError,match='ONE_SHOT'):contract.validate_launch(tmp_path,m,g,out,requested=True,worker=True)


@pytest.mark.parametrize('mutation',['duplicate','too_few','unavailable','unsorted','thread_mismatch','legacy_single_cpu'])
def test_parallel_affinity_binding(tmp_path,monkeypatch,mutation):
    m,g,out=sealed(tmp_path,monkeypatch)
    cpus=g['assigned_cpus']
    if mutation=='duplicate':g['assigned_cpus']=[cpus[0]]*4
    if mutation=='too_few':g['assigned_cpus']=cpus[:3]
    if mutation=='unavailable':g['assigned_cpus']=[max(os.sched_getaffinity(0))+i+1 for i in range(4)]
    if mutation=='unsorted':g['assigned_cpus']=list(reversed(cpus))
    if mutation=='thread_mismatch':m['assigned_resources']['numba_threads']=1;g['manifest_digest']=digest(m)
    if mutation=='legacy_single_cpu':g['assigned_cpu']=g.pop('assigned_cpus')[0]
    with pytest.raises(ValueError,match='CPU'):contract.validate_launch(tmp_path,m,g,out,requested=True)


def test_parallel_limits_in_dummy_child(tmp_path):
    cpus=sorted(os.sched_getaffinity(0))[:4]
    script="""import os,sys,json,resource
from trottertracks.resource_applicability.ax2b_h6_saved_completion_contract_v2 import install_parallel_limits
cpus=json.loads(sys.argv[1]);install_parallel_limits(cpus,8*2**30,32*2**20)
print(json.dumps({'cpus':sorted(os.sched_getaffinity(0)),'as':resource.getrlimit(resource.RLIMIT_AS),
 'fsize':resource.getrlimit(resource.RLIMIT_FSIZE),'numerical_imported':any(n in sys.modules for n in ('numpy','scipy','numba'))}))
"""
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),NUMBA_NUM_THREADS='4',OMP_NUM_THREADS='4')
    for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):env[k]='1'
    r=subprocess.run([sys.executable,'-S','-c',script,json.dumps(cpus)],env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    row=json.loads(r.stdout);assert row=={'cpus':cpus,'as':[8*2**30]*2,'fsize':[32*2**20]*2,'numerical_imported':False}


def test_parallel_receipt_tamper_rejected(tmp_path,monkeypatch):
    _,out=run_fixture(tmp_path,monkeypatch)
    p=out/'parallel_resource_receipt.json';row=json.loads(p.read_text());row['numba_threads']=1;p.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='PARALLEL_RUNTIME'):audit_saved(out)


def test_default_operator_passes_four_threads(tmp_path,monkeypatch):
    import trotterlib.df_hamiltonian as df
    import numba
    m,g,_=sealed(tmp_path,monkeypatch);out=tmp_path/'default_backend';out.mkdir()
    monkeypatch.setattr(os,'sched_getaffinity',lambda *_:set(g['assigned_cpus']))
    monkeypatch.setattr(numba,'get_num_threads',lambda:4)
    seen=[]
    def operator(h,s,**kwargs):seen.append(kwargs);return np.eye(400)*2,{}
    monkeypatch.setattr(df,'df_linear_operator',operator)
    writer=AtomicWriter(out,byte_cap=32*2**20)
    port=SavedCompletionPort(m,writer,CompletionProgress(writer,cap=256),root=tmp_path,
        grant_sha256='a'*64,payload=fixture_payload(),solver=fake_solver)
    port.execute()
    assert seen==[{'backend':'numba','num_threads':4,'block_chunk_size':1}]
    assert json.loads((out/'parallel_resource_receipt.json').read_text())['numba_threads']==4


def test_existing_numba_parallel_equals_serial_on_explicit_synthetic_arrays(tmp_path):
    # A subprocess permits NUMBA_NUM_THREADS=4 without changing the test process.
    # This builds no molecule, decomposition, eigensolver, signal or circuit.
    script="""import os,json,sys,numpy as np,numba
from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector,df_linear_operator
cpus=json.loads(sys.argv[1]);os.sched_setaffinity(0,set(cpus))
rng=np.random.default_rng(27491);gs=[]
for i in range(3):
 a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));a=(a+a.conj().T)/2
 g=np.zeros((8,8),complex);g[::2,::2]=g[1::2,1::2]=a;gs.append(g)
ham=DFHamiltonian(.3,np.eye(8)*.2,np.array([.2,-.1,0.]),tuple(gs),{})
sector=PhysicalSector.spin_sector(n_qubits=8,nelec_alpha=2,nelec_beta=2)
vectors=[rng.normal(size=sector.dimension)+1j*rng.normal(size=sector.dimension) for _ in range(3)]
serial,_=df_linear_operator(ham,sector,backend='numba',num_threads=1,block_chunk_size=1)
expected=[serial@v for v in vectors]
parallel,_=df_linear_operator(ham,sector,backend='numba',num_threads=4,block_chunk_size=1)
actual=[parallel@v for v in vectors]
assert numba.get_num_threads()==4
assert all(np.array_equal(a,b) for a,b in zip(expected,actual))
print(json.dumps({'fixture_modes':8,'sector_dimension':sector.dimension,'numba_threads':numba.get_num_threads(),
 'vectors_checked':3,'serial_parallel_exact_bytes_equal':True}))
"""
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),NUMBA_NUM_THREADS='4',OMP_NUM_THREADS='4',
             NUMBA_CACHE_DIR=str(tmp_path/'cache'),MPLCONFIGDIR=str(tmp_path/'mpl'),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    cpus=sorted(os.sched_getaffinity(0))[:4]
    r=subprocess.run([sys.executable,'-c',script,json.dumps(cpus)],env=env,capture_output=True,text=True,timeout=120)
    assert r.returncode==0,r.stderr
    row=json.loads(r.stdout);assert row['sector_dimension']==36 and row['serial_parallel_exact_bytes_equal']


@pytest.mark.parametrize('mutation',['old_grant','budget','parent','source','environment','output','bool_cpu','retry','resume','authorized_flag','unsealed'])
def test_grant_source_parent_environment_rejections(tmp_path,monkeypatch,mutation):
    m,g,out=sealed(tmp_path,monkeypatch)
    if mutation=='old_grant':g['schema']='track_a_h6_df_diagnostic_authorization_v1'
    if mutation=='budget':m['plan']['caps']['total_wall_seconds']+=1;g['manifest_digest']=digest(m)
    if mutation=='parent':monkeypatch.setattr(contract,'verify_parents',lambda *a:{'changed':True})
    if mutation=='source':monkeypatch.setattr(contract,'verify_sources',lambda *a:(_ for _ in ()).throw(ValueError('source')))
    if mutation=='environment':monkeypatch.setattr(contract,'environment',lambda:{'changed':True})
    if mutation=='output':out.mkdir(parents=True)
    if mutation=='bool_cpu':g['assigned_cpus']=[True,1,2,3]
    if mutation in ('retry','resume'):g[mutation]=True
    if mutation=='authorized_flag':m['science_authorized']=True;g['manifest_digest']=digest(m)
    if mutation=='unsealed':m['execution_plan_sealed']=False;g['manifest_digest']=digest(m)
    with pytest.raises(ValueError):contract.validate_launch(tmp_path,m,g,out,requested=True)


@pytest.mark.parametrize('mode',['sleep','complete','invalid_phase','log_cap'])
def test_dummy_watchdog(tmp_path,mode):
    out=tmp_path/'watchdog';out.mkdir();caps=deepcopy(contract.plan()['caps'])
    caps.update(total_wall_seconds=2.,phase_wall_seconds=dict.fromkeys(contract.PHASES,1.))
    script='import pathlib,json,time,sys\np=pathlib.Path(sys.argv[1])\n'
    if mode=='sleep':caps['phase_wall_seconds']['saved_input']=.1;script+='time.sleep(5)\n'
    elif mode=='invalid_phase':script+="(p/'phase_state_snapshot.json').write_text('{}');time.sleep(5)\n"
    elif mode=='log_cap':script+="print('x'*100000);time.sleep(5)\n"
    else:
        for phase in contract.PHASES:script+=f"(p/'phase_{phase}.json').write_text(json.dumps({{'phase':{phase!r},'elapsed':.01}}))\n"
        row={'status':'H6_SAVED_DF_COMPLETION_COMPLETE','df_receipt_saved':True,'snapshot_receipt_saved':True,
            'H6_input_accepted':True,'N':None,'G':None,'numerical_allowance_certified':False,
            'accuracy_eligibility':'UNDETERMINED','H6_status':'H6_NOT_AUTHORIZED',
            'contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}
        script+=f"(p/'worker_terminal.json').write_text(json.dumps({row!r}))\n"
    r=supervise([sys.executable,'-S','-c',script,str(out)],out,caps=caps)
    assert r['status']==('H6_SAVED_DF_COMPLETION_COMPLETE' if mode=='complete' else 'H6_SAVED_DF_COMPLETION_STOP')
    assert r['mandatory_stop'] and not r['next_stage_authorized']
