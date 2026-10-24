"""Artificial coefficient/phase/dispatch tests; no real QuantumCircuit exists."""
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
import cmath
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(Path(__file__).parent))
from test_research_c_c1_general_k_adapter import existing_data_bridges, frame, Gate
from trottertracks import c1_stage2_native as native
from trottertracks import c1_stage2_validation as validation
from trottertracks import c1_stage2_execution as execution


class CircuitDouble:
    def __init__(self, qubits, clbits=0, *, name='circuit'):
        self.num_qubits, self.num_clbits, self.name = qubits, clbits, name
        self.data, self.records, self.metadata = [], [], {}
        self.global_phase = 0.

    def _record(self, name, angle=None, qubits=()):
        self.records.append((name, angle, tuple(qubits)))
        self.data.append(SimpleNamespace(operation=SimpleNamespace(name=name)))

    def append(self, gate, qubits): self._record(gate.name, gate.params[0], qubits)
    def p(self, angle, q): self._record('p', angle, (q,))
    def x(self, q): self._record('x', None, (q,))
    def h(self, q): self._record('h', None, (q,))
    def sdg(self, q): self._record('sdg', None, (q,))
    def measure(self, q, c): self._record('measure', None, (q, c))
    def compose(self, other, qubits=None, inplace=False):
        assert inplace
        self._record('compose:'+other.name, None, tuple(qubits) if qubits is not None else ())
    def inverse(self):
        result = CircuitDouble(self.num_qubits, self.num_clbits, name=self.name+'†')
        result.records = [('actual_inverse_of', self.name, ())]
        return result


@contextmanager
def bridges():
    # Reuse the existing mocked bridge: all scientific constructors/norm/eigh
    # remain forbidden, and Gaussian construction is replaced by a recorder.
    with existing_data_bridges() as old:
        with patch('qiskit.QuantumCircuit', CircuitDouble), patch(
                'trottertracks.pr_dictionary_validation.circuits.native_gaussian',
                side_effect=lambda U: CircuitDouble(len(U), name='B'+str(U[0][0]))) as gaussian:
            yield old.ops, gaussian


def model(ops, *, same=False):
    u = ((1.+0j, 0j), (0j, 1.+0j))
    v = u if same else ((0j, 1j), (1j, 0j))
    p = ops.DiagonalEvolutionPrimitives(-.7, ((0, .3),), ((0, 1, -.2),))
    return native.NativeModel('ARTIFICIAL', (native.NativeBlock('one_body', u, p),
        native.NativeBlock('F0', v, p)), .375, 'artificial_source', 'artificial_input')


def test_import_and_default_cli_cannot_dispatch_science():
    script = "import sys;sys.path.insert(0,'src');from trottertracks import c1_stage2_native,c1_stage2_execution;assert not any(k in sys.modules for k in ['qiskit','numpy','scipy','ffsim'])"
    subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True)
    r = subprocess.run([sys.executable, str(ROOT/'scripts/run_research_c_c1_stage2_native.py'),
        '--contract', '/DO_NOT_READ', '--input', '/DO_NOT_READ', '--execute'], capture_output=True, text=True)
    assert r.returncode == 2 and 'Fresh output required' in r.stderr
    help = subprocess.run([sys.executable, str(ROOT/'scripts/run_research_c_c1_stage2_native.py'),
        '--help'],capture_output=True,text=True,check=True).stdout
    assert '--approval' not in help


def test_schedule_preserves_original_order_and_fractional_time():
    assert native.ordered_schedule(3, '1/10') == ((0, Fraction(1,20)), (1, Fraction(1,20)),
        (2, Fraction(1,10)), (1, Fraction(1,20)), (0, Fraction(1,20)))
    assert len(native.ordered_schedule(9, '1/10')) == 17
    assert len(native.ordered_schedule(11, '1/10')) == 21
    assert native.ordered_schedule(1, '-1/10') == ((0, Fraction(-1,10)),)


@pytest.mark.parametrize('count,time', [(0,'1'), (True,'1'), (2,'0')])
def test_invalid_schedule_rejected(count, time):
    with pytest.raises(ValueError): native.ordered_schedule(count, time)


@pytest.mark.parametrize('tau', ['1/20','1/10','-1/10'])
def test_artificial_diagonal_coefficients_include_both_orders_and_phase(tau):
    from trottertracks import c1_general_k_adapter as adapter
    f = frame(K=((2., -4.), (-4., 6.)))
    with bridges() as (ops, _):
        d = adapter.diagonal_primitives(f, float(Fraction(tau)))
        for state in range(4):
            n = [(state >> p)&1 for p in range(2)]; z = [1-2*v for v in n]
            energy = sum(f.K[a][b]*n[a]*n[b] for a in range(2) for b in range(2))
            phase = d.global_phase-sum(a*z[q]/2 for q,a in d.rz)-sum(a*z[p]*z[q]/2 for p,q,a in d.rzz)
            assert abs(cmath.exp(1j*phase)-cmath.exp(-1j*float(Fraction(tau))*energy)) < 1e-14


def test_builder_gaussian_direction_control_phase_and_scalar_once():
    with bridges() as (ops, gaussian):
        m = model(ops); builder = native.NativeBuilder()
        qc = builder.local(m, 0, '1/10')
        names = [row[0] for row in qc.records]
        assert names[0].startswith('compose:B') and names[0].endswith('†')
        assert names[-2].startswith('compose:B') and not names[-2].endswith('†')
        rzz = [row for row in qc.records if row[0] == 'c-rzz']
        assert len(rzz) == 1 and rzz[0][2] == (2, 0, 1)
        assert rzz[0][1] == pytest.approx(-.02)
        assert [row[1] for row in qc.records if row[0]=='p'] == pytest.approx([-.07,-.0375])
        frame_qc = builder.local(m, 1, '1/10')
        assert [row[1] for row in frame_qc.records if row[0]=='p'] == pytest.approx([-.07, 0.])
        assert gaussian.call_count == 2


def test_identical_basis_sharing_fuses_only_adjacent_runs_and_keeps_tiny_angles():
    with bridges() as (ops, gaussian):
        m = model(ops, same=True); seq = native.ordered_schedule(2, '1/10')
        assert len(native.runs(m, seq, optimized=False)) == 3
        assert len(native.runs(m, seq, optimized=True)) == 1
        d = native.combined_diagonal(m, native.runs(m, seq, optimized=True)[0])
        assert d.rz[0][1] == pytest.approx(.06)
        qc = native.NativeBuilder().ordered(m, '1/10', optimized=True)
        assert len([r for r in qc.records if r[0].startswith('compose:')]) == 2
        assert gaussian.call_count == 1
        tiny = ops.DiagonalEvolutionPrimitives(0., ((0,2.**-60),), ())
        m = native.NativeModel('tiny',(native.NativeBlock('one',m.blocks[0].rotation,tiny),),0.,'s','i')
        assert native.combined_diagonal(m,((0,Fraction(1)),)).rz == ((0,2.**-60),)


def test_distinct_basis_is_a_fence_and_basis_cache_keeps_provenance():
    with bridges() as (ops, gaussian):
        m = model(ops); seq = native.ordered_schedule(2, '1/10')
        assert len(native.runs(m,seq,optimized=True)) == 3
        builder=native.NativeBuilder();builder.ordered(m,'1/10',optimized=True)
        assert gaussian.call_count == 2
        other=native.NativeModel('other',m.blocks,m.constant,'other_source',m.input_identity)
        builder.ordered(other,'1/10',optimized=True)
        assert gaussian.call_count == 4


def test_actual_adjoint_and_existing_measured_XY_wrapper_dispatch():
    with bridges() as (ops, _):
        m = model(ops); qc = native.NativeBuilder().ordered(m,'1/10',optimized=False)
        assert native.actual_adjoint(qc).records == [('actual_inverse_of',qc.name,())]
        x=native.wrapper(qc,m,'1/10','X',optimized=False,occupied=(0,1))
        y=native.wrapper(qc,m,'1/10','Y',optimized=False,occupied=(0,1))
        assert [v[0] for v in x.records] == ['x','x','h','compose:circuit','h','measure']
        assert [v[0] for v in y.records] == ['x','x','h','compose:circuit','sdg','h','measure']
        assert y.records[-1][2] == (m.modes,0)
        with pytest.raises(ValueError): native.wrapper(qc,m,'1/10','X',optimized=False,occupied=(0,0))


def test_cdf_and_xdf_preparation_reuse_existing_diag_and_signed_square_primitives():
    from trottertracks import c1_general_k_adapter as adapter
    with bridges() as (ops, gaussian):
        decomp=importlib.import_module('trotterlib.df_trotter.decompose')
        diag=Mock(side_effect=lambda h, **kw: (np.eye(len(h)),np.diag(h)))
        with patch.object(decomp,'diag_hermitian',diag):
            m=adapter.from_ffsim_number_arrays(one_body_tensor=((.2,0.),(0.,-.3)),
                diag_coulomb_mats=(((2.,-4.),(-4.,6.)),),orbital_rotations=(((1.,0.),(0.,1.)),),
                constant=.375,z_representation=False,frame_ids=('F0',),group_ids=('G0',),
                source_identity='s',input_identity='i',one_body_role='ffsim_corrected_kappa',enable_general_k=True)
            cdf=native.prepare_cdf(m)
            xdf=native.prepare_xdf(np.diag([.2,-.3]),[-.7],[np.diag([.4,-.2])],.375,
                source_identity='x',input_identity='i')
            assert [b.block_id for b in cdf.blocks] == ['one_body','F0']
            assert len(cdf.blocks[0].rotation) == 4  # interleaved spin lift
            expected=ops.df_squared_diagonal_primitives(np.array([.4,-.2]),-.7,1.)
            assert xdf.blocks[1].diagonal == expected
            assert diag.call_count == 3 and gaussian.call_count == 0
            assert all(c.kwargs == {'assume_hermitian':True} for c in diag.call_args_list)


def test_artificial_original_CAR_reference_does_not_use_K_or_eigendecomposition():
    from trottertracks import c1_general_k_adapter as adapter
    b=np.array([[[[0.,0.],[0.,1.]]]])
    with patch.object(adapter,'general_k_coefficients',side_effect=AssertionError('candidate only')):
        assert np.array_equal(native.xdf_car_generator([[2.]],None,b),[[0.,0.],[0.,2.]])
        assert np.array_equal(native.xdf_car_generator([[2.]],-.5,b),[[0.,0.],[0.,-2.]])


def test_artificial_rank8_rank10_connections_and_complex_spin_order():
    from trottertracks import c1_general_k_adapter as adapter
    with bridges() as (ops, gaussian):
        decomp=importlib.import_module('trotterlib.df_trotter.decompose')
        diag=Mock(side_effect=lambda h, **kw: (np.eye(len(h)),np.diag(h)))
        with patch.object(decomp,'diag_hermitian',diag):
            U=np.eye(4,dtype=complex);U[0,0]=U[1,1]=0;U[0,1]=U[1,0]=1j
            m=adapter.from_ffsim_number_arrays(one_body_tensor=np.diag([.1,.2,.3,.4]),
                diag_coulomb_mats=np.zeros((8,4,4)),orbital_rotations=np.repeat(U[None],8,axis=0),
                constant=.375,z_representation=False,frame_ids=tuple(f'F{i}' for i in range(8)),
                group_ids=tuple(f'G{i}' for i in range(8)),source_identity='ARTIFICIAL',input_identity='ARTIFICIAL',
                one_body_role='ffsim_corrected_kappa',enable_general_k=True)
            cdf=native.prepare_cdf(m)
            xdf=native.prepare_xdf(np.eye(8),np.ones(10),np.repeat(np.eye(8)[None],10,axis=0),.375,
                source_identity='ARTIFICIAL_XDF',input_identity='ARTIFICIAL')
            assert cdf.modes==xdf.modes==8
            assert len(cdf.blocks)==9 and len(xdf.blocks)==11
            assert cdf.blocks[1].rotation[0][2]==1j and cdf.blocks[1].rotation[0][3]==0j
            assert diag.call_count==12 and gaussian.call_count==0


class BackendDouble:
    def __init__(self): self.calls=[]
    def __getattr__(self, key):
        def call(*args):
            self.calls.append((key,args))
            return (key,len(self.calls))
        return call


def proposal():
    return dict(total_time='1/10',local_times=['1/20','1/10'],product_formula='ordered_symmetric_S2_one_step')


def test_full_mock_validation_order_and_all_call_categories():
    backend=BackendDouble();counts=Counter();events=[]
    def call(kind,label,fn): counts[kind]+=1;events.append(kind);return fn()
    r=validation.validate(backend,proposal(),call)
    assert r['mandatory_stop'] and r['new_fit']==0
    assert counts == {k:v for k,v in validation.call_limits().items() if k in counts}
    assert counts['local_reference']==counts['local_controlled_check']==40
    assert counts['compile']==counts['metrics']==12
    assert max(i for i,k in enumerate(events) if k=='local_controlled_check') < events.index('ordered_build')
    assert max(i for i,k in enumerate(events) if k=='raw_signal_pair_check') < events.index('compile')
    assert sum(validation.call_limits().values()) == 243


@pytest.mark.parametrize('fail_kind', ['local_controlled_check','ordered_controlled_check','adjoint_check','raw_signal_pair_check','compiled_controlled_check'])
def test_fail_fast_has_no_dependent_calls_or_retry(fail_kind):
    events=[]
    def call(kind,label,fn):
        events.append(kind)
        if kind == fail_kind:raise RuntimeError('artificial gate failure')
        return fn()
    with pytest.raises(RuntimeError):validation.validate(BackendDouble(),proposal(),call)
    assert events.count(fail_kind)==1
    if fail_kind=='local_controlled_check': assert 'ordered_build' not in events
    if fail_kind!='compiled_controlled_check': assert 'compile' not in events
    else: assert 'metrics' not in events


def test_output_control_checks_source_drift_without_reading_permission_files(tmp_path):
    output=tmp_path/execution.FAMILY/'ARTIFICIAL'
    with patch.object(execution.subprocess,'check_output',return_value=b'source.py\n'),patch.object(
            execution.io,'read',side_effect=AssertionError('No permission file is read')):
        with pytest.raises(ValueError,match='SOURCE_DRIFT'):execution.check_output({'root':tmp_path},output)


def test_execution_rechecks_preflight_before_numeric_dispatch():
    with patch.object(execution,'preflight',side_effect=ValueError('artificial source drift')), patch.object(
            execution,'check_output') as output,patch.object(execution,'runtime') as runtime:
        with pytest.raises(ValueError):execution.execute(dict(root='r',contract_path='c',input='i'),'o')
        output.assert_not_called();runtime.assert_not_called()


def test_scientific_output_control_requires_task_namespace_and_fresh_output(tmp_path):
    output=tmp_path/execution.FAMILY/'ARTIFICIAL'
    with patch.object(execution.subprocess,'check_output',return_value=b''):
        execution.check_output({'root':tmp_path},output)
        with pytest.raises(ValueError,match='FRESH'):execution.check_output({'root':tmp_path},tmp_path/'wrong_namespace')
        output.mkdir(parents=True)
        with pytest.raises(ValueError,match='FRESH'):execution.check_output({'root':tmp_path},output)


@pytest.mark.parametrize('error', [ValueError('ARTIFICIAL_LOAD_BLOCK'), KeyboardInterrupt()])
def test_exception_preserves_prefix_ledger_claim_and_STOP_without_retry(tmp_path,error):
    v=dict(root=tmp_path,contract_path=tmp_path/'c',input=tmp_path/'ARTIFICIAL',
           contract=dict(call_limits=validation.call_limits()),source={'artificial':True})
    output=tmp_path/'fresh'
    with patch.object(execution,'preflight',return_value=v),patch.object(execution,'check_output'),patch.object(execution,'runtime',return_value={'artificial':True}),patch.object(np,'load',side_effect=error),patch.object(native,'prepare_cdf') as prepare,patch.object(validation,'validate') as validate:
        with pytest.raises(type(error)):execution.execute(v,output)
        prepare.assert_not_called();validate.assert_not_called()
    assert json.loads((output/'STOP.json').read_text())['mandatory_stop']
    assert json.loads((output/'failure.json').read_text())['call_counts']=={'input_load':1}
    assert len((output/'science_calls.jsonl').read_text().splitlines())==1
    assert (output/'evaluation.claim').exists() and not (output/'result.json').exists()


def test_real_backend_delegates_existing_operator_signal_compiler_and_metrics_APIs():
    from dataclasses import asdict
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    from trottertracks.pr_dictionary_validation import gates
    compiler=asdict(CompilerProfile())
    p=dict(compiler=compiler,operator_atol=1e-10,signal_atol=2e-10,occupied=[0])
    saved=[]
    with bridges() as (ops,_):
        backend=validation.QiskitBackend({}, {}, p, mark=Mock(),keep=lambda *v:saved.append(v))
        qc=CircuitDouble(3);target=np.eye(4)
        with patch.object(gates,'controlled_gate',return_value={'label':'artificial'}) as check:
            backend.controlled_check(qc,target,'local')
            check.assert_called_once_with(qc,target,1e-10)
        with patch.object(gates,'signal_gate',return_value={'error':0.}) as signal:
            backend.signal_check([.2,-.3],target,'XY')
            signal.assert_called_once_with(.2,-.3,1.,2e-10)
        with patch.object(gates,'compiled_wrapper',return_value=qc) as compile:
            assert backend.compile(qc) is qc
            compile.assert_called_once_with(qc,backend.profile)
        metric_module=importlib.import_module('trottertracks.pr_dictionary_validation.f2_execution')
        with patch.object(metric_module,'circuit_metrics',return_value={'rz_count':3}) as metrics:
            assert backend.metrics(qc)=={'rz_count':3}
            metrics.assert_called_once_with(qc)
        assert saved==[('compiled_0',qc)]
