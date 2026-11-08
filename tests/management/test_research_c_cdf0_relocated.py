"""Invented arrays and mock orchestration only; no real-H4 evaluation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import c_cdf0_relocated as cdf0


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def invented_contract(tmp_path):
    paths = dict(path='candidate.bin', reference='reference.bin', ground='ground.bin',
                 physical_record='physical.json', B1_contract='B1.json', stage_A_result='A.json')
    for name in ('candidate.bin', 'reference.bin', 'ground.bin', 'original.bin'):
        (tmp_path / name).write_bytes(('invented:' + name).encode())
    candidate = cdf0.sha(tmp_path / 'candidate.bin')
    original = cdf0.sha(tmp_path / 'original.bin')
    write(tmp_path / 'B1.json', dict(final_exports=dict(baseline='candidate.bin'),
          saved_inputs={'candidate.bin': candidate, 'original.bin': original}, stage_A_input='original.bin'))
    write(tmp_path / 'physical.json', dict(actual_rank=8, sector_error=1e-9, numerical_guard=2e-9))
    write(tmp_path / 'A.json', dict(eligible=dict(baseline=True),
          sector=dict(baseline=cdf0.read(tmp_path / 'physical.json'))))
    write(tmp_path / 'SOURCE.json', dict(files={}))
    c = dict(scope=cdf0.SCOPE, candidate=dict(paths, id=cdf0.CANDIDATE_ID, gamma=0, sha256=candidate),
             policy=copy.deepcopy(cdf0.POLICY), inputs={p: cdf0.sha(tmp_path / p) for p in paths.values()},
             SOURCE='SOURCE.json', SOURCE_sha256=cdf0.sha(tmp_path / 'SOURCE.json'),
             input_identity=original, original_input='original.bin', packages={},
             fresh_output='science/run01', retry=0, resume=0, mandatory_stop=True,
             scientific_execution_permission=False)
    c['inputs']['original.bin'] = original
    write(tmp_path / 'proposal.json', c)
    return c


def test_passive_import_does_not_import_numerics():
    code = "import sys; sys.path.insert(0,'src'); from trottertracks import c_cdf0_relocated; assert not any(k in sys.modules for k in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


def test_preflight_only_reads_bytes_and_metadata(tmp_path):
    c = invented_contract(tmp_path)
    assert cdf0.preflight(tmp_path, 'proposal.json') == c
    assert not (tmp_path / c['fresh_output']).exists()


@pytest.mark.parametrize('fault', ['candidate', 'source', 'old_output', 'identity', 'policy', 'handoff', 'escape'])
def test_preflight_rejects_bad_binding_and_old_run(tmp_path, fault):
    c = invented_contract(tmp_path)
    if fault == 'candidate':
        (tmp_path / 'candidate.bin').write_bytes(b'not the gamma-zero candidate')
    elif fault == 'source':
        (tmp_path / 'SOURCE.json').write_bytes(b'changed')
    elif fault == 'old_output':
        (tmp_path / c['fresh_output']).mkdir(parents=True)
    elif fault == 'identity':
        c['candidate']['id'] = 'ENTRY masquerading as CDF0'
    elif fault == 'policy':
        c['policy']['K'] = 2
    elif fault == 'handoff':
        c['handoff'] = {}
    elif fault == 'escape':
        c['fresh_output'] = '../outside'
    write(tmp_path / 'proposal.json', c)
    with pytest.raises((ValueError, FileExistsError)):
        cdf0.preflight(tmp_path, 'proposal.json')


def test_reference_schema_adapter_preserves_original_arguments(tmp_path, monkeypatch):
    from trottertracks import c_h4_relocated_downstream as bridge
    seen = {}
    def prepare(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return 'relocation', 'packet', 'report'
    monkeypatch.setattr(bridge, 'prepare_candidate', prepare)
    data = dict(J=object(), U=object())
    ref = dict(H=object(), generators=object())
    kappa = object()
    value = cdf0.connect(kappa, .71, data, ref,
        dict(sector_error=1e-9, numerical_guard=2e-9),
        row=dict(sha256='invented-cdf0'), input_identity='invented-input',
        output=tmp_path / 'A', call=None, shells={}, ground=tmp_path / 'ground')
    assert value == ('relocation', 'packet', 'report')
    assert seen['args'][0] is kappa and seen['args'][1] == .71
    assert seen['args'][4]['J'] is data['J'] and seen['args'][4]['H'] is ref['H']
    assert seen['args'][5]['candidate_sha256'] == 'invented-cdf0'
    assert seen['kwargs']['role'] == 'baseline'
    assert set(ref) == {'H', 'generators'}  # no mutation of saved-reference schema


def test_invented_CDF8_schema_reaches_real_relocation_and_finite_law(tmp_path):
    import numpy as np
    from fractions import Fraction
    from test_research_c_h4_relocated_downstream import fixture
    from trottertracks import c_h4_relocated_downstream as bridge
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks.c1_molecule_extension import SectorCAR
    invented = fixture(4, 8)
    ctx = SectorCAR(4)
    before, _ = bridge.raw_generators(invented, ctx=ctx)
    reference = dict(generators=np.asarray(before), H=sum(before))
    df = dict(actual_rank=8, sector_error=0., numerical_guard=1e-12,
              guard_policy='sector_operation_scale_guard8_proposal_v1')
    rel, packet, report = cdf0.connect(invented.kappa, .31,
        dict(J=invented.arm.J, U=invented.arm.U), reference, df,
        row=dict(sha256='invented-gamma-zero-final'), input_identity='invented-input',
        output=tmp_path / 'A', call=lambda k, l, f: f(), shells={}, ground=None)
    assert report['status'] == 'PASS'
    assert packet['candidate_identity'].startswith('invented-gamma-zero-final:')
    assert packet['arm'].general is None
    np.testing.assert_allclose(sum(packet['sector']['generators']), reference['H'], atol=1e-12)
    np.testing.assert_allclose(reference['H'], reference['H'].conj().T, atol=1e-12)
    tail, config, law = b.configure(packet['arm'], Fraction(1, 10), 16, 0)
    assert bridge.tail_gate(ctx, packet['arm'], packet['sector']['generators'], tail)['status'] == 'PASS'
    assert config.rte_steps == 16


def test_failure_has_one_shot_STOP_and_preserves_old_bytes(tmp_path, monkeypatch):
    from trottertracks import c_h4_relocated_downstream as bridge
    from trottertracks.c1_h4_native_fair import profile_dict
    c = invented_contract(tmp_path)
    c['compiler'] = profile_dict()
    write(tmp_path / 'proposal.json', c)
    old = (tmp_path / 'candidate.bin').read_bytes()
    def fail(root, contract, output, call):
        assert not call.stop.is_set()
        def failure():
            raise ValueError('INVENTED_CORRECTNESS_FAILURE')
        try:
            call('invented_correctness', 'invented', failure)
        finally:
            assert call.stop.is_set()
    monkeypatch.setattr(cdf0, 'evaluate', fail)
    with pytest.raises(ValueError, match='INVENTED_CORRECTNESS_FAILURE'):
        cdf0.execute(tmp_path, 'proposal.json')
    stop = cdf0.read(tmp_path / c['fresh_output'] / 'STOP.json')
    assert stop['failed'] and stop['retry'] == stop['resume'] == 0 and stop['mandatory_stop']
    assert (tmp_path / 'candidate.bin').read_bytes() == old
    with pytest.raises(FileExistsError):
        cdf0.execute(tmp_path, 'proposal.json')


def test_mock_execution_preserves_plan_and_does_not_reselect(tmp_path, monkeypatch):
    """Exercise actual orchestration with mocked numerical engines, not saved H4."""
    from trottertracks import c_h4_relocated_downstream as bridge
    from trottertracks import c_integrated_retuning_adapter as adapter
    from trottertracks import c_pr_aware_df_stage_b_runner as loader
    from trottertracks import c_near_r32_statistics as stats
    from trottertracks import c1_stage1 as stage
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks import c1_stage1_execution as io
    from qiskit import qpy
    c = dict(candidate=dict(id=cdf0.CANDIDATE_ID, path='invented', reference='invented',
              physical_record='physical.json', ground='invented'), kappa_export='invented',
             kappa_array_identity=dict(kappa={}, constant={}), input_identity='invented',
             SOURCE='SOURCE.json', SOURCE_sha256='invented-source', shell_reports='shells.json',
             shells={'head': dict(raw='raw.qpy', compiled='compiled.qpy')},
             ENTRY_summary='E2.json', statistics=dict(samples=64, confidence=.95))
    for p in ('raw.qpy', 'compiled.qpy'):
        (tmp_path / p).write_bytes(b'invented, never decoded')
    for p, value in [('SOURCE.json', {}), ('physical.json', {}), ('shells.json', {}),
                     ('E2.json', dict(rows=['saved-proxy'], bindings=dict(ideal_exact_native={})))]:
        write(tmp_path / p, value)
    arm = object()
    a = dict(candidate_identity='invented-cdf0-relocated', arm=arm)
    monkeypatch.setattr(loader, 'load', lambda *x: dict(kappa='kappa', constant=.3))
    monkeypatch.setattr(io, 'array_record', lambda x: {})
    monkeypatch.setattr(qpy, 'load', lambda stream: [SimpleNamespace(num_qubits=9)])
    monkeypatch.setattr(b2, 'shell_assessment', lambda *x: ({}, {}))
    monkeypatch.setattr(cdf0, 'connect', lambda *x, **kw: ('rel', a, {}))
    monkeypatch.setattr(stage, 'car_bilinears', lambda *x: 'car')
    monkeypatch.setattr(bridge, 'raw_generators', lambda *x, **kw: ([], []))
    monkeypatch.setattr(bridge, 'native_context', lambda *x, **kw: dict(arm=arm))
    refs = []
    cohorts = []
    def reference(packet, r, *args, **kwargs):
        refs.append(r)
        return packet, dict(r=r, native_admitted=r != 8), {}, 'cache'
    def cohort(root, runtime, out, call, packet, result, price, *, split, **kwargs):
        cohorts.append((result['r'], split, kwargs['cohort_key']))
        # Selection picks r16; final is deliberately worse. No final reselection.
        value = 10 if result['r'] == 16 else 20
        if split == 'final':
            value = 999
        result = dict(result, lanes=dict(ideal_exact_native=dict(
            conditional_totals=dict(totals=dict(rz_count=value)))))
        return result, dict(kernel_samples=['invented'] * (32 if split == 'selection' else 64))
    monkeypatch.setattr(bridge, 'reference_r', reference)
    monkeypatch.setattr(adapter, 'native_cohort', cohort)
    monkeypatch.setattr(stats, 'affine', lambda *x: dict(ideal_exact_native={}))
    monkeypatch.setattr(stats, 'primary', lambda *x, **kw: dict(mock_only=True))
    output = tmp_path / 'output'
    output.mkdir()
    result = cdf0.evaluate(tmp_path, c, output, lambda k, l, f: f())
    assert refs == [32, 8, 16, 64, 128]
    assert result['selection']['selected_r'] == 16
    assert [(r, s) for r, s, _ in cohorts] == [(32, 'selection'), (16, 'selection'),
        (64, 'selection'), (128, 'selection'), (16, 'final')]
    assert all(key.startswith('cdf0-20261108:') for _, _, key in cohorts)
    assert result['actual_CA_achieved'] is None
